// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

/**
 * @file
 *
 * @brief Definition of basicConvolveGPU, convolveLinearCombinationGPU
 * and convolveSpatiallyInvariantGPU functions declared in ConvolveGPU.h
 *
 * @author Kresimir Cosic
 * @author Original CPU convolution code by Russell Owen
 *
 * @ingroup afw
 */

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include "boost/cstdint.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image.h"
#include "lsst/afw/math.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math/detail/Convolve.h"

#include "lsst/afw/math/detail/ConvolveGPU.h"
#include "lsst/afw/math/detail/ImageBuffer.h"
#include "lsst/afw/math/detail/cudaConvWrapper.h"

namespace pexExcept = lsst::pex::exceptions;
namespace pexLog = lsst::pex::logging;
namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace mathDetail = lsst::afw::math::detail;

namespace {

typedef mathDetail::VarPixel VarPixel;
typedef mathDetail::MskPixel MskPixel;
typedef mathDetail::KerPixel KerPixel;

bool TryToSelectCudaDevice(afwMath::ConvolutionControl const& convolutionControl)
{
    afwMath::ConvolutionControl::DeviceSelection_t devSel;
    devSel=convolutionControl.getDeviceSelection();

    if (devSel!=afwMath::ConvolutionControl::AUTO_GPU_THROW) {
        bool done=mathDetail::gpu::SelectPreferredCudaDevice();
        if (!done)
            mathDetail::gpu::AutoSelectCudaDevice();

        return true;
    }

    bool done=mathDetail::gpu::SelectPreferredCudaDevice();
    if (done) return true;

    try {
        mathDetail::gpu::AutoSelectCudaDevice();
    }
    catch(...) {
        return false;
    }
    return true;
}


// copies data from MaskedImage to three image buffers
template <typename PixelT>
void CopyFromMaskedImage(afwImage::MaskedImage<PixelT, MskPixel, VarPixel> const& image,
                         mathDetail::ImageBuffer<PixelT>& img,
                         mathDetail::ImageBuffer<VarPixel>& var,
                         mathDetail::ImageBuffer<MskPixel>& msk
                        )
{
    int width=image.getWidth();
    int height=image.getHeight();
    img.Init(width,height);
    var.Init(width,height);
    msk.Init(width,height);

    typedef typename afwImage::MaskedImage<PixelT, MskPixel, VarPixel>
    ::x_iterator x_iterator;

    //copy input image data to buffer
    for (int i = 0; i < height; ++i) {
        x_iterator inPtr = image.x_at(0, i);
        PixelT*    imageDataPtr = img.GetImgLinePtr(i);
        MskPixel*  imageMaskPtr = msk.GetImgLinePtr(i);
        VarPixel*  imageVarPtr  = var.GetImgLinePtr(i);

        for (x_iterator cnvEnd = inPtr + width; inPtr != cnvEnd;
                ++inPtr, ++imageDataPtr, ++imageMaskPtr, ++imageVarPtr ) {
            *imageDataPtr = (*inPtr).image();
            *imageMaskPtr = (*inPtr).mask();
            *imageVarPtr  = (*inPtr).variance();
        }
    }
}

// copies data from three image buffers to MaskedImage
template <typename PixelT>
void CopyToImage(afwImage::MaskedImage<PixelT, MskPixel, VarPixel>& outImage,
                 int startX, int startY,
                 const mathDetail::ImageBuffer<PixelT>& img,
                 const mathDetail::ImageBuffer<VarPixel>& var,
                 const mathDetail::ImageBuffer<MskPixel>& msk
                )
{
    assert(img.height == var.height);
    assert(img.height == msk.height);
    assert(img.width == var.width);
    assert(img.width == msk.width);

    typedef typename afwImage::MaskedImage<PixelT, MskPixel, VarPixel>
    ::x_iterator x_iterator;

    for (int i = 0; i < img.height; ++i) {
        const PixelT*    outPtrImg = img.GetImgLinePtr(i);
        const MskPixel*  outPtrMsk = msk.GetImgLinePtr(i);
        const VarPixel*  outPtrVar = var.GetImgLinePtr(i);

        for (x_iterator cnvPtr = outImage.x_at(startX, i + startY),
                cnvEnd = cnvPtr + img.width;    cnvPtr != cnvEnd;    ++cnvPtr )
        {
            *cnvPtr = typename x_iterator::type(*outPtrImg,*outPtrMsk,*outPtrVar);
            ++outPtrImg;
            ++outPtrMsk;
            ++outPtrVar;
        }
    }
}

/*
 * @brief Compute the dot product of a kernel row or column and the overlapping portion of an %image
 *
 * @return computed dot product
 *
 * The pixel computed belongs at position imageIter + kernel center.
 *
 * @note Same as kernelDotProduct in basicConvolve.cc, copy-pasted
 */
template <typename OutPixelT, typename ImageIterT, typename KernelIterT, typename KernelPixelT>
inline OutPixelT kernelDotProduct(
    ImageIterT imageIter,       ///< start of input %image that overlaps kernel vector
    KernelIterT kernelIter,     ///< start of kernel vector
    int kWidth)                 ///< width of kernel
{
    OutPixelT outPixel(0);
    for (int x = 0; x < kWidth; ++x, ++imageIter, ++kernelIter) {
        KernelPixelT kVal = *kernelIter;
        if (kVal != 0) {

            outPixel += static_cast<OutPixelT>((*imageIter) * kVal);
        }
    }
    return outPixel;
}


/*
 * Assert that the dimensions of convolvedImage, inImage and kernel are compatible with convolution.
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if convolvedImage dimensions != inImage dim.
 * @throw lsst::pex::exceptions::InvalidParameterException if inImage smaller than kernel in width or h.
 * @throw lsst::pex::exceptions::InvalidParameterException if kernel width or height < 1
 *
 * @note Same as kernelDotProduct in basicConvolve.cc, copy-pasted
 */
template <typename OutImageT, typename InImageT>
void assertDimensionsOK(
    OutImageT const &convolvedImage,
    InImageT const &inImage,
    lsst::afw::math::Kernel const &kernel
) {
    if (convolvedImage.getDimensions() != inImage.getDimensions()) {
        std::ostringstream os;
        os << "convolvedImage dimensions = ( "
        << convolvedImage.getWidth() << ", " << convolvedImage.getHeight()
        << ") != (" << inImage.getWidth() << ", " << inImage.getHeight() << ") = inImage dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    if (afwGeom::any(inImage.getDimensions().lt(kernel.getDimensions()))) {
        std::ostringstream os;
        os << "inImage dimensions = ( "
        << inImage.getWidth() << ", " << inImage.getHeight()
        << ") smaller than (" << kernel.getWidth() << ", " << kernel.getHeight()
        << ") = kernel dimensions in width and/or height";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    if ((kernel.getWidth() < 1) || (kernel.getHeight() < 1)) {
        std::ostringstream os;
        os << "kernel dimensions = ( "
        << kernel.getWidth() << ", " << kernel.getHeight()
        << ") smaller than (1, 1) in width and/or height";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
}

}   // anonymous namespace

/**
 * @brief Low-level convolution function that does not set edge pixels.
 *
 * Will use GPU for convolution in following cases:
 * - kernel is spatially invariant  (calls convolveSpatiallyInvariantGPU)
 * - kernel is a linear combination (calls convolveLinearCombinationGPU)
 * Otherwise, delegates calculation to CPU versions of basicConvolve
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * - kernel.getCtrX() along the left edge
 * - kernel.getCtrY() along the bottom edge
 * - kernel.getWidth()  - 1 - kernel.getCtrX() along the right edge
 * - kernel.getHeight() - 1 - kernel.getCtrY() along the top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterException if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterException if kernel width or height < 1
 * @throw lsst::pex::exceptions::MemoryException when allocation of either CPU or GPU memory allocation fails
 * @throw lsst::pex::exceptions::RuntimeErrorException when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
bool mathDetail::basicConvolveGPU(
    OutImageT &convolvedImage,      ///< convolved %image
    InImageT const& inImage,        ///< %image to convolve
    afwMath::Kernel const& kernel,  ///< convolution kernel
    afwMath::ConvolutionControl const& convolutionControl)  ///< convolution control parameters
{
#ifndef GPU_BUILD
    throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Afw not compiled with GPU support");
#else
    // Because convolve isn't a method of Kernel we can't always use Kernel's vtbl to dynamically
    // dispatch the correct version of basicConvolve. The case that fails is convolving with a kernel
    // obtained from a pointer or reference to a Kernel (base class), e.g. as used in LinearCombinationKernel.
    if (IS_INSTANCE(kernel, afwMath::DeltaFunctionKernel)) {
        return false;
    } else if (IS_INSTANCE(kernel, afwMath::SeparableKernel)) {
        return false;
    } else if (IS_INSTANCE(kernel, afwMath::LinearCombinationKernel) && kernel.isSpatiallyVarying()) {
        pexLog::TTrace<4>("lsst.afw.math.convolve",
                          "generic basicConvolve: dispatch to convolveLinearCombinationGPU");
        return mathDetail::convolveLinearCombinationGPU(convolvedImage, inImage,
                *dynamic_cast<afwMath::LinearCombinationKernel const*>(&kernel),
                convolutionControl);
    }

    // use brute force
    pexLog::TTrace<3>("lsst.afw.math.convolve",
                      "generic basicConvolve: dispatch to convolveSpatiallyInvariantGPU");
    return mathDetail::convolveSpatiallyInvariantGPU(convolvedImage, inImage, kernel, convolutionControl);
#endif //GPU_BUILD
}

/**
 * @brief GPU convolution for MaskedImage used when convolving a LinearCombinationKernel
 *
 * Alocates input and output image buffers, extracts data from MaskedImage into image buffers,
 * prepares all other data required for GPU convolution.
 * calls GPU_ConvolutionMI_LinearCombinationKernel to perform the convolutiuon.
 *
 * Delegates calculation to basicConvolve when:
 * - kernel is too small (less than 30 pixels
 * - there is not enough shared memory on device to hold the image blocks
 *       (image blocks are larger than kernels)
 * - kernels don't match each other
 * - there are too many kernels ( > 100)
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterException if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterException if kernel width or height < 1
 * @throw lsst::pex::exceptions::InvalidParameterException if kernel width or height < 1
 * @throw lsst::pex::exceptions::MemoryException when allocation of either CPU or GPU memory allocation fails
 * @throw lsst::pex::exceptions::RuntimeErrorException when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InPixelT>
bool mathDetail::convolveLinearCombinationGPU(
    afwImage::MaskedImage<OutPixelT, MskPixel, VarPixel>& convolvedImage,      ///< convolved %image
    afwImage::MaskedImage<InPixelT , MskPixel, VarPixel> const& inImage,        ///< %image to convolve
    afwMath::LinearCombinationKernel const& kernel,         ///< convolution kernel
    afwMath::ConvolutionControl const & convolutionControl) ///< convolution control parameters
{
#ifndef GPU_BUILD
    throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Afw not compiled with GPU support");
#else
    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef afwImage::Image<KernelPixel> KernelImage;
    typedef ImageBuffer<KernelPixel> KernelBuffer;

    if (!kernel.isSpatiallyVarying()) {
        // use the standard algorithm for the spatially invariant case
        pexLog::TTrace<3>("lsst.afw.math.convolve",
                          "convolveLinearCombinationGPU: spatially invariant; will delegate");
        return mathDetail::convolveSpatiallyInvariantGPU(convolvedImage, inImage, kernel,
                convolutionControl.getDoNormalize());
    } else {
        // refactor the kernel if this is reasonable and possible;
        // then use the standard algorithm for the spatially varying case
        afwMath::Kernel::Ptr refKernelPtr; // possibly refactored version of kernel
        if (static_cast<int>(kernel.getNKernelParameters()) > kernel.getNSpatialParameters()) {
            // refactoring will speed convolution, so try it
            refKernelPtr = kernel.refactor();
            if (!refKernelPtr) {
                refKernelPtr = kernel.clone();
            }
        } else {
            // too few basis kernels for refactoring to be worthwhile
            refKernelPtr = kernel.clone();
        }

        /*return mathDetail::convolveWithBruteForce(convolvedImage, inImage, *refKernelPtr,
                convolutionControl.getDoNormalize());
        */

        {
            if (TryToSelectCudaDevice(convolutionControl)==false)
                return false;
            afwMath::LinearCombinationKernel* newKernel;

            newKernel = dynamic_cast<afwMath::LinearCombinationKernel*> (&(*refKernelPtr));

            //get copies of basis kernels
            afwMath::KernelList kernelList=newKernel->getKernelList();
            int kernelN=newKernel->getNBasisKernels();

            //if kernel is too small, call CPU convolution
            if (newKernel->getWidth() * newKernel->getHeight() <30 &&
                    convolutionControl.getDeviceSelection()!=ConvolutionControl::FORCE_GPU)
                return false;

            //if something is wrong, call CPU convolution
            bool shMemOk = gpu::IsSufficientSharedMemoryAvailableImgAndMask(
                               newKernel->getWidth(),newKernel->getHeight(),sizeof(KerPixel));
            if (!shMemOk) {
                //cannot fit kernels into shared memory, revert to convolution by CPU
                return false;
            }

            if (kernelN==0 || kernelN>100)
                return false;

            for (int i=0; i<kernelN; i++)
                if (kernelList[i]->getDimensions() != newKernel->getDimensions()
                        || kernelList[i]->getCtr() != newKernel->getCtr()
                   )
                    return false;

            std::vector< KernelBuffer >  basisKernels(kernelN);
            for (int i=0; i<kernelN; i++) {
                KernelImage kernelImage(kernelList[i]->getDimensions());
                (void)kernelList[i]->computeImage(kernelImage, false);
                basisKernels[i].Init(kernelImage);
            }

            int const inImageWidth = inImage.getWidth();
            int const inImageHeight = inImage.getHeight();
            int const cnvWidth = inImageWidth + 1 - newKernel->getWidth();
            int const cnvHeight = inImageHeight + 1 - newKernel->getHeight();
            int const cnvStartX = newKernel->getCtrX();
            int const cnvStartY = newKernel->getCtrY();

            std::vector<double> colPos(cnvWidth);
            std::vector<double> rowPos(cnvHeight);

            for (int i=0; i<cnvWidth; i++)
                colPos[i]=inImage.indexToPosition(i+cnvStartX, afwImage::X);

            for (int i=0; i<cnvHeight; i++)
                rowPos[i]=inImage.indexToPosition(i+cnvStartY, afwImage::Y);

            ImageBuffer<InPixelT>  inBufImg;
            ImageBuffer<VarPixel>  inBufVar;
            ImageBuffer<MskPixel>  inBufMsk;

            CopyFromMaskedImage(inImage, inBufImg, inBufVar, inBufMsk);

            ImageBuffer<OutPixelT> outBufImg(cnvWidth, cnvHeight);
            ImageBuffer<VarPixel>  outBufVar(cnvWidth, cnvHeight);
            ImageBuffer<MskPixel>  outBufMsk(cnvWidth, cnvHeight);

            GPU_ConvolutionMI_LinearCombinationKernel<OutPixelT,InPixelT>(
                inBufImg, inBufVar, inBufMsk,
                colPos, rowPos,
                newKernel->getSpatialFunctionList(),
                outBufImg, outBufVar, outBufMsk,
                basisKernels,
                convolutionControl.getDoNormalize()
            );

            CopyToImage(convolvedImage,cnvStartX,cnvStartY,
                        outBufImg, outBufVar, outBufMsk);
        }
    }
    return true;
#endif //GPU_BUILD
}

/**
 * @brief GPU convolution for Image class used when convolving a LinearCombinationKernel
 *
 * Alocates input and output image buffers, extracts data from Image into image buffers,
 * prepares all other data required for GPU convolution.
 * calls GPU_ConvolutionImage_LinearCombinationKernel to perform the convolutiuon.
 *
 * Delegates calculation to basicConvolve when:
 * - kernel is too small (less than 30 pixels
 * - there is not enough shared memory on device to hold the image blocks
 *       (image blocks are larger than kernels)
 * - kernels don't match each other
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterException if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterException if kernel width or height < 1
 * @throw lsst::pex::exceptions::InvalidParameterException if kernel width or height < 1
 * @throw lsst::pex::exceptions::MemoryException when allocation of either CPU or GPU memory allocation fails
 * @throw lsst::pex::exceptions::RuntimeErrorException when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InPixelT>
bool mathDetail::convolveLinearCombinationGPU(
    afwImage::Image<OutPixelT>& convolvedImage,      ///< convolved %image
    afwImage::Image<InPixelT > const& inImage,        ///< %image to convolve
    afwMath::LinearCombinationKernel const& kernel,         ///< convolution kernel
    afwMath::ConvolutionControl const & convolutionControl) ///< convolution control parameters
{
#ifndef GPU_BUILD
    throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Afw not compiled with GPU support");
#else
    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef afwImage::Image<KernelPixel> KernelImage;
    typedef ImageBuffer<KernelPixel> KernelBuffer;

    if (!kernel.isSpatiallyVarying()) {
        // use the standard algorithm for the spatially invariant case
        pexLog::TTrace<3>("lsst.afw.math.convolve",
                          "convolveLinearCombinationGPU: spatially invariant; delegate");
        return mathDetail::convolveSpatiallyInvariantGPU(convolvedImage, inImage, kernel,
                convolutionControl.getDoNormalize());
    } else {

        if (TryToSelectCudaDevice(convolutionControl)==false)
            return false;

        // refactor the kernel if this is reasonable and possible;
        // then use the standard algorithm for the spatially varying case
        afwMath::Kernel::Ptr refKernelPtr; // possibly refactored version of kernel
        if (static_cast<int>(kernel.getNKernelParameters()) > kernel.getNSpatialParameters()) {
            // refactoring will speed convolution, so try it
            refKernelPtr = kernel.refactor();
            if (!refKernelPtr) {
                refKernelPtr = kernel.clone();
            }
        } else {
            // too few basis kernels for refactoring to be worthwhile
            refKernelPtr = kernel.clone();
        }

        //return mathDetail::convolveWithBruteForce(convolvedImage, inImage, *refKernelPtr,
        //        convolutionControl.getDoNormalize());

        {
            afwMath::LinearCombinationKernel* newKernel;

            newKernel = dynamic_cast<afwMath::LinearCombinationKernel*> (&(*refKernelPtr));

            //get copies of basis kernels
            afwMath::KernelList kernelList=newKernel->getKernelList();
            int kernelN=newKernel->getNBasisKernels();

            //if kernel is too small, call CPU convolution
            if (newKernel->getWidth() * newKernel->getHeight() <20 &&
                    convolutionControl.getDeviceSelection()!=ConvolutionControl::FORCE_GPU)
                return false;

            //if something is wrong, call CPU convolution
            bool shMemOk = gpu::IsSufficientSharedMemoryAvailable(
                               newKernel->getWidth(),newKernel->getHeight(),sizeof(KerPixel));
            if (!shMemOk) {
                //cannot fit kernels into shared memory, revert to convolution by CPU
                return false;
            }

            if (kernelN==0)
                return false;

            for (int i=0; i<kernelN; i++)
                if (kernelList[i]->getDimensions() != newKernel->getDimensions()
                        || kernelList[i]->getCtr() != newKernel->getCtr()
                   )
                    return false;

            std::vector< KernelBuffer >  basisKernels(kernelN);
            for (int i=0; i<kernelN; i++) {
                KernelImage kernelImage(kernelList[i]->getDimensions());
                (void)kernelList[i]->computeImage(kernelImage, false);
                basisKernels[i].Init(kernelImage);
            }

            int const inImageWidth = inImage.getWidth();
            int const inImageHeight = inImage.getHeight();
            int const cnvWidth = inImageWidth + 1 - newKernel->getWidth();
            int const cnvHeight = inImageHeight + 1 - newKernel->getHeight();
            int const cnvStartX = newKernel->getCtrX();
            int const cnvStartY = newKernel->getCtrY();

            std::vector<double> colPos(cnvWidth);
            std::vector<double> rowPos(cnvHeight);

            for (int i=0; i<cnvWidth; i++)
                colPos[i]=inImage.indexToPosition(i+cnvStartX, afwImage::X);

            for (int i=0; i<cnvHeight; i++)
                rowPos[i]=inImage.indexToPosition(i+cnvStartY, afwImage::Y);

            ImageBuffer<InPixelT>  inBuf(inImage);
            ImageBuffer<OutPixelT> outBuf(cnvWidth, cnvHeight);

            GPU_ConvolutionImage_LinearCombinationKernel<OutPixelT,InPixelT>(
                inBuf, colPos, rowPos,
                newKernel->getSpatialFunctionList(),
                outBuf,
                basisKernels,
                convolutionControl.getDoNormalize()
            );

            outBuf.CopyToImage(convolvedImage,cnvStartX,cnvStartY);
        }
    }
    return true;
#endif //GPU_BUILD
}

/**
 * @brief Convolve an Image with a spatially invariant Kernel.
 *
 * Alocates input and output image buffers, extracts data from Image into image buffers,
 * prepares all other data required for GPU convolution.
 * calls GPU_ConvolutionImage_SpatiallyInvariantKernel to perform the convolutiuon.
 *
* Delegates calculation to basicConvolve when:
 * - kernel is too small (less than 20 pixels)
 * - there is not enough shared memory on device to hold the image blocks
 *       (image blocks are larger than kernels)
 *
 * @warning Low-level convolution function that does not set edge pixels.
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * - kernel.getCtrX() along the left edge
 * - kernel.getCtrY() along the bottom edge
 * - kernel.getWidth()  - 1 - kernel.getCtrX() along the right edge
 * - kernel.getHeight() - 1 - kernel.getCtrY() along the top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterException if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterException if kernel width or height < 1
 * @throw lsst::pex::exceptions::MemoryException when allocation of either CPU or GPU memory allocation fails
 * @throw lsst::pex::exceptions::RuntimeErrorException when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InPixelT>
bool mathDetail::convolveSpatiallyInvariantGPU(
    afwImage::Image<OutPixelT>& convolvedImage,      ///< convolved %image
    afwImage::Image<InPixelT > const& inImage,        ///< %image to convolve
    afwMath::Kernel const& kernel,  ///< convolution kernel
    afwMath::ConvolutionControl const & convolutionControl) ///< convolution control parameters
{
#ifndef GPU_BUILD
    throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Afw not compiled with GPU support");
#else
    bool doNormalize=convolutionControl.getDoNormalize();

    if (TryToSelectCudaDevice(convolutionControl)==false)
        return false;

    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef afwImage::Image<KernelPixel> KernelImage;
    typedef typename KernelImage::const_x_iterator KernelXIterator;
    typedef typename KernelImage::const_xy_locator KernelXYLocator;

    if (kernel.isSpatiallyVarying())
        return false;

    assertDimensionsOK(convolvedImage, inImage, kernel);

    int const inImageWidth = inImage.getWidth();
    int const inImageHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    int const cnvWidth = inImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = inImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();

    KernelImage kernelImage(kernel.getDimensions());
    KernelXYLocator const kernelLoc = kernelImage.xy_at(0,0);

    pexLog::TTrace<5>("lsst.afw.math.convolve",
                      "convolveSpatiallyInvariantGPU: plain Image, kernel is spatially invariant");
    (void)kernel.computeImage(kernelImage, doNormalize);

    typedef afwImage::Image<InPixelT  > InImageT;
    typedef afwImage::Image<OutPixelT > OutImageT;

    bool shMemOk = gpu::IsSufficientSharedMemoryAvailable(kWidth,kHeight,sizeof(KerPixel));
    if (!shMemOk) {
        //cannot fit kernels into shared memory, revert to convolution by CPU
        return false;
    }
    //if kernel is too small, call CPU convolution
    if (kWidth * kHeight <25 &&
            convolutionControl.getDeviceSelection()!=ConvolutionControl::FORCE_GPU)
        return false;

    ImageBuffer<InPixelT>  inBuf(inImage);
    ImageBuffer<OutPixelT> outBuf(cnvWidth, cnvHeight);
    ImageBuffer<KernelPixel> kernelBuf(kernelImage);

    GPU_ConvolutionImage_SpatiallyInvariantKernel<OutPixelT,InPixelT>(inBuf,outBuf,kernelBuf);

    outBuf.CopyToImage(convolvedImage,cnvStartX,cnvStartY);
    return true;
#endif //GPU_BUILD
}

/**
 * @brief Convolve an MaskedImage with a spatially invariant Kernel.
 *
 * Alocates input and output image buffers, extracts data from Image into image buffers,
 * prepares all other data required for GPU convolution.
 * calls GPU_ConvolutionMI_SpatiallyInvariantKernel to perform the convolutiuon.
 *
* Delegates calculation to basicConvolve when:
 * - kernel is too small (less than 20 pixels)
 * - there is not enough shared memory on device to hold the image blocks
 *       (image blocks are larger than kernels)
 *
 * @warning Low-level convolution function that does not set edge pixels.
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * - kernel.getCtrX() along the left edge
 * - kernel.getCtrY() along the bottom edge
 * - kernel.getWidth()  - 1 - kernel.getCtrX() along the right edge
 * - kernel.getHeight() - 1 - kernel.getCtrY() along the top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterException if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterException if kernel width or height < 1
 * @throw lsst::pex::exceptions::MemoryException when allocation of either CPU or GPU memory allocation fails
 * @throw lsst::pex::exceptions::RuntimeErrorException when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InPixelT>
bool mathDetail::convolveSpatiallyInvariantGPU(
    afwImage::MaskedImage<OutPixelT, MskPixel, VarPixel>& convolvedImage,      ///< convolved %image
    afwImage::MaskedImage<InPixelT , MskPixel, VarPixel> const& inImage,        ///< %image to convolve
    afwMath::Kernel const& kernel,  ///< convolution kernel
    afwMath::ConvolutionControl const & convolutionControl) ///< convolution control parameters
{
#ifndef GPU_BUILD
    throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Afw not compiled with GPU support");
#else
    bool doNormalize=convolutionControl.getDoNormalize();

    typedef afwImage::MaskedImage<InPixelT  > InImageT;
    typedef afwImage::MaskedImage<OutPixelT > OutImageT;
    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef afwImage::Image<KernelPixel> KernelImage;
    typedef typename KernelImage::const_x_iterator KernelXIterator;
    typedef typename KernelImage::const_xy_locator KernelXYLocator;

    if (kernel.isSpatiallyVarying()) {
        return false;
    }

    assertDimensionsOK(convolvedImage, inImage, kernel);

    int const inImageWidth = inImage.getWidth();
    int const inImageHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    int const cnvWidth = inImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = inImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();

    if (TryToSelectCudaDevice(convolutionControl)==false)
        return false;

    bool shMemOk = gpu::IsSufficientSharedMemoryAvailable(kWidth,kHeight,sizeof(KerPixel));
    if (!shMemOk) {
        //cannot fit kernels into shared memory, revert to convolution by CPU
        return false;
    }

    //if kernel is too small, call CPU convolution
    if (kWidth * kHeight <20
            && convolutionControl.getDeviceSelection()!=ConvolutionControl::FORCE_GPU)
        return false;

    KernelImage kernelImage(kernel.getDimensions());
    KernelXYLocator const kernelLoc = kernelImage.xy_at(0,0);

    pexLog::TTrace<5>("lsst.afw.math.convolve",
                      "convolveSpatiallyInvariantGPU: MaskedImage, kernel is spatially invariant");
    (void)kernel.computeImage(kernelImage, doNormalize);

    ImageBuffer<InPixelT>  inBufImg;
    ImageBuffer<VarPixel>  inBufVar;
    ImageBuffer<MskPixel>  inBufMsk;
    CopyFromMaskedImage(inImage, inBufImg, inBufVar, inBufMsk);

    ImageBuffer<OutPixelT> outBufImg(cnvWidth, cnvHeight);
    ImageBuffer<VarPixel>  outBufVar(cnvWidth, cnvHeight);
    ImageBuffer<MskPixel>  outBufMsk(cnvWidth, cnvHeight);

    ImageBuffer<KernelPixel> kernelBuf(kernelImage);

    GPU_ConvolutionMI_SpatiallyInvariantKernel<OutPixelT,InPixelT>(
        inBufImg,inBufVar,inBufMsk,
        outBufImg,outBufVar, outBufMsk,
        kernelBuf
    );

    CopyToImage(convolvedImage,cnvStartX,cnvStartY,
                outBufImg, outBufVar, outBufMsk);
    return true;
#endif //GPU_BUILD
}

/*
 * Explicit instantiation
 */
/// \cond
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define NL /* */
// Instantiate Image or MaskedImage versions
#define INSTANTIATE_IM_OR_MI(IMGMACRO, OUTPIXTYPE, INPIXTYPE) \
    template bool mathDetail::basicConvolveGPU( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::Kernel const&, \
            afwMath::ConvolutionControl const&); NL \
    template bool mathDetail::convolveLinearCombinationGPU( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::LinearCombinationKernel const&, \
            afwMath::ConvolutionControl const&); NL \
    template bool mathDetail::convolveSpatiallyInvariantGPU( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::Kernel const&, \
            afwMath::ConvolutionControl const&);
// Instantiate both Image and MaskedImage versions
#define INSTANTIATE(OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(IMAGE,       OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(MASKEDIMAGE, OUTPIXTYPE, INPIXTYPE)

INSTANTIATE(double, double)
INSTANTIATE(double, float)
INSTANTIATE(double, int)
INSTANTIATE(double, boost::uint16_t)
INSTANTIATE(float, float)
INSTANTIATE(float, int)
INSTANTIATE(float, boost::uint16_t)
INSTANTIATE(int, int)
INSTANTIATE(boost::uint16_t, boost::uint16_t)
/// \endcond


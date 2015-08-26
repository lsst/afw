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
#include <cstdint>
#include <sstream>
#include <vector>

#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/ConvolveImage.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math/detail/ConvCpuGpuShared.h"
#include "lsst/afw/math/detail/Convolve.h"

#include "lsst/afw/math/detail/ConvolveGPU.h"
#include "lsst/afw/math/detail/convCUDA.h"
#include "lsst/afw/gpu/detail/GpuBuffer2D.h"
#include "lsst/afw/math/detail/cudaConvWrapper.h"
#include "lsst/afw/gpu/detail/CudaSelectGpu.h"
#include "lsst/afw/gpu/IsGpuBuild.h"
#include "lsst/afw/gpu/GpuExceptions.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace afwGpu = lsst::afw::gpu;
namespace mathDetail = lsst::afw::math::detail;
namespace gpuDetail = lsst::afw::gpu::detail;

namespace {

typedef mathDetail::VarPixel VarPixel;
typedef mathDetail::MskPixel MskPixel;
typedef mathDetail::KerPixel KerPixel;

// copies data from MaskedImage to three image buffers
template <typename PixelT>
void CopyFromMaskedImage(afwImage::MaskedImage<PixelT, MskPixel, VarPixel> const& image,
                         gpuDetail::GpuBuffer2D<PixelT>& img,
                         gpuDetail::GpuBuffer2D<VarPixel>& var,
                         gpuDetail::GpuBuffer2D<MskPixel>& msk
                        )
{
    int width = image.getWidth();
    int height = image.getHeight();
    img.Init(width, height);
    var.Init(width, height);
    msk.Init(width, height);

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
                 const gpuDetail::GpuBuffer2D<PixelT>& img,
                 const gpuDetail::GpuBuffer2D<VarPixel>& var,
                 const gpuDetail::GpuBuffer2D<MskPixel>& msk
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
            *cnvPtr = typename x_iterator::type(*outPtrImg, *outPtrMsk, *outPtrVar);
            ++outPtrImg;
            ++outPtrMsk;
            ++outPtrVar;
        }
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
 * @throw lsst::pex::exceptions::InvalidParameterError if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterError if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterError if kernel width or height < 1
 * @throw lsst::afw::gpu::GpuMemoryError when allocation or transfer to/from GPU memory fails
 * @throw lsst::pex::exceptions::MemoryError when allocation of CPU memory fails
 * @throw lsst::afw::gpu::GpuRuntimeError when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
mathDetail::ConvolveGpuStatus::ReturnCode mathDetail::basicConvolveGPU(
    OutImageT &convolvedImage,      ///< convolved %image
    InImageT const& inImage,        ///< %image to convolve
    afwMath::Kernel const& kernel,  ///< convolution kernel
    afwMath::ConvolutionControl const& convolutionControl)  ///< convolution control parameters
{
    if (!afwGpu::isGpuBuild()) {
        throw LSST_EXCEPT(afwGpu::GpuRuntimeError, "Afw not compiled with GPU support");
    }

    // Because convolve isn't a method of Kernel we can't always use Kernel's vtbl to dynamically
    // dispatch the correct version of basicConvolve. The case that fails is convolving with a kernel
    // obtained from a pointer or reference to a Kernel (base class), e.g. as used in LinearCombinationKernel.
    if (IS_INSTANCE(kernel, afwMath::DeltaFunctionKernel)) {
        return mathDetail::ConvolveGpuStatus::UNSUPPORTED_KERNEL;
    } else if (IS_INSTANCE(kernel, afwMath::SeparableKernel)) {
        return mathDetail::ConvolveGpuStatus::UNSUPPORTED_KERNEL;
    } else if (IS_INSTANCE(kernel, afwMath::LinearCombinationKernel) && kernel.isSpatiallyVarying()) {
        LOGF_TRACE4("lsst.afw.math.convolve",
                          "generic basicConvolve (GPU): dispatch to convolveLinearCombinationGPU");
        return mathDetail::convolveLinearCombinationGPU(convolvedImage, inImage,
                *dynamic_cast<afwMath::LinearCombinationKernel const*>(&kernel),
                convolutionControl);
    }

    // use brute force
    LOGF_TRACE3("lsst.afw.math.convolve",
                      "generic basicConvolve (GPU): dispatch to convolveSpatiallyInvariantGPU");
    return mathDetail::convolveSpatiallyInvariantGPU(convolvedImage, inImage, kernel, convolutionControl);
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
 * - there are too many kernels ( > maxGpuSfCount=100)
 *
 * @throw lsst::pex::exceptions::InvalidParameterError if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterError if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterError if kernel width or height < 1
 * @throw lsst::pex::exceptions::InvalidParameterError if kernel width or height < 1
 * @throw lsst::afw::gpu::GpuMemoryError when allocation or transfer to/from GPU memory fails
 * @throw lsst::pex::exceptions::MemoryError when allocation of CPU memory fails
 * @throw lsst::afw::gpu::GpuRuntimeError when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InPixelT>
mathDetail::ConvolveGpuStatus::ReturnCode mathDetail::convolveLinearCombinationGPU(
    afwImage::MaskedImage<OutPixelT, MskPixel, VarPixel>& convolvedImage,      ///< convolved %image
    afwImage::MaskedImage<InPixelT , MskPixel, VarPixel> const& inImage,        ///< %image to convolve
    afwMath::LinearCombinationKernel const& kernel,         ///< convolution kernel
    afwMath::ConvolutionControl const & convolutionControl) ///< convolution control parameters
{
    if (!afwGpu::isGpuBuild()) {
        throw LSST_EXCEPT(afwGpu::GpuRuntimeError, "Afw not compiled with GPU support");
    }
    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef afwImage::Image<KernelPixel> KernelImage;
    typedef gpuDetail::GpuBuffer2D<KernelPixel> KernelBuffer;

    if (!kernel.isSpatiallyVarying()) {
        // use the standard algorithm for the spatially invariant case
        LOGF_TRACE3("lsst.afw.math.convolve",
                          "convolveLinearCombinationGPU: spatially invariant; will delegate");
        return mathDetail::convolveSpatiallyInvariantGPU(convolvedImage, inImage, kernel,
                convolutionControl.getDoNormalize());
    } else {
        bool throwExceptionsOn=convolutionControl.getDevicePreference() == afwGpu::USE_GPU;
        if (afwGpu::detail::TryToSelectCudaDevice(!throwExceptionsOn) == false){
            return mathDetail::ConvolveGpuStatus::NO_GPU;
        }

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
        assertDimensionsOK(convolvedImage, inImage, kernel);

        const afwMath::LinearCombinationKernel* newKernel =
            dynamic_cast<afwMath::LinearCombinationKernel*> (refKernelPtr.get());
        assert(newKernel!=NULL);

        const int kernelN = newKernel->getNBasisKernels();
        const std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn = newKernel->getSpatialFunctionList();
        if (sFn.size() < 1) {
            return mathDetail::ConvolveGpuStatus::SFN_COUNT_ERROR;
        }
        if (int(sFn.size()) != kernelN) {
            return mathDetail::ConvolveGpuStatus::SFN_COUNT_ERROR;
        }
        bool isAllCheby = true;
        for (int i = 0; i < kernelN; i++) {
            if (! IS_INSTANCE( *sFn[i], afwMath::Chebyshev1Function2<double> ) ) {
                isAllCheby = false;
            }
        }
        bool isAllPoly = true;
        for (int i = 0; i < kernelN; i++) {
            if (! IS_INSTANCE( *sFn[i], afwMath::PolynomialFunction2<double> ) ) {
                isAllPoly = false;
            }
        }

        int order = 0;
#ifdef GPU_BUILD
        SpatialFunctionType_t sfType;
#endif
        if (isAllPoly) {
            order = dynamic_cast<const afwMath::PolynomialFunction2<double>*>( sFn[0].get() ) ->getOrder();
#ifdef GPU_BUILD
            sfType = sftPolynomial;
#endif
        } else if(isAllCheby) {
            order = dynamic_cast<const afwMath::Chebyshev1Function2<double>*>( sFn[0].get() ) ->getOrder();
#ifdef GPU_BUILD
            sfType = sftChebyshev;
#endif
        } else
            return mathDetail::ConvolveGpuStatus::SFN_TYPE_ERROR;

        //get copies of basis kernels
        const afwMath::KernelList kernelList = newKernel->getKernelList();

        //if kernel is too small, call CPU convolution
        const int minKernelSize = 25;
        if (newKernel->getWidth() * newKernel->getHeight() < minKernelSize &&
                convolutionControl.getDevicePreference() != lsst::afw::gpu::USE_GPU) {
            return mathDetail::ConvolveGpuStatus::KERNEL_TOO_SMALL;
        }

        //if something is wrong, call CPU convolution
        const bool shMemOkA = IsSufficientSharedMemoryAvailable_ForImgAndMaskBlock(
                                  newKernel->getWidth(), newKernel->getHeight(), sizeof(double));
        const bool shMemOkB = IsSufficientSharedMemoryAvailable_ForSfn(order, kernelN);
        if (!shMemOkA || !shMemOkB) {
            //cannot fit kernels into shared memory, revert to convolution by CPU
            return mathDetail::ConvolveGpuStatus::KERNEL_TOO_BIG;
        }

        if (kernelN == 0 || kernelN > detail::gpu::maxGpuSfCount) {
            return mathDetail::ConvolveGpuStatus::KERNEL_COUNT_ERROR;
        }
        for (int i = 0; i < kernelN; i++) {
            if (kernelList[i]->getDimensions() != newKernel->getDimensions()
                    || kernelList[i]->getCtr() != newKernel->getCtr()
               ) {
                return mathDetail::ConvolveGpuStatus::INVALID_KERNEL_DATA;
            }
        }
        LOGF_TRACE3("lsst.afw.math.convolve",
                "MaskedImage, convolveLinearCombinationGPU: will use GPU acceleration");

        std::vector< KernelBuffer >  basisKernels(kernelN);
        for (int i = 0; i < kernelN; i++) {
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

        for (int i = 0; i < cnvWidth; i++) {
            colPos[i] = inImage.indexToPosition(i + cnvStartX, afwImage::X);
        }
        for (int i = 0; i < cnvHeight; i++) {
            rowPos[i] = inImage.indexToPosition(i + cnvStartY, afwImage::Y);
        }
        gpuDetail::GpuBuffer2D<InPixelT>  inBufImg;
        gpuDetail::GpuBuffer2D<VarPixel>  inBufVar;
        gpuDetail::GpuBuffer2D<MskPixel>  inBufMsk;

        CopyFromMaskedImage(inImage, inBufImg, inBufVar, inBufMsk);

        gpuDetail::GpuBuffer2D<OutPixelT> outBufImg(cnvWidth, cnvHeight);
        gpuDetail::GpuBuffer2D<VarPixel>  outBufVar(cnvWidth, cnvHeight);
        gpuDetail::GpuBuffer2D<MskPixel>  outBufMsk(cnvWidth, cnvHeight);

        LOGF_TRACE3("lsst.afw.math.convolve",
                "MaskedImage, convolveLinearCombinationGPU: will use GPU acceleration");

#ifdef GPU_BUILD
        GPU_ConvolutionMI_LinearCombinationKernel<OutPixelT, InPixelT>(
            inBufImg, inBufVar, inBufMsk,
            colPos, rowPos,
            sFn,
            outBufImg, outBufVar, outBufMsk,
            basisKernels,
            sfType,
            convolutionControl.getDoNormalize()
        );
#endif

        CopyToImage(convolvedImage, cnvStartX, cnvStartY,
                    outBufImg, outBufVar, outBufMsk);
    }
    return mathDetail::ConvolveGpuStatus::OK;
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
 * @throw lsst::pex::exceptions::InvalidParameterError if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterError if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterError if kernel width or height < 1
 * @throw lsst::pex::exceptions::InvalidParameterError if kernel width or height < 1
 * @throw lsst::afw::gpu::GpuMemoryError when allocation or transfer to/from GPU memory fails
 * @throw lsst::pex::exceptions::MemoryError when allocation of CPU memory fails
 * @throw lsst::afw::gpu::GpuRuntimeError when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InPixelT>
mathDetail::ConvolveGpuStatus::ReturnCode mathDetail::convolveLinearCombinationGPU(
    afwImage::Image<OutPixelT>& convolvedImage,      ///< convolved %image
    afwImage::Image<InPixelT > const& inImage,        ///< %image to convolve
    afwMath::LinearCombinationKernel const& kernel,         ///< convolution kernel
    afwMath::ConvolutionControl const & convolutionControl) ///< convolution control parameters
{
    if (!afwGpu::isGpuBuild()) {
        throw LSST_EXCEPT(afwGpu::GpuRuntimeError, "Afw not compiled with GPU support");
    }
    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef afwImage::Image<KernelPixel> KernelImage;
    typedef gpuDetail::GpuBuffer2D<KernelPixel> KernelBuffer;

    if (!kernel.isSpatiallyVarying()) {
        // use the standard algorithm for the spatially invariant case
        LOGF_TRACE3("lsst.afw.math.convolve",
                          "convolveLinearCombinationGPU: spatially invariant; delegate");
        return mathDetail::convolveSpatiallyInvariantGPU(convolvedImage, inImage, kernel,
                convolutionControl.getDoNormalize());
    } else {
        bool throwExceptionsOn=convolutionControl.getDevicePreference() == afwGpu::USE_GPU;
        if (afwGpu::detail::TryToSelectCudaDevice(!throwExceptionsOn) == false){
            return mathDetail::ConvolveGpuStatus::NO_GPU;
        }

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

        {
            assertDimensionsOK(convolvedImage, inImage, kernel);

            const afwMath::LinearCombinationKernel* newKernel =
                dynamic_cast<afwMath::LinearCombinationKernel*> (refKernelPtr.get());
            assert(newKernel!=NULL);

            const int kernelN = newKernel->getNBasisKernels();
            const std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn = newKernel->getSpatialFunctionList();
            if (sFn.size() < 1) {
                return mathDetail::ConvolveGpuStatus::SFN_COUNT_ERROR;
            }
            if (int(sFn.size()) != kernelN) {
                return mathDetail::ConvolveGpuStatus::SFN_COUNT_ERROR;
            }

            bool isAllCheby = true;
            for (int i = 0; i < kernelN; i++) {
                if (! IS_INSTANCE( *sFn[i], afwMath::Chebyshev1Function2<double> ) ) {
                    isAllCheby = false;
                }
            }
            bool isAllPoly = true;
            for (int i = 0; i < kernelN; i++) {
                if (! IS_INSTANCE( *sFn[i], afwMath::PolynomialFunction2<double> ) ) {
                    isAllPoly = false;
                }
            }
            if (!isAllPoly && !isAllCheby) {
                return mathDetail::ConvolveGpuStatus::SFN_TYPE_ERROR;
            }

            int order = 0;
#ifdef GPU_BUILD
            SpatialFunctionType_t sfType;
#endif
            if (isAllPoly) {
                order = dynamic_cast<const afwMath::PolynomialFunction2<double>*>( sFn[0].get() ) ->getOrder();
#ifdef GPU_BUILD
                sfType = sftPolynomial;
#endif
            } else if(isAllCheby) {
                order = dynamic_cast<const afwMath::Chebyshev1Function2<double>*>( sFn[0].get() ) ->getOrder();
#ifdef GPU_BUILD
                sfType = sftChebyshev;
#endif
            } else {
                return mathDetail::ConvolveGpuStatus::SFN_TYPE_ERROR;
            }
            //get copies of basis kernels
            const afwMath::KernelList kernelList = newKernel->getKernelList();

            //if kernel is too small, call CPU convolution
            const int minKernelSize = 20;
            if (newKernel->getWidth() * newKernel->getHeight() < minKernelSize &&
                    convolutionControl.getDevicePreference() != lsst::afw::gpu::USE_GPU) {
                return mathDetail::ConvolveGpuStatus::KERNEL_TOO_SMALL;
            }

            //if something is wrong, call CPU convolution
            const bool shMemOkA = IsSufficientSharedMemoryAvailable_ForImgBlock(
                                      newKernel->getWidth(), newKernel->getHeight(), sizeof(double));
            const bool shMemOkB = IsSufficientSharedMemoryAvailable_ForSfn(order, kernelN);
            if (!shMemOkA || !shMemOkB) {
                //cannot fit kernels into shared memory, revert to convolution by CPU
                return mathDetail::ConvolveGpuStatus::KERNEL_TOO_BIG;
            }

            if (kernelN == 0) {
                return mathDetail::ConvolveGpuStatus::KERNEL_COUNT_ERROR;
            }

            for (int i = 0; i < kernelN; i++) {
                if (kernelList[i]->getDimensions() != newKernel->getDimensions()
                        || kernelList[i]->getCtr() != newKernel->getCtr()
                   ) {
                    return mathDetail::ConvolveGpuStatus::INVALID_KERNEL_DATA;
                }
            }

            std::vector< KernelBuffer >  basisKernels(kernelN);
            for (int i = 0; i < kernelN; i++) {
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

            for (int i = 0; i < cnvWidth; i++) {
                colPos[i] = inImage.indexToPosition(i + cnvStartX, afwImage::X);
            }
            for (int i = 0; i < cnvHeight; i++) {
                rowPos[i] = inImage.indexToPosition(i + cnvStartY, afwImage::Y);
            }
            gpuDetail::GpuBuffer2D<InPixelT>  inBuf(inImage);
            gpuDetail::GpuBuffer2D<OutPixelT> outBuf(cnvWidth, cnvHeight);

            LOGF_TRACE3("lsst.afw.math.convolve",
                "plain Image, convolveLinearCombinationGPU: will use GPU acceleration");

#ifdef GPU_BUILD
            GPU_ConvolutionImage_LinearCombinationKernel<OutPixelT, InPixelT>(
                inBuf, colPos, rowPos,
                sFn,
                outBuf,
                basisKernels,
                sfType,
                convolutionControl.getDoNormalize()
            );
#endif

            outBuf.CopyToImage(convolvedImage, cnvStartX, cnvStartY);
        }
    }
    return mathDetail::ConvolveGpuStatus::OK;
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
 * @throw lsst::pex::exceptions::InvalidParameterError if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterError if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterError if kernel width or height < 1
 * @throw lsst::afw::gpu::GpuMemoryError when allocation or transfer to/from GPU memory fails
 * @throw lsst::pex::exceptions::MemoryError when allocation of CPU memory fails
 * @throw lsst::afw::gpu::GpuRuntimeError when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InPixelT>
mathDetail::ConvolveGpuStatus::ReturnCode mathDetail::convolveSpatiallyInvariantGPU(
    afwImage::Image<OutPixelT>& convolvedImage,      ///< convolved %image
    afwImage::Image<InPixelT > const& inImage,        ///< %image to convolve
    afwMath::Kernel const& kernel,  ///< convolution kernel
    afwMath::ConvolutionControl const & convolutionControl) ///< convolution control parameters
{
    if (!afwGpu::isGpuBuild()) {
        throw LSST_EXCEPT(afwGpu::GpuRuntimeError, "Afw not compiled with GPU support");
    }
    const bool doNormalize = convolutionControl.getDoNormalize();

    const bool throwExceptionsOn=convolutionControl.getDevicePreference() == afwGpu::USE_GPU;
    if (afwGpu::detail::TryToSelectCudaDevice(!throwExceptionsOn) == false){
        return mathDetail::ConvolveGpuStatus::NO_GPU;
    }

    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef afwImage::Image<KernelPixel> KernelImage;
    typedef typename KernelImage::const_x_iterator KernelXIterator;
    typedef typename KernelImage::const_xy_locator KernelXYLocator;

    if (kernel.isSpatiallyVarying()) {
        return mathDetail::ConvolveGpuStatus::UNSUPPORTED_KERNEL;
    }

    assertDimensionsOK(convolvedImage, inImage, kernel);

    const int minKernelSize = 25;

    int const inImageWidth = inImage.getWidth();
    int const inImageHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    int const cnvWidth = inImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = inImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();

    KernelImage kernelImage(kernel.getDimensions());

    LOGF_TRACE3("lsst.afw.math.convolve",
                      "convolveSpatiallyInvariantGPU: using GPU acceleration, "
                      "plain Image, kernel is spatially invariant");
    (void)kernel.computeImage(kernelImage, doNormalize);

    typedef afwImage::Image<InPixelT  > InImageT;
    typedef afwImage::Image<OutPixelT > OutImageT;

    const bool shMemOk = IsSufficientSharedMemoryAvailable_ForImgBlock(kWidth, kHeight, sizeof(double));
    if (!shMemOk) {
        //cannot fit kernels into shared memory, revert to convolution by CPU
        return mathDetail::ConvolveGpuStatus::KERNEL_TOO_BIG;
    }
    //if kernel is too small, call CPU convolution
    if (kWidth * kHeight < minKernelSize &&
            convolutionControl.getDevicePreference() != lsst::afw::gpu::USE_GPU) {
        return mathDetail::ConvolveGpuStatus::KERNEL_TOO_SMALL;
    }

    gpuDetail::GpuBuffer2D<InPixelT>  inBuf(inImage);
    gpuDetail::GpuBuffer2D<OutPixelT> outBuf(cnvWidth, cnvHeight);
    gpuDetail::GpuBuffer2D<KernelPixel> kernelBuf(kernelImage);

#ifdef GPU_BUILD
    GPU_ConvolutionImage_SpatiallyInvariantKernel<OutPixelT, InPixelT>(inBuf, outBuf, kernelBuf);
#endif
    outBuf.CopyToImage(convolvedImage, cnvStartX, cnvStartY);
    return mathDetail::ConvolveGpuStatus::OK;
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
 * @throw lsst::pex::exceptions::InvalidParameterError if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterError if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterError if kernel width or height < 1
 * @throw lsst::afw::gpu::GpuMemoryError when allocation or transfer to/from GPU memory fails
 * @throw lsst::pex::exceptions::MemoryError when allocation of CPU memory fails
 * @throw lsst::afw::gpu::GpuRuntimeError when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InPixelT>
mathDetail::ConvolveGpuStatus::ReturnCode mathDetail::convolveSpatiallyInvariantGPU(
    afwImage::MaskedImage<OutPixelT, MskPixel, VarPixel>& convolvedImage,      ///< convolved %image
    afwImage::MaskedImage<InPixelT , MskPixel, VarPixel> const& inImage,        ///< %image to convolve
    afwMath::Kernel const& kernel,  ///< convolution kernel
    afwMath::ConvolutionControl const & convolutionControl) ///< convolution control parameters
{
    if (!afwGpu::isGpuBuild()) {
        throw LSST_EXCEPT(afwGpu::GpuRuntimeError, "Afw not compiled with GPU support");
    }
    bool doNormalize = convolutionControl.getDoNormalize();

    typedef afwImage::MaskedImage<InPixelT  > InImageT;
    typedef afwImage::MaskedImage<OutPixelT > OutImageT;
    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef afwImage::Image<KernelPixel> KernelImage;
    typedef typename KernelImage::const_x_iterator KernelXIterator;
    typedef typename KernelImage::const_xy_locator KernelXYLocator;

    if (kernel.isSpatiallyVarying()) {
        return mathDetail::ConvolveGpuStatus::UNSUPPORTED_KERNEL;
    }

    assertDimensionsOK(convolvedImage, inImage, kernel);

    const int minKernelSize = 20;

    int const inImageWidth = inImage.getWidth();
    int const inImageHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    int const cnvWidth = inImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = inImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();

    const bool throwExceptionsOn=convolutionControl.getDevicePreference() == afwGpu::USE_GPU;
    if (afwGpu::detail::TryToSelectCudaDevice(!throwExceptionsOn) == false){
        return mathDetail::ConvolveGpuStatus::NO_GPU;
    }

    const bool shMemOk = IsSufficientSharedMemoryAvailable_ForImgAndMaskBlock(kWidth, kHeight, sizeof(double));
    if (!shMemOk) {
        //cannot fit kernels into shared memory, revert to convolution by CPU
        return mathDetail::ConvolveGpuStatus::KERNEL_TOO_BIG;
    }

    //if kernel is too small, call CPU convolution
    if (kWidth * kHeight < minKernelSize
            && convolutionControl.getDevicePreference() != lsst::afw::gpu::USE_GPU) {
        return mathDetail::ConvolveGpuStatus::KERNEL_TOO_SMALL;
    }

    KernelImage kernelImage(kernel.getDimensions());

    LOGF_TRACE3("lsst.afw.math.convolve",
                      "convolveSpatiallyInvariantGPU: using GPU acceleration, "
                      "MaskedImage, kernel is spatially invariant");
    (void)kernel.computeImage(kernelImage, doNormalize);

    gpuDetail::GpuBuffer2D<InPixelT>  inBufImg;
    gpuDetail::GpuBuffer2D<VarPixel>  inBufVar;
    gpuDetail::GpuBuffer2D<MskPixel>  inBufMsk;
    CopyFromMaskedImage(inImage, inBufImg, inBufVar, inBufMsk);

    gpuDetail::GpuBuffer2D<OutPixelT> outBufImg(cnvWidth, cnvHeight);
    gpuDetail::GpuBuffer2D<VarPixel>  outBufVar(cnvWidth, cnvHeight);
    gpuDetail::GpuBuffer2D<MskPixel>  outBufMsk(cnvWidth, cnvHeight);

    gpuDetail::GpuBuffer2D<KernelPixel> kernelBuf(kernelImage);
#ifdef GPU_BUILD
    GPU_ConvolutionMI_SpatiallyInvariantKernel<OutPixelT, InPixelT>(
        inBufImg, inBufVar, inBufMsk,
        outBufImg, outBufVar, outBufMsk,
        kernelBuf
    );
#endif
    CopyToImage(convolvedImage, cnvStartX, cnvStartY,
                outBufImg, outBufVar, outBufMsk);
    return mathDetail::ConvolveGpuStatus::OK;
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
    template mathDetail::ConvolveGpuStatus::ReturnCode mathDetail::basicConvolveGPU( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::Kernel const&, \
            afwMath::ConvolutionControl const&); NL \
    template mathDetail::ConvolveGpuStatus::ReturnCode mathDetail::convolveLinearCombinationGPU( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::LinearCombinationKernel const&, \
            afwMath::ConvolutionControl const&); NL \
    template mathDetail::ConvolveGpuStatus::ReturnCode mathDetail::convolveSpatiallyInvariantGPU( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::Kernel const&, \
            afwMath::ConvolutionControl const&);
// Instantiate both Image and MaskedImage versions
#define INSTANTIATE(OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(IMAGE,       OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(MASKEDIMAGE, OUTPIXTYPE, INPIXTYPE)

INSTANTIATE(double, double)
INSTANTIATE(double, float)
INSTANTIATE(double, int)
INSTANTIATE(double, std::uint16_t)
INSTANTIATE(float, float)
INSTANTIATE(float, int)
INSTANTIATE(float, std::uint16_t)
INSTANTIATE(int, int)
INSTANTIATE(std::uint16_t, std::uint16_t)
/// \endcond



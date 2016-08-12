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
 * @brief Definition of basicConvolve and convolveWithBruteForce functions declared in detail/ConvolveImage.h
 *
 * @author Russell Owen
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
#include "lsst/afw/geom.h"
#include "lsst/afw/math/detail/Convolve.h"
#include "lsst/afw/math/detail/ConvCpuGpuShared.h"
#include "lsst/afw/math/detail/ConvolveGPU.h"
#include "lsst/afw/gpu/IsGpuBuild.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace mathDetail = lsst::afw::math::detail;

namespace {

    /*
     * @brief Compute the dot product of a kernel row or column and the overlapping portion of an %image
     *
     * @return computed dot product
     *
     * The pixel computed belongs at position imageIter + kernel center.
     *
     * @todo get rid of KernelPixelT parameter if possible by not computing local variable kVal,
     * or by using iterator traits:
     *     typedef typename std::iterator_traits<KernelIterT>::value_type KernelPixel;
     * Unfortunately, in either case compilation fails with this sort of message:
\verbatim
include/lsst/afw/image/Pixel.h: In instantiation of ‘lsst::afw::image::pixel::exprTraits<boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > > >’:
include/lsst/afw/image/Pixel.h:385:   instantiated from ‘lsst::afw::image::pixel::BinaryExpr<lsst::afw::image::pixel::Pixel<int, short unsigned int, float>, boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >, std::multiplies<int>, lsst::afw::image::pixel::bitwise_or<short unsigned int>, lsst::afw::image::pixel::variance_multiplies<float> >’
src/math/ConvolveImage.cc:59:   instantiated from ‘OutPixelT<unnamed>::kernelDotProduct(ImageIterT, KernelIterT, int) [with OutPixelT = lsst::afw::image::pixel::SinglePixel<int, short unsigned int, float>, ImageIterT = lsst::afw::image::MaskedImage<int, short unsigned int, float>::const_MaskedImageIterator<boost::gil::gray32s_pixel_t*, boost::gil::gray16_pixel_t*, boost::gil::gray32f_noscale_pixel_t*>, KernelIterT = const boost::gil::gray64f_noscalec_pixel_t*]’
src/math/ConvolveImage.cc:265:   instantiated from ‘void lsst::afw::math::basicConvolve(OutImageT&, const InImageT&, const lsst::afw::math::Kernel&, bool) [with OutImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>, InImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>]’
src/math/ConvolveImage.cc:451:   instantiated from ‘void lsst::afw::math::convolve(OutImageT&, const InImageT&, const KernelT&, bool, int) [with OutImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>, InImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>, KernelT = lsst::afw::math::AnalyticKernel]’
src/math/ConvolveImage.cc:587:   instantiated from here
include/lsst/afw/image/Pixel.h:210: error: no type named ‘ImagePixelT’ in ‘struct boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >’
include/lsst/afw/image/Pixel.h:211: error: no type named ‘MaskPixelT’ in ‘struct boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >’
include/lsst/afw/image/Pixel.h:212: error: no type named ‘VariancePixelT’ in ‘struct boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >’
\endverbatim
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

    /**
     * @brief Throws exception when trying to USE_GPU without GPU support
     *
     * If GPU support was not included at compile time, USE_GPU option will cause
     * this function to throw an exception
     *
     * @throw lsst::pex::exceptions::RuntimeError when USE_GPU enabled with no GPU support
     *
     * @ingroup afw
     */
    void CheckForceGpuOnNoGpu(afwMath::ConvolutionControl const& convolutionControl)
    {
        #ifndef GPU_BUILD
        if (lsst::afw::gpu::isGpuEnabled()==true
            && convolutionControl.getDevicePreference()==lsst::afw::gpu::USE_GPU) {
            throw LSST_EXCEPT(pexExcept::RuntimeError,
                    "Gpu acceleration must be enabled at compiling for lsst::afw::gpu::USE_GPU");
        }
        #endif
    }
    /**
     * @brief Throws exception whenever trying to USE_GPU
     *
     * USE_GPU option will cause this function to throw an exception
     *
     * @throw lsst::pex::exceptions::InvalidParameterError when USE_GPU is selected
     *
     * @ingroup afw
     */
    void CheckForceGpuOnUnsupportedKernel(afwMath::ConvolutionControl const& convolutionControl)
    {
        if (lsst::afw::gpu::isGpuEnabled()==true
            && convolutionControl.getDevicePreference()==lsst::afw::gpu::USE_GPU) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterError, "Gpu can not process this type of kernel");
        }
    }

}   // anonymous namespace

/**
 * @brief Low-level convolution function that does not set edge pixels.
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
 * @throw lsst::pex::exceptions::MemoryError when allocation of CPU memory fails
 * @throw lsst::afw::gpu::GpuMemoryError when allocation or transfer to/from GPU memory fails
 * @throw lsst::afw::gpu::GpuRuntimeError when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void mathDetail::basicConvolve(
        OutImageT &convolvedImage,      ///< convolved %image
        InImageT const& inImage,        ///< %image to convolve
        afwMath::Kernel const& kernel,  ///< convolution kernel
        afwMath::ConvolutionControl const& convolutionControl)  ///< convolution control parameters
{
    // Because convolve isn't a method of Kernel we can't always use Kernel's vtbl to dynamically
    // dispatch the correct version of basicConvolve. The case that fails is convolving with a kernel
    // obtained from a pointer or reference to a Kernel (base class), e.g. as used in LinearCombinationKernel.
    if (IS_INSTANCE(kernel, afwMath::DeltaFunctionKernel)) {
        LOGL_TRACE("lsst.afw.math.convolve.basic",
            "generic basicConvolve: dispatch to DeltaFunctionKernel basicConvolve");
        mathDetail::basicConvolve(convolvedImage, inImage,
            *dynamic_cast<afwMath::DeltaFunctionKernel const*>(&kernel),
            convolutionControl);
        return;
    } else if (IS_INSTANCE(kernel, afwMath::SeparableKernel)) {
        LOGL_TRACE("lsst.afw.math.convolve.basic",
            "generic basicConvolve: dispatch to SeparableKernel basicConvolve");
        mathDetail::basicConvolve(convolvedImage, inImage,
            *dynamic_cast<afwMath::SeparableKernel const*>(&kernel),
            convolutionControl);
        return;
    } else if (IS_INSTANCE(kernel, afwMath::LinearCombinationKernel) && kernel.isSpatiallyVarying()) {
        LOGL_TRACE("lsst.afw.math.convolve.basic",
            "generic basicConvolve: dispatch to spatially varying LinearCombinationKernel basicConvolve");
        mathDetail::basicConvolve(convolvedImage, inImage,
            *dynamic_cast<afwMath::LinearCombinationKernel const*>(&kernel),
            convolutionControl);
        return;
    }
    // OK, use general (and slower) form
    if (kernel.isSpatiallyVarying() && (convolutionControl.getMaxInterpolationDistance() > 1)) {
        // use linear interpolation
        LOGL_DEBUG("lsst.afw.math.convolve.basic", "generic basicConvolve: using linear interpolation");
        mathDetail::convolveWithInterpolation(convolvedImage, inImage, kernel, convolutionControl);

    } else {
        // use brute force
        LOGL_DEBUG("lsst.afw.math.convolve.basic", "generic basicConvolve: using brute force");
        mathDetail::convolveWithBruteForce(convolvedImage, inImage, kernel,convolutionControl);
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving delta function kernels
 *
 * @throw lsst::pex::exceptions::InvalidParameterError when GPU acceleration forced
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void mathDetail::basicConvolve(
        OutImageT& convolvedImage,      ///< convolved %image
        InImageT const& inImage,        ///< %image to convolve
        afwMath::DeltaFunctionKernel const &kernel, ///< convolution kernel
        afwMath::ConvolutionControl const &convolutionControl)       ///< convolution control parameters
{
    assert (!kernel.isSpatiallyVarying());
    assertDimensionsOK(convolvedImage, inImage, kernel);

    CheckForceGpuOnUnsupportedKernel(convolutionControl);

    int const mImageWidth = inImage.getWidth(); // size of input region
    int const mImageHeight = inImage.getHeight();
    int const cnvWidth = mImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = mImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const inStartX = kernel.getPixel().getX();
    int const inStartY = kernel.getPixel().getY();

    LOGL_DEBUG("lsst.afw.math.convolve.basic", "DeltaFunctionKernel basicConvolve");

    for (int i = 0; i < cnvHeight; ++i) {
        typename InImageT::x_iterator inPtr = inImage.x_at(inStartX, i +  inStartY);
        for (typename OutImageT::x_iterator cnvPtr = convolvedImage.x_at(cnvStartX, i + cnvStartY),
                 cnvEnd = cnvPtr + cnvWidth; cnvPtr != cnvEnd; ++cnvPtr, ++inPtr){
            *cnvPtr = *inPtr;
        }
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving a LinearCombinationKernel
 *
 * The Algorithm:
 * - If the kernel is spatially varying and contains only DeltaFunctionKernels
 *   then convolves the input Image by each basis kernel in turn, solves the spatial model
 *   for that component and adds in the appropriate amount of the convolved %image.
 * - In all other cases uses normal convolution
 *
 * @throw lsst::pex::exceptions::InvalidParameterError if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterError if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterError if kernel width or height < 1
 * @throw lsst::pex::exceptions::MemoryError when allocation of CPU memory fails
 * @throw lsst::afw::gpu::GpuMemoryError when allocation or transfer to/from GPU memory fails
 * @throw lsst::afw::gpu::GpuRuntimeError when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void mathDetail::basicConvolve(
    OutImageT& convolvedImage,      ///< convolved %image
    InImageT const& inImage,        ///< %image to convolve
    afwMath::LinearCombinationKernel const& kernel,         ///< convolution kernel
    afwMath::ConvolutionControl const & convolutionControl) ///< convolution control parameters
{
    if (!kernel.isSpatiallyVarying()) {
        // use the standard algorithm for the spatially invariant case
        LOGL_DEBUG("lsst.afw.math.convolve.basic",
            "basicConvolve for LinearCombinationKernel: spatially invariant; using brute force");
        return mathDetail::convolveWithBruteForce(convolvedImage, inImage, kernel,
            convolutionControl.getDoNormalize());
    } else {
        CheckForceGpuOnNoGpu(convolutionControl);
        if (lsst::afw::gpu::isGpuBuild() && lsst::afw::gpu::isGpuEnabled()==true) {
            if (convolutionControl.getDevicePreference() == lsst::afw::gpu::AUTO_WITH_CPU_FALLBACK) {
                try {
                    mathDetail::ConvolveGpuStatus::ReturnCode rc =
                               mathDetail::convolveLinearCombinationGPU(convolvedImage,inImage,kernel,
                                                                                convolutionControl);
                    if (rc == mathDetail::ConvolveGpuStatus::OK) return;
                } catch(lsst::afw::gpu::GpuMemoryError) { }
                catch(pexExcept::MemoryError) { }
                catch(lsst::afw::gpu::GpuRuntimeError) { }
            } else if (convolutionControl.getDevicePreference() != lsst::afw::gpu::USE_CPU) {
                mathDetail::ConvolveGpuStatus::ReturnCode rc =
                              mathDetail::convolveLinearCombinationGPU(convolvedImage,inImage,kernel,
                                                                            convolutionControl);
                if (rc == mathDetail::ConvolveGpuStatus::OK) return;
                if (convolutionControl.getDevicePreference() == lsst::afw::gpu::USE_GPU) {
                    throw LSST_EXCEPT(pexExcept::RuntimeError, "Gpu will not process this kernel");
                }
            }
        }

        // refactor the kernel if this is reasonable and possible;
        // then use the standard algorithm for the spatially varying case
        PTR(afwMath::Kernel) refKernelPtr; // possibly refactored version of kernel
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
        if (convolutionControl.getMaxInterpolationDistance() > 1) {
            LOGL_DEBUG("lsst.afw.math.convolve.basic",
                "basicConvolve for LinearCombinationKernel: using interpolation");
            return mathDetail::convolveWithInterpolation(convolvedImage, inImage, *refKernelPtr, convolutionControl);
        } else {
            LOGL_DEBUG("lsst.afw.math.convolve.basic",
                "basicConvolve for LinearCombinationKernel: maxInterpolationError < 0; using brute force");
            return mathDetail::convolveWithBruteForce(convolvedImage, inImage, *refKernelPtr,
                convolutionControl.getDoNormalize());
        }
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving separable kernels
 *
 * @throw lsst::pex::exceptions::InvalidParameterError when GPU acceleration forced
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void mathDetail::basicConvolve(
        OutImageT& convolvedImage,      ///< convolved %image
        InImageT const& inImage,        ///< %image to convolve
        afwMath::SeparableKernel const &kernel, ///< convolution kernel
        afwMath::ConvolutionControl const & convolutionControl) ///< convolution control parameters
{
    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef typename std::vector<KernelPixel> KernelVector;
    typedef KernelVector::const_iterator KernelIterator;
    typedef typename InImageT::const_x_iterator InXIterator;
    typedef typename InImageT::const_xy_locator InXYLocator;
    typedef typename OutImageT::x_iterator OutXIterator;
    typedef typename OutImageT::y_iterator OutYIterator;
    typedef typename OutImageT::SinglePixel OutPixel;

    assertDimensionsOK(convolvedImage, inImage, kernel);

    CheckForceGpuOnUnsupportedKernel(convolutionControl);

    afwGeom::Box2I const fullBBox = inImage.getBBox(image::LOCAL);
    afwGeom::Box2I const goodBBox = kernel.shrinkBBox(fullBBox);

    KernelVector kernelXVec(kernel.getWidth());
    KernelVector kernelYVec(kernel.getHeight());

    if (kernel.isSpatiallyVarying()) {
        LOGL_DEBUG("lsst.afw.math.convolve.basic",
            "SeparableKernel basicConvolve: kernel is spatially varying");

        for (int cnvY = goodBBox.getMinY(); cnvY <= goodBBox.getMaxY(); ++cnvY) {
            double const rowPos = inImage.indexToPosition(cnvY, afwImage::Y);

            InXYLocator inImLoc = inImage.xy_at(0, cnvY - goodBBox.getMinY());
            OutXIterator cnvXIter = convolvedImage.row_begin(cnvY) + goodBBox.getMinX();
            for (int cnvX = goodBBox.getMinX(); cnvX <= goodBBox.getMaxX();
                ++cnvX, ++inImLoc.x(), ++cnvXIter) {
                double const colPos = inImage.indexToPosition(cnvX, afwImage::X);

                KernelPixel kSum = kernel.computeVectors(kernelXVec, kernelYVec,
                    convolutionControl.getDoNormalize(), colPos, rowPos);

                // why does this trigger warnings? It did not in the past.
                *cnvXIter = afwMath::convolveAtAPoint<OutImageT, InImageT>(inImLoc, kernelXVec, kernelYVec);
                if (convolutionControl.getDoNormalize()) {
                    *cnvXIter = *cnvXIter/kSum;
                }
            }
        }
    } else {
        // kernel is spatially invariant
        // The basic sequence:
        // - For each output row:
        // - Compute x-convolved data: a kernel height's strip of input image convolved with kernel x vector
        // - Compute one row of output by dotting each column of x-convolved data with the kernel y vector
        // The x-convolved data is stored in a kernel-height by good-width buffer.
        // This is circular buffer along y (to avoid shifting pixels before setting each new row);
        // so for each new row the kernel y vector is rotated to match the order of the x-convolved data.

        LOGL_DEBUG("lsst.afw.math.convolve.basic",
            "SeparableKernel basicConvolve: kernel is spatially invariant");

        kernel.computeVectors(kernelXVec, kernelYVec, convolutionControl.getDoNormalize());
        KernelIterator const kernelXVecBegin = kernelXVec.begin();
        KernelIterator const kernelYVecBegin = kernelYVec.begin();

        // buffer for x-convolved data
        OutImageT buffer(afwGeom::Extent2I(goodBBox.getWidth(), kernel.getHeight()));

        // pre-fill x-convolved data buffer with all but one row of data
        int yInd = 0; // during initial fill bufY = inImageY
        int const yPrefillEnd = buffer.getHeight() - 1;
        for (; yInd < yPrefillEnd; ++yInd) {
            OutXIterator bufXIter = buffer.x_at(0, yInd);
            OutXIterator const bufXEnd = buffer.x_at(goodBBox.getWidth(), yInd);
            InXIterator inXIter = inImage.x_at(0, yInd);
            for ( ; bufXIter != bufXEnd; ++bufXIter, ++inXIter) {
                *bufXIter = kernelDotProduct<OutPixel, InXIterator, KernelIterator, KernelPixel>(
                    inXIter, kernelXVecBegin, kernel.getWidth());
            }
        }

        // compute output pixels using the sequence described above
        int inY = yPrefillEnd;
        int bufY = yPrefillEnd;
        int cnvY = goodBBox.getMinY();
        while (true) {
            // fill next buffer row and compute output row
            InXIterator inXIter = inImage.x_at(0, inY);
            OutXIterator bufXIter = buffer.x_at(0, bufY);
            OutXIterator cnvXIter = convolvedImage.x_at(goodBBox.getMinX(), cnvY);
            for (int bufX = 0; bufX < goodBBox.getWidth(); ++bufX, ++cnvXIter, ++bufXIter, ++inXIter) {
                // note: bufXIter points to the row of the buffer that is being updated,
                // whereas bufYIter points to row 0 of the buffer
                *bufXIter = kernelDotProduct<OutPixel, InXIterator, KernelIterator, KernelPixel>(
                    inXIter, kernelXVecBegin, kernel.getWidth());

                OutYIterator bufYIter = buffer.y_at(bufX, 0);
                *cnvXIter = kernelDotProduct<OutPixel, OutYIterator, KernelIterator, KernelPixel>(
                    bufYIter, kernelYVecBegin, kernel.getHeight());
            }

            // test for done now, instead of the start of the loop,
            // to avoid an unnecessary extra rotation of the kernel Y vector
            if (cnvY >= goodBBox.getMaxY()) break;

            // update y indices, including bufY, and rotate the kernel y vector to match
            ++inY;
            bufY = (bufY + 1) % kernel.getHeight();
            ++cnvY;
            std::rotate(kernelYVec.begin(), kernelYVec.end()-1, kernelYVec.end());
        }
    }
}

/**
 * @brief Convolve an Image or MaskedImage with a Kernel by computing the kernel image
 * at every point. (If the kernel is not spatially varying then only compute it once).
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
 * @throw lsst::pex::exceptions::InvalidParameterError when GPU acceleration forced on spatially varying kernel
 * @throw lsst::pex::exceptions::MemoryError when allocation of CPU memory fails
 * @throw lsst::afw::gpu::GpuMemoryError when allocation or transfer to/from GPU memory fails
 * @throw lsst::afw::gpu::GpuRuntimeError when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void mathDetail::convolveWithBruteForce(
        OutImageT &convolvedImage,      ///< convolved %image
        InImageT const& inImage,        ///< %image to convolve
        afwMath::Kernel const& kernel,  ///< convolution kernel
        afwMath::ConvolutionControl const & convolutionControl) ///< convolution control parameters
{
    bool doNormalize=convolutionControl.getDoNormalize();

    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef afwImage::Image<KernelPixel> KernelImage;

    typedef typename KernelImage::const_x_iterator KernelXIterator;
    typedef typename KernelImage::const_xy_locator KernelXYLocator;
    typedef typename InImageT::const_x_iterator InXIterator;
    typedef typename InImageT::const_xy_locator InXYLocator;
    typedef typename OutImageT::x_iterator OutXIterator;
    typedef typename OutImageT::SinglePixel OutPixel;

    assertDimensionsOK(convolvedImage, inImage, kernel);

    int const inImageWidth = inImage.getWidth();
    int const inImageHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    int const cnvWidth = inImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = inImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const cnvEndX = cnvStartX + cnvWidth;  // end index + 1
    int const cnvEndY = cnvStartY + cnvHeight; // end index + 1

    KernelImage kernelImage(kernel.getDimensions());
    KernelXYLocator const kernelLoc = kernelImage.xy_at(0,0);

    if (kernel.isSpatiallyVarying()) {
        LOGL_TRACE("lsst.afw.math.convolve.WithBruteForce",
            "convolveWithBruteForce: kernel is spatially varying");

        CheckForceGpuOnUnsupportedKernel(convolutionControl);

        for (int cnvY = cnvStartY; cnvY != cnvEndY; ++cnvY) {
            double const rowPos = inImage.indexToPosition(cnvY, afwImage::Y);

            InXYLocator  inImLoc =  inImage.xy_at(0, cnvY - cnvStartY);
            OutXIterator cnvXIter = convolvedImage.x_at(cnvStartX, cnvY);
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++inImLoc.x(), ++cnvXIter) {
                double const colPos = inImage.indexToPosition(cnvX, afwImage::X);

                KernelPixel kSum = kernel.computeImage(kernelImage, false, colPos, rowPos);
                *cnvXIter = afwMath::convolveAtAPoint<OutImageT, InImageT>(
                    inImLoc, kernelLoc, kWidth, kHeight);
                if (doNormalize) {
                    *cnvXIter = *cnvXIter/kSum;
                }
            }
        }
    } else {
        LOGL_TRACE("lsst.afw.math.convolve.WithBruteForce",
            "convolveWithBruteForce: kernel is spatially invariant");

        CheckForceGpuOnNoGpu(convolutionControl);
        if (lsst::afw::gpu::isGpuBuild() && lsst::afw::gpu::isGpuEnabled()==true) {
            if (convolutionControl.getDevicePreference() == lsst::afw::gpu::AUTO_WITH_CPU_FALLBACK) {
                try {
                    mathDetail::ConvolveGpuStatus::ReturnCode rc =
                              mathDetail::convolveSpatiallyInvariantGPU(convolvedImage,inImage,kernel,
                                                                                 convolutionControl);
                    if (rc == mathDetail::ConvolveGpuStatus::OK) return;
                } catch(lsst::afw::gpu::GpuMemoryError) { }
                catch(pexExcept::MemoryError) { }
                catch(lsst::afw::gpu::GpuRuntimeError) { }
            } else if (convolutionControl.getDevicePreference() != lsst::afw::gpu::USE_CPU) {
                mathDetail::ConvolveGpuStatus::ReturnCode rc =
                            mathDetail::convolveSpatiallyInvariantGPU(convolvedImage,inImage,kernel,
                                                                             convolutionControl);
                if (rc == mathDetail::ConvolveGpuStatus::OK) return;
                if (convolutionControl.getDevicePreference() == lsst::afw::gpu::USE_GPU) {
                    throw LSST_EXCEPT(pexExcept::RuntimeError, "Gpu will not process this kernel");
                }
            }
        }

        (void)kernel.computeImage(kernelImage, doNormalize);

        for (int inStartY = 0, cnvY = cnvStartY; inStartY < cnvHeight; ++inStartY, ++cnvY) {
            KernelXIterator kernelXIter = kernelImage.x_at(0, 0);
            InXIterator inXIter = inImage.x_at(0, inStartY);
            OutXIterator cnvXIter = convolvedImage.x_at(cnvStartX, cnvY);
            for (int x = 0; x < cnvWidth; ++x, ++cnvXIter, ++inXIter) {
                *cnvXIter = kernelDotProduct<OutPixel, InXIterator, KernelXIterator, KernelPixel>(
                    inXIter, kernelXIter, kWidth);
            }
            for (int kernelY = 1, inY = inStartY + 1; kernelY < kHeight; ++inY, ++kernelY) {
                KernelXIterator kernelXIter = kernelImage.x_at(0, kernelY);
                InXIterator inXIter = inImage.x_at(0, inY);
                OutXIterator cnvXIter = convolvedImage.x_at(cnvStartX, cnvY);
                for (int x = 0; x < cnvWidth; ++x, ++cnvXIter, ++inXIter) {
                    *cnvXIter += kernelDotProduct<OutPixel, InXIterator, KernelXIterator, KernelPixel>(
                        inXIter, kernelXIter, kWidth);
                }
            }
        }
    }
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
    template void mathDetail::basicConvolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::Kernel const&, \
            afwMath::ConvolutionControl const&); NL \
    template void mathDetail::basicConvolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::DeltaFunctionKernel const&, \
            afwMath::ConvolutionControl const&); NL \
    template void mathDetail::basicConvolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::LinearCombinationKernel const&, \
            afwMath::ConvolutionControl const&); NL \
    template void mathDetail::basicConvolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::SeparableKernel const&, \
            afwMath::ConvolutionControl const&); NL \
    template void mathDetail::convolveWithBruteForce( \
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

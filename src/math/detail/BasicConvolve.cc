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

/*
 * Definition of basicConvolve and convolveWithBruteForce functions declared in detail/ConvolveImage.h
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <vector>

#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/ConvolveImage.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/detail/Convolve.h"

namespace pexExcept = lsst::pex::exceptions;

namespace {

/**
 * @internal Assert that the dimensions of convolvedImage, inImage and kernel are compatible with convolution.
 *
 * @throws lsst::pex::exceptions::InvalidParameterError if convolvedImage dimensions != inImage dim.
 * @throws lsst::pex::exceptions::InvalidParameterError if inImage smaller than kernel in width or h.
 * @throws lsst::pex::exceptions::InvalidParameterError if kernel width or height < 1
 *
 */
template <typename OutImageT, typename InImageT>
void assertDimensionsOK(OutImageT const& convolvedImage, InImageT const& inImage,
                        lsst::afw::math::Kernel const& kernel) {
    if (convolvedImage.getDimensions() != inImage.getDimensions()) {
        std::ostringstream os;
        os << "convolvedImage dimensions = ( " << convolvedImage.getWidth() << ", "
           << convolvedImage.getHeight() << ") != (" << inImage.getWidth() << ", " << inImage.getHeight()
           << ") = inImage dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    if (inImage.getWidth() < kernel.getWidth() || inImage.getHeight() < kernel.getHeight()) {
        std::ostringstream os;
        os << "inImage dimensions = ( " << inImage.getWidth() << ", " << inImage.getHeight()
           << ") smaller than (" << kernel.getWidth() << ", " << kernel.getHeight()
           << ") = kernel dimensions in width and/or height";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    if ((kernel.getWidth() < 1) || (kernel.getHeight() < 1)) {
        std::ostringstream os;
        os << "kernel dimensions = ( " << kernel.getWidth() << ", " << kernel.getHeight()
           << ") smaller than (1, 1) in width and/or height";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
}
/**
 * @internal Compute the dot product of a kernel row or column and the overlapping portion of an %image
 *
 * @returns computed dot product
 *
 * The pixel computed belongs at position imageIter + kernel center.
 *
 * @todo get rid of KernelPixelT parameter if possible by not computing local variable kVal,
 * or by using iterator traits:
 *     typedef typename std::iterator_traits<KernelIterT>::value_type KernelPixel;
 * Unfortunately, in either case compilation fails with this sort of message:
@verbatim
include/lsst/afw/image/Pixel.h: In instantiation of
‘lsst::afw::image::pixel::exprTraits<boost::gil::pixel<double,
boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > > >’:
include/lsst/afw/image/Pixel.h:385:   instantiated from
‘lsst::afw::image::pixel::BinaryExpr<lsst::afw::image::pixel::Pixel<int, short unsigned int, float>,
boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>,
boost::mpl::range_c<int, 0, 1> > >, std::multiplies<int>, lsst::afw::image::pixel::bitwise_or<short unsigned
int>, lsst::afw::image::pixel::variance_multiplies<float> >’
src/math/ConvolveImage.cc:59:   instantiated from ‘OutPixelT<unnamed>::kernelDotProduct(ImageIterT,
KernelIterT, int) [with OutPixelT = lsst::afw::image::pixel::SinglePixel<int, short unsigned int, float>,
ImageIterT = lsst::afw::image::MaskedImage<int, short unsigned int,
float>::const_MaskedImageIterator<boost::gil::gray32s_pixel_t*, boost::gil::gray16_pixel_t*,
boost::gil::gray32f_noscale_pixel_t*>, KernelIterT = const boost::gil::gray64f_noscalec_pixel_t*]’
src/math/ConvolveImage.cc:265:   instantiated from ‘void lsst::afw::math::basicConvolve(OutImageT&, const
InImageT&, const lsst::afw::math::Kernel&, bool) [with OutImageT = lsst::afw::image::MaskedImage<int, short
unsigned int, float>, InImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>]’
src/math/ConvolveImage.cc:451:   instantiated from ‘void lsst::afw::math::convolve(OutImageT&, const
InImageT&, const KernelT&, bool, int) [with OutImageT = lsst::afw::image::MaskedImage<int, short unsigned int,
float>, InImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>, KernelT =
lsst::afw::math::AnalyticKernel]’
src/math/ConvolveImage.cc:587:   instantiated from here
include/lsst/afw/image/Pixel.h:210: error: no type named ‘ImagePixelT’ in ‘struct boost::gil::pixel<double,
boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >’
include/lsst/afw/image/Pixel.h:211: error: no type named ‘MaskPixelT’ in ‘struct boost::gil::pixel<double,
boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >’
include/lsst/afw/image/Pixel.h:212: error: no type named ‘VariancePixelT’ in ‘struct boost::gil::pixel<double,
boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >’
@endverbatim
 */
template <typename OutPixelT, typename ImageIterT, typename KernelIterT, typename KernelPixelT>
inline OutPixelT kernelDotProduct(
        ImageIterT imageIter,    ///< @internal start of input %image that overlaps kernel vector
        KernelIterT kernelIter,  ///< @internal start of kernel vector
        int kWidth)              ///< @internal width of kernel
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
}  // anonymous namespace

namespace lsst {
namespace afw {
namespace math {
namespace detail {

template <typename OutImageT, typename InImageT>
void basicConvolve(OutImageT& convolvedImage, InImageT const& inImage, math::Kernel const& kernel,
                   math::ConvolutionControl const& convolutionControl) {
    // Because convolve isn't a method of Kernel we can't always use Kernel's vtbl to dynamically
    // dispatch the correct version of basicConvolve. The case that fails is convolving with a kernel
    // obtained from a pointer or reference to a Kernel (base class), e.g. as used in LinearCombinationKernel.
    if (IS_INSTANCE(kernel, math::DeltaFunctionKernel)) {
        LOGL_DEBUG("TRACE3.afw.math.convolve.basicConvolve",
                   "generic basicConvolve: dispatch to DeltaFunctionKernel basicConvolve");
        basicConvolve(convolvedImage, inImage, *dynamic_cast<math::DeltaFunctionKernel const*>(&kernel),
                      convolutionControl);
        return;
    } else if (IS_INSTANCE(kernel, math::SeparableKernel)) {
        LOGL_DEBUG("TRACE3.afw.math.convolve.basicConvolve",
                   "generic basicConvolve: dispatch to SeparableKernel basicConvolve");
        basicConvolve(convolvedImage, inImage, *dynamic_cast<math::SeparableKernel const*>(&kernel),
                      convolutionControl);
        return;
    } else if (IS_INSTANCE(kernel, math::LinearCombinationKernel) && kernel.isSpatiallyVarying()) {
        LOGL_DEBUG(
                "TRACE3.afw.math.convolve.basicConvolve",
                "generic basicConvolve: dispatch to spatially varying LinearCombinationKernel basicConvolve");
        basicConvolve(convolvedImage, inImage, *dynamic_cast<math::LinearCombinationKernel const*>(&kernel),
                      convolutionControl);
        return;
    }
    // OK, use general (and slower) form
    if (kernel.isSpatiallyVarying() && (convolutionControl.getMaxInterpolationDistance() > 1)) {
        // use linear interpolation
        LOGL_DEBUG("TRACE2.afw.math.convolve.basicConvolve",
                   "generic basicConvolve: using linear interpolation");
        convolveWithInterpolation(convolvedImage, inImage, kernel, convolutionControl);

    } else {
        // use brute force
        LOGL_DEBUG("TRACE2.afw.math.convolve.basicConvolve", "generic basicConvolve: using brute force");
        convolveWithBruteForce(convolvedImage, inImage, kernel, convolutionControl);
    }
}

template <typename OutImageT, typename InImageT>
void basicConvolve(OutImageT& convolvedImage, InImageT const& inImage,
                   math::DeltaFunctionKernel const& kernel,
                   math::ConvolutionControl const& convolutionControl) {
    assert(!kernel.isSpatiallyVarying());
    assertDimensionsOK(convolvedImage, inImage, kernel);

    int const mImageWidth = inImage.getWidth();  // size of input region
    int const mImageHeight = inImage.getHeight();
    int const cnvWidth = mImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = mImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const inStartX = kernel.getPixel().getX();
    int const inStartY = kernel.getPixel().getY();

    LOGL_DEBUG("TRACE2.afw.math.convolve.basicConvolve", "DeltaFunctionKernel basicConvolve");

    for (int i = 0; i < cnvHeight; ++i) {
        typename InImageT::x_iterator inPtr = inImage.x_at(inStartX, i + inStartY);
        for (typename OutImageT::x_iterator cnvPtr = convolvedImage.x_at(cnvStartX, i + cnvStartY),
                                            cnvEnd = cnvPtr + cnvWidth;
             cnvPtr != cnvEnd; ++cnvPtr, ++inPtr) {
            *cnvPtr = *inPtr;
        }
    }
}

template <typename OutImageT, typename InImageT>
void basicConvolve(OutImageT& convolvedImage, InImageT const& inImage,
                   math::LinearCombinationKernel const& kernel,
                   math::ConvolutionControl const& convolutionControl) {
    if (!kernel.isSpatiallyVarying()) {
        // use the standard algorithm for the spatially invariant case
        LOGL_DEBUG("TRACE2.afw.math.convolve.basicConvolve",
                   "basicConvolve for LinearCombinationKernel: spatially invariant; using brute force");
        return convolveWithBruteForce(convolvedImage, inImage, kernel, convolutionControl.getDoNormalize());
    } else {
        // refactor the kernel if this is reasonable and possible;
        // then use the standard algorithm for the spatially varying case
        std::shared_ptr<Kernel> refKernelPtr;  // possibly refactored version of kernel
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
            LOGL_DEBUG("TRACE2.afw.math.convolve.basicConvolve",
                       "basicConvolve for LinearCombinationKernel: using interpolation");
            return convolveWithInterpolation(convolvedImage, inImage, *refKernelPtr, convolutionControl);
        } else {
            LOGL_DEBUG("TRACE2.afw.math.convolve.basicConvolve",
                       "basicConvolve for LinearCombinationKernel: maxInterpolationError < 0; using brute "
                       "force");
            return convolveWithBruteForce(convolvedImage, inImage, *refKernelPtr,
                                          convolutionControl.getDoNormalize());
        }
    }
}

template <typename OutImageT, typename InImageT>
void basicConvolve(OutImageT& convolvedImage, InImageT const& inImage, math::SeparableKernel const& kernel,
                   math::ConvolutionControl const& convolutionControl) {
    typedef typename math::Kernel::Pixel KernelPixel;
    typedef typename std::vector<KernelPixel> KernelVector;
    typedef KernelVector::const_iterator KernelIterator;
    typedef typename InImageT::const_x_iterator InXIterator;
    typedef typename InImageT::const_xy_locator InXYLocator;
    typedef typename OutImageT::x_iterator OutXIterator;
    typedef typename OutImageT::y_iterator OutYIterator;
    typedef typename OutImageT::SinglePixel OutPixel;

    assertDimensionsOK(convolvedImage, inImage, kernel);

    lsst::geom::Box2I const fullBBox = inImage.getBBox(image::LOCAL);
    lsst::geom::Box2I const goodBBox = kernel.shrinkBBox(fullBBox);

    KernelVector kernelXVec(kernel.getWidth());
    KernelVector kernelYVec(kernel.getHeight());

    if (kernel.isSpatiallyVarying()) {
        LOGL_DEBUG("TRACE2.afw.math.convolve.basicConvolve",
                   "SeparableKernel basicConvolve: kernel is spatially varying");

        for (int cnvY = goodBBox.getMinY(); cnvY <= goodBBox.getMaxY(); ++cnvY) {
            double const rowPos = inImage.indexToPosition(cnvY, image::Y);

            InXYLocator inImLoc = inImage.xy_at(0, cnvY - goodBBox.getMinY());
            OutXIterator cnvXIter = convolvedImage.row_begin(cnvY) + goodBBox.getMinX();
            for (int cnvX = goodBBox.getMinX(); cnvX <= goodBBox.getMaxX();
                 ++cnvX, ++inImLoc.x(), ++cnvXIter) {
                double const colPos = inImage.indexToPosition(cnvX, image::X);

                KernelPixel kSum = kernel.computeVectors(kernelXVec, kernelYVec,
                                                         convolutionControl.getDoNormalize(), colPos, rowPos);

                // why does this trigger warnings? It did not in the past.
                *cnvXIter = math::convolveAtAPoint<OutImageT, InImageT>(inImLoc, kernelXVec, kernelYVec);
                if (convolutionControl.getDoNormalize()) {
                    *cnvXIter = *cnvXIter / kSum;
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

        LOGL_DEBUG("TRACE2.afw.math.convolve.basicConvolve",
                   "SeparableKernel basicConvolve: kernel is spatially invariant");

        kernel.computeVectors(kernelXVec, kernelYVec, convolutionControl.getDoNormalize());
        KernelIterator const kernelXVecBegin = kernelXVec.begin();
        KernelIterator const kernelYVecBegin = kernelYVec.begin();

        // buffer for x-convolved data
        OutImageT buffer(lsst::geom::Extent2I(goodBBox.getWidth(), kernel.getHeight()));

        // pre-fill x-convolved data buffer with all but one row of data
        int yInd = 0;  // during initial fill bufY = inImageY
        int const yPrefillEnd = buffer.getHeight() - 1;
        for (; yInd < yPrefillEnd; ++yInd) {
            OutXIterator bufXIter = buffer.x_at(0, yInd);
            OutXIterator const bufXEnd = buffer.x_at(goodBBox.getWidth(), yInd);
            InXIterator inXIter = inImage.x_at(0, yInd);
            for (; bufXIter != bufXEnd; ++bufXIter, ++inXIter) {
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
            std::rotate(kernelYVec.begin(), kernelYVec.end() - 1, kernelYVec.end());
        }
    }
}

template <typename OutImageT, typename InImageT>
void convolveWithBruteForce(OutImageT& convolvedImage, InImageT const& inImage, math::Kernel const& kernel,
                            math::ConvolutionControl const& convolutionControl) {
    bool doNormalize = convolutionControl.getDoNormalize();

    typedef typename math::Kernel::Pixel KernelPixel;
    typedef image::Image<KernelPixel> KernelImage;

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
    int const cnvEndX = cnvStartX + cnvWidth;   // end index + 1
    int const cnvEndY = cnvStartY + cnvHeight;  // end index + 1

    KernelImage kernelImage(kernel.getDimensions());
    KernelXYLocator const kernelLoc = kernelImage.xy_at(0, 0);

    if (kernel.isSpatiallyVarying()) {
        LOGL_DEBUG("TRACE4.afw.math.convolve.convolveWithBruteForce",
                   "convolveWithBruteForce: kernel is spatially varying");

        for (int cnvY = cnvStartY; cnvY != cnvEndY; ++cnvY) {
            double const rowPos = inImage.indexToPosition(cnvY, image::Y);

            InXYLocator inImLoc = inImage.xy_at(0, cnvY - cnvStartY);
            OutXIterator cnvXIter = convolvedImage.x_at(cnvStartX, cnvY);
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++inImLoc.x(), ++cnvXIter) {
                double const colPos = inImage.indexToPosition(cnvX, image::X);

                KernelPixel kSum = kernel.computeImage(kernelImage, false, colPos, rowPos);
                *cnvXIter = math::convolveAtAPoint<OutImageT, InImageT>(inImLoc, kernelLoc, kWidth, kHeight);
                if (doNormalize) {
                    *cnvXIter = *cnvXIter / kSum;
                }
            }
        }
    } else {
        LOGL_DEBUG("TRACE4.afw.math.convolve.convolveWithBruteForce",
                   "convolveWithBruteForce: kernel is spatially invariant");

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
/// @cond
#define IMAGE(PIXTYPE) image::Image<PIXTYPE>
#define MASKEDIMAGE(PIXTYPE) image::MaskedImage<PIXTYPE, image::MaskPixel, image::VariancePixel>
#define NL /* */
// Instantiate Image or MaskedImage versions
#define INSTANTIATE_IM_OR_MI(IMGMACRO, OUTPIXTYPE, INPIXTYPE)                                              \
    template void basicConvolve(IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const &, math::Kernel const&,   \
                                math::ConvolutionControl const&);                                          \
    NL template void basicConvolve(IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const &,                     \
                                   math::DeltaFunctionKernel const&, math::ConvolutionControl const&);     \
    NL template void basicConvolve(IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const &,                     \
                                   math::LinearCombinationKernel const&, math::ConvolutionControl const&); \
    NL template void basicConvolve(IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const &,                     \
                                   math::SeparableKernel const&, math::ConvolutionControl const&);         \
    NL template void convolveWithBruteForce(IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const &,            \
                                            math::Kernel const&, math::ConvolutionControl const&);
// Instantiate both Image and MaskedImage versions
#define INSTANTIATE(OUTPIXTYPE, INPIXTYPE)             \
    INSTANTIATE_IM_OR_MI(IMAGE, OUTPIXTYPE, INPIXTYPE) \
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
/// @endcond
}  // namespace detail
}  // namespace math
}  // namespace afw
}  // namespace lsst

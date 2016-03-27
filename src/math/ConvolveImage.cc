// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief Definition of functions declared in ConvolveImage.h
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <limits>
#include <vector>
#include <string>

#include "boost/cstdint.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/detail/Convolve.h"

namespace pexExcept = lsst::pex::exceptions;
namespace pexLog = lsst::pex::logging;
namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace mathDetail = lsst::afw::math::detail;

namespace {

    /*
    * @brief Set the edge pixels of a convolved Image based on size of the convolution kernel used
    *
    * Separate specializations for Image and MaskedImage are required to set the EDGE bit of the Mask plane
    * (if there is one) when doCopyEdge is true.
    */
    template <typename OutImageT, typename InImageT>
    inline void setEdgePixels(
            OutImageT& outImage,        ///< %image whose edge pixels are to be set
            afwMath::Kernel const &kernel, ///< convolution kernel; kernel size is used to determine the edge
            InImageT const &inImage,
                ///< %image whose edge pixels are to be copied; ignored if doCopyEdge is false
            bool doCopyEdge,            ///< if false (default), set edge pixels to the standard edge pixel;
                                        ///< if true, copy edge pixels from input and set EDGE bit of mask
            lsst::afw::image::detail::Image_tag)
                ///< lsst::afw::image::detail::image_traits<ImageT>::image_category()

    {
        const unsigned int imWidth = outImage.getWidth();
        const unsigned int imHeight = outImage.getHeight();
        const unsigned int kWidth = kernel.getWidth();
        const unsigned int kHeight = kernel.getHeight();
        const unsigned int kCtrX = kernel.getCtrX();
        const unsigned int kCtrY = kernel.getCtrY();

        const typename OutImageT::SinglePixel edgePixel = afwMath::edgePixel<OutImageT>(
            typename lsst::afw::image::detail::image_traits<OutImageT>::image_category()
        );
        std::vector<afwGeom::Box2I> bboxList;

        // create a list of bounding boxes describing edge regions, in this order:
        // bottom edge, top edge (both edge to edge),
        // left edge, right edge (both omitting pixels already in the bottom and top edge regions)
        int const numHeight = kHeight - (1 + kCtrY);
        int const numWidth = kWidth - (1 + kCtrX);
        bboxList.push_back(
            afwGeom::Box2I(afwGeom::Point2I(0, 0), afwGeom::Extent2I(imWidth, kCtrY))
        );
        bboxList.push_back(
            afwGeom::Box2I(afwGeom::Point2I(0, imHeight - numHeight), afwGeom::Extent2I(imWidth, numHeight))
        );
        bboxList.push_back(
            afwGeom::Box2I(afwGeom::Point2I(0, kCtrY), afwGeom::Extent2I(kCtrX, imHeight + 1 - kHeight))
        );
        bboxList.push_back(
            afwGeom::Box2I(afwGeom::Point2I(imWidth - numWidth, kCtrY), afwGeom::Extent2I(numWidth, imHeight + 1 - kHeight))
        );

        for (std::vector<afwGeom::Box2I>::const_iterator bboxIter = bboxList.begin();
            bboxIter != bboxList.end(); ++bboxIter
        ) {
            OutImageT outView(outImage, *bboxIter, afwImage::LOCAL);
            if (doCopyEdge) {
                // note: set only works with data of the same type
                // so convert the input image to output format
                outView.assign(OutImageT(InImageT(inImage, *bboxIter, afwImage::LOCAL), true));
            } else {
                outView = edgePixel;
            }
        }
    }

    /*
    * @brief Set the edge pixels of a convolved MaskedImage based on size of the convolution kernel used
    *
    * Separate specializations for Image and MaskedImage are required to set the EDGE bit of the Mask plane
    * (if there is one) when doCopyEdge is true.
    */
    template <typename OutImageT, typename InImageT>
    inline void setEdgePixels(
            OutImageT& outImage,        ///< %image whose edge pixels are to be set
            afwMath::Kernel const &kernel,  ///< convolution kernel; kernel size is used to determine the edge
            InImageT const &inImage,
                ///< %image whose edge pixels are to be copied; ignored if doCopyEdge false
            bool doCopyEdge,            ///< if false (default), set edge pixels to the standard edge pixel;
                                        ///< if true, copy edge pixels from input and set EDGE bit of mask
            lsst::afw::image::detail::MaskedImage_tag)
                ///< lsst::afw::image::detail::image_traits<MaskedImageT>::image_category()

    {
        const unsigned int imWidth = outImage.getWidth();
        const unsigned int imHeight = outImage.getHeight();
        const unsigned int kWidth = kernel.getWidth();
        const unsigned int kHeight = kernel.getHeight();
        const unsigned int kCtrX = kernel.getCtrX();
        const unsigned int kCtrY = kernel.getCtrY();

        const typename OutImageT::SinglePixel edgePixel = afwMath::edgePixel<OutImageT>(
            typename lsst::afw::image::detail::image_traits<OutImageT>::image_category()
        );
        std::vector<afwGeom::Box2I> bboxList;

        // create a list of bounding boxes describing edge regions, in this order:
        // bottom edge, top edge (both edge to edge),
        // left edge, right edge (both omitting pixels already in the bottom and top edge regions)
        int const numHeight = kHeight - (1 + kCtrY);
        int const numWidth = kWidth - (1 + kCtrX);
        bboxList.push_back(
            afwGeom::Box2I(
                afwGeom::Point2I(0, 0),
                afwGeom::Extent2I(imWidth, kCtrY)
            )
        );
        bboxList.push_back(
            afwGeom::Box2I(
                afwGeom::Point2I(0, imHeight - numHeight),
                afwGeom::Extent2I(imWidth, numHeight)
            )
        );
        bboxList.push_back(
            afwGeom::Box2I(
                afwGeom::Point2I(0, kCtrY),
                afwGeom::Extent2I(kCtrX, imHeight + 1 - kHeight)
            )
        );
        bboxList.push_back(
            afwGeom::Box2I(
                afwGeom::Point2I(imWidth - numWidth, kCtrY),
                afwGeom::Extent2I(numWidth, imHeight + 1 - kHeight)
            )
        );

        afwImage::MaskPixel const edgeMask = afwImage::Mask<afwImage::MaskPixel>::getPlaneBitMask("EDGE");
        for (std::vector<afwGeom::Box2I>::const_iterator bboxIter = bboxList.begin();
            bboxIter != bboxList.end(); ++bboxIter) {
            OutImageT outView(outImage, *bboxIter, afwImage::LOCAL);
            if (doCopyEdge) {
                // note: set only works with data of the same type
                // so convert the input image to output format
                outView.assign(OutImageT(InImageT(inImage, *bboxIter, afwImage::LOCAL), true));
                *(outView.getMask()) |= edgeMask;
            } else {
                outView = edgePixel;
            }
        }
    }

}   // anonymous namespace

/**
 * Compute the scaled sum of two images
 *
 * outImage = c1 inImage1 + c2 inImage2
 *
 * For example to linearly interpolate between two images set:
 *   c1 = 1.0 - fracDist
 *   c2 = fracDist
 * where fracDist is the fractional distance of outImage from inImage1:
 *              location of outImage - location of inImage1
 *   fracDist = -------------------------------------------
 *              location of inImage2 - location of inImage1
 *
 * @throw lsst::pex::exceptions::InvalidParameterError if outImage is not same dimensions
 * as inImage1 and inImage2.
 */
template <typename OutImageT, typename InImageT>
void afwMath::scaledPlus(
        OutImageT &outImage,        ///< output image
        double c1,                  ///< coefficient for image 1
        InImageT const &inImage1,   ///< input image 1
        double c2,                  ///< coefficient for image 2
        InImageT const &inImage2)   ///< input image 2
{
    if (outImage.getDimensions() != inImage1.getDimensions()) {
        std::ostringstream os;
        os << "outImage dimensions = ( " << outImage.getWidth() << ", " << outImage.getHeight()
            << ") != (" << inImage1.getWidth() << ", " << inImage1.getHeight()
            << ") = inImage1 dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    } else if (inImage1.getDimensions() != inImage2.getDimensions()) {
        std::ostringstream os;
        os << "inImage1 dimensions = ( " << inImage1.getWidth() << ", " << inImage1.getHeight()
            << ") != (" << inImage2.getWidth() << ", " << inImage2.getHeight()
            << ") = inImage2 dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }

    typedef typename InImageT::const_x_iterator InConstXIter;
    typedef typename OutImageT::x_iterator OutXIter;
    for (int y = 0; y != inImage1.getHeight(); ++y) {
        InConstXIter const end1 = inImage1.row_end(y);
        InConstXIter inIter1 = inImage1.row_begin(y);
        InConstXIter inIter2 = inImage2.row_begin(y);
        OutXIter outIter = outImage.row_begin(y);
        for (; inIter1 != end1; ++inIter1, ++inIter2, ++outIter) {
            *outIter = (*inIter1 * c1) + (*inIter2 * c2);
        }
    }
}

/**
 * @brief Convolve an Image or MaskedImage with a Kernel, setting pixels of an existing output %image.
 *
 * Various convolution kernels are available, including:
 * - FixedKernel: a kernel based on an %image
 * - AnalyticKernel: a kernel based on a Function
 * - SeparableKernel: a kernel described by the product of two one-dimensional Functions: f0(x) * f1(y)
 * - LinearCombinationKernel: a linear combination of a set of spatially invariant basis kernels.
 * - DeltaFunctionKernel: a kernel that is all zeros except one pixel whose value is 1.
 *   Typically used as a basis kernel for LinearCombinationKernel.
 *
 * If a kernel is spatially varying, its spatial model is computed at each pixel position on the image
 * (pixel position, not pixel index). At present (2009-09-24) this position is computed relative
 * to the lower left corner of the sub-image, but it will almost certainly change to be
 * the lower left corner of the parent image.
 *
 * All convolution is performed in real space. This allows convolution to handle masked pixels
 * and spatially varying kernels. Although convolution of an Image with a spatially invariant kernel could,
 * in fact, be performed in Fourier space, the code does not do this.
 *
 * Note that mask bits are smeared by convolution; all nonzero pixels in the kernel smear the mask, even
 * pixels that have very small values. Larger kernels smear the mask more and are also slower to convolve.
 * Use the smallest kernel that will do the job.
 *
 * convolvedImage has a border of edge pixels which cannot be computed normally. Normally these pixels
 * are set to the standard edge pixel, as returned by edgePixel(). However, if your code cannot handle
 * nans in the %image or infs in the variance, you may set doCopyEdge true, in which case the edge pixels
 * are set to the corresponding pixels of the input %image and (if there is a mask) the mask EDGE bit is set.
 *
 * The border of edge pixels has size:
 * - kernel.getCtrX() along the left edge
 * - kernel.getCtrY() along the bottom edge
 * - kernel.getWidth()  - 1 - kernel.getCtrX() along the right edge
 * - kernel.getHeight() - 1 - kernel.getCtrY() along the top edge
 * You can obtain a bounding box for the good pixels in the convolved image
 * from a bounding box for the entire image using the Kernel method shrinkBBox.
 *
 * Convolution has been optimized for the various kinds of kernels, as follows (listed approximately
 * in order of decreasing speed):
 * - DeltaFunctionKernel convolution is a simple %image shift.
 * - SeparableKernel convolution is performed by convolving the input by one of the two functions,
 *   then the result by the other function. Thus convolution with a kernel of size nCols x nRows becomes
 *   convolution with a kernel of size nCols x 1, followed by convolution with a kernel of size 1 x nRows.
 * - Convolution with spatially invariant versions of the other kernels is performed by computing
 *   the kernel %image once and convolving with that. The code has been optimized for cache performance
 *   and so should be fairly efficient.
 * - Convolution with a spatially varying LinearCombinationKernel is performed by convolving the %image
 *   by each basis kernel and combining the result by solving the spatial model. This will be efficient
 *   provided the kernel does not contain too many or very large basis kernels.
 * - Convolution with spatially varying AnalyticKernel is likely to be slow. The code simply computes
 *   the output one pixel at a time by computing the AnalyticKernel at that point and applying it to
 *   the input %image. This is not favorable for cache performance (especially for large kernels)
 *   but avoids recomputing the AnalyticKernel. It is probably possible to do better.
 *
 * Additional convolution functions include:
 *  - convolveAtAPoint(): convolve a Kernel to an Image or MaskedImage at a point.
 *  - basicConvolve(): convolve a Kernel with an Image or MaskedImage, but do not set the edge pixels
 *    of the output. Optimization of convolution for different types of Kernel are handled by different
 *    specializations of basicConvolve().
 *
 * afw/examples offers programs that time convolution including timeConvolve and timeSpatiallyVaryingConvolve.
 *
 * \note This function is able to use GPU acceleration (for spatially invariant kernels and
 *       for LinearCombinationKernel).
 *       There is a limit on maximum kernel size, but kernels sized at most 17x17 should be accelerated
 *       on all supported GPU hardware (SM 1.3 and better). SM 2.x can accelerate kernels sized up to 22x22.
 *
 * @throw lsst::pex::exceptions::InvalidParameterError if convolvedImage is not the same size as inImage
 * @throw lsst::pex::exceptions::InvalidParameterError if inImage is smaller than kernel
 *  in columns and/or rows.
 * @throw lsst::pex::exceptions::MemoryError when allocation of CPU memory fails
 * @throw lsst::afw::gpu::GpuMemoryError when allocation or transfer to/from GPU memory fails
 * @throw lsst::afw::gpu::GpuRuntimeError when GPU code run fails
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT, typename KernelT>
void afwMath::convolve(
        OutImageT& convolvedImage,  ///< convolved %image; must be the same size as inImage
        InImageT const& inImage,    ///< %image to convolve
        KernelT const& kernel,      ///< convolution kernel
        ConvolutionControl const& convolutionControl)   ///< convolution control parameters
{
    mathDetail::basicConvolve(convolvedImage, inImage, kernel, convolutionControl);
    setEdgePixels(convolvedImage, kernel, inImage, convolutionControl.getDoCopyEdge(),
        typename lsst::afw::image::detail::image_traits<OutImageT>::image_category()
    );
    convolvedImage.setXY0(inImage.getXY0());
}

/**
 * @brief Old, deprecated version of convolve.
 *
 * @deprecated This version has no ability to control interpolation parameters.
 */
template <typename OutImageT, typename InImageT, typename KernelT>
void afwMath::convolve(
        OutImageT& convolvedImage,  ///< convolved %image; must be the same size as inImage
        InImageT const& inImage,    ///< %image to convolve
        KernelT const& kernel,      ///< convolution kernel
        bool doNormalize,           ///< if true, normalize the kernel, else use "as is"
        bool doCopyEdge)            ///< if false (default), set edge pixels to the standard edge pixel;
                                    ///< if true, copy edge pixels from input and set EDGE bit of mask
{
    ConvolutionControl convolutionControl;
    convolutionControl.setDoNormalize(doNormalize);
    convolutionControl.setDoCopyEdge(doCopyEdge);
    afwMath::convolve(convolvedImage, inImage, kernel, convolutionControl);
}




/*
 * Explicit instantiation of all convolve functions.
 *
 * This code needs to be compiled with full optimization, and there's no reason why
 * it should be instantiated in the swig wrappers.
 */
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
//
// Next a macro to generate needed instantiations for IMGMACRO (either IMAGE or MASKEDIMAGE)
// and the specified pixel types
//
/* NL's a newline for debugging -- don't define it and say
 g++ -C -E -I$(eups list -s -d boost)/include Convolve.cc | perl -pe 's| *NL *|\n|g'
*/
#define NL /* */
//
// Instantiate one kernel-specific specializations of convolution functions for Image or MaskedImage
// IMGMACRO = IMAGE or MASKEDIMAGE
// KERNELTYPE = a kernel class
//
#define INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, KERNELTYPE) \
    template void afwMath::convolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, KERNELTYPE const&, bool, bool); NL \
    template void afwMath::convolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, KERNELTYPE const&, ConvolutionControl const&); NL
//
// Instantiate Image or MaskedImage versions of all functions defined in this file.
// Call INSTANTIATE_IM_OR_MI_KERNEL once for each kernel class.
// IMGMACRO = IMAGE or MASKEDIMAGE
//
#define INSTANTIATE_IM_OR_MI(IMGMACRO, OUTPIXTYPE, INPIXTYPE) \
    template void afwMath::scaledPlus( \
        IMGMACRO(OUTPIXTYPE)&, double, IMGMACRO(INPIXTYPE) const&, double, IMGMACRO(INPIXTYPE) const&); NL \
    INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, afwMath::AnalyticKernel) \
    INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, afwMath::DeltaFunctionKernel) \
    INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, afwMath::FixedKernel) \
    INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, afwMath::LinearCombinationKernel) \
    INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, afwMath::SeparableKernel) \
    INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, afwMath::Kernel) \
//
// Instantiate all functions defined in this file for one specific output and input pixel type
//
/// \cond
#define INSTANTIATE(OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(IMAGE,       OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(MASKEDIMAGE, OUTPIXTYPE, INPIXTYPE)
//
// Instantiate all functions defined in this file
//
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

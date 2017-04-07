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

#ifndef LSST_AFW_MATH_CONVOLVEIMAGE_H
#define LSST_AFW_MATH_CONVOLVEIMAGE_H
/*
 * Convolve and convolveAtAPoint functions for Image and Kernel
 *
 * @todo Consider adding a flag to convolve indicating which specialized version of basicConvolve was used.
 *   This would only be used for unit testing and trace messages suffice (barely), so not a high priority.
 */
#include <limits>
#include <sstream>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"

namespace lsst {
namespace afw {
namespace math {

    /**
     * Parameters to control convolution
     *
     * @ingroup afw
     */
    class ConvolutionControl {
    public:
        ConvolutionControl(
                bool doNormalize = true,    ///< normalize the kernel to sum=1?
                bool doCopyEdge = false,    ///< copy edge pixels from source image
                    ///< instead of setting them to the standard edge pixel?
                int maxInterpolationDistance = 10  ///< maximum width or height of a region
                    ///< over which to use linear interpolation interpolate
                )
        :
            _doNormalize(doNormalize),
            _doCopyEdge(doCopyEdge),
            _maxInterpolationDistance(maxInterpolationDistance)
        { }

        bool getDoNormalize() const { return _doNormalize; }
        bool getDoCopyEdge() const { return _doCopyEdge; }
        int getMaxInterpolationDistance() const { return _maxInterpolationDistance; };

        void setDoNormalize(bool doNormalize) {_doNormalize = doNormalize; }
        void setDoCopyEdge(bool doCopyEdge) { _doCopyEdge = doCopyEdge; }
        void setMaxInterpolationDistance(int maxInterpolationDistance) {
            _maxInterpolationDistance = maxInterpolationDistance; }

    private:
        bool _doNormalize;  ///< normalize the kernel to sum=1?
        bool _doCopyEdge;   ///< copy edge pixels from source image
                    ///< instead of setting them to the standard edge pixel?
        int _maxInterpolationDistance;  ///< maximum width or height of a region
                    ///< over which to attempt interpolation
    };

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
     * @param[out] outImage output image
     * @param[in] c1 coefficient for image 1
     * @param[in] inImage1 input image 1
     * @param[in] c2 coefficient for image 2
     * @param[in] inImage2 input image 2
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if outImage is not same dimensions
     * as inImage1 and inImage2.
     */
    template <typename OutImageT, typename InImageT>
    void scaledPlus(
            OutImageT &outImage,
            double c1,
            InImageT const &inImage1,
            double c2,
            InImageT const &inImage2);

    template <typename OutImageT, typename InImageT>
    inline typename OutImageT::SinglePixel convolveAtAPoint(
            typename InImageT::const_xy_locator inImageLocator,
            typename lsst::afw::image::Image<lsst::afw::math::Kernel::Pixel>::const_xy_locator kernelLocator,
            int kWidth,
            int kHeight);

    template <typename OutImageT, typename InImageT>
    inline typename OutImageT::SinglePixel convolveAtAPoint(
            typename InImageT::const_xy_locator inImageLocator,
            std::vector<lsst::afw::math::Kernel::Pixel> const& kernelColList,
            std::vector<lsst::afw::math::Kernel::Pixel> const& kernelRowList);

    /**
     * Convolve an Image or MaskedImage with a Kernel, setting pixels of an existing output %image.
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
     * @param[out] convolvedImage convolved %image; must be the same size as inImage
     * @param[in] inImage %image to convolve
     * @param[in] kernel convolution kernel
     * @param[in] convolutionControl convolution control parameters
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if convolvedImage is not the same size as inImage
     * @throws lsst::pex::exceptions::InvalidParameterError if inImage is smaller than kernel
     *  in columns and/or rows.
     * @throws lsst::pex::exceptions::MemoryError when allocation of CPU memory fails
     */
    template <typename OutImageT, typename InImageT, typename KernelT>
    void convolve(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            KernelT const& kernel,
            ConvolutionControl const& convolutionControl = ConvolutionControl());

    /**
     * Old, deprecated version of convolve.
     *
     * @deprecated This version has no ability to control interpolation parameters.
     *
     * @param[out] convolvedImage convolved %image; must be the same size as inImage
     * @param[in] inImage %image to convolve
     * @param[in] kernel convolution kernel
     * @param[in] doNormalize if true, normalize the kernel, else use "as is"
     * @param[in] doCopyEdge if false (default), set edge pixels to the standard edge pixel; if true,
     *                       copy edge pixels from input and set EDGE bit of mask
     */
    template <typename OutImageT, typename InImageT, typename KernelT>
    void convolve(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            KernelT const& kernel,
            bool doNormalize,
            bool doCopyEdge = false);

    /**
     * Return an off-the-edge pixel appropriate for a given Image type
     *
     * The value is quiet_NaN if that exists for the pixel type, else 0
     */
    template <typename ImageT>
    typename ImageT::SinglePixel edgePixel(
            lsst::afw::image::detail::Image_tag
                ///< lsst::afw::image::detail::image_traits<ImageT>::image_category()
    ) {
        typedef typename ImageT::SinglePixel SinglePixelT;
        return SinglePixelT(
            std::numeric_limits<SinglePixelT>::has_quiet_NaN ?
                std::numeric_limits<SinglePixelT>::quiet_NaN() : 0);
    }

    /**
     * Return an off-the-edge pixel appropriate for a given MaskedImage type
     *
     * The components are:
     * - %image = quiet_NaN if that exists for the pixel type, else 0
     * - mask = NO_DATA bit set
     * - variance = infinity
     */
    template <typename MaskedImageT>
    typename MaskedImageT::SinglePixel edgePixel(
            lsst::afw::image::detail::MaskedImage_tag
            ///< lsst::afw::image::detail::image_traits<MaskedImageT>::image_category()
    ) {
        typedef typename MaskedImageT::Image::Pixel ImagePixelT;
        typedef typename MaskedImageT::Variance::Pixel VariancePixelT;

        return typename MaskedImageT::SinglePixel(
            std::numeric_limits<ImagePixelT>::has_quiet_NaN ?
                std::numeric_limits<ImagePixelT>::quiet_NaN() : 0,
            MaskedImageT::Mask::getPlaneBitMask("NO_DATA"),
            std::numeric_limits<VariancePixelT>::infinity());
    }
}}}   // lsst::afw::math

/*
 * Define inline functions
 */

/**
 * Apply convolution kernel to an %image at one point
 *
 * @note This subroutine sacrifices convenience for performance. The user is expected to figure out
 * the kernel center and adjust the supplied pixel accessors accordingly.
 * For an example of how to do this see convolve().
 *
 * @param inImageLocator locator for %image pixel that overlaps pixel (0,0) of kernel
 *                       (the origin of the kernel, not its center)
 * @param kernelLocator locator for (0,0) pixel of kernel (the origin of the kernel,
 *                      not its center)
 * @param kWidth number of columns in kernel
 * @param kHeight number of rows in kernel
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
inline typename OutImageT::SinglePixel lsst::afw::math::convolveAtAPoint(
        typename InImageT::const_xy_locator inImageLocator,
        lsst::afw::image::Image<lsst::afw::math::Kernel::Pixel>::const_xy_locator kernelLocator,
        int kWidth,
        int kHeight)
{
    typename OutImageT::SinglePixel outValue = 0;
    for (int kRow = 0; kRow != kHeight; ++kRow) {
        for (lsst::afw::image::Image<lsst::afw::math::Kernel::Pixel>::const_xy_locator kEnd =
            kernelLocator + lsst::afw::image::detail::difference_type(kWidth, 0);
            kernelLocator != kEnd; ++inImageLocator.x(), ++kernelLocator.x()) {
            typename lsst::afw::math::Kernel::Pixel const kVal = kernelLocator[0];
            if (kVal != 0) {
                outValue += *inImageLocator*kVal;
            }
        }

        inImageLocator  += lsst::afw::image::detail::difference_type(-kWidth, 1);
        kernelLocator += lsst::afw::image::detail::difference_type(-kWidth, 1);
    }

    return outValue;
}

/**
 * Apply separable convolution kernel to an %image at one point
 *
 * @note This subroutine sacrifices convenience for performance. The user is expected to figure out
 * the kernel center and adjust the supplied pixel accessors accordingly.
 * For an example of how to do this see convolve().
 *
 * @param inImageLocator locator for %image pixel that overlaps pixel (0,0) of
 *                       kernel (the origin of the kernel, not its center)
 * @param kernelXList kernel column vector
 * @param kernelYList kernel row vector
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
inline typename OutImageT::SinglePixel lsst::afw::math::convolveAtAPoint(
        typename InImageT::const_xy_locator inImageLocator,
        std::vector<lsst::afw::math::Kernel::Pixel> const &kernelXList,
        std::vector<lsst::afw::math::Kernel::Pixel> const &kernelYList)
{
    typedef typename std::vector<lsst::afw::math::Kernel::Pixel>::const_iterator k_iter;

    typedef typename OutImageT::SinglePixel OutT;
    OutT outValue = 0;
    for (k_iter kernelYIter = kernelYList.begin(), yEnd = kernelYList.end();
         kernelYIter != yEnd; ++kernelYIter) {

        OutT outValueY = 0;
        for (k_iter kernelXIter = kernelXList.begin(), xEnd = kernelXList.end();
             kernelXIter != xEnd; ++kernelXIter, ++inImageLocator.x()) {
            typename lsst::afw::math::Kernel::Pixel const kValX = *kernelXIter;
            if (kValX != 0) {
                outValueY += *inImageLocator*kValX;
            }
        }

        double const kValY = *kernelYIter;
        if (kValY != 0) {
            outValue += outValueY*kValY;
        }

        inImageLocator += lsst::afw::image::detail::difference_type(-kernelXList.size(), 1);
    }

    return outValue;
}


#endif // !defined(LSST_AFW_MATH_CONVOLVEIMAGE_H)

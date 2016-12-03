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
/**
 * @file
 *
 * @brief Convolve and convolveAtAPoint functions for Image and Kernel
 *
 * @todo Consider adding a flag to convolve indicating which specialized version of basicConvolve was used.
 *   This would only be used for unit testing and trace messages suffice (barely), so not a high priority.
 *
 * @author Russell Owen
 *
 * @ingroup afw
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
     * @brief Parameters to control convolution
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

    template <typename OutImageT, typename InImageT, typename KernelT>
    void convolve(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            KernelT const& kernel,
            ConvolutionControl const& convolutionControl = ConvolutionControl());

    template <typename OutImageT, typename InImageT, typename KernelT>
    void convolve(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            KernelT const& kernel,
            bool doNormalize,
            bool doCopyEdge = false);

    /**
     * \brief Return an off-the-edge pixel appropriate for a given Image type
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
     * \brief Return an off-the-edge pixel appropriate for a given MaskedImage type
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
 * @brief Apply convolution kernel to an %image at one point
 *
 * @note This subroutine sacrifices convenience for performance. The user is expected to figure out
 * the kernel center and adjust the supplied pixel accessors accordingly.
 * For an example of how to do this see convolve().
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
inline typename OutImageT::SinglePixel lsst::afw::math::convolveAtAPoint(
        typename InImageT::const_xy_locator inImageLocator, ///< locator for %image pixel that overlaps
            ///< pixel (0,0) of kernel (the origin of the kernel, not its center)
        lsst::afw::image::Image<lsst::afw::math::Kernel::Pixel>::const_xy_locator kernelLocator,
                        ///< locator for (0,0) pixel of kernel (the origin of the kernel, not its center)
        int kWidth,     ///< number of columns in kernel
        int kHeight)    ///< number of rows in kernel
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
 * @brief Apply separable convolution kernel to an %image at one point
 *
 * @note This subroutine sacrifices convenience for performance. The user is expected to figure out
 * the kernel center and adjust the supplied pixel accessors accordingly.
 * For an example of how to do this see convolve().
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
inline typename OutImageT::SinglePixel lsst::afw::math::convolveAtAPoint(
        typename InImageT::const_xy_locator inImageLocator,   ///< locator for %image pixel that overlaps
            ///< pixel (0,0) of kernel (the origin of the kernel, not its center)
        std::vector<lsst::afw::math::Kernel::Pixel> const &kernelXList,  ///< kernel column vector
        std::vector<lsst::afw::math::Kernel::Pixel> const &kernelYList)  ///< kernel row vector
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

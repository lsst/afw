// -*- LSST-C++ -*- // fixed format comment for emacs

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
 * \file
 *
 * \ingroup afw
 *
 * \brief Support for warping an image to a new WCS.
 *
 * \author Nicole M. Silvestri and Russell Owen, University of Washington
 */

#ifndef LSST_AFW_MATH_WARPEXPOSURE_H
#define LSST_AFW_MATH_WARPEXPOSURE_H

#include <memory>
#include <string>

#include "lsst/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/gpu/DevicePreference.h"
#include "lsst/afw/image/LsstImageTypes.h"
#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/math/ConvolveImage.h"
#include "lsst/afw/math/Function.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/math/Kernel.h"

namespace lsst {
namespace afw {
namespace image {
    class Wcs;
}
namespace math {

    /**
    * \brief Lanczos warping: accurate but slow and can introduce ringing artifacts.
    *
    * This kernel is the product of two 1-dimensional Lanczos functions.
    * The number of minima and maxima in the 1-dimensional Lanczos function is 2*order + 1.
    * The kernel has one pixel per function minimum or maximum; but as applied to warping,
    * the first or last pixel is always zero and can be omitted. Thus the kernel size is 2*order x 2*order.
    *
    * For more information about warping kernels see makeWarpingKernel
    *
    * @todo: make a new class WarpingKernel and make this a subclass.
    */
    class LanczosWarpingKernel : public SeparableKernel {
    public:
        explicit LanczosWarpingKernel(
            int order ///< order of Lanczos function
        )
        :
            SeparableKernel(2 * order, 2 * order,
                LanczosFunction1<Kernel::Pixel>(order), LanczosFunction1<Kernel::Pixel>(order))
        {}

        virtual ~LanczosWarpingKernel() {}

        virtual PTR(Kernel) clone() const;

        int getOrder() const;

    protected:
        virtual void setKernelParameter(unsigned int ind, double value) const;
    };

    /**
    * \brief Bilinear warping: fast; good for undersampled data.
    *
    * The kernel size is 2 x 2.
    *
    * For more information about warping kernels see makeWarpingKernel
    *
    * @todo: make a new class WarpingKernel and make this a subclass.
    */
    class BilinearWarpingKernel : public SeparableKernel {
    public:
        explicit BilinearWarpingKernel()
        :
            SeparableKernel(2, 2, BilinearFunction1(0.0), BilinearFunction1(0.0))
        {}

        virtual ~BilinearWarpingKernel() {}

        virtual PTR(Kernel) clone() const;

        /**
         * \brief 1-dimensional bilinear interpolation function.
         *
         * Optimized for bilinear warping so only accepts two values: 0 and 1
         * (which is why it defined in the BilinearWarpingKernel class instead of
         * being made available as a standalone function).
         */
        class BilinearFunction1: public Function1<Kernel::Pixel> {
        public:
            typedef PTR(Function1<Kernel::Pixel>) Function1Ptr;

            /**
             * \brief Construct a Bilinear interpolation function
             */
            explicit BilinearFunction1(
                double fracPos)    ///< fractional position; must be >= 0 and < 1
            :
                Function1<Kernel::Pixel>(1)
            {
                this->_params[0] = fracPos;
            }
            virtual ~BilinearFunction1() {}

            virtual Function1Ptr clone() const {
                return Function1Ptr(new BilinearFunction1(this->_params[0]));
            }

            virtual Kernel::Pixel operator() (double x) const;

            virtual std::string toString(std::string const& ="") const;
        };

    protected:
        virtual void setKernelParameter(unsigned int ind, double value) const;
    };

    /**
    * \brief Nearest neighbor warping: fast; good for undersampled data.
    *
    * The kernel size is 2 x 2.
    *
    * For more information about warping kernels see makeWarpingKernel
    *
    * @todo: make a new class WarpingKernel and make this a subclass.
    */
    class NearestWarpingKernel : public SeparableKernel {
    public:
        explicit NearestWarpingKernel()
        :
            SeparableKernel(2, 2, NearestFunction1(0.0), NearestFunction1(0.0))
        {}

        virtual ~NearestWarpingKernel() {}

        virtual PTR(Kernel) clone() const;

        /**
         * \brief 1-dimensional nearest neighbor interpolation function.
         *
         * Optimized for nearest neighbor warping so only accepts two values: 0 and 1
         * (which is why it defined in the NearestWarpingKernel class instead of
         * being made available as a standalone function).
         */
        class NearestFunction1: public Function1<Kernel::Pixel> {
        public:
            typedef PTR(Function1<Kernel::Pixel>) Function1Ptr;

            /**
             * \brief Construct a Nearest interpolation function
             */
            explicit NearestFunction1(
                double fracPos)    ///< fractional position
            :
                Function1<Kernel::Pixel>(1)
            {
                this->_params[0] = fracPos;
            }
            virtual ~NearestFunction1() {}

            virtual Function1Ptr clone() const {
                return Function1Ptr(new NearestFunction1(this->_params[0]));
            }

            virtual Kernel::Pixel operator() (double x) const;

            virtual std::string toString(std::string const& ="") const;
        };

    protected:
        virtual void setKernelParameter(unsigned int ind, double value) const;
    };

    /**
     * \brief Return a warping kernel given its name.
     *
     * Intended for use with warpImage() and warpExposure().
     *
     * Allowed names are:
     * - bilinear: return a BilinearWarpingKernel
     * - lanczos#: return a LanczosWarpingKernel of order #, e.g. lanczos4
     * - nearest: return a NearestWarpingKernel
     *
     * A warping kernel is a subclass of SeparableKernel with the following properties
     * (though for the sake of speed few, if any, of these are enforced):
     * - Width and height are even. This is unusual for a kernel, but it is more efficient
     *   because if the extra pixel was included it would always have value 0.
     * - The center pixels should be adjacent to the kernel center.
     *   Again, this avoids extra pixels that are sure to have value 0.
     * - It has two parameters: fractional x and fractional row position on the source %image.
     *   The fractional position is the offset of the pixel position on the source
     *   from the center of a nearby source pixel:
     *   - The pixel whose center is just below or to the left of the source position:
     *     0 <= fractional x and y < 0 and the kernel center is the default (size-1)/2.
     *   - The pixel whose center is just above or to the right of the source position:
     *     -1.0 < fractional x and y <= 0 and the kernel center must be set to (size+1)/2.
     */
    PTR(SeparableKernel) makeWarpingKernel(std::string name);

    /**
     * \brief Parameters to control convolution
     *
     * \note padValue is not member of this class to avoid making this a templated class.
     *
     * \warning: GPU acceleration requires interpLength > 0
     *
     * \ingroup afw
     */
    class WarpingControl {
    public:
        /**
         * @brief Construct a WarpingControl object
         *
         * @warning: the GPU code does not yet support warping the mask with
         * a separate kernel. Thus if maskWarpingKernelName is provided
         * the GPU is disabled (or an exception is raised if the GPU is required)
         *
         * @throw pex_exceptions InvalidParameterError if the warping kernel
         * is smaller than the mask warping kernel.
         * @throw pex_exceptions InvalidParameterError if GPU is required
         * and maskWarpingKernelName supplied.
         */
        explicit WarpingControl(
            std::string const &warpingKernelName,   ///< name of warping kernel;
                ///< used as the argument to makeWarpingKernel
            std::string const &maskWarpingKernelName = "",  ///< name of warping kernel used for
                ///< the mask plane; if "" then the regular warping kernel is used.
                ///< Intended so one can use a bilinear kernel or other compact kernel for the mask plane
                ///< to avoid smearing mask bits too far. The theory is that bad pixels are already
                ///< interpolated over, so we don't need to worry about bad values spreading very far.
            int cacheSize = 0,      ///< cache size for warping kernel; no cache if 0
                ///< (used as the argument to the warping kernels' computeCache method)
            int interpLength = 0,   ///< distance over which the WCS can be linearly interpolated
            lsst::afw::gpu::DevicePreference devicePreference = lsst::afw::gpu::DEFAULT_DEVICE_PREFERENCE,
                ///< use GPU acceleration?
            lsst::afw::image::MaskPixel growFullMask = 0
                ///< mask bits to grow to full width of image/variance kernel
        ) :
            _warpingKernelPtr(makeWarpingKernel(warpingKernelName)),
            _maskWarpingKernelPtr(),
            _cacheSize(cacheSize),
            _interpLength(interpLength),
            _devicePreference(devicePreference),
            _growFullMask(growFullMask)
        {
            setMaskWarpingKernelName(maskWarpingKernelName);
            _testDevicePreference(_devicePreference, _warpingKernelPtr);
        }


        virtual ~WarpingControl() {};

        /**
         * @brief get the cache size for the interpolation kernel(s)
         */
        int getCacheSize() const { return _cacheSize; };

        /**
         * @brief set the cache size for the interpolation kernel(s)
         *
         * A value of 0 disables the cache for maximum accuracy.
         * 10,000 typically results in a warping error of a fraction of a count.
         * 100,000 typically results in a warping error of less than 0.01 count.
         * Note the new cache is not computed until getWarpingKernel or getMaskWarpingKernel is called.
         */
        void setCacheSize(
            int cacheSize ///< cache size
        ) { _cacheSize = cacheSize; };

        /**
         * @brief get the interpolation length (pixels)
         */
        int getInterpLength() const { return _interpLength; };

        /**
         * @brief set the interpolation length
         *
         * Interpolation length is the distance over which the WCS can be linearly interpolated, in pixels:
         * * 0 means no interpolation and uses an optimized branch of the code
         * * 1 also performs no interpolation but it runs the interpolation code branch
         *   (and so is only intended for unit tests)
         */
        void setInterpLength(
            int interpLength ///< interpolation length (pixels)
        ) { _interpLength = interpLength; };

        /**
         * @brief get the GPU device preference
         */
        lsst::afw::gpu::DevicePreference getDevicePreference() const { return _devicePreference; };

        /**
         * @brief set the GPU device preference
         */
        void setDevicePreference(
            lsst::afw::gpu::DevicePreference devicePreference  ///< device preference
        ) {
            _testDevicePreference(devicePreference, _warpingKernelPtr);
            _devicePreference = devicePreference;
        }

        /**
         * @brief get the warping kernel
         */
        PTR(SeparableKernel) getWarpingKernel() const;

        /**
         * @brief set the warping kernel by name
         */
        void setWarpingKernelName(
            std::string const &warpingKernelName    ///< name of warping kernel
        );

        /**
         * @brief set the warping kernel
         *
         * @throw lsst::pex::exceptions::InvalidParameterError if new kernel pointer is empty.
         */
        void setWarpingKernel(
            SeparableKernel const &warpingKernel   ///< warping kernel
        );

        /**
         * @brief get the mask warping kernel
         */
        PTR(SeparableKernel) getMaskWarpingKernel() const;

        /**
         * @brief return true if there is a mask kernel
         */
        bool hasMaskWarpingKernel() const { return static_cast<bool>(_maskWarpingKernelPtr); }

        /**
         * @brief set or clear the mask warping kernel by name
         */
        void setMaskWarpingKernelName(
            std::string const &maskWarpingKernelName
                ///< name of mask warping kernel; use "" to clear the kernel
        );

        /**
         * @brief set the mask warping kernel
         *
         * @note To clear the mask warping kernel use setMaskWarpingKernelName("").
         */
        void setMaskWarpingKernel(
            SeparableKernel const &maskWarpingKernel    ///< mask warping kernel
        );

        /**
         * @brief get mask bits to grow to full width of image/variance kernel
         */
        lsst::afw::image::MaskPixel getGrowFullMask() const { return _growFullMask; };

        /**
         * @brief set mask bits to grow to full width of image/variance kernel
         */
        void setGrowFullMask(
            lsst::afw::image::MaskPixel growFullMask  ///< device preference
        ) { _growFullMask = growFullMask; }

    private:
        /**
         * @brief Throw an exception if the two kernels are not compatible in shape
         *
         * @throw lsst::pex::exceptions::InvalidParameterError if the two kernels
         * are not compatible in shape
         */
        void _testWarpingKernels(
            SeparableKernel const &warpingKernel,       ///< warping kernel
            SeparableKernel const &maskWarpingKernel    ///< mask warping kernel
        ) const;

        /**
         * @brief test if GPU device preference and main warping kernel are compatible
         *
         * @throw lsst::pex::exceptions::InvalidParameterError if the parameters are incompatible
         */
        void _testDevicePreference(
            lsst::afw::gpu::DevicePreference const &devicePreference,   ///< GPU device preference
            CONST_PTR(SeparableKernel) const &warpingKernelPtr          ///< warping kernel
        ) const;

        PTR(SeparableKernel) _warpingKernelPtr;
        PTR(SeparableKernel) _maskWarpingKernelPtr;
        int _cacheSize;
        int _interpLength;
        lsst::afw::gpu::DevicePreference _devicePreference; ///< choose CPU or GPU acceleration
        lsst::afw::image::MaskPixel _growFullMask;
    };


    /**
     * \brief Warp (remap) one exposure to another.
     *
     * This is a convenience wrapper around warpImage().
     */
    template<typename DestExposureT, typename SrcExposureT>
        int warpExposure(
        DestExposureT &destExposure,        ///< Remapped exposure. Wcs and xy0 are read, MaskedImage is set,
                                            ///< and Calib and Filter are copied from srcExposure.
                                            ///< All other attributes are left alone (including Detector and Psf)
        SrcExposureT const &srcExposure,    ///< Source exposure
        WarpingControl const &control,      ///< control parameters
        lsst::afw::image::Image<float> &covImage,
        typename DestExposureT::MaskedImageT::SinglePixel padValue =
            lsst::afw::math::edgePixel<typename DestExposureT::MaskedImageT>(
                typename lsst::afw::image::detail::image_traits<
                    typename DestExposureT::MaskedImageT>::image_category())
            ///< use this value for undefined (edge) pixels
    );

    /**
     * \brief Warp an Image or MaskedImage to a new Wcs. See also convenience function
     * warpExposure() to warp an Exposure.
     *
     * Edge pixels are set to padValue; these are pixels that cannot be computed because they
     * are too near the edge of srcImage or miss srcImage entirely.
     *
     * \return the number of valid pixels in destImage (those that are not edge pixels).
     *
     * \note This function is able to use GPU acceleration when interpLength > 0.
     *
     * \b Algorithm Without Interpolation:
     *
     * For each integer pixel position in the remapped Exposure:
     * - The associated pixel position on srcImage is determined using the destination and source WCS
     * - The warping kernel's parameters are set based on the fractional part of the pixel position on srcImage
     * - The warping kernel is applied to srcImage at the integer portion of the pixel position
     *   to compute the remapped pixel value
     * - A flux-conservation factor is determined from the source and destination WCS
     *   and is applied to the remapped pixel
     *
     * The scaling of intensity for relative area of source and destination uses two minor approximations:
     * - The area of the sky marked out by a pixel on the destination %image
     *   corresponds to a parallellogram on the source %image.
     * - The area varies slowly enough across the %image that we can get away with computing
     *   the source area shifted by half a pixel up and to the left of the true area.
     *
     * \b Algorithm With Interpolation:
     *
     * Interpolation simply reduces the number of times WCS is used to map between destination and source
     * pixel position. This computation is only made at a grid of points on the destination image,
     * separated by interpLen pixels along rows and columns. All other source pixel positions are determined
     * by linear interpolation between those grid points. Everything else remains the same.
     *
     * \throw lsst::pex::exceptions::InvalidParameterError if destImage is srcImage
     * \throw lsst::pex::exceptions::MemoryError when allocation of CPU memory fails
     * \throw lsst::afw::gpu::GpuMemoryError when allocation or transfer to/from GPU memory fails
     * \throw lsst::afw::gpu::GpuRuntimeError when GPU code run fails
     *
     * \todo Should support an additional color-based position correction in the remapping
     *   (differential chromatic refraction). This can be done either object-by-object or pixel-by-pixel.
     *
     * \todo Need to deal with oversampling and/or weight maps. If done we can use faster kernels than sinc.
     */
    template<typename DestImageT, typename SrcImageT>
    int warpImage(
        DestImageT &destImage,                  ///< remapped %image
        lsst::afw::image::Wcs const &destWcs,   ///< WCS of remapped %image
        SrcImageT const &srcImage,              ///< source %image
        lsst::afw::image::Wcs const &srcWcs,    ///< WCS of source %image
        WarpingControl const &control,          ///< control parameters
        lsst::afw::image::Image<float> &covImage,
        typename DestImageT::SinglePixel padValue = lsst::afw::math::edgePixel<DestImageT>(
              typename lsst::afw::image::detail::image_traits<DestImageT>::image_category())
              ///< use this value for undefined (edge) pixels
    );

    /**
     * \brief A variant of warpImage that uses an XYTransform instead of a pair of WCS
     * to describe the transformation.
     */
    template<typename DestImageT, typename SrcImageT>
    int warpImage(
        DestImageT &destImage,              ///< remapped %image
        SrcImageT const &srcImage,          ///< source %image
        lsst::afw::geom::XYTransform const &xyTransform, ///<  xy transform mapping source position
            ///< to destination position in the forward direction (but only the reverse direction is used)
        WarpingControl const &control,      ///< control parameters
        lsst::afw::image::Image<float> &covImage,
        typename DestImageT::SinglePixel padValue = lsst::afw::math::edgePixel<DestImageT>(
            typename lsst::afw::image::detail::image_traits<DestImageT>::image_category())
            ///< use this value for undefined (edge) pixels
     );


    /**
     * @brief Warp an image with a LinearTranform about a specified point.
     *        This enables warping an image of e.g. a PSF without translating the centroid.
     */
    template<typename DestImageT, typename SrcImageT>
    int warpCenteredImage(
        DestImageT &destImage,              ///< remapped %image
        SrcImageT const &srcImage,          ///< source %image
        lsst::afw::geom::LinearTransform const &linearTransform, ///< linear transformation to apply
        lsst::afw::geom::Point2D const &centerPosition, ///< pixel position for location of linearTransform
        WarpingControl const &control,      ///< control parameters
        lsst::afw::image::Image<float> &covImage,
        typename DestImageT::SinglePixel padValue = lsst::afw::math::edgePixel<DestImageT>(
            typename lsst::afw::image::detail::image_traits<DestImageT>::image_category())
            ///< use this value for undefined (edge) pixels
    );

    namespace details {
        template <typename A, typename B>
        bool isSameObject(A const&, B const&) { return false; }

        template <typename A>
        bool isSameObject(A const& a, A const& b) { return &a == &b; }
    }

}}} // lsst::afw::math

#endif // !defined(LSST_AFW_MATH_WARPEXPOSURE_H)

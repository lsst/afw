// -*- LSST-C++ -*- // fixed format comment for emacs
/**
  * @file 
  *
  * @ingroup afw
  *
  * @brief Implementation of the templated utility function, warpExposure, for
  * Astrometric Image Remapping for the LSST.
  *
  * @author Nicole M. Silvestri, University of Washington
  *
  * Contact: nms@astro.washington.edu 
  *
  * @version
  *
  * @todo
  * * Modify WarpingKernel so the class is not templated but the method computePixel is.
  *   That was the original design, but it tended to hide code errors so I switched for now.
  */

#ifndef LSST_AFW_MATH_WARPEXPOSURE_H
#define LSST_AFW_MATH_WARPEXPOSURE_H

#include <string>

#include <boost/shared_ptr.hpp>

#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/math/Function.h"
#include "lsst/afw/math/FunctionLibrary.h"

namespace lsst {
namespace afw {
namespace math {
       
    typedef boost::uint16_t maskPixelType;
    
    /**
    * @brief Virtual base warping kernel. Used as an input to warpExposure
    * to control (and implement) the warping algorithm.
    *
    * A warping kernel encapsulates a convolution kernel in such a way
    * that it can be used to compute one pixel of a warped exposure.
    */
    template<typename MaskedImageT>
    class WarpingKernel {
    public:
        explicit WarpingKernel(int width, int height)
        :
            _xList(width),
            _yList(height)
        {};
        
        virtual ~WarpingKernel();

        /**
        * @brief Return width of warping kernel
        */
        int getWidth() const { return _xList.size(); };
        
        /**
        * @brief Return height of warping kernel
        */
        int getHeight() const { return _yList.size(); };
        
        /**
         * @brief Return index of the center column
         */
        int getCtrX() const {
            return (getWidth() - 1) / 2;
        };

        /**
         * @brief Return index of the center row
         */
        int getCtrY() const {
            return (getHeight() - 1) / 2;
        };

        /**
        * @brief Compute one pixel of a warped exposure
        *
        * @warning The warped value assumes the source and destination pixels have equal area on the sky;
        * you must adjust for differences in pixel area afterwards.
        */
        virtual void computePixel(
            typename MaskedImageT::Pixel &destPixel,    ///< destination pixel (warped)
            typename MaskedImageT::const_xy_locator const &srcLoc,  ///< locator for source image pixel at lower left corner
            std::vector<double> const &fracXY   ///< fractional src pixel offset; 0 if none; must be <= 0 and < 1
        );
    protected:
        std::vector<lsst::afw::math::Kernel::PixelT> _xList;
        std::vector<lsst::afw::math::Kernel::PixelT> _yList;
    };
    
    /**
    * @brief Lanczos warping: accurate but slow; can introduce ringing artifacts.
    */
    template<typename MaskedImageT>
    class LanczosWarpingKernel : public WarpingKernel<MaskedImageT> {
    public:
        explicit LanczosWarpingKernel(int order)
        :
            WarpingKernel<MaskedImageT>(2 * order, 2 * order),
            _order(order),
            _kernel(2 * order, 2 * order, LanczosFunction(order), LanczosFunction(order))
        {};
        
        virtual ~LanczosWarpingKernel();

        /**
        * @brief Return order of Lanczos kernel
        */
        int getOrder() { return _order; };

        virtual void computePixel(
            typename MaskedImageT::Pixel &destPixel,
            typename MaskedImageT::const_xy_locator const &srcLoc,
            std::vector<double> const & fracXY
        ) {
            _kernel.setKernelParameters(fracXY);
            double kSum = _kernel.computeVectors(this->_xList, this->_yList, false);
            destPixel = lsst::afw::math::convolveAtAPoint<MaskedImageT, MaskedImageT>(srcLoc, this->_xList, this->_yList);
            destPixel /= static_cast<typename MaskedImageT::Image::Pixel>(kSum);
        };
    private:
        typedef lsst::afw::math::LanczosFunction1<lsst::afw::math::Kernel::PixelT> LanczosFunction;
        int _order;
        lsst::afw::math::SeparableKernel _kernel;
    };

    /**
    * @brief Nearest neighbor warping: fast; has good noise conservation but can introduce aliasing.
    * Best used for weight maps.
    */
    template<typename MaskedImageT>
    class NearestNeighborWarpingKernel : public WarpingKernel<MaskedImageT> {
    public:
        explicit NearestNeighborWarpingKernel()
        :
            WarpingKernel<MaskedImageT>(2, 2)
        {};

        virtual ~NearestNeighborWarpingKernel();

        virtual void computePixel(
            typename MaskedImageT::Pixel &destPixel,
            typename MaskedImageT::const_xy_locator const &srcLoc,
            std::vector<double> const & fracXY
        ) {
            int xOff = fracXY[0] < 0.5 ? 0 : 1;
            int yOff = fracXY[1] < 0.5 ? 0 : 1;
            destPixel = *srcLoc(xOff, yOff);
        };
    };

    /**
    * @brief Bilinear warping: fast; good for undersampled data.
    */
    template<typename MaskedImageT>
    class BilinearWarpingKernel : public WarpingKernel<MaskedImageT> {
    public:
        explicit BilinearWarpingKernel()
        :
            WarpingKernel<MaskedImageT>(2, 2)
        {};

        virtual ~BilinearWarpingKernel();

        virtual void computePixel(
            typename MaskedImageT::Pixel &destPixel,
            typename MaskedImageT::const_xy_locator const &srcLoc,
            std::vector<double> const & fracXY
        ) {
            this->_xList[0] = 1.0 - fracXY[0];
            this->_xList[1] = fracXY[0];
            this->_yList[0] = 1.0 - fracXY[1];
            this->_yList[1] = fracXY[1];
            destPixel = lsst::afw::math::convolveAtAPoint<MaskedImageT, MaskedImageT>(srcLoc, this->_xList, this->_yList);
        };
    };


    template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
    int warpExposure(
        lsst::afw::image::Exposure<ImagePixelT, MaskPixelT, VariancePixelT> &destExposure,
        lsst::afw::image::Exposure<ImagePixelT, MaskPixelT, VariancePixelT> const &srcExposure,
        WarpingKernel<lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> > &WarpingKernel
        );
       
}}} // lsst::afw::math

#endif // !defined(LSST_AFW_MATH_WARPEXPOSURE_H)

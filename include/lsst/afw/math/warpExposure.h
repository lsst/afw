// -*- LSST-C++ -*- // fixed format comment for emacs
/**
  * @file 
  *
  * @ingroup afw
  *
  * @brief Implementation of the templated utility function, warpExposure, for
  * Astrometric Image Remapping for the LSST.
  *
  * @author Nicole M. Silvestri and Russell Owen, University of Washington
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
#include "lsst/afw/math/ConvolveImage.h"
#include "lsst/afw/math/Function.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/math/Kernel.h"

namespace lsst {
namespace afw {
namespace math {
       
    /**
    * @brief Lanczos warping: accurate but slow; can introduce ringing artifacts.
    */
    class LanczosWarpingKernel : public SeparableKernel {
    public:
        explicit LanczosWarpingKernel(int order)
        :
            SeparableKernel(2 * order, 2 * order,
                LanczosFunction1<Kernel::PixelT>(order), LanczosFunction1<Kernel::PixelT>(order))
        {};
        
        virtual ~LanczosWarpingKernel() {};
    };


    /**
    * @brief Bilinear warping: fast; good for undersampled data.
    */
    class BilinearWarpingKernel : public SeparableKernel {
    public:
        explicit BilinearWarpingKernel()
        :
            SeparableKernel(2, 2, BilinearFunction1(0.0), BilinearFunction1(0.0))
        {};

        virtual ~BilinearWarpingKernel() {};

        /**
         * @brief 1-dimensional bilinear interpolation function.
         *
         * Optimized for bilinear warping so only accepts two values: 0 and 1
         * (which is why it defined in the BilinearWarpingKernel class instead of standalone)
         */
        class BilinearFunction1: public Function1<Kernel::PixelT> {
        public:
            typedef Function1<Kernel::PixelT>::Ptr Function1Ptr;
    
            /**
             * @brief Construct a Bilinear interpolation function
             */
            explicit BilinearFunction1(
                double fracPos)    ///< fractional position; must be >= 0 and < 1
            :
                Function1<Kernel::PixelT>(1)
            {
                this->_params[0] = fracPos;
            }
            virtual ~BilinearFunction1() {};
            
            virtual Function1Ptr copy() const {
                return Function1Ptr(new BilinearFunction1(this->_params[0]));
            }
            
            virtual Kernel::PixelT operator() (double x) const;
            
            virtual std::string toString(void) const;
        };
    };

    template<typename DestExposureT, typename SrcExposureT>
    int warpExposure(
        DestExposureT &destExposure,
        SrcExposureT const &srcExposure,
        SeparableKernel &warpingKernel
        );
       
}}} // lsst::afw::math

#endif // !defined(LSST_AFW_MATH_WARPEXPOSURE_H)

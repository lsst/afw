// -*- LSST-C++ -*- // fixed format comment for emacs
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
    * \brief Lanczos warping: accurate but slow and can introduce ringing artifacts.
    *
    * This kernel is the product of two 1-dimensional Lanczos functions.
    * The number of minima and maxima in the 1-dimensional Lanczos function is 2*order + 1.
    * The kernel has one pixel per function minimum or maximum; but as applied to warping,
    * the first or last pixel is always zero and can be omitted. Thus the kernel size is 2*order x 2*order.
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
        
        virtual Kernel::Ptr clone() const;
        
        int getOrder() const;
    };


    /**
    * \brief Bilinear warping: fast; good for undersampled data.
    *
    * The kernel size is 2 x 2.
    */
#if defined(SWIG)
    #pragma SWIG nowarn=SWIGWARN_PARSE_NESTED_CLASS
#endif
    class BilinearWarpingKernel : public SeparableKernel {
    public:
        explicit BilinearWarpingKernel()
        :
            SeparableKernel(2, 2, BilinearFunction1(0.0), BilinearFunction1(0.0))
        {}

        virtual ~BilinearWarpingKernel() {}
        
        virtual Kernel::Ptr clone() const;

        /**
         * \brief 1-dimensional bilinear interpolation function.
         *
         * Optimized for bilinear warping so only accepts two values: 0 and 1
         * (which is why it defined in the BilinearWarpingKernel class instead of
         * being made available as a standalone function).
         */
        class BilinearFunction1: public Function1<Kernel::Pixel> {
        public:
            typedef Function1<Kernel::Pixel>::Ptr Function1Ptr;
    
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
    };
    
    boost::shared_ptr<SeparableKernel> makeWarpingKernel(std::string name);

    template<typename DestExposureT, typename SrcExposureT>
    int warpExposure(
        DestExposureT &destExposure,
        SrcExposureT const &srcExposure,
        SeparableKernel &warpingKernel);

    template<typename DestImageT, typename SrcImageT>
    int warpImage(
        DestImageT &destImage,
        lsst::afw::image::Wcs const &destWcs,
        SrcImageT const &srcImage,
        lsst::afw::image::Wcs const &srcWcs,
        SeparableKernel &warpingKernel);

    namespace details {
        template <typename A, typename B>
        bool isSameObject(A const&, B const&) { return false; }
        
        template <typename A>
        bool isSameObject(A const& a, A const& b) { return &a == &b; }
    }
       
}}} // lsst::afw::math

#endif // !defined(LSST_AFW_MATH_WARPEXPOSURE_H)

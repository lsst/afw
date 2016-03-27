/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#if !defined(LSST_AFW_MATH_OFFSETIMAGE_H)
#define LSST_AFW_MATH_OFFSETIMAGE_H 1

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Statistics.h"

namespace lsst {
namespace afw {
namespace math {

template<typename ImageT>
typename ImageT::Ptr offsetImage(ImageT const& image, float dx, float dy, 
                                 std::string const& algorithmName="lanczos5", unsigned int buffer=0);
template<typename ImageT>
typename ImageT::Ptr rotateImageBy90(ImageT const& image, int nQuarter);

template<typename ImageT>
PTR(ImageT) flipImage(ImageT const& inImage, ///< The %image to flip
                      bool flipLR,           ///< Flip left <--> right?
                      bool flipTB            ///< Flip top <--> bottom?
                     );
template<typename ImageT>
PTR(ImageT) binImage(ImageT const& inImage, int const binX, int const binY,
                     lsst::afw::math::Property const flags=lsst::afw::math::MEAN);
template<typename ImageT>
PTR(ImageT) binImage(ImageT const& inImage, int const binsize,
                     lsst::afw::math::Property const flags=lsst::afw::math::MEAN);

    
}}}
#endif

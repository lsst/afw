#if !defined(LSST_AFW_MATH_OFFSETIMAGE_H)
#define LSST_AFW_MATH_OFFSETIMAGE_H 1

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/warpExposure.h"

namespace lsst {
namespace afw {
namespace math {

template<typename ImageT>
typename ImageT::Ptr offsetImage(std::string const& algorithmName, ImageT const& image, float dx, float dy);
    
}}}
#endif

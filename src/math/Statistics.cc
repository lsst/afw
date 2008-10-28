#include <limits>
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Statistics.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

namespace {
    double NaN = std::numeric_limits<double>::quiet_NaN();
}

template<typename Image>
math::Statistics<Image>::Statistics(Image const& img, ///< Image (or MaskedImage) whose properties we want
                                    int const flags   ///< Describe what we want to calculate
                                   ) : _flags(flags) {
    ;
}

template<typename Image>
std::pair<double, double> math::Statistics<Image>::getParameter(math::Property const prop ///< Desired property
                                             ) const {
    if (!(prop & _flags)) {             // we didn't calculate it
        throw lsst::pex::exceptions::InvalidParameter(boost::format("You didn't ask me to calculate %d") % prop);
    }

    value_type ret(NaN, NAN);
    if (prop == STDEV) {
        ret.first = 666;
        if (_flags & ERRORS) {
            ret.second = 111;
        }
    }

    return ret;
}

template<typename Image>
double math::Statistics<Image>::getValue(math::Property const prop ///< Desired property
                     ) const {
    return getParameter(prop).first;
}

template<typename Image>
double math::Statistics<Image>::getError(math::Property const prop ///< Desired property
                     ) const {
    return getParameter(prop).second;
}

/************************************************************************************************************/
//
// Explicit instantiations
//
template class math::Statistics<image::Image<double> >;
template class math::Statistics<image::Image<float> >;
template class math::Statistics<image::Image<int> >;

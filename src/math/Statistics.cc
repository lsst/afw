#include <limits>
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Statistics.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

namespace {
    double const NaN = std::numeric_limits<double>::quiet_NaN();
}

template<typename Image>
math::Statistics<Image>::Statistics(Image const& img, ///< Image (or MaskedImage) whose properties we want
                                    int const flags   ///< Describe what we want to calculate
                                   ) : _flags(flags),
                                       _mean(NaN), _variance(NaN) {
    _n = img.getWidth()*img.getHeight();
    if (_n == 0) {
        throw lsst::pex::exceptions::InvalidParameter("Image contains no pixels");
    }
    // Check that an int's large enough to hold the number of pixels
    assert(img.getWidth()*static_cast<double>(img.getHeight()) < std::numeric_limits<int>::max());
    //
    // Get a crude estimate of the mean
    //
    double sum = 0;
    int n = 0;
    for (int y = 0; y < img.getHeight(); y += 10) {
        for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth(); ptr != end; ++ptr) {
            sum += *ptr;
            ++n;
        }
    }
    double const crude_mean = sum/n;    // a crude estimate of the mean, used for numerical stability
    ;                                   // in estimating the variance
    //
    // Estimate the full precision variance using that crude mean
    //
    sum = 0;
    double sumx2 = 0;                   // sum of (data - crude_mean)^2
    for (int y = 0; y < img.getHeight(); y++) {
        for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth(); ptr != end; ++ptr) {
            double const delta = *ptr - crude_mean;
            sum += delta;
            sumx2 += delta*delta;
        }
    }
    _mean = crude_mean + sum/_n;
    _variance = sumx2/_n - sum*sum/(static_cast<double>(_n)*_n);
}

template<typename Image>
std::pair<double, double> math::Statistics<Image>::getParameter(math::Property const prop ///< Desired property
                                             ) const {
    if (!(prop & _flags)) {             // we didn't calculate it
        throw lsst::pex::exceptions::InvalidParameter(boost::format("You didn't ask me to calculate %d") % prop);
    }

    value_type ret(NaN, NAN);
    if (prop == MEAN) {
        ret.first = _mean;
        if (_flags & ERRORS) {
            ret.second = _variance/sqrt(_n);
        }
    } else if (prop == STDEV || prop == VARIANCE) {
        if (_n == 1) {
            throw lsst::pex::exceptions::InvalidParameter("Image contains one pixels; population st. dev. is undefined");
        }

        ret.first = _variance*_n/(_n - 1);
        if (_flags & ERRORS) {
            ret.second = 2*(_n - 1)*ret.first*ret.first/(static_cast<double>(_n)*_n); // assumes a Gaussian
        }

        if (prop == STDEV) {
            ret.first = sqrt(ret.first);
            ret.second = sqrt(ret.first)/2;
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

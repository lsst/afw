/**
 * \file
 * \brief Support statistical operations on images
 */
#include <limits>
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Statistics.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

namespace {
    double const NaN = std::numeric_limits<double>::quiet_NaN();
}

/**
 * Constructor for Statistics
 *
 * Most of the actual work is done in this constructor; the results
 * are retrieved using \c get_value etc.
 */
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

    if (flags & (STDEV | VARIANCE)) {
        if (_n == 1) {
            throw lsst::pex::exceptions::InvalidParameter("Image contains only one pixel; "
                                                          "population st. dev. is undefined");
        }
    }

    _variance = sumx2/(_n - 1) - sum*sum/(static_cast<double>(_n - 1)*_n); // estimate of population variance
}

/// Return the value and error in the specified statistic (e.g. MEAN)
///
/// Only quantities requested in the constructor may be retrieved
///
/// \sa getValue and getError
template<typename Image>
std::pair<double, double> math::Statistics<Image>::getResult(math::Property const prop ///< Desired property
                                             ) const {
    if (!(prop & _flags)) {             // we didn't calculate it
        throw lsst::pex::exceptions::InvalidParameter(boost::format("You didn't ask me to calculate %d") % prop);
    }

    value_type ret(NaN, NAN);
    if (prop == NPOINT) {
        ret.first = _n;
        if (_flags & ERRORS) {
            ret.second = 0;
        }
    } else if (prop == MEAN) {
        ret.first = _mean;
        if (_flags & ERRORS) {
            ret.second = sqrt(_variance/_n);
        }
    } else if (prop == STDEV || prop == VARIANCE) {
        ret.first = _variance;
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

/// Return the value of the desired property (if specified in the constructor)
template<typename Image>
double math::Statistics<Image>::getValue(math::Property const prop ///< Desired property
                     ) const {
    return getResult(prop).first;
}

/// Return the error in the desired property (if specified in the constructor)
template<typename Image>
double math::Statistics<Image>::getError(math::Property const prop ///< Desired property
                     ) const {
    return getResult(prop).second;
}

/************************************************************************************************************/
//
// Explicit instantiations
//
template class math::Statistics<image::Image<double> >;
template class math::Statistics<image::Image<float> >;
template class math::Statistics<image::Image<int> >;

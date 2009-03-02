// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Support statistical operations on images
 *
 * @author Steve Bickerton
 * @ingroup afw
 */
#include <iostream>
#include <limits>
#include <cmath>
#include "boost/tuple/tuple.hpp"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Statistics.h"

using namespace std;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;
namespace ex = lsst::pex::exceptions;

namespace {
    double const NaN = std::numeric_limits<double>::quiet_NaN();
    double const iqToStdev = 0.741301109252802;   // 1 sigma in units of iqrange (assume Gaussian)
}

/**
 * @brief Constructor for Statistics object
 *
 * @note Most of the actual work is done in this constructor; the results
 * are retrieved using \c getValue etc.
 *
 */
template<typename Image>
math::Statistics::Statistics(Image const& img, ///< Image (or MaskedImage) whose properties we want
                             int const flags, ///< Describe what we want to calculate
                             StatisticsControl const& sctrl ///< Control how things are calculated
                            ) : _flags(flags),
                                _mean(NaN), _variance(NaN), _min(NaN), _max(NaN),
                                _meanclip(NaN), _varianceclip(NaN), _median(NaN), _iqrange(NaN) {
    
    _n = img.getWidth()*img.getHeight();
    if (_n == 0) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Image contains no pixels");
    }
    
    // Check that an int's large enough to hold the number of pixels
    assert(img.getWidth()*static_cast<double>(img.getHeight()) < std::numeric_limits<int>::max());

    // get the standard statistics
    boost::tuple<double, double, double, double> standard = _getStandard(img, flags);

    _mean = standard.get<0>();
    _variance = standard.get<1>();
    _min = standard.get<2>();
    _max = standard.get<3>();

    // ==========================================================
    // now only calculate it if it's specifically requested - these all cost more!

    // copy the image for any routines that will use median or quantiles
    if (flags & (MEDIAN | IQRANGE | MEANCLIP | STDEVCLIP | VARIANCECLIP)) {
        
        typename Image::Ptr imgcp = typename Image::Ptr(new Image(img, true));  // deep copy
        
        if (flags & (MEDIAN | MEANCLIP | STDEVCLIP | VARIANCECLIP)) {
            _median = _quickSelect(*imgcp, 0.5);
        }
        if (flags & (IQRANGE | MEANCLIP | STDEVCLIP | VARIANCECLIP)) {
            _iqrange = std::fabs(_quickSelect(*imgcp, 0.75) - _quickSelect(*imgcp, 0.25));
        }
        
        if (flags & (MEANCLIP | STDEVCLIP | VARIANCECLIP)) {            
            for(int i_i = 0; i_i < sctrl.getNumIter(); ++i_i) {
                
                double const center = (i_i > 0) ? _meanclip : _median;
                double const hwidth = (i_i > 0) ?
                    sctrl.getNumSigmaClip()*std::sqrt(_varianceclip) : sctrl.getNumSigmaClip()*iqToStdev*_iqrange;
                std::pair<double,double> const clipinfo(center, hwidth);
                
                // returns a 4-tuple but we'll ignore clipped min and max;
                boost::tuple<double, double, double, double> clipped = _getStandard(img, flags, clipinfo);
                
                _meanclip = clipped.get<0>();
                _varianceclip = clipped.get<1>();
            }
        }
    }
}


/* =========================================================================
 * _getStandard(img, flags)
 * *brief Compute the standard stats: mean, variance, min, max
 *
 * *param img    an afw::Image to compute the stats over
 * *param flags  an integer (bit field indicating which statistics are to be computed
 *
 * *note An overloaded version below is used to get clipped versions
 */
template<typename Image>
boost::tuple<double, double, double, double> math::Statistics::_getStandard(Image const &img, int const flags) {
    
    // =====================================================
    // Get a crude estimate of the mean
    int n = 0;
    double sum = 0;
    for (int y=0; y<img.getHeight(); y+=10) {
        for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth(); ptr != end; ++ptr) {
            sum += *ptr;
            ++n;
        }
    }
    double const crude_mean = sum/n;    // a crude estimate of the mean, used for numerical stability of variance
    
    // =======================================================
    // Estimate the full precision variance using that crude mean
    // - get the min and max as well
    sum = 0;
    n = 0;
    double sumx2 = 0;                   // sum of (data - crude_mean)^2
    double min = crude_mean;
    double max = crude_mean;
    
    // If we want max or min (you get both)
    if (flags & (MIN | MAX)){
        for (int y = 0; y < img.getHeight(); ++y) {
            for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth(); ptr != end; ++ptr) {
                double const delta = *ptr - crude_mean;
                sum   += delta;
                sumx2 += delta*delta;
                if ( *ptr < min ) { min = *ptr; }
                if ( *ptr > max ) { max = *ptr; }
                n++;
            }
        }
    // fast loop ... just the mean & variance
    } else {
        min = max = NaN;
        for (int y = 0; y < img.getHeight(); ++y) {
            for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth(); ptr != end; ++ptr) {
                double const delta = *ptr - crude_mean;
                sum   += delta;
                sumx2 += delta*delta;
                n++;
            }
        }

    }


    if (n == 0) {
        throw LSST_EXCEPT(ex::InvalidParameterException,"Image has no valid pixels; mean is undefined.");
    }
    double mean = crude_mean + sum/n;
    
    if (n == 1) {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Image contains only one pixel; population st. dev. is undefined");
    }
    double variance = sumx2/(n - 1) - sum*sum/(static_cast<double>(n - 1)*n); // estimate of population variance
    
    boost::tuple<double, double, double, double> standard = boost::make_tuple(mean, variance, min, max);
    return standard;
}


/* ==========================================================
 * *overload _getStandard(img, flags, clipinfo)
 *
 * *param img      an afw::Image to compute stats for
 * *param flags    an int (bit field indicating which stats to compute
 * *param clipinfo the center and cliplimit for the first clip iteration
 *
 * *brief A routine to get standard stats: mean, variance, min, max with
 *   clipping on std::pair<double,double> = center, cliplimit
 */
template<typename Image>
boost::tuple<double, double, double, double> math::Statistics::_getStandard(Image const &img, int const flags, std::pair<double,double> const clipinfo) {
    
    double const center = clipinfo.first;
    double const cliplimit = clipinfo.second;
    assert(! isnan(center) && ! isnan(cliplimit) );
    
    double const crude_mean = center;    // a crude estimate of the mean for numerical stability of variance

    // =======================================================
    // Estimate the full precision variance using that crude mean
    double sum = 0;
    int n = 0;
    double sumx2 = 0;                   // sum of (data - crude_mean)^2
    double min = crude_mean;
    double max = crude_mean;

    // If we want max or min (you get both)
    if (flags & (MIN | MAX)){
        for (int y = 0; y < img.getHeight(); ++y) {
            for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth(); ptr != end; ++ptr) {
                
                if ( fabs(*ptr - center) > cliplimit ) { continue; }  // clip
                double const delta = *ptr - crude_mean;
                sum += delta;
                sumx2 += delta*delta;
                if ( *ptr < min ) { min = *ptr; }
                if ( *ptr > max ) { max = *ptr; }
                n++;
            }
        }
    // fast loop ... just the mean & variance, no if() for max/min
    } else {
        for (int y = 0; y < img.getHeight(); ++y) {
            for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth(); ptr != end; ++ptr) {
                if ( fabs(*ptr - center) > cliplimit ) { continue; }  // clip
                double const delta = *ptr - crude_mean;
                sum += delta;
                sumx2 += delta*delta;
                n++;
            }
        }

    }
    
    if (n == 0) {
        throw LSST_EXCEPT(ex::InvalidParameterException,"Image has no valid pixels; mean is undefined.");
    }
    double mean = crude_mean + sum/n;
    
    if (n == 1) {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Image contains only one pixel; population st. dev. is undefined");
    }
    double variance = sumx2/(n - 1) - sum*sum/(static_cast<double>(n - 1)*n); // estimate of population variance

    boost::tuple<double, double, double, double> standard = boost::make_tuple(mean, variance, min, max);
    return standard;
    
}


/* _quickSelect()
 *
 * *brief A fast algorithm for computing percentiles for an image
 *
 * *param img       an afw::Image
 * *param quartile  the desired percentile.
 *
 * *note Uses the Floyd & Rivest _quickSelect algorithm for fast computation of a median
 * *note Implementation adapted from Numerical recipes (3rd ed.) Press et al. 2007.
 */
template<typename Image>
double math::Statistics::_quickSelect(Image const &img, double const quartile) {
    
    int const n = img.getWidth() * img.getHeight();
    int const q = static_cast<int>(quartile * n);

    // this routine should only be called with a (contiguous) copy of the image
    // so this declaration should be fine to treat like a vector
    typename Image::fast_iterator arr = img.begin(true);
    
    // apply the algorithm
    int i_mid;
    int i_left = 0;                     // left-most index
    int i_right = n - 1;                // right-most index

    for (;;) {

        // if there are only 1 or 2 elements remaining
        if ( i_right <= i_left + 1 ) {

            // if exactly 2 elements ... switch order as necessary
            if ( ( i_right == i_left + 1 ) && ( arr[i_right] < arr[i_left] ) ) {
                std::swap(arr[i_left], arr[i_right]);
            }
            return arr[q];

        } else {                        // array has > 2 elements

            // use midpoint value for starting partition element
            i_mid = ( i_left + i_right ) >> 1;       // shift to get midpoint betw L and R
            std::swap(arr[i_mid], arr[i_left + 1]);  

            // make sure arr[i_left] <= arr[i_left+1] <= arr[i_right]
            if ( arr[i_left]     > arr[i_right] )    { std::swap( arr[i_left], arr[i_right] ); }
            if ( arr[i_left + 1] > arr[i_right] )    { std::swap( arr[i_left+1], arr[i_right] ); }
            if ( arr[i_left]     > arr[i_left + 1] ) { std::swap( arr[i_left], arr[i_left+1] ); }

            int j_left  = i_left + 1;   // inner loop left-index
            int j_right = i_right;      // inner loop right-index

            typename Image::Pixel const a = arr[i_left + 1]; // the value of the partition element
            
            // partition this sub-array
            for (;;) {
                do { j_left++; } while ( arr[j_left] < a ); // scan up to find elem > a
                do { j_right--; } while ( arr[j_right] > a ); // scan down to find elem < a
                if ( j_right < j_left ) { break; } // break if indices meet
                std::swap( arr[j_left], arr[j_right] ); // switch the two that are out of order.
            }
            arr[i_left + 1] = arr[j_right]; // value in j_right is < 'a', so stash it in i_left+1
            arr[j_right] = a;               // put the partition element in its position

            // keep partition containing q, slide appropriate end of curr partition in to midpoint
            if ( j_right >= q ) { i_right = j_right - 1; }
            if ( j_right <= q ) { i_left = j_left; }
        }
    }
    
}


/* @brief Return the value and error in the specified statistic (e.g. MEAN)
 *
 * @param prop the property (see Statistics.h header) to retrieve
 *
 * @note Only quantities requested in the constructor may be retrieved
 *
 * @sa getValue and getError
 *
 * @todo uncertainties on MEANCLIP,STDEVCLIP are sketchy.  _n != _nClip
 *
 */
std::pair<double, double> math::Statistics::getResult(math::Property const prop ///< Desired property
                                                     ) const {
    if (!(prop & _flags)) {             // we didn't calculate it
        throw LSST_EXCEPT(ex::InvalidParameterException, (boost::format("You didn't ask me to calculate %d") % prop).str());
    }
    
    
    value_type ret(NaN, NaN);
    switch (prop) {

      case ( NPOINT ):
          ret.first = static_cast<double>(_n);
          if (_flags & ERRORS) { ret.second = 0; }
          break;

          // == means ==
      case ( MEAN ):
          ret.first = _mean;
          if (_flags & ERRORS) { ret.second = sqrt(_variance/_n); }
          break;
      case ( MEANCLIP ):
          ret.first = _meanclip;
          if ( _flags & ERRORS ) { ret.second = sqrt(_varianceclip/_n); }  // this is a bug ... _nClip != _n
          break;

          // == stdevs & variances ==
      case ( VARIANCE ):
          ret.first = _variance;
          if (_flags & ERRORS) { ret.second = _varianceError(ret.first, _n); }
          break;
      case ( STDEV ):
          ret.first = sqrt(_variance);
          if (_flags & ERRORS) { ret.second = 0.5*_varianceError(_variance, _n)/ret.first; }
          break;
      case ( VARIANCECLIP ):
          ret.first = _varianceclip;
          if (_flags & ERRORS) { ret.second = _varianceError(ret.first, _n); }
          break;
      case ( STDEVCLIP ):
          ret.first = sqrt(_varianceclip);  // bug: nClip != _n
          if (_flags & ERRORS) { ret.second = 0.5*_varianceError(_varianceclip, _n)/ret.first; }
          break;

          // == other stats ==
      case ( MIN ):
          ret.first = _min;
          if ( _flags & ERRORS ) { ret.second = 0; }
          break;
      case ( MAX ):
          ret.first = _max;
          if ( _flags & ERRORS ) { ret.second = 0; }
          break;
      case ( MEDIAN ):
          ret.first = _median;
          if ( _flags & ERRORS ) { ret.second = 0; }
          break;
      case ( IQRANGE ):
          ret.first = _iqrange;
          if ( _flags & ERRORS ) { ret.second = 0; }
          break;

          // no-op to satisfy the compiler
      case ( ERRORS ):
          break;
          // default: redundant as 'ret' is initialized to NaN, NaN
    }
     return ret;
}

/* @brief Return the value of the desired property (if specified in the constructor)
 * @param prop - the property (see Statistics.h) to retrieve
 */
double math::Statistics::getValue(math::Property const prop ///< Desired property
                                 ) const {
    return getResult(prop).first;
}


/* @brief Return the error in the desired property (if specified in the constructor)
 * @param prop - the property (see Statistics.h) to retrieve
 */
double math::Statistics::getError(math::Property const prop ///< Desired property
                                 ) const {
    return getResult(prop).second;
}


/**
 * @brief Explicit instantiations
 *
 * explicit Statistics(Image const& img, int const flags,
 *                        StatisticsControl const& sctrl=StatisticsControl());
 */

#define INSTANTIATE_STATISTICS(TYPE) \
    template math::Statistics::Statistics(image::Image<TYPE> const& img, int const flags, StatisticsControl const& sctrl);\
    template boost::tuple<double, double, double, double> math::Statistics::_getStandard(image::Image<TYPE> const& img, int const flags);\
    template boost::tuple<double, double, double, double> math::Statistics::_getStandard(image::Image<TYPE> const& img, int const flags, std::pair<double,double> clipinfo);\
    template double math::Statistics::_quickSelect(image::Image<TYPE> const& img, double const quartile);

INSTANTIATE_STATISTICS(double);
INSTANTIATE_STATISTICS(float);
INSTANTIATE_STATISTICS(int);
INSTANTIATE_STATISTICS(unsigned short);


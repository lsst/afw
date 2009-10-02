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
#include "boost/shared_ptr.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Statistics.h"

using namespace std;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;
namespace ex = lsst::pex::exceptions;

namespace {
    double const NaN = std::numeric_limits<double>::quiet_NaN();
    double const MaxDouble = std::numeric_limits<double>::max();
    double const iqToStdev = 0.741301109252802;   // 1 sigma in units of iqrange (assume Gaussian)
}


/**
 * @brief Constructor for Statistics object
 *
 * @note Most of the actual work is done in this constructor; the results
 * are retrieved using \c getValue etc.
 *
 */
template<typename Image, typename Mask>
math::Statistics::Statistics(Image const &img, ///< Image whose properties we want
                             Mask const &msk,   ///< Mask to control which pixels are included
                             int const flags, ///< Describe what we want to calculate
                             StatisticsControl const& sctrl ///< Control how things are calculated
                            ) : _flags(flags),
                                _mean(NaN), _variance(NaN), _min(NaN), _max(NaN), _sum(NaN),
                                _meanclip(NaN), _varianceclip(NaN), _median(NaN), _iqrange(NaN),
                                _sctrl(sctrl) {
    
    _n = img.getWidth()*img.getHeight();
    if (_n == 0) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Image contains no pixels");
    }
    
    // Check that an int's large enough to hold the number of pixels
    assert(img.getWidth()*static_cast<double>(img.getHeight()) < std::numeric_limits<int>::max());

    // get the standard statistics
    StandardReturnT standard = _getStandard(img, msk, flags);

    _mean = standard.get<0>();
    _variance = standard.get<1>();
    _min = standard.get<2>();
    _max = standard.get<3>();
    _sum = standard.get<4>();

    // ==========================================================
    // now only calculate it if it's specifically requested - these all cost more!

    // copy the image for any routines that will use median or quantiles
    if (flags & (MEDIAN | IQRANGE | MEANCLIP | STDEVCLIP | VARIANCECLIP)) {

        // make a vector copy of the image to get the median and quartiles (will move values)
        boost::shared_ptr<std::vector<typename Image::Pixel> > imgcp(new std::vector<typename Image::Pixel>(0));
        
        if (_sctrl.useNanSafe()) {
            for (int i_y = 0; i_y < img.getHeight(); ++i_y) {
                typename Mask::x_iterator mptr = msk.row_begin(i_y);
                for (typename Image::x_iterator ptr = img.row_begin(i_y); ptr != img.row_end(i_y); ++ptr, ++mptr) {
                    if ( !isnan(*ptr) && !(*mptr & _sctrl.getAndMask()) ) {
                        imgcp->push_back(*ptr);
                    }
                }
            }
        } else {
            for (int i_y = 0; i_y < img.getHeight(); ++i_y) {
                typename Mask::x_iterator mptr = msk.row_begin(i_y);
                for (typename Image::x_iterator ptr = img.row_begin(i_y); ptr != img.row_end(i_y); ++ptr, ++mptr) {
                    if ( ! (*mptr & _sctrl.getAndMask()) ) {
                        imgcp->push_back(*ptr);
                    }
                }
            }
        }

        //typename Image::Ptr imgcp = typename Image::Ptr(new Image(img, true));  // deep copy
        
        if (flags & (MEDIAN | MEANCLIP | STDEVCLIP | VARIANCECLIP)) {
            _median = _percentile(*imgcp, 0.5);
        }
        if (flags & (IQRANGE | MEANCLIP | STDEVCLIP | VARIANCECLIP)) {
            _iqrange = std::fabs(_percentile(*imgcp, 0.75) - _percentile(*imgcp, 0.25));
        }
        
        if (flags & (MEANCLIP | STDEVCLIP | VARIANCECLIP)) {            
            for(int i_i = 0; i_i < _sctrl.getNumIter(); ++i_i) {
                
                double const center = (i_i > 0) ? _meanclip : _median;
                double const hwidth = (i_i > 0) ?
                    _sctrl.getNumSigmaClip()*std::sqrt(_varianceclip) : _sctrl.getNumSigmaClip()*iqToStdev*_iqrange;
                std::pair<double,double> const clipinfo(center, hwidth);
                
                // returns a tuple but we'll ignore clipped min, max, and sum;
                StandardReturnT clipped = _getStandard(img, msk, flags, clipinfo);
                
                _meanclip = clipped.get<0>();
                _varianceclip = clipped.get<1>();
            }
        }
    }
}


/* =========================================================================
 * _getStandard(img, flags)
 * @brief Compute the standard stats: mean, variance, min, max
 *
 * @param img    an afw::Image to compute the stats over
 * @param flags  an integer (bit field indicating which statistics are to be computed
 *
 * @note An overloaded version below is used to get clipped versions
 */
template<typename Image, typename Mask>
math::Statistics::StandardReturnT math::Statistics::_getStandard(Image const &img,
                                                                 Mask const &msk,   
                                                                 int const flags) {

    
    // =====================================================
    // Get a crude estimate of the mean
    int n = 0;
    double sum = 0;
    if ( _sctrl.useNanSafe()) {

        for (int y=0; y<img.getHeight(); y+=10) {
            typename Mask::x_iterator mptr = msk.row_begin(y);
            for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth();
                 ptr != end; ++ptr, ++mptr) {
                if ( !isnan(*ptr) && !(*mptr & _sctrl.getAndMask()) ) {
                    sum += *ptr;
                    ++n;
                }
            }
        }
    } else {
        
        for (int y=0; y<img.getHeight(); y+=10) {
            typename Mask::x_iterator mptr = msk.row_begin(y);
            for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth();
                 ptr != end; ++ptr, ++mptr) {
                if ( ! (*mptr & _sctrl.getAndMask()) ) {
                    sum += *ptr;
                    ++n;
                }
            }
        }
    }

    // a crude estimate of the mean, used for numerical stability of variance
    double crude_mean = 0.0;
    if ( n > 0 ) { crude_mean = sum/n; }

    // =======================================================
    // Estimate the full precision variance using that crude mean
    // - get the min and max as well
    sum = 0;
    n = 0;
    double sumx2 = 0;                   // sum of (data - crude_mean)^2
    double min = (n) ? crude_mean : MaxDouble;
    double max = (n) ? crude_mean : -MaxDouble;
    
    // If we want max or min (you get both)
    if (flags & (MIN | MAX)){
        for (int y = 0; y < img.getHeight(); ++y) {
            
            typename Mask::x_iterator mptr = msk.row_begin(y);
            for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth();
                 ptr != end; ++ptr, ++mptr) {

                if ( (! isnan(*ptr)) &&
                     (! (*mptr & _sctrl.getAndMask())) ) {
                    double const delta = *ptr - crude_mean;
                    sum   += delta;
                    sumx2 += delta*delta;
                    if ( *ptr < min ) { min = *ptr; }
                    if ( *ptr > max ) { max = *ptr; }
                    n++;
                }
                
            }
        }
        if (n == 0) {
            min = NaN;
            max = NaN;
        }
    // fast loop ... just the mean & variance
    } else {
        min = max = NaN;

        if (_sctrl.useNanSafe()) {
            for (int y = 0; y < img.getHeight(); ++y) {
                typename Mask::x_iterator mptr = msk.row_begin(y);
                for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth();
                     ptr != end; ++ptr, ++mptr) {
                    
                    if ( (! isnan(*ptr)) &&
                         (! (*mptr & _sctrl.getAndMask())) ){
                        double const delta = *ptr - crude_mean;
                        sum   += delta;
                        sumx2 += delta*delta;
                        n++;
                    }
                }
            }
        } else {
            for (int y = 0; y < img.getHeight(); ++y) {
                typename Mask::x_iterator mptr = msk.row_begin(y);
                for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth();
                     ptr != end; ++ptr, ++mptr) {
                    
                    if ( ! (*mptr & _sctrl.getAndMask()) ){
                        double const delta = *ptr - crude_mean;
                        sum   += delta;
                        sumx2 += delta*delta;
                        n++;
                    }
                }
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

    _n = n;
    
    return boost::make_tuple(mean, variance, min, max, sum + n*crude_mean);
}


/* ==========================================================
 * *overload _getStandard(img, flags, clipinfo)
 *
 * @param img      an afw::Image to compute stats for
 * @param flags    an int (bit field indicating which stats to compute
 * @param clipinfo the center and cliplimit for the first clip iteration
 *
 * @brief A routine to get standard stats: mean, variance, min, max with
 *   clipping on std::pair<double,double> = center, cliplimit
 */
template<typename Image, typename Mask>
math::Statistics::StandardReturnT math::Statistics::_getStandard(Image const &img,
                                                                 Mask const &msk,   
                                                                 int const flags,
                                                                 std::pair<double,double> const clipinfo) {
    
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
            typename Mask::x_iterator mptr = msk.row_begin(y);
            for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth();
                 ptr != end; ++ptr, ++mptr) {
                
                if ( ! (*mptr & _sctrl.getAndMask()) ){                
                    if ( !isnan(*ptr) &&
                         (fabs(*ptr - center) <= cliplimit) ) { // clip
                        double const delta = *ptr - crude_mean;
                        sum += delta;
                        sumx2 += delta*delta;
                        if ( *ptr < min ) { min = *ptr; }
                        if ( *ptr > max ) { max = *ptr; }
                        
                        n++;
                    }
                }
            }
        }
    // fast loop ... just the mean & variance, no if() for max/min
    } else {
        for (int y = 0; y < img.getHeight(); ++y) {
            typename Mask::x_iterator mptr = msk.row_begin(y);
            for (typename Image::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth();
                 ptr != end; ++ptr, ++mptr) {
                
                if ( ! (*mptr & _sctrl.getAndMask()) ){
                    if ( !isnan(*ptr) &&
                         (fabs(*ptr - center) <= cliplimit) ) { // clip
                        double const delta = *ptr - crude_mean;
                        sum += delta;
                        sumx2 += delta*delta;
                        n++;
                    }
                }
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

    _n = n;
    
    return boost::make_tuple(mean, variance, min, max, sum + crude_mean*n);
}


/* _percentile()
 *
 * @brief A wrapper using the nth_element() built-in to compute percentiles for an image
 *
 * @param img       an afw::Image
 * @param quartile  the desired percentile.
 *
 */
template<typename Pixel>
double math::Statistics::_percentile(std::vector<Pixel> &img,
                                     double const quartile) {
    
    int const n = img.size();
    int const q = static_cast<int>(quartile * n);
    
    std::nth_element(img.begin(), img.begin()+q, img.begin()+n-1);
    return img[q];
    
}



/* @brief Return the value and error in the specified statistic (e.g. MEAN)
 *
 * @param prop the property (see Statistics.h header) to retrieve. If NOTHING (default) and you only asked for one
 * property (and maybe its error), that property is returned
 *
 * @note Only quantities requested in the constructor may be retrieved
 *
 * @sa getValue and getError
 *
 * @todo uncertainties on MEANCLIP,STDEVCLIP are sketchy.  _n != _nClip
 *
 */
std::pair<double, double> math::Statistics::getResult(math::Property const iProp ///< Desired property
                                                         ) const {
    // if iProp == NOTHING try to return their heart's delight, as specified in the constructor
    math::Property const prop = (iProp == NOTHING) ? static_cast<math::Property>(_flags & ~ERRORS) : iProp;
    
    if (!(prop & _flags)) {             // we didn't calculate it
        throw LSST_EXCEPT(ex::InvalidParameterException, (boost::format("You didn't ask me to calculate %d") % prop).str());
    }
    
    
    value_type ret(NaN, NaN);
    switch (prop) {

      case ( NPOINT ):
          ret.first = static_cast<double>(_n);
          if (_flags & ERRORS) { ret.second = 0; }
          break;

      case SUM:
          ret.first = static_cast<double>(_sum);
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
      default:                          // we must have set prop to _flags
        assert (iProp == 0);
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          "You may only call getValue without a parameter if you asked for only one statistic");
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

/************************************************************************************************************/
/**
 * Specialisation for Masks; just calculate the "Sum" as the bitwise OR of all pixels
 */

namespace lsst { namespace afw { namespace math {
template<>
Statistics::Statistics(
                       image::Mask<image::MaskPixel> const& msk, ///< Mask whose properties we want
                       image::Mask<image::MaskPixel> const& dmsk, ///< A mask (currently dummy) to control which pixels
                       int const flags,                          ///< Describe what we want to calculate
                       StatisticsControl const& sctrl            ///< Control how things are calculated
                      ) :
    _flags(flags),
    _mean(NaN), _variance(NaN), _min(NaN), _max(NaN),
    _meanclip(NaN), _varianceclip(NaN), _median(NaN), _iqrange(NaN),
    _sctrl(sctrl) {
    
    if ((flags & ~(NPOINT | SUM)) != 0x0) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Statistics<Mask> only supports NPOINT and SUM");
    }
    
    typedef image::Mask<image::MaskPixel> MaskT;
    
    _n = msk.getWidth()*msk.getHeight();
    if (_n == 0) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Image contains no pixels");
    }
    
    // Check that an int's large enough to hold the number of pixels
    assert(msk.getWidth()*static_cast<double>(msk.getHeight()) < std::numeric_limits<int>::max());
    
    image::MaskPixel sum = 0x0;
    for (int y = 0; y != msk.getHeight(); ++y) {
        for (MaskT::x_iterator ptr = msk.row_begin(y), end = msk.row_end(y); ptr != end; ++ptr) {
            sum |= (*ptr)[0];
        }
    }
    _sum = sum;
}

/*
 * @brief Specialization to handle Masks
 * @note Although short, the definition can't be in the header as it must follow the specialization definition
 *       (g++ complained when this was in the header.)
 *
 */            
Statistics makeStatistics(image::Mask<image::MaskPixel> const &msk, ///< Image (or MaskedImage) whose properties we want
                          int const flags,   ///< Describe what we want to calculate
                          StatisticsControl const& sctrl ///< Control how things are calculated
                         ) {
    return Statistics(msk, msk, flags, sctrl);
}

}}}

/************************************************************************************************************/
/*
 * Explicit instantiations
 *
 * explicit Statistics(MaskedImage const& img, int const flags,
 *                        StatisticsControl const& sctrl=StatisticsControl());
 */

//
#define INSTANTIATE_MASKEDIMAGE_STATISTICS(TYPE) \
    template math::Statistics::Statistics(image::Image<TYPE> const &img, image::Mask<image::MaskPixel> const &msk, int const flags, StatisticsControl const& sctrl); \
    template math::Statistics::StandardReturnT math::Statistics::_getStandard(image::Image<TYPE> const &img, image::Mask<image::MaskPixel> const &msk, int const flags); \
    template math::Statistics::StandardReturnT math::Statistics::_getStandard(image::Image<TYPE> const &img, image::Mask<image::MaskPixel> const &msk, int const flags, std::pair<double,double> clipinfo); \
    template double math::Statistics::_percentile(std::vector<TYPE> &img, double const quartile);

//
#define INSTANTIATE_REGULARIMAGE_STATISTICS(TYPE) \
    template math::Statistics::Statistics(image::Image<TYPE> const &img, math::MaskImposter<image::MaskPixel> const &msk, int const flags, StatisticsControl const& sctrl); \
    template math::Statistics::StandardReturnT math::Statistics::_getStandard(image::Image<TYPE> const &img, math::MaskImposter<image::MaskPixel> const &msk, int const flags); \
    template math::Statistics::StandardReturnT math::Statistics::_getStandard(image::Image<TYPE> const &img, math::MaskImposter<image::MaskPixel> const &msk, int const flags, std::pair<double,double> clipinfo);

//
#define INSTANTIATE_VECTOR_STATISTICS(TYPE) \
    template math::Statistics::Statistics(math::ImageImposter<TYPE> const &img, math::MaskImposter<image::MaskPixel> const &msk, int const flags, StatisticsControl const& sctrl); \
    template math::Statistics::StandardReturnT math::Statistics::_getStandard(math::ImageImposter<TYPE> const &img, math::MaskImposter<image::MaskPixel> const &msk, int const flags); \
    template math::Statistics::StandardReturnT math::Statistics::_getStandard(math::ImageImposter<TYPE> const &img, math::MaskImposter<image::MaskPixel> const &msk, int const flags, std::pair<double,double> clipinfo);

#define INSTANTIATE_IMAGE_STATISTICS(T) \
    INSTANTIATE_MASKEDIMAGE_STATISTICS(T); \
    INSTANTIATE_REGULARIMAGE_STATISTICS(T);     \
    INSTANTIATE_VECTOR_STATISTICS(T);


INSTANTIATE_IMAGE_STATISTICS(double);
INSTANTIATE_IMAGE_STATISTICS(float);
INSTANTIATE_IMAGE_STATISTICS(int);
INSTANTIATE_IMAGE_STATISTICS(unsigned short);




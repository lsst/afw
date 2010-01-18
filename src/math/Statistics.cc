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
#include "boost/shared_ptr.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Statistics.h"

using namespace std;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace ex = lsst::pex::exceptions;

namespace {
    
double const NaN = std::numeric_limits<double>::quiet_NaN();
double const MAX_DOUBLE = std::numeric_limits<double>::max();
double const IQ_TO_STDEV = 0.741301109252802;   // 1 sigma in units of iqrange (assume Gaussian)



/**
 * @brief A boolean functor which always returns true (for templated conditionals)
 */
class AlwaysTrue {
public:
    template<typename T>
    bool operator()(T) const {
        return true;
    }
    template<typename Ta, typename Tb>
    bool operator()(Ta, Tb) const {
        return true;
    }
    template<typename Ta, typename Tb, typename Tc>
    bool operator()(Ta, Tb, Tc) const {
        return true;
    }
};

/**
 * @brief A boolean functor which always returns false (for templated conditionals)
 */
class AlwaysFalse {
public:
    template<typename T>
    bool operator()(T) const {
        return false;
    }
    template<typename Ta, typename Tb>
    bool operator()(Ta, Tb) const {
        return false;
    }
    template<typename Ta, typename Tb, typename Tc>
    bool operator()(Ta, Tb, Tc) const {
        return false;
    }
};

/**
 * @brief A boolean functor to check for NaN (for templated conditionals)
 */    
class CheckFinite {
public:
    template<typename T>
    bool operator()(T val) const {
        return !std::isnan(static_cast<float>(val));
    }
};

/**
 * @brief A boolean functor to test val < min (for templated conditionals)
 */    
class CheckValueLtMin {
public:
    template<typename Tval, typename Tmin>
    bool operator()(Tval val, Tmin min) const {
        return (static_cast<Tmin>(val) < min);
    }
};

/**
 * @brief A boolean functor to test val > max (for templated conditionals)
 */    
class CheckValueGtMax {
public:
    template<typename Tval, typename Tmax>
    bool operator()(Tval val, Tmax max) const {
        return (static_cast<Tmax>(val) > max);
    }
};

/**
 * @brief A boolean functor to test |val| < cliplimit  (for templated conditionals)
 */    
class CheckClipRange {
public:
    template<typename Tval, typename Tcen, typename Tmax>
    bool operator()(Tval val, Tcen center, Tmax cliplimit) const {
        Tmax tmp = fabs(val - center);
        return (tmp <= cliplimit);
    }
};
    
// define some abbreviated typenames for the test templates
typedef CheckFinite     ChkFin;
typedef CheckValueLtMin ChkMin;
typedef CheckValueGtMax ChkMax;
typedef CheckClipRange  ChkClip;    
typedef AlwaysTrue      AlwaysT;
typedef AlwaysFalse     AlwaysF;
    
}


/**
 * @brief A private function to copy an image into a vector
 *
 * This is used for percentile and iq_range as these must reorder the values.
 * Because it loops over the pixels, it's been templated over the NaN test to avoid
 * code repetition of the loops.
 */
template<typename IsFinite, typename ImageT, typename MaskT, typename VarianceT>
boost::shared_ptr<std::vector<typename ImageT::Pixel> > afwMath::Statistics::_makeVectorCopy(
        ImageT const &img,
        MaskT const &msk,
        VarianceT const &,
        int const
                                                                                         )
{

    boost::shared_ptr<std::vector<typename ImageT::Pixel> > imgcp(new std::vector<typename ImageT::Pixel>(0));
    for (int i_y = 0; i_y < img.getHeight(); ++i_y) {
        typename MaskT::x_iterator mptr = msk.row_begin(i_y);
        for (typename ImageT::x_iterator ptr = img.row_begin(i_y); ptr != img.row_end(i_y); ++ptr) {
            if ( IsFinite()(*ptr) && !(*mptr & _sctrl.getAndMask()) ) {
                imgcp->push_back(*ptr);
            }
            ++mptr;
        }
    }
    return imgcp;
}


/**
 * @brief Constructor for Statistics object
 *
 * @note Most of the actual work is done in this constructor; the results
 * are retrieved using \c getValue etc.
 *
 */
template<typename ImageT, typename MaskT, typename VarianceT>
afwMath::Statistics::Statistics(
    ImageT const &img,             ///< Image whose properties we want
    MaskT const &msk,              ///< Mask to control which pixels are included
    VarianceT const &var,          ///< Variances corresponding to values in Image
    int const flags,               ///< Describe what we want to calculate
    StatisticsControl const& sctrl ///< Control how things are calculated
                               ) :
    _flags(flags),
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
    StandardReturn standard = _getStandard(img, msk, var, flags);

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
        boost::shared_ptr<std::vector<typename ImageT::Pixel> > imgcp;
        if (_sctrl.getNanSafe()) {
            imgcp = _makeVectorCopy<ChkFin>(img, msk, var, flags);
        } else {
            imgcp = _makeVectorCopy<AlwaysT>(img, msk, var, flags);
        }

        if (flags & (MEDIAN | MEANCLIP | STDEVCLIP | VARIANCECLIP)) {
            _median = _percentile(*imgcp, 0.5);
        }
        if (flags & (IQRANGE | MEANCLIP | STDEVCLIP | VARIANCECLIP)) {
            _iqrange = std::fabs(_percentile(*imgcp, 0.75) - _percentile(*imgcp, 0.25));
        }
        
        if (flags & (MEANCLIP | STDEVCLIP | VARIANCECLIP)) {            
            for (int i_i = 0; i_i < _sctrl.getNumIter(); ++i_i) {
                
                double const center = (i_i > 0) ? _meanclip : _median;
                double const hwidth = (i_i > 0) ?
                    _sctrl.getNumSigmaClip()*std::sqrt(_varianceclip) :
                    _sctrl.getNumSigmaClip()*IQ_TO_STDEV*_iqrange;
                std::pair<double, double> const clipinfo(center, hwidth);
                
                // returns a tuple but we'll ignore clipped min, max, and sum;
                StandardReturn clipped = _getStandard(img, msk, var, flags, clipinfo);
                
                _meanclip = clipped.get<0>();
                _varianceclip = clipped.get<1>();
            }
        }
    }
}


/**
 * @brief This function handles the inner summation loop, with tests templated
 *
 * The idea here is to allow different conditionals in the inner loop, but avoid repeating code.
 * Each test is actually a functor which is handled through a template.  If the
 * user requests a test (eg check for NaNs), the function is instantiated with the appropriate functor.
 * Otherwise, an 'AlwaysTrue' or 'AlwaysFalse' object is passed in.  The compiler then compiles-out
 * a test which is always false, or removes the conditional for a test which is always true.
 */

template<typename IsFinite,
         typename HasValueLtMin,
         typename HasValueGtMax,
         typename InClipRange,
         bool IsWeighted,
         typename ImageT, typename MaskT, typename VarianceT>
afwMath::Statistics::SumReturn afwMath::Statistics::_sumImage(ImageT const &img,
                                                              MaskT const &msk,
                                                              VarianceT const &var,
                                                              int const,
                                                              int const nCrude,
                                                              int const stride,
                                                              double const meanCrude,
                                                              double const cliplimit) {
    int n = 0;
    double wsum = 0.0;
    double sum = 0, sumx2 = 0;
    double min = (nCrude) ? meanCrude : MAX_DOUBLE;
    double max = (nCrude) ? meanCrude : -MAX_DOUBLE;

    for (int iY = 0; iY < img.getHeight(); iY += stride) {
        
        typename MaskT::x_iterator mptr = msk.row_begin(iY);
        typename VarianceT::x_iterator vptr = var.row_begin(iY);
        
        for (typename ImageT::x_iterator ptr = img.row_begin(iY), end = ptr + img.getWidth();
             ptr != end; ++ptr, ++mptr, ++vptr) {
            
            if (IsFinite()(*ptr) &&
                !(*mptr & _sctrl.getAndMask()) &&
                InClipRange()(*ptr, meanCrude, cliplimit) ) { // clip
                
                double const delta = (*ptr - meanCrude);

                if (IsWeighted) {
                    if ( _sctrl.getMultiplyWeights()) {
                        sum   += (*vptr)*delta;
                        sumx2 += (*vptr)*delta*delta;
                        wsum  += (*vptr);
                    } else {
                        if (*vptr > 0) {
                            sum   += delta/(*vptr);
                            sumx2 += delta*delta/(*vptr);
                            wsum  += 1.0/(*vptr);
                        }
                    }
                    
                } else {
                    sum += delta;
                    sumx2 += delta*delta;
                }

                if ( HasValueLtMin()(*ptr, min) ) { min = *ptr; }
                if ( HasValueGtMax()(*ptr, max) ) { max = *ptr; }
                n++;
                
            }
        }
    }
    if (n == 0) {
        min = NaN;
        max = NaN;
    }

    return afwMath::Statistics::SumReturn(n, sum, sumx2, min, max, wsum);
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
template<typename ImageT, typename MaskT, typename VarianceT>
afwMath::Statistics::StandardReturn afwMath::Statistics::_getStandard(ImageT const &img,
                                                                      MaskT const &msk,
                                                                      VarianceT const &var,
                                                                      int const flags) {


    // =====================================================
    // a crude estimate of the mean, used for numerical stability of variance
    SumReturn loopValues;
    
    int nCrude       = 0;
    double meanCrude = 0.0;

    // for small numbers of values, use a small stride
    int const nPix = img.getWidth()*img.getHeight();
    int strideCrude;
    if (nPix < 100) {
        strideCrude = 2;
    } else {
        strideCrude = 10;
    }
    if (_sctrl.getNanSafe()) {
        if (_sctrl.getWeighted()){
            loopValues = _sumImage<ChkFin, AlwaysF, AlwaysF, AlwaysT, true>(img, msk, var, flags,
                                                                            nCrude, strideCrude, meanCrude);
        } else {
            loopValues = _sumImage<ChkFin, AlwaysF, AlwaysF, AlwaysT, false>(img, msk, var, flags,
                                                                             nCrude, strideCrude, meanCrude);
        }
    } else {
        if (_sctrl.getWeighted()) {
            loopValues = _sumImage<AlwaysT, AlwaysF, AlwaysF, AlwaysT, true>(img, msk, var, flags,
                                                                             nCrude, strideCrude, meanCrude);
        } else {
            loopValues = _sumImage<AlwaysT, AlwaysF, AlwaysF, AlwaysT, false>(img, msk, var, flags,
                                                                              nCrude, strideCrude, meanCrude);
        }
    }
    nCrude = loopValues.get<0>();

    double sumCrude = loopValues.get<1>();
    meanCrude = 0.0;
    if ( nCrude > 0 ) {
        meanCrude = sumCrude/nCrude;
    }

    // =======================================================
    // Estimate the full precision variance using that crude mean
    // - get the min and max as well
    
    // If we want max or min (you get both)
    if (flags & (MIN | MAX)){
        
        if (_sctrl.getWeighted()) {
            loopValues = _sumImage<ChkFin, ChkMin, ChkMax, AlwaysT,true>(img, msk, var,
                                                                         flags, nCrude, 1, meanCrude);
            // fast loop ... just the mean & variance
        } else {
            loopValues = _sumImage<ChkFin, ChkMin, ChkMax, AlwaysT,false>(img, msk, var,
                                                                          flags, nCrude, 1, meanCrude);
        }
    } else {
        
        if (_sctrl.getNanSafe()) {
            if ( _sctrl.getWeighted()) {
                loopValues = _sumImage<ChkFin, AlwaysF, AlwaysF, AlwaysT,true>(img, msk, var,
                                                                               flags, nCrude, 1, meanCrude);
            } else {
                loopValues = _sumImage<ChkFin, AlwaysF, AlwaysF, AlwaysT,false>(img, msk, var,
                                                                                flags, nCrude, 1, meanCrude);
            }
        } else {
            if ( _sctrl.getWeighted()) {
                loopValues = _sumImage<AlwaysT, AlwaysF, AlwaysF, AlwaysT,true>(img, msk, var, flags,
                                                                                nCrude, 1, meanCrude);
            } else {
                loopValues = _sumImage<AlwaysT, AlwaysF, AlwaysF, AlwaysT,false>(img, msk, var, flags,
                                                                                 nCrude, 1, meanCrude);
            }
        }
    }

    int n        = loopValues.get<0>();
    double sum   = loopValues.get<1>();
    double sumx2 = loopValues.get<2>();
    double min   = loopValues.get<3>();
    double max   = loopValues.get<4>();
    double wsum  = loopValues.get<5>();

    // estimate of population mean and variance
    double mean, variance;
    if (_sctrl.getWeighted()) {
        mean = (wsum) ? meanCrude + sum/wsum : NaN;
        variance = (n > 1) ? sumx2/(wsum - wsum/n) - sum*sum/(static_cast<double>(wsum - wsum/n)*wsum) : NaN; 
    } else {
        mean = (n) ? meanCrude + sum/n : NaN;
        variance = (n > 1) ? sumx2/(n - 1) - sum*sum/(static_cast<double>(n - 1)*n) : NaN; 
    }
    _n = n;
    
    return afwMath::Statistics::StandardReturn(mean, variance, min, max, sum + n*meanCrude);
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
template<typename ImageT, typename MaskT, typename VarianceT>
afwMath::Statistics::StandardReturn afwMath::Statistics::_getStandard(
    ImageT const &img,
    MaskT const &msk,
    VarianceT const &var,
    int const flags,
    std::pair<double, double> const clipinfo
                                                                     ) {
    
    double const center = clipinfo.first;
    double const cliplimit = clipinfo.second;

    if (isnan(center) || isnan(cliplimit)) {
        //return afwMath::Statistics::StandardReturn(mean, variance, min, max, sum + center*n);
        return afwMath::Statistics::StandardReturn(NaN, NaN, NaN, NaN, NaN);
    }
    
    // =======================================================
    // Estimate the full precision variance using that crude mean
    SumReturn loopValues;

    int const stride = 1;
    int nCrude = 0;
    
    // If we want max or min (you get both)
    if (flags & (MIN | MAX)){
        if ( _sctrl.getWeighted()) {
            loopValues = _sumImage<ChkFin, ChkMin, ChkMax, ChkClip, true>(img, msk, var,
                                                                          flags, nCrude, stride,
                                                                          center, cliplimit);
        } else {
            loopValues = _sumImage<ChkFin, ChkMin, ChkMax, ChkClip, false>(img, msk, var,
                                                                           flags, nCrude, stride,
                                                                           center, cliplimit);
        }
    // fast loop ... just the mean & variance
    } else {
        if ( _sctrl.getWeighted()) {
            
            if (_sctrl.getNanSafe()) {
                loopValues = _sumImage<ChkFin, AlwaysF, AlwaysF, ChkClip, true>(img, msk, var,
                                                                                flags, nCrude, stride,
                                                                                center, cliplimit);
            } else {
                loopValues = _sumImage<AlwaysT, AlwaysF, AlwaysF, ChkClip, true>(img, msk, var,
                                                                                 flags, nCrude, stride,
                                                                                 center, cliplimit);
            }
        } else {
            
            if (_sctrl.getNanSafe()) {
                loopValues = _sumImage<ChkFin, AlwaysF, AlwaysF, ChkClip, false>(img, msk, var,
                                                                                 flags, nCrude, stride,
                                                                                 center, cliplimit);
            } else {
                loopValues = _sumImage<AlwaysT, AlwaysF, AlwaysF, ChkClip, false>(img, msk, var,
                                                                                  flags, nCrude, stride,
                                                                                  center, cliplimit);
            }
        }
    }
    
    int n        = loopValues.get<0>();
    double sum   = loopValues.get<1>();
    double sumx2 = loopValues.get<2>();
    double min   = loopValues.get<3>();
    double max   = loopValues.get<4>();
    double wsum  = loopValues.get<5>();
    
    // estimate of population variance
    double mean, variance;
    if (_sctrl.getWeighted()) {
        mean = (wsum > 0) ? center + sum/wsum : NaN;
        variance = (n > 1) ? sumx2/(wsum - wsum/n) - sum*sum/(static_cast<double>(wsum - wsum/n)*n) : NaN;
    } else {
        mean = (n) ? center + sum/n : NaN;
        variance = (n > 1) ? sumx2/(n - 1) - sum*sum/(static_cast<double>(n - 1)*n) : NaN;
    }
    _n = n;
    
    return afwMath::Statistics::StandardReturn(mean, variance, min, max, sum + center*n);
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
double afwMath::Statistics::_percentile(std::vector<Pixel> &img,
                                        double const percentile) {
    
    int const n = img.size();

    if (n > 1) {
        double const idx = percentile*(n - 1);
        
        // interpolate linearly between the adjacent values
        
        int const q1 = static_cast<int>(idx);
        typename std::vector<Pixel>::iterator midMinus1 = img.begin() + q1;
        std::nth_element(img.begin(), midMinus1, img.end());
        double val1 = static_cast<double>(*midMinus1);
        
        int const q2 = q1 + 1;
        typename std::vector<Pixel>::iterator midPlus1 = img.begin() + q2;
        std::nth_element(img.begin(), midPlus1, img.end());
        double val2 = static_cast<double>(*midPlus1);
        
        double w1 = (static_cast<double>(q2) - idx);
        double w2 = (idx - static_cast<double>(q1));
        
        return w1*val1 + w2*val2;
        
    } else if (n == 1) {
        return img[0];
    } else {
        return NaN;
    }
            

}



/* @brief Return the value and error in the specified statistic (e.g. MEAN)
 *
 * @param prop the property (see Statistics.h header) to retrieve.
 * If NOTHING (default) and you only asked for one
 * property (and maybe its error), that property is returned
 *
 * @note Only quantities requested in the constructor may be retrieved
 *
 * @sa getValue and getError
 *
 * @todo uncertainties on MEANCLIP,STDEVCLIP are sketchy.  _n != _nClip
 *
 */
std::pair<double, double> afwMath::Statistics::getResult(
        afwMath::Property const iProp ///< Desired property
                                                        ) const {
    
    // if iProp == NOTHING try to return their heart's delight, as specified in the constructor
    afwMath::Property const prop =
        (iProp == NOTHING) ? static_cast<afwMath::Property>(_flags & ~ERRORS) : iProp;
    
    if (!(prop & _flags)) {             // we didn't calculate it
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          (boost::format("You didn't ask me to calculate %d") % prop).str());
    }

    
    Value ret(NaN, NaN);
    switch (prop) {
        
      case ( NPOINT ):
        ret.first = static_cast<double>(_n);
        if (_flags & ERRORS) {
            ret.second = 0;
        }
        break;
        
      case SUM:
        ret.first = static_cast<double>(_sum);
        if (_flags & ERRORS) {
            ret.second = 0;
        }
        break;
        
        // == means ==
      case ( MEAN ):
        ret.first = _mean;
        if (_flags & ERRORS) {
            ret.second = sqrt(_variance/_n);
        }
        break;
      case ( MEANCLIP ):
        ret.first = _meanclip;
        if ( _flags & ERRORS ) {
            ret.second = sqrt(_varianceclip/_n);  // this is a bug ... _nClip != _n
        }
        break;
        
        // == stdevs & variances ==
      case ( VARIANCE ):
        ret.first = _variance;
        if (_flags & ERRORS) {
            ret.second = _varianceError(ret.first, _n);
        }
        break;
      case ( STDEV ):
        ret.first = sqrt(_variance);
        if (_flags & ERRORS) {
            ret.second = 0.5*_varianceError(_variance, _n)/ret.first;
        }
        break;
      case ( VARIANCECLIP ):
        ret.first = _varianceclip;
        if (_flags & ERRORS) {
            ret.second = _varianceError(ret.first, _n);
        }
        break;
      case ( STDEVCLIP ):
        ret.first = sqrt(_varianceclip);  // bug: nClip != _n
        if (_flags & ERRORS) {
            ret.second = 0.5*_varianceError(_varianceclip, _n)/ret.first;
        }
        break;
        
        // == other stats ==
      case ( MIN ):
        ret.first = _min;
        if ( _flags & ERRORS ) {
            ret.second = 0;
        }
        break;
      case ( MAX ):
        ret.first = _max;
        if ( _flags & ERRORS ) {
            ret.second = 0;
        }
        break;
      case ( MEDIAN ):
        ret.first = _median;
        if ( _flags & ERRORS ) {
            ret.second = 0;
        }
        break;
      case ( IQRANGE ):
        ret.first = _iqrange;
        if ( _flags & ERRORS ) {
            ret.second = 0;
        }
        break;
        
        // no-op to satisfy the compiler
      case ( ERRORS ):
        break;
        // default: redundant as 'ret' is initialized to NaN, NaN
      default:                          // we must have set prop to _flags
        assert (iProp == 0);
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          "getValue() may only be called without a parameter"
                          " if you asked for only one statistic");
    }
    return ret;
}

/* @brief Return the value of the desired property (if specified in the constructor)
 * @param prop - the property (see Statistics.h) to retrieve
 */
double afwMath::Statistics::getValue(afwMath::Property const prop ///< Desired property
                                     ) const {
    return getResult(prop).first;
}


/* @brief Return the error in the desired property (if specified in the constructor)
 * @param prop - the property (see Statistics.h) to retrieve
 */
double afwMath::Statistics::getError(afwMath::Property const prop ///< Desired property
                                     ) const {
    return getResult(prop).second;
}

/************************************************************************************************/
/**
 * Specialisation for Masks; just calculate the "Sum" as the bitwise OR of all pixels
 */

namespace lsst {
namespace afw {
namespace math {
    
template<>
Statistics::Statistics(
    afwImage::Mask<afwImage::MaskPixel> const& msk, ///< Mask whose properties we want
    afwImage::Mask<afwImage::MaskPixel> const&,     ///< A mask to control which pixels
    afwImage::Mask<afwImage::MaskPixel> const&,     ///< A variance
    int const flags,                                ///< Describe what we want to calculate
    StatisticsControl const& sctrl                  ///< Control how things are calculated
                      ) :
    _flags(flags),
    _mean(NaN), _variance(NaN), _min(NaN), _max(NaN),
    _meanclip(NaN), _varianceclip(NaN), _median(NaN), _iqrange(NaN),
    _sctrl(sctrl) {
    
    if ((flags & ~(NPOINT | SUM)) != 0x0) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Statistics<Mask> only supports NPOINT and SUM");
    }
    
    typedef afwImage::Mask<afwImage::MaskPixel> Mask;
    
    _n = msk.getWidth()*msk.getHeight();
    if (_n == 0) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Image contains no pixels");
    }
    
    // Check that an int's large enough to hold the number of pixels
    assert(msk.getWidth()*static_cast<double>(msk.getHeight()) < std::numeric_limits<int>::max());
    
    afwImage::MaskPixel sum = 0x0;
    for (int y = 0; y != msk.getHeight(); ++y) {
        for (Mask::x_iterator ptr = msk.row_begin(y), end = msk.row_end(y); ptr != end; ++ptr) {
            sum |= (*ptr)[0];
        }
    }
    _sum = sum;
}

/*
 * @brief Specialization to handle Masks
 * @note Although short, the definition can't be in the header as it must
 *       follow the specialization definition
 *       (g++ complained when this was in the header.)
 *
 */            
Statistics makeStatistics(
    afwImage::Mask<afwImage::MaskPixel> const &msk, ///< Image (or MaskedImage) whose properties we want
    int const flags,                          ///< Describe what we want to calculate
    StatisticsControl const& sctrl            ///< Control how things are calculated
                         ) {
    return Statistics(msk, msk, msk, flags, sctrl);
}

}}}

/****************************************************************************************************/
/*
 * Explicit instantiations
 *
 * explicit Statistics(MaskedImage const& img, int const flags,
 *                        StatisticsControl const& sctrl=StatisticsControl());
 */

//
#define STAT afwMath::Statistics

typedef afwImage::VariancePixel VPixel;

#define INSTANTIATE_MASKEDIMAGE_STATISTICS(TYPE)                       \
    template STAT::Statistics(afwImage::Image<TYPE> const &img,            \
                              afwImage::Mask<afwImage::MaskPixel> const &msk, \
                              afwImage::Image<VPixel> const &var,               \
                              int const flags, StatisticsControl const& sctrl); \
    template STAT::StandardReturn STAT::_getStandard(afwImage::Image<TYPE> const &img, \
                                                     afwImage::Mask<afwImage::MaskPixel> const &msk, \
                                                     afwImage::Image<VPixel> const &var, \
                                                     int const flags);  \
    template STAT::StandardReturn STAT::_getStandard(afwImage::Image<TYPE> const &img, \
                                                     afwImage::Mask<afwImage::MaskPixel> const &msk, \
                                                     afwImage::Image<VPixel> const &var, \
                                                     int const flags, std::pair<double, double> clipinfo); \
    template double STAT::_percentile(std::vector<TYPE> &img, double const percentile);


#define INSTANTIATE_MASKEDIMAGE_STATISTICS_NO_MASK(TYPE)                       \
    template STAT::Statistics(afwImage::Image<TYPE> const &img,            \
                              afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                              afwImage::Image<VPixel> const &var,               \
                              int const flags, StatisticsControl const& sctrl); \
    template STAT::StandardReturn STAT::_getStandard(afwImage::Image<TYPE> const &img, \
                                                     afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                                                     afwImage::Image<VPixel> const &var, \
                                                     int const flags);  \
    template STAT::StandardReturn STAT::_getStandard(afwImage::Image<TYPE> const &img, \
                                                     afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                                                     afwImage::Image<VPixel> const &var, \
                                                     int const flags, std::pair<double, double> clipinfo);


#define INSTANTIATE_MASKEDIMAGE_STATISTICS_NO_VAR(TYPE)                       \
    template STAT::Statistics(afwImage::Image<TYPE> const &img,            \
                              afwImage::Mask<afwImage::MaskPixel> const &msk, \
                              afwMath::MaskImposter<VPixel> const &var,          \
                              int const flags, StatisticsControl const& sctrl); \
    template STAT::StandardReturn STAT::_getStandard(afwImage::Image<TYPE> const &img, \
                                                     afwImage::Mask<afwImage::MaskPixel> const &msk, \
                                                     afwMath::MaskImposter<VPixel> const &var, \
                                                     int const flags);  \
    template STAT::StandardReturn STAT::_getStandard(afwImage::Image<TYPE> const &img, \
                                                     afwImage::Mask<afwImage::MaskPixel> const &msk, \
                                                     afwMath::MaskImposter<VPixel> const &var, \
                                                     int const flags, std::pair<double, double> clipinfo);


//
#define INSTANTIATE_REGULARIMAGE_STATISTICS(TYPE)                      \
    template STAT::Statistics(afwImage::Image<TYPE> const &img,            \
                              afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                              afwMath::MaskImposter<VPixel> const &var, \
                              int const flags, StatisticsControl const& sctrl); \
    template STAT::StandardReturn STAT::_getStandard(afwImage::Image<TYPE> const &img, \
                                                     afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                                                     afwMath::MaskImposter<VPixel> const &var, \
                                                     int const flags);  \
    template STAT::StandardReturn STAT::_getStandard(afwImage::Image<TYPE> const &img, \
                                                     afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                                                     afwMath::MaskImposter<VPixel> const &var, \
                                                     int const flags, std::pair<double, double> clipinfo);

//
#define INSTANTIATE_VECTOR_STATISTICS(TYPE)                         \
    template STAT::Statistics(afwMath::ImageImposter<TYPE> const &img,     \
                              afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                              afwMath::MaskImposter<VPixel> const &var,      \
                              int const flags, StatisticsControl const& sctrl); \
    template STAT::StandardReturn STAT::_getStandard(afwMath::ImageImposter<TYPE> const &img, \
                                                     afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                                                     afwMath::MaskImposter<VPixel> const &var, \
                                                     int const flags);  \
    template STAT::StandardReturn STAT::_getStandard(afwMath::ImageImposter<TYPE> const &img, \
                                                     afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                                                     afwMath::MaskImposter<VPixel> const &var, \
                                                     int const flags, std::pair<double, double> clipinfo);


#define INSTANTIATE_IMAGE_STATISTICS(T) \
    INSTANTIATE_MASKEDIMAGE_STATISTICS(T); \
    INSTANTIATE_MASKEDIMAGE_STATISTICS_NO_VAR(T); \
    INSTANTIATE_MASKEDIMAGE_STATISTICS_NO_MASK(T); \
    INSTANTIATE_REGULARIMAGE_STATISTICS(T); \
    INSTANTIATE_VECTOR_STATISTICS(T);


INSTANTIATE_IMAGE_STATISTICS(double);
INSTANTIATE_IMAGE_STATISTICS(float);
INSTANTIATE_IMAGE_STATISTICS(int);
INSTANTIATE_IMAGE_STATISTICS(unsigned short);




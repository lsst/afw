// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
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
#include "lsst/utils/ieee.h"

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
        return !lsst::utils::isnan(static_cast<float>(val));
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
 * @brief Conversion function to switch a string to a Property (see Statistics.h)
 */
afwMath::Property afwMath::stringToStatisticsProperty(std::string const property) {
    static std::map<std::string, Property> statisticsProperty;
    if (statisticsProperty.size() == 0) {
        statisticsProperty["NOTHING"]      = NOTHING;
        statisticsProperty["ERRORS"]       = ERRORS;
        statisticsProperty["NPOINT"]       = NPOINT;
        statisticsProperty["MEAN"]         = MEAN;
        statisticsProperty["STDEV"]        = STDEV;
        statisticsProperty["VARIANCE"]     = VARIANCE;
        statisticsProperty["MEDIAN"]       = MEDIAN;
        statisticsProperty["IQRANGE"]      = IQRANGE;
        statisticsProperty["MEANCLIP"]     = MEANCLIP;
        statisticsProperty["STDEVCLIP"]    = STDEVCLIP;
        statisticsProperty["VARIANCECLIP"] = VARIANCECLIP;
        statisticsProperty["MIN"]          = MIN;
        statisticsProperty["MAX"]          = MAX;
        statisticsProperty["SUM"]          = SUM;
        statisticsProperty["MEANSQUARE"]   = MEANSQUARE;
        statisticsProperty["ORMASK"]       = ORMASK;
    }
    return statisticsProperty[property];
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

    // Note to self: I'm not going to keep track of allPixelOrMask here ... sumImage() does that
    // and it always gets called
    boost::shared_ptr<std::vector<typename ImageT::Pixel> > imgcp(new std::vector<typename ImageT::Pixel>(0));
    for (int i_y = 0; i_y < img.getHeight(); ++i_y) {
        typename MaskT::x_iterator mptr = msk.row_begin(i_y);
        for (typename ImageT::x_iterator ptr = img.row_begin(i_y), end = img.row_end(i_y); ptr != end; ++ptr) {
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
    _allPixelOrMask = standard.get<5>();

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

        // if we *only* want the median, just use _percentile(), otherwise use _medianAndQuartiles()
        if ( (flags & (MEDIAN)) && !(flags & (IQRANGE | MEANCLIP | STDEVCLIP | VARIANCECLIP)) ) {
            _median = _percentile(*imgcp, 0.5);
        } else {
            MedianQuartileReturn mq = _medianAndQuartiles(*imgcp);
            _median = mq.get<0>();
            _iqrange = mq.get<2>() - mq.get<1>();
        }
        
        
        if (flags & (MEANCLIP | STDEVCLIP | VARIANCECLIP)) {            
            for (int i_i = 0; i_i < _sctrl.getNumIter(); ++i_i) {
                
                double const center = (i_i > 0) ? _meanclip : _median;
                double const hwidth = (i_i > 0 && _n > 1) ?
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

    afwImage::MaskPixel allPixelOrMask = 0x0;
    
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

                allPixelOrMask |= *mptr;
                
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

    return afwMath::Statistics::SumReturn(n, sum, sumx2, min, max, wsum, allPixelOrMask);
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
    afwImage::MaskPixel allPixelOrMask = loopValues.get<6>();
    
    // estimate of population mean and variance
    double mean, variance;
    if (_sctrl.getWeighted()) {
        mean = (wsum > 0) ? meanCrude + sum/wsum : NaN;
        variance = (n > 1) ? sumx2/(wsum - wsum/n) - sum*sum/(static_cast<double>(wsum - wsum/n)*wsum) : NaN;
        sum += meanCrude*wsum;
    } else {
        mean = (n) ? meanCrude + sum/n : NaN;
        variance = (n > 1) ? sumx2/(n - 1) - sum*sum/(static_cast<double>(n - 1)*n) : NaN;
        sum += n*meanCrude;
    }
    _n = n;
    
    return afwMath::Statistics::StandardReturn(mean, variance, min, max, sum, allPixelOrMask);
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

    if (lsst::utils::isnan(center) || lsst::utils::isnan(cliplimit)) {
        //return afwMath::Statistics::StandardReturn(mean, variance, min, max, sum + center*n);
        return afwMath::Statistics::StandardReturn(NaN, NaN, NaN, NaN, NaN, ~0x0);
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
    afwImage::MaskPixel allPixelOrMask = loopValues.get<6>();
    
    // estimate of population variance
    double mean, variance;
    if (_sctrl.getWeighted()) {
        mean = (wsum > 0) ? center + sum/wsum : NaN;
        variance = (n > 1) ? sumx2/(wsum - wsum/n) - sum*sum/(static_cast<double>(wsum - wsum/n)*n) : NaN;
        sum += center*wsum;
    } else {
        mean = (n) ? center + sum/n : NaN;
        variance = (n > 1) ? sumx2/(n - 1) - sum*sum/(static_cast<double>(n - 1)*n) : NaN;
        sum += n*center;
    }
    _n = n;
    
    return afwMath::Statistics::StandardReturn(mean, variance, min, max, sum, allPixelOrMask);
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
        // For efficiency:
        // - if we're asked for a percentile > 0.5,
        //    we'll do the second partial sort on shorter (upper) portion
        // - otherwise, the shorter portion will be the lower one, we'll partial-sort that.
        
        int const q1 = static_cast<int>(idx);
        int const q2 = q1 + 1;

        typename std::vector<Pixel>::iterator mid1 = img.begin() + q1;
        typename std::vector<Pixel>::iterator mid2 = img.begin() + q2;
        if ( percentile > 0.5 ) {
            std::nth_element(img.begin(), mid1, img.end());
            std::nth_element(mid1, mid2, img.end());
        } else {
            std::nth_element(img.begin(), mid2, img.end());
            std::nth_element(img.begin(), mid1, mid2);
        }
    
        double val1 = static_cast<double>(*mid1);
        double val2 = static_cast<double>(*mid2);
        double w1 = (static_cast<double>(q2) - idx);
        double w2 = (idx - static_cast<double>(q1));
        return w1*val1 + w2*val2;
        
    } else if (n == 1) {
        return img[0];
    } else {
        return NaN;
    }
            

}



/* _medianAndQuartiles()
 *
 * @brief A wrapper using the nth_element() built-in to compute median and Quartiles for an image
 *
 * @param img       an afw::Image
 * @param quartile  the desired percentile.
 *
 */
template<typename Pixel>
afwMath::Statistics::MedianQuartileReturn afwMath::Statistics::_medianAndQuartiles(std::vector<Pixel> &img) {
    
    int const n = img.size();

    if (n > 1) {
        
        double const idx50 = 0.50*(n - 1);
        double const idx25 = 0.25*(n - 1);
        double const idx75 = 0.75*(n - 1);
        
        // For efficiency:
        // - partition at 50th, then partition the two half further to get 25th and 75th
        // - to get the adjacent points (for interpolation), partition between 25/50, 50/75, 75/end
        //   these should be much smaller partitions
        
        int const q50a = static_cast<int>(idx50);
        int const q50b = q50a + 1;
        int const q25a = static_cast<int>(idx25);
        int const q25b = q25a + 1;
        int const q75a = static_cast<int>(idx75);
        int const q75b = q75a + 1;
        
        typename std::vector<Pixel>::iterator mid50a = img.begin() + q50a;
        typename std::vector<Pixel>::iterator mid50b = img.begin() + q50b;
        typename std::vector<Pixel>::iterator mid25a = img.begin() + q25a;
        typename std::vector<Pixel>::iterator mid25b = img.begin() + q25b;
        typename std::vector<Pixel>::iterator mid75a = img.begin() + q75a;
        typename std::vector<Pixel>::iterator mid75b = img.begin() + q75b;

        // get the 50th percentile, then get the 25th and 75th on the smaller partitions
        std::nth_element(img.begin(), mid50a, img.end());
        std::nth_element(mid50a, mid75a, img.end());
        std::nth_element(img.begin(), mid25a, mid50a);

        // and the adjacent points for each ... use the smallest segments available.
        std::nth_element(mid50a, mid50b, mid75a);
        std::nth_element(mid25a, mid25b, mid50a);
        std::nth_element(mid75a, mid75b, img.end());

        // interpolate linearly between the adjacent values
        double val50a = static_cast<double>(*mid50a);
        double val50b = static_cast<double>(*mid50b);
        double w50a = (static_cast<double>(q50b) - idx50);
        double w50b = (idx50 - static_cast<double>(q50a));
        double median = w50a*val50a + w50b*val50b;

        double val25a = static_cast<double>(*mid25a);
        double val25b = static_cast<double>(*mid25b);
        double w25a = (static_cast<double>(q25b) - idx25);
        double w25b = (idx25 - static_cast<double>(q25a));
        double q1 = w25a*val25a + w25b*val25b;
        
        double val75a = static_cast<double>(*mid75a);
        double val75b = static_cast<double>(*mid75b);
        double w75a = (static_cast<double>(q75b) - idx75);
        double w75b = (idx75 - static_cast<double>(q75a));
        double q3 = w75a*val75a + w75b*val75b;

        return MedianQuartileReturn(median, q1, q3);
    } else if (n == 1) {
        return MedianQuartileReturn(img[0], img[0], img[0]);
    } else {
        return MedianQuartileReturn(NaN, NaN, NaN);
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
        static_cast<afwMath::Property>(((iProp == NOTHING) ? _flags : iProp) & ~ERRORS);
    
    if (!(prop & _flags)) {             // we didn't calculate it
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          (boost::format("You didn't ask me to calculate %d") % prop).str());
    }

    
    Value ret(NaN, NaN);
    switch (prop) {
        
      case NPOINT:
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
      case MEAN:
        ret.first = _mean;
        if (_flags & ERRORS) {
            ret.second = sqrt(_variance/_n);
        }
        break;
      case MEANCLIP:
        ret.first = _meanclip;
        if ( _flags & ERRORS ) {
            ret.second = sqrt(_varianceclip/_n);  // this is a bug ... _nClip != _n
        }
        break;
        
        // == stdevs & variances ==
      case VARIANCE:
        ret.first = _variance;
        if (_flags & ERRORS) {
            ret.second = _varianceError(ret.first, _n);
        }
        break;
      case STDEV:
        ret.first = sqrt(_variance);
        if (_flags & ERRORS) {
            ret.second = 0.5*_varianceError(_variance, _n)/ret.first;
        }
        break;
      case VARIANCECLIP:
        ret.first = _varianceclip;
        if (_flags & ERRORS) {
            ret.second = _varianceError(ret.first, _n);
        }
        break;
      case STDEVCLIP:
        ret.first = sqrt(_varianceclip);  // bug: nClip != _n
        if (_flags & ERRORS) {
            ret.second = 0.5*_varianceError(_varianceclip, _n)/ret.first;
        }
        break;

      case MEANSQUARE:
        ret.first = (_n - 1)/static_cast<double>(_n)*_variance + _mean*_mean;
        if (_flags & ERRORS) {
            ret.second = ::sqrt(2*ret.first*ret.first/(_n*_n)); // assumes Gaussian
        }
        break;
        
        // == other stats ==
      case MIN:
        ret.first = _min;
        if ( _flags & ERRORS ) {
            ret.second = 0;
        }
        break;
      case MAX:
        ret.first = _max;
        if ( _flags & ERRORS ) {
            ret.second = 0;
        }
        break;
      case MEDIAN:
        ret.first = _median;
        if ( _flags & ERRORS ) {
            ret.second = sqrt(M_PI/2*_variance/_n); // assumes Gaussian
        }
        break;
      case IQRANGE:
        ret.first = _iqrange;
        if ( _flags & ERRORS ) {
            ret.second = 0;
        }
        break;

        // no-op to satisfy the compiler
      case ERRORS:
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
                                                     int const flags, std::pair<double, double> clipinfo);


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
                                                     int const flags, std::pair<double, double> clipinfo)


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
                                                     int const flags, std::pair<double, double> clipinfo)


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
                                                     int const flags, std::pair<double, double> clipinfo)

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
                                                     int const flags, std::pair<double, double> clipinfo)


#define INSTANTIATE_IMAGE_STATISTICS(T) \
    INSTANTIATE_MASKEDIMAGE_STATISTICS(T); \
    INSTANTIATE_MASKEDIMAGE_STATISTICS_NO_VAR(T); \
    INSTANTIATE_MASKEDIMAGE_STATISTICS_NO_MASK(T); \
    INSTANTIATE_REGULARIMAGE_STATISTICS(T); \
    INSTANTIATE_VECTOR_STATISTICS(T)


INSTANTIATE_IMAGE_STATISTICS(double);
INSTANTIATE_IMAGE_STATISTICS(float);
INSTANTIATE_IMAGE_STATISTICS(int);
INSTANTIATE_IMAGE_STATISTICS(boost::uint16_t);


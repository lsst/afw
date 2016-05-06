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
#include <cassert>
#include <memory>
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Statistics.h"
#include "lsst/afw/geom/Angle.h"

using namespace std;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace afwGeom = lsst::afw::geom;
namespace pexExceptions = lsst::pex::exceptions;

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
     * @brief A boolean functor to check for NaN or infinity (for templated conditionals)
     */    
    class CheckFinite {
    public:
        template<typename T>
        bool operator()(T val) const {
            return std::isfinite(static_cast<float>(val));
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
    
    // Return the variance of a variance, assuming a Gaussian
    // There is apparently an attempt to correct for bias in the factor (n - 1)/n.  RHL
    inline double varianceError(double const variance, int const n)
    {
        return 2*(n - 1)*variance*variance/static_cast<double>(n*n);
    }

    /*********************************************************************************************************/
    // return type for processPixels
    typedef boost::tuple<int,                        // n
                         double,                     // sum
                         afwMath::Statistics::Value, // mean
                         afwMath::Statistics::Value, // variance
                         double,                     // min
                         double,                     // max
                         lsst::afw::image::MaskPixel // allPixelOrMask
                         > StandardReturn;

    /*
     * Functions which convert the booleans into calls to the proper templated types, one type per
     * recursion level
     */
    /*
     * This function handles the inner summation loop, with tests templated
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
             bool useWeights,
             typename ImageT, typename MaskT, typename VarianceT, typename WeightT>
    StandardReturn processPixels(ImageT const &img,
                                MaskT const &msk,
                                VarianceT const &var,
                                WeightT const &weights,
                                int const,
                                int const nCrude,
                                int const stride,
                                double const meanCrude,
                                double const cliplimit,
                                bool const weightsAreMultiplicative,
                                int const andMask,
                                bool const calcErrorFromInputVariance,
                                std::vector<double> const & maskPropagationThresholds
                               )
    {
        int n = 0;
        double sumw = 0.0;              // sum(weight)  (N.b. weight will be 1.0 if !useWeights)
        double sumw2 = 0.0;             // sum(weight^2)
        double sumx = 0;                // sum(data*weight)
        double sumx2 = 0;               // sum(data*weight^2)
#if 1
        double sumvw2 = 0.0;            // sum(variance*weight^2)
#endif
        double min = (nCrude) ? meanCrude : MAX_DOUBLE;
        double max = (nCrude) ? meanCrude : -MAX_DOUBLE;

        afwImage::MaskPixel allPixelOrMask = 0x0;

        std::vector<double> rejectedWeightsByBit(maskPropagationThresholds.size(), 0.0);
    
        for (int iY = 0; iY < img.getHeight(); iY += stride) {

            typename MaskT::x_iterator mptr = msk.row_begin(iY);
            typename VarianceT::x_iterator vptr = var.row_begin(iY);
            typename WeightT::x_iterator wptr = weights.row_begin(iY);
        
            for (typename ImageT::x_iterator ptr = img.row_begin(iY), end = ptr + img.getWidth();
                 ptr != end; ++ptr, ++mptr, ++vptr, ++wptr) {
            
                if (IsFinite()(*ptr) && !(*mptr & andMask) &&
                    InClipRange()(*ptr, meanCrude, cliplimit) ) { // clip
                
                    double const delta = (*ptr - meanCrude);

                    if (useWeights) {
                        double weight = *wptr;
                        if (weightsAreMultiplicative) {
                            ;
                        } else {
                            if (*wptr <= 0) {
                                continue;
                            }
                            weight = 1/weight;
                        }
                        
                        sumw   += weight;
                        sumw2  += weight*weight;
                        sumx   += weight*delta;
                        sumx2  += weight*delta*delta;
                        
                        if (calcErrorFromInputVariance) {
                            double const var = *vptr;
                            sumvw2 += var*weight*weight;
                        }
                    } else {
                        sumx += delta;
                        sumx2 += delta*delta;

                        if (calcErrorFromInputVariance) {
                            double const var = *vptr;
                            sumvw2 += var;
                        }
                    }

                    allPixelOrMask |= *mptr;
                
                    if (HasValueLtMin()(*ptr, min)) { min = *ptr; }
                    if (HasValueGtMax()(*ptr, max)) { max = *ptr; }
                    n++;
                } else { // pixel has been clipped, rejected, etc.
                    for (int bit = 0, nBits=maskPropagationThresholds.size(); bit < nBits; ++bit) {
                        lsst::afw::image::MaskPixel mask = 1 << bit;
                        if (*mptr & mask) {
                            double weight = 1.0;
                            if (useWeights) {
                                weight = *wptr;
                                if (!weightsAreMultiplicative) {
                                    if (*wptr <= 0) {
                                        continue;
                                    }
                                    weight = 1.0 / weight;
                                }
                            }
                            rejectedWeightsByBit[bit] += weight;
                        }
                    }
                }
            }
        }
        if (n == 0) {
            min = NaN;
            max = NaN;
        }

        // estimate of population mean and variance.
        double mean, variance;
        if (!useWeights) {
            sumw = sumw2 = n;
        }

        for (int bit = 0, nBits=maskPropagationThresholds.size(); bit < nBits; ++bit) {
            double hypotheticalTotalWeight = sumw + rejectedWeightsByBit[bit];
            rejectedWeightsByBit[bit] /= hypotheticalTotalWeight;
            if (rejectedWeightsByBit[bit] > maskPropagationThresholds[bit]) {
                allPixelOrMask |= (1 << bit);
            }
        }


        // N.b. if sumw == 0 or sumw*sumw == sumw2 (e.g. n == 1) we'll get NaNs
        // N.b. the estimator of the variance assumes that the sample points all have the same variance;
        // otherwise, what is it that we're estimating?
        mean = sumx/sumw;
        variance = sumx2/sumw - ::pow(mean, 2);    // biased estimator
        variance *= sumw*sumw/(sumw*sumw - sumw2); // debias

        double meanVar;                 // (standard error of mean)^2
        if (calcErrorFromInputVariance) {
            meanVar = sumvw2/(sumw*sumw);
        } else {
            meanVar = variance*sumw2/(sumw*sumw);
        }                
        
        double varVar = varianceError(variance, n); // error in variance; incorrect if useWeights is true

        sumx += sumw*meanCrude;
        mean += meanCrude;

        return StandardReturn(n, sumx,
                              afwMath::Statistics::Value(mean, meanVar),
                              afwMath::Statistics::Value(variance, varVar), min, max, allPixelOrMask);
    }

    template<typename IsFinite,
             typename HasValueLtMin,
             typename HasValueGtMax,
             typename InClipRange,
             bool useWeights,
             typename ImageT, typename MaskT, typename VarianceT, typename WeightT>
    StandardReturn processPixels(ImageT const &img,
                                MaskT const &msk,
                                VarianceT const &var,
                                WeightT const &weights,
                                int const flags,
                                int const nCrude,
                                int const stride,
                                double const meanCrude,
                                double const cliplimit,
                                bool const weightsAreMultiplicative,
                                int const andMask,
                                bool const calcErrorFromInputVariance,
                                bool doGetWeighted,
                                std::vector<double> const & maskPropagationThresholds
                               )
    {
        if (doGetWeighted) {
            return processPixels<IsFinite, HasValueLtMin, HasValueGtMax, InClipRange, true>(
                                 img, msk, var, weights,
                                 flags, nCrude, 1, meanCrude, cliplimit, weightsAreMultiplicative, andMask,
                                 calcErrorFromInputVariance, maskPropagationThresholds);
        } else {
            return processPixels<IsFinite, HasValueLtMin, HasValueGtMax, InClipRange, false>(
                                 img, msk, var, weights,
                                 flags, nCrude, 1, meanCrude, cliplimit, weightsAreMultiplicative, andMask,
                                 calcErrorFromInputVariance, maskPropagationThresholds);
        }
    }

    template<typename IsFinite,
             typename HasValueLtMin,
             typename HasValueGtMax,
             typename InClipRange,
             bool useWeights,
             typename ImageT, typename MaskT, typename VarianceT, typename WeightT>
    StandardReturn processPixels(ImageT const &img,
                                MaskT const &msk,
                                VarianceT const &var,
                                WeightT const &weights,
                                int const flags,
                                int const nCrude,
                                int const stride,
                                double const meanCrude,
                                double const cliplimit,
                                bool const weightsAreMultiplicative,
                                int const andMask,
                                bool const calcErrorFromInputVariance,                                
                                bool doCheckFinite,
                                bool doGetWeighted,
                                std::vector<double> const & maskPropagationThresholds
                               )
    {
        if (doCheckFinite) {
            return processPixels<CheckFinite, HasValueLtMin, HasValueGtMax, InClipRange, useWeights>(
                                 img, msk, var, weights,
                                 flags, nCrude, 1, meanCrude, cliplimit,
                                 weightsAreMultiplicative, andMask, calcErrorFromInputVariance,
                                 doGetWeighted, maskPropagationThresholds);
        } else {
            return processPixels<AlwaysTrue, HasValueLtMin, HasValueGtMax, InClipRange, useWeights>(
                                 img, msk, var, weights,
                                 flags, nCrude, 1, meanCrude, cliplimit,
                                 weightsAreMultiplicative, andMask, calcErrorFromInputVariance,
                                 doGetWeighted, maskPropagationThresholds);
        }
    }

    /** =========================================================================
     * @brief Compute the standard stats: mean, variance, min, max
     *
     * @param img    an afw::Image to compute the stats over
     * @param flags  an integer (bit field indicating which statistics are to be computed
     *
     * @note An overloaded version below is used to get clipped versions
     */
    template<typename ImageT, typename MaskT, typename VarianceT, typename WeightT>
    StandardReturn getStandard(ImageT const &img,              // image
                               MaskT const &msk,               // mask
                               VarianceT const &var,           // variance
                               WeightT const &weights,         // weights to apply to each pixel
                               int const flags,                // what to measure
                               bool const weightsAreMultiplicative,  // weights are multiplicative (not inverse)
                               int const andMask,              // mask of bad pixels
                               bool const calcErrorFromInputVariance, // estimate errors from variance
                               bool doCheckFinite,             // check for NaN/Inf
                               bool doGetWeighted,             // use the weights
                               std::vector<double> const & maskPropagationThresholds
                              )
    {
        // =====================================================
        // a crude estimate of the mean, used for numerical stability of variance
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

        double cliplimit = -1;              // unused
        StandardReturn values = processPixels<ChkFin, AlwaysF, AlwaysF, AlwaysT, true>(
                                              img, msk, var, weights,
                                              flags, nCrude, strideCrude, meanCrude,
                                              cliplimit,
                                              weightsAreMultiplicative, andMask, calcErrorFromInputVariance,
                                              doCheckFinite, doGetWeighted,
                                              maskPropagationThresholds);
        nCrude = values.get<0>();
        double sumCrude = values.get<1>();
        
        meanCrude = 0.0;
        if (nCrude > 0) {
            meanCrude = sumCrude/nCrude;
        }

        // =======================================================
        // Estimate the full precision variance using that crude mean
        // - get the min and max as well
    
        if (flags & (afwMath::MIN | afwMath::MAX)) {
            return processPixels<ChkFin, ChkMin, ChkMax, AlwaysT, true>(
                                 img, msk, var, weights,
                                 flags, nCrude, 1, meanCrude,
                                 cliplimit,
                                 weightsAreMultiplicative, andMask, calcErrorFromInputVariance,
                                 true, doGetWeighted, maskPropagationThresholds);
        } else {
            return processPixels<ChkFin, AlwaysF, AlwaysF, AlwaysT,true>(
                                 img, msk, var, weights,
                                 flags, nCrude, 1, meanCrude,
                                 cliplimit,
                                 weightsAreMultiplicative, andMask, calcErrorFromInputVariance,
                                 doCheckFinite, doGetWeighted, maskPropagationThresholds);
        }
    }

    /** ==========================================================
     *
     * @brief A routine to get standard stats: mean, variance, min, max with
     *   clipping on std::pair<double,double> = center, cliplimit
     */
    template<typename ImageT, typename MaskT, typename VarianceT, typename WeightT>
    StandardReturn getStandard(ImageT const &img,                        // image
                               MaskT const &msk,                         // mask
                               VarianceT const &var,                     // variance
                               WeightT const &weights,                   // weights to apply to each pixel
                               int const flags,                          // what to measure
                               std::pair<double, double> const clipinfo, // the center and cliplimit for the
                                                                         // first clip iteration 
                               bool const weightsAreMultiplicative,  // weights are multiplicative (not inverse)
                               int const andMask,              // mask of bad pixels
                               bool const calcErrorFromInputVariance, // estimate errors from variance
                               bool doCheckFinite,             // check for NaN/Inf
                               bool doGetWeighted,             // use the weights,
                               std::vector<double> const & maskPropagationThresholds
                              )
    {
        double const center = clipinfo.first;
        double const cliplimit = clipinfo.second;

        if (std::isnan(center) || std::isnan(cliplimit)) {
            return StandardReturn(0, NaN,
                                  afwMath::Statistics::Value(NaN, NaN),
                                  afwMath::Statistics::Value(NaN, NaN), NaN, NaN, ~0x0);
        }
    
        // =======================================================
        // Estimate the full precision variance using that crude mean
        int const stride = 1;
        int nCrude = 0;
    
        if (flags & (afwMath::MIN | afwMath::MAX)) {
            return processPixels<ChkFin, ChkMin, ChkMax, ChkClip, true>(
                                 img, msk, var, weights,
                                 flags, nCrude, stride,
                                 center, cliplimit,
                                 weightsAreMultiplicative, andMask, calcErrorFromInputVariance,
                                 true, doGetWeighted, maskPropagationThresholds);
        } else {                            // fast loop ... just the mean & variance
            return processPixels<ChkFin, AlwaysF, AlwaysF, ChkClip, true>(
                                 img, msk, var, weights,
                                 flags, nCrude, stride,
                                 center, cliplimit,
                                 weightsAreMultiplicative, andMask, calcErrorFromInputVariance,
                                 doCheckFinite, doGetWeighted, maskPropagationThresholds);
        }
    }

    /** percentile()
     *
     * @brief A wrapper using the nth_element() built-in to compute percentiles for an image
     *
     * @param img       an afw::Image
     * @param quartile  the desired percentile.
     *
     */
    template<typename Pixel>
    double percentile(std::vector<Pixel> &img,
                      double const fraction)
    {
        assert (fraction >= 0.0 && fraction <= 1.0);

        int const n = img.size();

        if (n > 1) {
        
            double const idx = fraction*(n - 1);
        
            // interpolate linearly between the adjacent values
            // For efficiency:
            // - if we're asked for a fraction > 0.5,
            //    we'll do the second partial sort on shorter (upper) portion
            // - otherwise, the shorter portion will be the lower one, we'll partial-sort that.
        
            int const q1 = static_cast<int>(idx);
            int const q2 = q1 + 1;

            typename std::vector<Pixel>::iterator mid1 = img.begin() + q1;
            typename std::vector<Pixel>::iterator mid2 = img.begin() + q2;
            if (fraction > 0.5) {
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


/** medianAndQuartiles()
 *
 * @brief A wrapper using the nth_element() built-in to compute median and Quartiles for an image
 *
 * @param img       an afw::Image
 * @param quartile  the desired percentile.
 *
 */
    typedef boost::tuple<double, double, double> MedianQuartileReturn;

    template<typename Pixel>
    MedianQuartileReturn medianAndQuartiles(std::vector<Pixel> &img)
    {
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

    /*********************************************************************************************************/
    /**
     * A function to copy an image into a vector
     *
     * This is used for percentile and iq_range as these must reorder the values.
     * Because it loops over the pixels, it's been templated over the NaN test to avoid
     * code repetition of the loops.
     */
    template<typename IsFinite, typename ImageT, typename MaskT, typename VarianceT>
    std::shared_ptr<std::vector<typename ImageT::Pixel> > makeVectorCopy(ImageT const &img,
                                                                           MaskT const &msk,
                                                                           VarianceT const &,
                                                                           int const andMask
                                                                          )
    {
        // Note need to keep track of allPixelOrMask here ... processPixels() does that
        // and it always gets called
        std::shared_ptr<std::vector<typename ImageT::Pixel> >
            imgcp(new std::vector<typename ImageT::Pixel>(0));
        
        for (int i_y = 0; i_y < img.getHeight(); ++i_y) {
            typename MaskT::x_iterator mptr = msk.row_begin(i_y);
            for (typename ImageT::x_iterator ptr = img.row_begin(i_y), end = img.row_end(i_y);
                 ptr != end; ++ptr) {
                if (IsFinite()(*ptr) && !(*mptr & andMask)) {
                    imgcp->push_back(*ptr);
                }
                ++mptr;
            }
        }
        
        return imgcp;
    }
}



double afwMath::StatisticsControl::getMaskPropagationThreshold(int bit) const {
    int oldSize = _maskPropagationThresholds.size();
    if (oldSize < bit) {
        return 1.0;
    }
    return _maskPropagationThresholds[bit];
}

void afwMath::StatisticsControl::setMaskPropagationThreshold(int bit, double threshold) {
    int oldSize = _maskPropagationThresholds.size();
    if (oldSize <= bit) {
        int newSize = bit + 1;
        _maskPropagationThresholds.resize(newSize);
        for (int i = oldSize; i < bit; ++i) {
            _maskPropagationThresholds[i] = 1.0;
        }
    }
    _maskPropagationThresholds[bit] = threshold;
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
 * @brief Constructor for Statistics object
 *
 * @note Most of the actual work is done in this constructor; the results
 * are retrieved using \c getValue etc.
 *
 */
template<typename ImageT, typename MaskT, typename VarianceT>
afwMath::Statistics::Statistics(
        ImageT const &img,                      ///< Image whose properties we want
        MaskT const &msk,                       ///< Mask to control which pixels are included
        VarianceT const &var,                   ///< Variances corresponding to values in Image
        int const flags,                        ///< Describe what we want to calculate
        afwMath::StatisticsControl const& sctrl ///< Control how things are calculated
                                        ) :
    _flags(flags), _mean(NaN, NaN), _variance(NaN, NaN), _min(NaN), _max(NaN), _sum(NaN),
    _meanclip(NaN, NaN), _varianceclip(NaN, NaN), _median(NaN, NaN), _iqrange(NaN),
    _sctrl(sctrl), _weightsAreMultiplicative(false)
{
    doStatistics(img, msk, var, var, _flags, _sctrl);
}

namespace {
    template<typename T>
    bool isEmpty(T const& t) { return t.empty(); }

    template<typename T>
    bool isEmpty(afwImage::Image<T> const& im) { return (im.getWidth() == 0 && im.getHeight() == 0); }

    // Asserts that image dimensions are equal
    template<typename ImageT1, typename ImageT2>
    void checkDimensions(ImageT1 const& image1, ImageT2 const& image2)
    {
        if (image1.getDimensions() != image2.getDimensions()) {
            throw LSST_EXCEPT(pexExceptions::InvalidParameterError,
                              (boost::format("Image sizes don't match: %s vs %s") %
                               image1.getDimensions() % image2.getDimensions()).str());
        }
    }

    // Overloads for MaskImposter (which doesn't have a size)
    template<typename ImageT, typename PixelT>
    void checkDimensions(ImageT const& image1, afwMath::MaskImposter<PixelT> const& image2) {}
    template<typename ImageT, typename PixelT>
    void checkDimensions(afwMath::MaskImposter<PixelT> const& image1, ImageT const& image2) {}

}

template<typename ImageT, typename MaskT, typename VarianceT, typename WeightT>
afwMath::Statistics::Statistics(
        ImageT const &img,                      ///< Image whose properties we want
        MaskT const &msk,                       ///< Mask to control which pixels are included
        VarianceT const &var,                   ///< Variances corresponding to values in Image
        WeightT const &weights,                 ///< Weights to use corresponding to values in Image
        int const flags,                        ///< Describe what we want to calculate
        afwMath::StatisticsControl const& sctrl ///< Control how things are calculated
                                        ) :
    _flags(flags), _mean(NaN, NaN), _variance(NaN, NaN), _min(NaN), _max(NaN), _sum(NaN),
    _meanclip(NaN, NaN), _varianceclip(NaN, NaN), _median(NaN, NaN), _iqrange(NaN),
    _sctrl(sctrl), _weightsAreMultiplicative(true)
{
    if (!isEmpty(weights)) {
        if (_sctrl.getWeightedIsSet() && !_sctrl.getWeighted()) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              "You must use the weights if you provide them");
        }

        _sctrl.setWeighted(true);
    }
    doStatistics(img, msk, var, weights, _flags, _sctrl);
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WeightT>
void afwMath::Statistics::doStatistics(
    ImageT const &img,             ///< Image whose properties we want
    MaskT const &msk,              ///< Mask to control which pixels are included
    VarianceT const &var,          ///< Variances corresponding to values in Image
    WeightT const &weights,        ///< Weights to use corresponding to values in Image
    int const flags,               ///< Describe what we want to calculate
    afwMath::StatisticsControl const& sctrl ///< Control how things are calculated
                               )
{
    _n = img.getWidth()*img.getHeight();
    if (_n == 0) {
        throw LSST_EXCEPT(pexExceptions::InvalidParameterError, "Image contains no pixels");
    }
    checkDimensions(img, msk);
    checkDimensions(img, var);
    if (sctrl.getWeighted()) {
        checkDimensions(img, weights);
    }
    
    // Check that an int's large enough to hold the number of pixels
    assert(img.getWidth()*static_cast<double>(img.getHeight()) < std::numeric_limits<int>::max());

    // get the standard statistics
    StandardReturn standard = getStandard(img, msk, var, weights, flags,
                                          _weightsAreMultiplicative,
                                          _sctrl.getAndMask(),
                                          _sctrl.getCalcErrorFromInputVariance(),
                                          _sctrl.getNanSafe(), _sctrl.getWeighted(),
                                          _sctrl._maskPropagationThresholds);

    _n = standard.get<0>();
    _sum = standard.get<1>();
    _mean = standard.get<2>();
    _variance = standard.get<3>();
    _min = standard.get<4>();
    _max = standard.get<5>();
    _allPixelOrMask = standard.get<6>();

    // ==========================================================
    // now only calculate it if it's specifically requested - these all cost more!

    // copy the image for any routines that will use median or quantiles
    if (flags & (MEDIAN | IQRANGE | MEANCLIP | STDEVCLIP | VARIANCECLIP)) {

        // make a vector copy of the image to get the median and quartiles (will move values)
        std::shared_ptr<std::vector<typename ImageT::Pixel> > imgcp;
        if (_sctrl.getNanSafe()) {
            imgcp = makeVectorCopy<ChkFin>(img, msk, var, _sctrl.getAndMask());
        } else {
            imgcp = makeVectorCopy<AlwaysT>(img, msk, var, _sctrl.getAndMask());
        }

        // if we *only* want the median, just use percentile(), otherwise use medianAndQuartiles()
        if ( (flags & (MEDIAN)) && !(flags & (IQRANGE | MEANCLIP | STDEVCLIP | VARIANCECLIP)) ) {
            _median = Value(percentile(*imgcp, 0.5), NaN);
        } else {
            MedianQuartileReturn mq = medianAndQuartiles(*imgcp);
            _median = Value(mq.get<0>(), NaN);
            _iqrange = mq.get<2>() - mq.get<1>();
        }
        
        
        if (flags & (MEANCLIP | STDEVCLIP | VARIANCECLIP)) {            
            for (int i_i = 0; i_i < _sctrl.getNumIter(); ++i_i) {
                double const center = ((i_i > 0) ? _meanclip : _median).first;
                double const hwidth = (i_i > 0 && _n > 1) ?
                    _sctrl.getNumSigmaClip()*std::sqrt(_varianceclip.first) :
                    _sctrl.getNumSigmaClip()*IQ_TO_STDEV*_iqrange;
                std::pair<double, double> const clipinfo(center, hwidth);
                
                StandardReturn clipped = getStandard(img, msk, var, weights, flags, clipinfo,
                                                     _weightsAreMultiplicative,
                                                     _sctrl.getAndMask(),
                                                     _sctrl.getCalcErrorFromInputVariance(),
                                                     _sctrl.getNanSafe(), _sctrl.getWeighted(),
                                                     _sctrl._maskPropagationThresholds);
                
                int const nClip = clipped.get<0>();
                _meanclip = clipped.get<2>();     // clipped mean
                double const varClip = clipped.get<3>().first;  // clipped variance

                _varianceclip = Value(varClip, varianceError(varClip, nClip));
                // ... ignore other values
            }
        }
    }
}

/************************************************************************************************************/
/** @brief Return the value and error in the specified statistic (e.g. MEAN)
 *
 * @note Only quantities requested in the constructor may be retrieved; in particular
 * errors may not be available if you didn't specify ERROR in the constructor
 *
 * @sa getValue and getError
 *
 * @todo uncertainties on MEANCLIP,STDEVCLIP are sketchy.  _n != _nClip
 *
 */
std::pair<double, double> afwMath::Statistics::getResult(
		afwMath::Property const iProp ///< the afw::math::Property to retrieve.
                                        ///< If NOTHING (default) and you only asked for one
                                        ///< property (and maybe its error) in the constructor,
                                        ///< that property is returned
                                                        ) const {
    
    // if iProp == NOTHING try to return their heart's delight, as specified in the constructor
    afwMath::Property const prop =
        static_cast<afwMath::Property>(((iProp == NOTHING) ? _flags : iProp) & ~ERRORS);
    
    if (!(prop & _flags)) {             // we didn't calculate it
        throw LSST_EXCEPT(pexExceptions::InvalidParameterError,
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
        ret.first = _mean.first;
        if (_flags & ERRORS) {
            ret.second = ::sqrt(_mean.second);
        }
        break;
      case MEANCLIP:
        ret.first = _meanclip.first;
        if ( _flags & ERRORS ) {
            ret.second = ::sqrt(_meanclip.second);
        }
        break;
        
        // == stdevs & variances ==
      case VARIANCE:
        ret.first = _variance.first;
        if (_flags & ERRORS) {
            ret.second = ::sqrt(_variance.second);
        }
        break;
      case STDEV:
        ret.first = sqrt(_variance.first);
        if (_flags & ERRORS) {
            ret.second = 0.5*::sqrt(_variance.second)/ret.first;
        }
        break;
      case VARIANCECLIP:
        ret.first = _varianceclip.first;
        if (_flags & ERRORS) {
            ret.second = ret.second;
        }
        break;
      case STDEVCLIP:
        ret.first = sqrt(_varianceclip.first);
        if (_flags & ERRORS) {
            ret.second = 0.5*::sqrt(_varianceclip.second)/ret.first;
        }
        break;

      case MEANSQUARE:
        ret.first = (_n - 1)/static_cast<double>(_n)*_variance.first + ::pow(_mean.first, 2);
        if (_flags & ERRORS) {
            ret.second = ::sqrt(2*::pow(ret.first/_n, 2)); // assumes Gaussian
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
        ret.first = _median.first;
        if ( _flags & ERRORS ) {
            ret.second = sqrt(afwGeom::HALFPI*_variance.first/_n); // assumes Gaussian
        }
        break;
      case IQRANGE:
        ret.first = _iqrange;
        if ( _flags & ERRORS ) {
            ret.second = NaN;           // we're not estimating this properly
        }
        break;

        // no-op to satisfy the compiler
      case ERRORS:
        break;
        // default: redundant as 'ret' is initialized to NaN, NaN
      default:                          // we must have set prop to _flags
        assert (iProp == 0);
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "getValue() may only be called without a parameter"
                          " if you asked for only one statistic");
    }
    return ret;
}

/** @brief Return the value of the desired property (if specified in the constructor)
 */
double afwMath::Statistics::getValue(
		afwMath::Property const prop ///< the afw::math::Property to retrieve.
                                        ///< If NOTHING (default) and you only asked for one
                                        ///< property in the constructor, that property is returned
                                     ) const {
    return getResult(prop).first;
}


/** @brief Return the error in the desired property (if specified in the constructor)
 */
double afwMath::Statistics::getError(
		afwMath::Property const prop ///< the afw::math::Property to retrieve.
                                        ///< If NOTHING (default) and you only asked for one
                                        ///< property in the constructor, that property's error is returned
                                        ///< \note You may have needed to specify ERROR to the ctor
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
    _mean(NaN, NaN), _variance(NaN, NaN), _min(NaN), _max(NaN),
    _meanclip(NaN, NaN), _varianceclip(NaN, NaN), _median(NaN, NaN), _iqrange(NaN),
    _sctrl(sctrl) {
    
    if ((flags & ~(NPOINT | SUM)) != 0x0) {
        throw LSST_EXCEPT(pexExceptions::InvalidParameterError, "Statistics<Mask> only supports NPOINT and SUM");
    }
    
    typedef afwImage::Mask<afwImage::MaskPixel> Mask;
    
    _n = msk.getWidth()*msk.getHeight();
    if (_n == 0) {
        throw LSST_EXCEPT(pexExceptions::InvalidParameterError, "Image contains no pixels");
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

/**
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
/// \cond
//
#define STAT afwMath::Statistics

typedef afwImage::VariancePixel VPixel;

#define INSTANTIATE_MASKEDIMAGE_STATISTICS(TYPE)                       \
    template STAT::Statistics(afwImage::Image<TYPE> const &img,            \
                              afwImage::Mask<afwImage::MaskPixel> const &msk, \
                              afwImage::Image<VPixel> const &var,               \
                              int const flags, StatisticsControl const& sctrl); \
    template STAT::Statistics(afwImage::Image<TYPE> const &img,            \
                              afwImage::Mask<afwImage::MaskPixel> const &msk, \
                              afwImage::Image<VPixel> const &var,               \
                              afwImage::Image<VPixel> const &weights,   \
                              int const flags, StatisticsControl const& sctrl); \
    template STAT::Statistics(afwImage::Image<TYPE> const &img,            \
                              afwImage::Mask<afwImage::MaskPixel> const &msk, \
                              afwImage::Image<VPixel> const &var,               \
                              afwMath::ImageImposter<VPixel> const &weights,   \
                              int const flags, StatisticsControl const& sctrl)

#define INSTANTIATE_MASKEDIMAGE_STATISTICS_NO_MASK(TYPE)                       \
    template STAT::Statistics(afwImage::Image<TYPE> const &img,            \
                              afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                              afwImage::Image<VPixel> const &var,               \
                              int const flags, StatisticsControl const& sctrl); \
    template STAT::Statistics(afwImage::Image<TYPE> const &img,            \
                              afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                              afwImage::Image<VPixel> const &var,               \
                              afwImage::Image<VPixel> const &weights,   \
                              int const flags, StatisticsControl const& sctrl)

#define INSTANTIATE_MASKEDIMAGE_STATISTICS_NO_VAR(TYPE)                       \
    template STAT::Statistics(afwImage::Image<TYPE> const &img,            \
                              afwImage::Mask<afwImage::MaskPixel> const &msk, \
                              afwMath::MaskImposter<VPixel> const &var,          \
                              int const flags, StatisticsControl const& sctrl); \
    template STAT::Statistics(afwImage::Image<TYPE> const &img,            \
                              afwImage::Mask<afwImage::MaskPixel> const &msk, \
                              afwMath::MaskImposter<VPixel> const &var,          \
                              afwImage::Image<VPixel> const &weights,   \
                              int const flags, StatisticsControl const& sctrl); \
    template STAT::Statistics(afwImage::Image<TYPE> const &img,            \
                              afwImage::Mask<afwImage::MaskPixel> const &msk, \
                              afwMath::MaskImposter<VPixel> const &var,          \
                              afwMath::ImageImposter<VPixel> const &weights,   \
                              int const flags, StatisticsControl const& sctrl)

#define INSTANTIATE_REGULARIMAGE_STATISTICS(TYPE)                      \
    template STAT::Statistics(afwImage::Image<TYPE> const &img,            \
                              afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                              afwMath::MaskImposter<VPixel> const &var, \
                              int const flags, StatisticsControl const& sctrl)

#define INSTANTIATE_VECTOR_STATISTICS(TYPE)                         \
    template STAT::Statistics(afwMath::ImageImposter<TYPE> const &img,     \
                              afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                              afwMath::MaskImposter<VPixel> const &var,      \
                              int const flags, StatisticsControl const& sctrl); \
    template STAT::Statistics(afwMath::ImageImposter<TYPE> const &img,     \
                              afwMath::MaskImposter<afwImage::MaskPixel> const &msk, \
                              afwMath::MaskImposter<VPixel> const &var,      \
                              afwMath::ImageImposter<VPixel> const &weights,   \
                              int const flags, StatisticsControl const& sctrl)

#define INSTANTIATE_IMAGE_STATISTICS(T)            \
    INSTANTIATE_MASKEDIMAGE_STATISTICS(T);         \
    INSTANTIATE_MASKEDIMAGE_STATISTICS_NO_VAR(T);  \
    INSTANTIATE_MASKEDIMAGE_STATISTICS_NO_MASK(T); \
    INSTANTIATE_REGULARIMAGE_STATISTICS(T);        \
    INSTANTIATE_VECTOR_STATISTICS(T)

INSTANTIATE_IMAGE_STATISTICS(double);
INSTANTIATE_IMAGE_STATISTICS(float);
INSTANTIATE_IMAGE_STATISTICS(int);
INSTANTIATE_IMAGE_STATISTICS(boost::uint16_t);
INSTANTIATE_IMAGE_STATISTICS(boost::uint64_t);

/// \endcond

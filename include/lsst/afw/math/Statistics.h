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
 
#if !defined(LSST_AFW_MATH_STATISTICS_H)
#define LSST_AFW_MATH_STATISTICS_H
/**
 * @file Statistics.h
 * @brief Compute Image Statistics
 * @ingroup afw
 * @author Steve Bickerton
 *
 * @note The Statistics class itself can only handle image::MaskedImage() types.
 *       The philosophy has been to handle other types by making them look like
 *       image::MaskedImage() and reusing that code.
 *       Users should have no need to instantiate a Statistics object directly,
 *       but should use the overloaded makeStatistics() factory functions.
 */

#include <cassert>
#include <limits>
#include "boost/iterator/iterator_adaptor.hpp"
#include "boost/tuple/tuple.hpp"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/MaskedVector.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

namespace lsst {
namespace afw {
namespace math {
            
/**
 * @brief control what is calculated
 */
enum Property {
    NOTHING = 0x0,         ///< We don't want anything
    ERRORS = 0x1,          ///< Include errors of requested quantities
    NPOINT = 0x2,          ///< number of sample points
    MEAN = 0x4,            ///< estimate sample mean
    STDEV = 0x8,           ///< estimate sample standard deviation
    VARIANCE = 0x10,       ///< estimate sample variance
    MEDIAN = 0x20,         ///< estimate sample median
    IQRANGE = 0x40,        ///< estimate sample inter-quartile range
    MEANCLIP = 0x80,       ///< estimate sample N-sigma clipped mean (N set in StatisticsControl, default=3)
    STDEVCLIP = 0x100,     ///< estimate sample N-sigma clipped stdev (N set in StatisticsControl, default=3)
    VARIANCECLIP = 0x200,  ///< estimate sample N-sigma clipped variance
                           ///<  (N set in StatisticsControl, default=3)
    MIN = 0x400,           ///< estimate sample minimum
    MAX = 0x800,           ///< estimate sample maximum
    SUM = 0x1000,          ///< find sum of pixels in the image
    MEANSQUARE = 0x2000    ///< find mean value of square of pixel values
};

    
/**
 * @brief Pass parameters to a Statistics object
 * @ingroup afw
 *
 * A class to pass parameters which control how the stats are calculated.
 * 
 */
class StatisticsControl {
public:
    StatisticsControl(
        double numSigmaClip = 3.0, ///< number of standard deviations to clip at
        int numIter = 3,           ///< Number of iterations
        image::MaskPixel andMask = 0x0, ///< and-Mask: defines which mask bits cause a value to be ignored
        bool isNanSafe = true,     ///< flag NaNs
        bool isWeighted = false    ///< use inverse Variance plane for weights
                     ) :
        _numSigmaClip(numSigmaClip),
        _numIter(numIter),
        _andMask(andMask),
        _isNanSafe(isNanSafe),
        _isWeighted(isWeighted),
        _isMultiplyingWeights(false) {
        
        assert(_numSigmaClip > 0);
        assert(_numIter > 0);
    }

    double getNumSigmaClip() const { return _numSigmaClip; }
    int getNumIter() const { return _numIter; }
    image::MaskPixel getAndMask() const { return _andMask; }
    bool getNanSafe() const { return _isNanSafe; }
    bool getWeighted() const { return _isWeighted; }
    bool getMultiplyWeights() const { return _isMultiplyingWeights; }
    
    
    void setNumSigmaClip(double numSigmaClip) { assert(numSigmaClip > 0); _numSigmaClip = numSigmaClip; }
    void setNumIter(int numIter) { assert(numIter > 0); _numIter = numIter; }
    void setAndMask(image::MaskPixel andMask) { _andMask = andMask; }
    void setNanSafe(bool isNanSafe) { _isNanSafe = isNanSafe; }
    void setWeighted(bool isWeighted) { _isWeighted = isWeighted; }
    void setMultiplyWeights(bool isMultiplyingWeights) { _isMultiplyingWeights = isMultiplyingWeights; }
    

private:
    double _numSigmaClip;                 // Number of standard deviations to clip at
    int _numIter;                         // Number of iterations
    image::MaskPixel _andMask;            // and-Mask to specify which mask planes to pay attention to
    bool _isNanSafe;                      // Check for NaNs before running (slower)
    bool _isWeighted;                     // Use inverse variance to weight statistics.
    bool _isMultiplyingWeights;           // Treat variance plane as weights and multiply instead of dividing
};

            
/**
 * @ingroup afw
 *
 * A class to evaluate %image statistics
 *
 * The basic strategy is to construct a Statistics object from an Image and
 * a statement of what we want to know.  The desired results can then be
 * returned using Statistics methods.  A StatisticsControl object is used to
 * pass parameters.  The statistics currently implemented are listed in the
 * enum Properties in Statistics.h.
 *
 * @code
        math::StatisticsControl sctrl(3.0, 3); // sets NumSigclip (3.0), and NumIter (3) for clipping
        sctrl.setNumSigmaClip(4.0);            // reset number of standard deviations for N-sigma clipping
        sctrl.setNumIter(5);                   // reset number of iterations for N-sigma clipping
        sctrl.setAndMask(0x1);                 // ignore pixels with these mask bits set
        sctrl.setNanSafe(true);                // check for NaNs, a bit slower (default=true)
        
        math::Statistics statobj =
              math::makeStatistics(*img, math::NPOINT | math::MEAN | math::MEANCLIP, sctrl);
        
        double const n = statobj.getValue(math::NPOINT);
        std::pair<double, double> const mean = statobj.getResult(math::MEAN); // Returns (value, error)
        double const meanError = statobj.getError(math::MEAN);                // just the error
 * @endcode
 *
 * @note Factory function: We used a helper function, \c makeStatistics, rather that the constructor
 *       directly so that the compiler could deduce the types -- cf. \c std::make_pair)
 *
 * @note Inputs: The class Statistics is templatized, and makeStatistics() can take either:
 *       (1) an image, (2) a maskedImage, or (3) a std::vector<>
 *       Overloaded makeStatistics() functions then wrap what they were passed in Image/Mask-like classes
 *       and call the Statistics constructor.
 * @note Clipping: The clipping is done iteratively with numSigmaClip and numIter specified in
 *       the StatisticsControl object.  The first clip (ie. the first iteration) is performed at:
 *       median +/- numSigmaClip*IQ_TO_STDEV*IQR, where IQ_TO_STDEV=~0.74 is the conversion factor
 *       between the IQR and sigma for a Gaussian distribution.  All subsequent iterations perform
 *       clips at mean +/- numSigmaClip*stdev.
 *
 */
class Statistics {
public:
    /// The type used to report (value, error) for desired statistics
    typedef std::pair<double, double> Value;
    
    template<typename ImageT, typename MaskT, typename VarianceT>
    explicit Statistics(ImageT const &img,
                        MaskT const &msk,
                        VarianceT const &var,
                        int const flags,
                        StatisticsControl const& sctrl = StatisticsControl());

    Value getResult(Property const prop = NOTHING) const;
    
    double getError(Property const prop = NOTHING) const;
    double getValue(Property const prop = NOTHING) const;
    
private:

    // return type for _getStandard
    typedef boost::tuple<double, double, double, double, double> StandardReturn; 
    typedef boost::tuple<int, double, double, double, double, double> SumReturn; 
    typedef boost::tuple<double, double, double> MedianQuartileReturn;
    
    long _flags;                        // The desired calculation

    int _n;                             // number of pixels in the image
    double _mean;                       // the image's mean
    double _variance;                   // the image's variance
    double _min;                        // the image's minimum
    double _max;                        // the image's maximum
    double _sum;                        // the sum of all the image's pixels
    double _meanclip;                   // the image's N-sigma clipped mean
    double _varianceclip;               // the image's N-sigma clipped variance
    double _median;                     // the image's median
    double _iqrange;                    // the image's interquartile range

    StatisticsControl _sctrl;           // the control structure

    template<typename IsFinite, typename ImageT, typename MaskT, typename VarianceT>
    boost::shared_ptr<std::vector<typename ImageT::Pixel> >  _makeVectorCopy(ImageT const &img,
                                                                             MaskT const &msk,
                                                                             VarianceT const &var,
                                                                             int const flags);
        
    template<typename IsFinite,
             typename HasValueLtMin,
             typename HasValueGtMax,
             typename InClipRange,
             bool MultiplyWeight,
             typename ImageT, typename MaskT, typename VarianceT>
    SumReturn _sumImage(ImageT const &img, MaskT const &msk, VarianceT const &var, int const flags,
                        int const nCrude, int const stride = 1, double const meanCrude = 0,
                        double const cliplimit = std::numeric_limits<double>::quiet_NaN());
    
    template<typename ImageT, typename MaskT, typename VarianceT>
    StandardReturn _getStandard(ImageT const &img, MaskT const &msk, VarianceT const &var, int const flags);
    template<typename ImageT, typename MaskT, typename VarianceT>
    StandardReturn _getStandard(ImageT const &img, MaskT const &msk, VarianceT const &var,
                                int const flags, std::pair<double, double> clipinfo);

    template<typename Pixel>
    double _percentile(std::vector<Pixel> &img, double const percentile);   

    template<typename Pixel>
    MedianQuartileReturn _medianAndQuartiles(std::vector<Pixel> &img);
    
    inline double _varianceError(double const variance, int const n) const {
        return 2*(n - 1)*variance*variance/(static_cast<double>(n)*n); // assumes a Gaussian
    }

};


            
/*************************************  The factory functions **********************************/
/**
 * @brief This iterator will never increment.  It is returned by row_begin() in the MaskImposter class
 *        (below) to allow phony mask pixels to be iterated over for non-mask images within Statistics.
 * @note As the iterator always returns 0x0, the comparisons in Statistics::_getStandard() should always
 *       evaluate to 0x0 and they should be compiled out for calls to non-masked Images.
 *
 */
template <typename ValueT>
class infinite_iterator
    : public boost::iterator_adaptor<infinite_iterator<ValueT>,
                                     const ValueT*, const ValueT,
                                     boost::forward_traversal_tag> {
public:
    infinite_iterator() : infinite_iterator::iterator_adaptor_(0) {}
    explicit infinite_iterator(const ValueT* p) : infinite_iterator::iterator_adaptor_(p) {}
private:
    friend class boost::iterator_core_access;
    void increment() { ; }              // never actually advance the iterator
};
/**
 * @brief A Mask wrapper to provide an infinite_iterator for Mask::row_begin().  This allows a fake
 *        Mask to be passed in to Statistics with a regular (non-masked) Image.
 */
template<typename ValueT>
class MaskImposter {
public:
    typedef infinite_iterator<ValueT> x_iterator;
    explicit MaskImposter(ValueT val = 0) { _val[0] = val; }
    x_iterator row_begin(int) const { return x_iterator(_val); }
private:
    ValueT _val[1];
};

    
/**
 * @brief Handle a watered-down front-end to the constructor (no variance)
 * @relates Statistics
 */
template<typename Pixel>
Statistics makeStatistics(image::Image<Pixel> const &img,
                          image::Mask<image::MaskPixel> const &msk, 
                          int const flags,  
                          StatisticsControl const& sctrl = StatisticsControl() 
                         ) {
    MaskImposter<image::VariancePixel> var;
    return Statistics(img, msk, var, flags, sctrl);
}


/**
 * @brief Handle a straigh front-end to the constructor
 * @relates Statistics
 */
template<typename ImageT, typename MaskT, typename VarianceT>
Statistics makeStatistics(ImageT const &img,
                          MaskT const &msk,
                          VarianceT const &var,
                          int const flags,  
                          StatisticsControl const& sctrl = StatisticsControl() 
                         ) {
    return Statistics(img, msk, var, flags, sctrl);
}

    
/**
 * @brief Handle MaskedImages, just pass the getImage() and getMask() values right on through.
 * @relates Statistics
 */
template<typename Pixel>
Statistics makeStatistics(image::MaskedImage<Pixel, image::MaskPixel, image::VariancePixel> const &mimg, 
                          int const flags,  
                          StatisticsControl const& sctrl = StatisticsControl() 
                         ) {
    if (sctrl.getWeighted()) {
        return Statistics(*mimg.getImage(), *mimg.getMask(), *mimg.getVariance(), flags, sctrl);
    } else {
        MaskImposter<image::VariancePixel> var;
        return Statistics(*mimg.getImage(), *mimg.getMask(), var, flags, sctrl);
    }
}

/**
 * @brief Front end for specialization to handle Masks
 * @note The definition (in Statistics.cc) simply calls the specialized constructor
 * @relates Statistics
 */            
Statistics makeStatistics(image::Mask<image::MaskPixel> const &msk, 
                          int const flags,  
                          StatisticsControl const& sctrl = StatisticsControl());
            
                        
            
/**
 * @brief The makeStatistics() overload to handle regular (non-masked) Images
 * @relates Statistics
 */
template<typename Pixel>
Statistics makeStatistics(image::Image<Pixel> const &img, ///< Image (or Image) whose properties we want
                          int const flags,   ///< Describe what we want to calculate
                          StatisticsControl const& sctrl = StatisticsControl() ///< Control calculation
                         ) {
    // make a phony mask that will be compiled out
    MaskImposter<image::MaskPixel> const msk;
    MaskImposter<image::VariancePixel> const var;
    return Statistics(img, msk, var, flags, sctrl);
}


/**
 * @brief A vector wrapper to provide a vector with the necessary methods and typedefs to
 *        be processed by Statistics as though it were an Image.
 */
template<typename ValueT>
class ImageImposter {
public:
    
    // types we'll use in Statistics
    typedef typename std::vector<ValueT>::const_iterator x_iterator;
    typedef typename std::vector<ValueT>::const_iterator fast_iterator;
    typedef ValueT Pixel;

    // constructors for std::vector<>, and copy constructor
    // These are both shallow! ... no actual copying of values
    explicit ImageImposter(std::vector<ValueT> const &v) : _v(v) { }
    explicit ImageImposter(ImageImposter<ValueT> const &img) : _v(img._getVector()) {}

    // The methods we'll use in Statistics
    x_iterator row_begin(int) const { return _v.begin(); }
    x_iterator row_end(int) const { return _v.end(); }
    int getWidth() const { return _v.size(); }
    int getHeight() const { return 1; }
    
private:
    std::vector<ValueT> const &_v;                  // a private reference to the data
    std::vector<ValueT> const &_getVector() const { return _v; } // get the ref for the copyCon
};

/**
 * @brief The makeStatistics() overload to handle std::vector<>
 * @relates Statistics
 */
template<typename EntryT>
Statistics makeStatistics(std::vector<EntryT> const &v, ///< Image (or MaskedImage) whose properties we want
                          int const flags,   ///< Describe what we want to calculate
                          StatisticsControl const& sctrl = StatisticsControl() ///< Control calculation
                         ) {
    ImageImposter<EntryT> img(v);           // wrap the vector in a fake image
    MaskImposter<image::MaskPixel> msk;     // instantiate a fake mask that will be compiled out.
    MaskImposter<image::VariancePixel> var;
    return Statistics(img, msk, var, flags, sctrl);
}


/**
 * @brief The makeStatistics() overload to handle math::MaskedVector<>
 * @relates Statistics
 */
template<typename EntryT>
Statistics makeStatistics(math::MaskedVector<EntryT> const &mv, ///< MaskedVector
                          int const flags,   ///< Describe what we want to calculate
                          StatisticsControl const& sctrl = StatisticsControl() ///< Control calculation
                         ) {
    if (sctrl.getWeighted()) {
        return Statistics(*mv.getImage(), *mv.getMask(), *mv.getVariance(), flags, sctrl);
    } else {
        MaskImposter<image::VariancePixel> var;
        return Statistics(*mv.getImage(), *mv.getMask(), var, flags, sctrl);
    }
}


    
}}}

#endif

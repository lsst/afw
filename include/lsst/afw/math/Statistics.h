// -*- LSST-C++ -*-
#if !defined(LSST_AFW_MATH_STATISTICS_H)
#define LSST_AFW_MATH_STATISTICS_H
/**
 * @file Statistics.h
 * @brief Compute Image Statistics
 * @ingroup afw
 * @author Steve Bickerton
 */

#include <cassert>
#include <boost/iterator/iterator_adaptor.hpp>
#include "boost/tuple/tuple.hpp"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"

namespace image = lsst::afw::image;


namespace lsst { namespace afw { namespace math {
    
/* @brief control what is calculated
 */
enum Property {
    NOTHING = 0x0,                      ///< We don't want anything
    ERRORS = 0x1,                       ///< Include errors of requested quantities
    NPOINT = 0x2,                       ///< number of sample points
    MEAN = 0x4,                         ///< estimate sample mean
    STDEV = 0x8,                        ///< estimate sample standard deviation
    VARIANCE = 0x10,                    ///< estimate sample variance
    MEDIAN = 0x20,                      ///< estimate sample median
    IQRANGE = 0x40,                     ///< estimate sample inter-quartile range
    MEANCLIP = 0x80,                    ///< estimate sample 3 sigma clipped mean
    STDEVCLIP = 0x100,                  ///< estimate sample 3 sigma clipped stdev
    VARIANCECLIP = 0x200,               ///< estimate sample 3 sigma clipped variance
    MIN = 0x400,                        ///< estimate sample minimum
    MAX = 0x800,                        ///< estimate sample maximum
    SUM = 0x1000,                       ///< find sum of pixels in the image
};

    
/* @class Pass parameters to a Statistics object
 * @ingroup afw
 *
 * A class to pass parameters which control how the stats are calculated.
 * 
 */
class StatisticsControl {
public:
    StatisticsControl(double numSigmaClip = 3.0, ///< number of standard deviations to clip at
                      int numIter = 3,   ///< Number of iterations
                      int andMask = 0x1  ///< and-Mask to specify which mask planes to pay attention to
                     ) :
        _numSigmaClip(numSigmaClip),
        _numIter(numIter),
        _andMask(andMask)
        {
        assert(_numSigmaClip > 0);
        assert(_numIter > 0);
        }

    double getNumSigmaClip() const { return _numSigmaClip; }
    int getNumIter() const { return _numIter; }
    int getAndMask() const { return _andMask; }
    
    void setNumSigmaClip(double numSigmaClip) { assert(numSigmaClip > 0); _numSigmaClip = numSigmaClip; }
    void setNumIter(int numIter) { assert(numIter > 0); _numIter = numIter; }
    void setAndMask(int andMask) { _andMask = andMask; }

private:
    double _numSigmaClip;                 // Number of standard deviations to clip at
    int _numIter;                         // Number of iterations
    int _andMask;                         // and-Mask to specify which mask planes to pay attention to
};
            
/**
 * @class Statistics
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

        math::Statistics statobj = math::makeStatistics(*img, math::NPOINT | math::MEAN | math::MEANCLIP, sctrl);
        
        double const n = statobj.getValue(math::NPOINT);
        std::pair<double, double> const mean = statobj.getResult(math::MEAN); // Returns (value, error)
        double const meanError = statobj.getError(math::MEAN);                // just the error
 * @endcode
 *
 * @note we used a helper function, \c makeStatistics, rather that the constructor directly so that
 *       the compiler could deduce the types -- cf. \c std::make_pair)
 */
class Statistics {
public:
    /// The type used to report (value, error) for desired statistics
    typedef std::pair<double, double> value_type;
    
    template<typename Image, typename Mask>
    explicit Statistics(Image &img,
                        Mask &msk,
                        int const flags,
                        StatisticsControl const& sctrl=StatisticsControl());
    
    value_type getResult(Property const prop=NOTHING) const;
    
    double getError(Property const prop=NOTHING) const;
    double getValue(Property const prop=NOTHING) const;
    
private:
    typedef boost::tuple<double, double, double, double, double> StandardReturnT; // return type for _getStandard

    long _flags;                        // The desired calculation

    int _n;                             // number of pixels in the image
    double _mean;                       // the image's mean
    double _variance;                   // the image's variance
    double _min;                        // the image's minimum
    double _max;                        // the image's maximum
    double _sum;                        // the sum of all the image's pixels
    double _meanclip;                   // the image's 3-sigma clipped mean
    double _varianceclip;               // the image's 3-sigma clipped variance
    double _median;                     // the image's median
    double _iqrange;                    // the image's interquartile range

    StatisticsControl _sctrl;           // the control structure
    
    template<typename Image, typename Mask>
    StandardReturnT _getStandard(Image &img, Mask &msk, int const flags);
    template<typename Image, typename Mask>
    StandardReturnT _getStandard(Image &img, Mask &msk,
                                 int const flags, std::pair<double,double> clipinfo);
    
    template<typename Image>
    double _percentile(Image &img,
                       double const quartile);   // compute median with quickselect (Press et al.)
    
    inline double _varianceError(double const variance, int const n) const {
        return 2*(n - 1)*variance*variance/(static_cast<double>(n)*n); // assumes a Gaussian
    }

};






/*
 * MaskedImage
 */
template<typename Pixel>
Statistics makeStatistics(image::MaskedImage<Pixel> &mimg, ///< Image (or MaskedImage) whose properties we want
                          int const flags,   ///< Describe what we want to calculate
                          StatisticsControl const& sctrl=StatisticsControl() ///< Control how things are calculated
                         ) {
    return Statistics(*mimg.getImage(), *mimg.getMask(), flags, sctrl);
}


            
/*
 * Image
 */
            
template <typename ValueT>
class infinite_iterator
    : public boost::iterator_adaptor<infinite_iterator<ValueT>, ValueT*, ValueT, boost::forward_traversal_tag> {
public:
    infinite_iterator() : infinite_iterator::iterator_adaptor_(0) {}
    explicit infinite_iterator(ValueT* p) : infinite_iterator::iterator_adaptor_(p) {}
private:
    friend class boost::iterator_core_access;
    void increment() { ; }              // never actually advance the iterator
};

template<typename ValueT>
class MaskImposter {
    ValueT _val[1];
public:
    typedef infinite_iterator<ValueT> x_iterator;
    MaskImposter(ValueT val = 0) { _val[0] = val; }
    x_iterator row_begin(int) {
        return x_iterator(_val);
    }
};
            

/* @brief 
 * @ingroup afw
 */
template<typename Pixel>
Statistics makeStatistics(image::Image<Pixel> &img, ///< Image (or Image) whose properties we want
                          int const flags,   ///< Describe what we want to calculate
                          StatisticsControl const& sctrl=StatisticsControl() ///< Control how things are calculated
                         ) {
    MaskImposter<image::MaskPixel> msk;
    return Statistics(img, msk, flags, sctrl);
}


/*
 * Vector
 */


template<typename ValueT>
class ImageImposter : public std::vector<ValueT> {
public:
    
    typedef typename std::vector<ValueT>::iterator x_iterator;
    typedef typename std::vector<ValueT>::iterator fast_iterator;
    typedef typename boost::shared_ptr<ImageImposter<ValueT> > Ptr;
    
    ImageImposter(std::vector<ValueT> &v) : std::vector<ValueT>(v.begin(), v.end()) {}
    ImageImposter(ImageImposter<ValueT> &v) : 
        std::vector<ValueT>(v.row_begin(0), v.row_end(0)) {}
    ImageImposter(ImageImposter<ValueT> &v, bool deep) :
        std::vector<ValueT>(v.row_begin(0), v.row_end(0)) {}

    //~ImageImposter() {}
    
    fast_iterator begin(bool) { return std::vector<ValueT>::begin(); }
    x_iterator row_begin(int) { return std::vector<ValueT>::begin(); }
    x_iterator row_end(int) { return std::vector<ValueT>::end();   }
    int getWidth()  { return std::vector<ValueT>::size(); }
    int getHeight() { return 1; }

private:
};
            

template<typename EntryT>
Statistics makeStatistics(std::vector<EntryT> &v, ///< Image (or MaskedImage) whose properties we want
                          int const flags,   ///< Describe what we want to calculate
                          StatisticsControl const& sctrl=StatisticsControl() ///< Control how things are calculated
                         ) {
    ImageImposter<EntryT> img(v);
    MaskImposter<image::MaskPixel> msk;
    return Statistics(img, msk, flags, sctrl);
}
            
}}}

#endif

// -*- LSST-C++ -*-
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

namespace image = lsst::afw::image;

namespace lsst {
namespace afw {
namespace math {
            
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
                      int numIter = 3,     ///< Number of iterations
                      image::MaskPixel andMask = ~0x0, ///< and-Mask to specify planes to use
                      bool isNanSafe = true  ///< flag NaNs
                     ) :
        _numSigmaClip(numSigmaClip),
        _numIter(numIter),
        _andMask(andMask),
        _isNanSafe(isNanSafe) {
        
        assert(_numSigmaClip > 0);
        assert(_numIter > 0);
    }

    double getNumSigmaClip() const { return _numSigmaClip; }
    int getNumIter() const { return _numIter; }
    image::MaskPixel getAndMask() const { return _andMask; }
    bool getNanSafe() const { return _isNanSafe; }
    
    void setNumSigmaClip(double numSigmaClip) { assert(numSigmaClip > 0); _numSigmaClip = numSigmaClip; }
    void setNumIter(int numIter) { assert(numIter > 0); _numIter = numIter; }
    void setAndMask(image::MaskPixel andMask) { _andMask = andMask; }
    void setNanSafe(bool isNanSafe) { _isNanSafe = isNanSafe; }

private:
    double _numSigmaClip;                 // Number of standard deviations to clip at
    int _numIter;                         // Number of iterations
    image::MaskPixel _andMask;            // and-Mask to specify which mask planes to pay attention to
    bool _isNanSafe;                         // Check for NaNs before running (slower)
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
        sctrl.setAndMask(0x1);                 // ignore pixels with these mask bits set
        sctrl.setNanSafe(true);                // check for NaNs, a bit slower (default=true)
        
        math::Statistics statobj =
              math::makeStatistics(*img, math::NPOINT | math::MEAN | math::MEANCLIP, sctrl);
        
        double const n = statobj.getValue(math::NPOINT);
        std::pair<double, double> const mean = statobj.getResult(math::MEAN); // Returns (value, error)
        double const meanError = statobj.getError(math::MEAN);                // just the error
 * @endcode
 *
 * @note we used a helper function, \c makeStatistics, rather that the constructor directly so that
 *       the compiler could deduce the types -- cf. \c std::make_pair)
 *
 * @note The class Statistics is templatized over Image and Mask, and makeStatistics() can take either:
 *       (1) an image, (2) a maskedImage, or (3) a std::vector<>
 *       Overloaded makeStatistics() functions then wrap what they were passed in Image/Mask-like classes
 *       and call the Statistics constructor.
 */
class Statistics {
public:
    /// The type used to report (value, error) for desired statistics
    typedef std::pair<double, double> Value;
    
    template<typename Image, typename Mask>
    explicit Statistics(Image const &img,
                        Mask const &msk,
                        int const flags,
                        StatisticsControl const& sctrl = StatisticsControl());

    Value getResult(Property const prop = NOTHING) const;
    
    double getError(Property const prop = NOTHING) const;
    double getValue(Property const prop = NOTHING) const;
    
private:
    
    // return type for _getStandard
    typedef boost::tuple<double, double, double, double, double> StandardReturn; 
    typedef boost::tuple<int, double, double, double, double> SumReturn; 

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
    SumReturn _sumImage(Image const &img, Mask const &msk, int const flags,
                        int const nCrude, double const meanCrude = 0);

    template<typename Image, typename Mask>
    StandardReturn _getStandard(Image const &img, Mask const &msk, int const flags);
    template<typename Image, typename Mask>
    StandardReturn _getStandard(Image const &img, Mask const &msk,
                                int const flags, std::pair<double, double> clipinfo);

    template<typename Pixel>
    double _percentile(std::vector<Pixel> &img, double const quartile);   
    
    inline double _varianceError(double const variance, int const n) const {
        return 2*(n - 1)*variance*variance/(static_cast<double>(n)*n); // assumes a Gaussian
    }

};


            
/*************************************  The factory functions **********************************/

/*
 * @brief Handle MaskedImages, just pass the getImage() and getMask() values right on through.
 *
 */
template<typename Pixel>
Statistics makeStatistics(image::MaskedImage<Pixel, image::MaskPixel, image::VariancePixel> const &mimg, 
                          int const flags,  
                          StatisticsControl const& sctrl = StatisticsControl() 
                         ) {
    return Statistics(*mimg.getImage(), *mimg.getMask(), flags, sctrl);
}

/*
 * @brief Front end for specialization to handle Masks
 * @note The definition (in Statistics.cc) simply calls the specialized constructor
 *
 */            
Statistics makeStatistics(image::Mask<image::MaskPixel> const &msk, 
                          int const flags,  
                          StatisticsControl const& sctrl = StatisticsControl());
            

/*
 * @class infinite_iterator
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

            
/*
 * @class MaskImposter
 * @brief A Mask wrapper to provide an infinite_iterator for Mask::row_begin().  This allows a fake
 *        Mask to be passed in to Statistics with a regular (non-masked) Image.
 */
template<typename ValueT>
class MaskImposter {
public:
    typedef infinite_iterator<ValueT> x_iterator;
    MaskImposter(ValueT val = 0) { _val[0] = val; }
    x_iterator row_begin(int dummyCol) const { return x_iterator(_val); }
private:
    ValueT _val[1];
};
            
            
/*
 * @brief The makeStatistics() overload to handle regular (non-masked) Images
 */
template<typename Pixel>
Statistics makeStatistics(image::Image<Pixel> const &img, ///< Image (or Image) whose properties we want
                          int const flags,   ///< Describe what we want to calculate
                          StatisticsControl const& sctrl = StatisticsControl() ///< Control calculation
                         ) {
    // make a phony mask that will be compiled out
    MaskImposter<image::MaskPixel> msk;
    return Statistics(img, msk, flags, sctrl);
}


/*
 * @class ImageImposter
 *
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
    x_iterator row_begin(int dummyCol) const { return _v.begin(); }
    x_iterator row_end(int dummyCol) const { return _v.end(); }
    int getWidth() const { return _v.size(); }
    int getHeight() const { return 1; }
    
private:
    std::vector<ValueT> const &_v;                  // a private reference to the data
    std::vector<ValueT> const &_getVector() const { return _v; } // get the ref for the copyCon
};

/*
 * @brief The makeStatistics() overload to handle std::vector<> 
 */
template<typename EntryT>
Statistics makeStatistics(std::vector<EntryT> const &v, ///< Image (or MaskedImage) whose properties we want
                          int const flags,   ///< Describe what we want to calculate
                          StatisticsControl const& sctrl = StatisticsControl() ///< Control calculation
                         ) {
    
    ImageImposter<EntryT> img(v);           // wrap the vector in a fake image
    MaskImposter<image::MaskPixel> msk;     // instantiate a fake mask that will be compiled out. 
    return Statistics(img, msk, flags, sctrl);
}
            
}}}

#endif

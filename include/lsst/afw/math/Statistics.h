#if !defined(LSST_AFW_MATH_STATISTICS_H)
#define LSST_AFW_MATH_STATISTICS_H
/**
 * \file
 * \brief Image Statistics
 */

#include "boost/tuple/tuple.hpp"

namespace lsst { namespace afw { namespace math {
    
/// \brief control what is calculated
enum Property {
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
};

/// \brief Pass parameters to a Statistics object
class StatisticsControl {
public:
    StatisticsControl(double numSigmaClip = 3.0, ///< number of standard deviations to clip at
                      int numIter = 3   ///< Number of iterations
                     ) : _numSigmaClip(numSigmaClip), _numIter(numIter) {
        assert(_numSigmaClip > 0);
        assert(_numIter > 0);
    }
    double getNumSigmaClip() const { return _numSigmaClip; }
    int getNumIter() const { return _numIter; }
    void setNumSigmaClip(double numSigmaClip) { assert(numSigmaClip > 0); _numSigmaClip = numSigmaClip; }
    void setNumIter(int numIter) { assert(numIter > 0); _numIter = numIter; }

private:
    double _numSigmaClip;                 // Number of standard deviations to clip at
    int _numIter;                         // Number of iterations
};
            
/**
 * A class to evaluate %image statistics
 *
 * The basic strategy is to construct a Statistics object from an Image and
 * a statement of what we want to know.  The desired results can then be
 * returned using Statistics methods.  A StatisticsControl object is used to
 * pass parameters.  The statistics currently implemented are listed in the
 * enum Properties in Statistics.h.
 * \code
        math::StatisticsControl sctrl(3.0, 3); // sets NumSigclip (3.0), and NumIter (3) for clipping
        sctrl.setNumSigmaClip(4.0);            // reset number of standard deviations for N-sigma clipping
        sctrl.setNumIter(5);                   // reset number of iterations for N-sigma clipping

        math::Statistics statobj = math::make_Statistics(*img, math::NPOINT | math::MEAN | math::MEANCLIP, sctrl);
        
        double const n = statobj.getValue(math::NPOINT);
        std::pair<double, double> const mean = statobj.getResult(math::MEAN); // Returns (value, error)
        double const meanError = statobj.getError(math::MEAN);                // just the error
 * \endcode
 *
 * (Note that we used a helper function, \c make_Statistics, rather that the constructor directly so that
 * the compiler could deduce the types -- cf. \c std::make_pair)
 */
class Statistics {
public:
    /// The type used to report (value, error) for desired statistics
    typedef std::pair<double, double> value_type;
    
    template<typename Image>
    explicit Statistics(Image const& img, int const flags,
                        StatisticsControl const& sctrl=StatisticsControl());
    
    value_type getResult(Property const prop) const;
    
    double getError(Property const prop) const;
    double getValue(Property const prop) const;
    
private:
    long _flags;                        // The desired calculation

    int _n;                             // number of pixels in the image
    double _mean;                       // the image's mean
    double _variance;                   // the image's variance
    double _min;                        // the image's minimum
    double _max;                        // the image's maximum
    double _meanclip;                   // the image's 3-sigma clipped mean
    double _varianceclip;               // the image's 3-sigma clipped variance
    double _median;                     // the image's median
    double _iqrange;                    // the image's interquartile range

    template<typename Image>
    boost::tuple<double, double, double, double> _getStandard(Image const& img, int const flags);
    template<typename Image>
    boost::tuple<double, double, double, double> _getStandard(Image const& img, int const flags,
                                                              std::pair<double,double> clipinfo);
    
    template<typename Image>
    double _quickSelect(Image const& img, double const quartile);   // compute median with quickselect (Press et al.)
    
    inline double _varianceError(double const variance, int const n) const {
        return 2*(n - 1)*variance*variance/(static_cast<double>(n)*n); // assumes a Gaussian
    }

};

/// A convenience function that uses function overloading to make the correct type of Statistics
///
/// cf. std::make_pair()
template<typename Image>
Statistics make_Statistics(Image const& img, ///< Image (or MaskedImage) whose properties we want
                           int const flags,   ///< Describe what we want to calculate
                           StatisticsControl const& sctrl=StatisticsControl() ///< Control how things are calculated
                          ) {
    return Statistics(img, flags, sctrl);
}

}}}

#endif

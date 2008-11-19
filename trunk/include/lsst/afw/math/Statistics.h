#if !defined(LSST_AFW_MATH_STATISTICS_H)
#define LSST_AFW_MATH_STATISTICS_H
/**
 * \file
 * \brief Image Statistics
 */
namespace lsst { namespace afw { namespace math {
/// \brief control what is calculated
enum Property {
    ERRORS = 0x1,                       ///< Include errors of requested quantities
    NPOINT = 0x2,                       ///< number of sample points
    MEAN = 0x4,                         ///< estimate sample mean
    STDEV = 0x8,                        ///< estimate sample standard deviation
    VARIANCE = 0x10,                    ///< estimate sample variance
};
            
/**
 * A class to evaluate %image statistics
 *
 * The basic strategy is to construct a Statistics object from an Image and
 * a statement of what we want to know.  The desired results can then be
 * returned using Statistics methods:
 * \code
        math::Statistics<ImageT> stats = math::make_Statistics(*img, math::NPOINT | math::MEAN);
        
        double const n = stats.getValue(math::NPOINT);
        std::pair<double, double> const mean = stats.getResult(math::MEAN); // Returns (value, error)
        double const meanError = stats.getError(math::MEAN);                // just the error
 * \endcode
 *
 * (Note that we used a helper function, \c make_Statistics, rather that the constructor directly so that
 * the compiler could deduce the types -- cf. \c std::make_pair)
 */
template<typename Image>
class Statistics {
public:
    /// The type used to report (value, error) for desired statistics
    typedef std::pair<double, double> value_type;
    
    explicit Statistics(Image const& img, int const flags);
    value_type getResult(Property const prop) const;

    double getError(Property const prop) const;
    double getValue(Property const prop) const;
    
private:
    long _flags;                        // The desired calculation

    int _n;                             // number of pixels in the image
    double _mean;                       // the image's mean
    double _variance;                   // the image's variance
};

/// A convenience function that uses function overloading to make the correct type of Statistics
///
/// cf. std::make_pair()
template<typename Image>
Statistics<Image> make_Statistics(Image const& img, ///< Image (or MaskedImage) whose properties we want
                                  int const flags   ///< Describe what we want to calculate
                                 ) {
    return Statistics<Image>(img, flags);
}

}}}

#endif

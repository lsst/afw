#if !defined(LSST_AFW_MATH_STATISTICS_H)
#define LSST_AFW_MATH_STATISTICS_H

namespace lsst { namespace afw { namespace math {
/// @brief control what is calculated
enum Property {
    ERRORS = 0x1,                       ///< Include errors of requested quantities
    NPOINT = 0x2,                       ///< number of sample points
    MEAN = 0x4,                         ///< estimate sample mean
    STDEV = 0x8,                        ///< estimate sample standard deviation
    VARIANCE = 0x10,                    ///< estimate sample variance
};
            
template<typename Image>
class Statistics {
public:
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

template<typename Image>
Statistics<Image> make_Statistics(Image const& img, ///< Image (or MaskedImage) whose properties we want
                                  int const flags   ///< Describe what we want to calculate
                                 ) {
    return Statistics<Image>(img, flags);
}

}}}

#endif

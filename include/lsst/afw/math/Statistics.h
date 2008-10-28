#if !defined(LSST_AFW_MATH_STATISTICS_H)
#define LSST_AFW_MATH_STATISTICS_H

namespace lsst { namespace afw { namespace math {

enum Property { ERRORS = 0x1, MEAN = 0x2, STDEV = 0x4, };
            
template<typename Image>
class Statistics {
public:
    typedef std::pair<double, double> value_type;
    
    explicit Statistics(Image const& img, int const flags);
    value_type getParameter(Property const prop) const;

    double getError(Property const prop) const;
    double getValue(Property const prop) const;
    
private:
    long _flags;                        // The desired calculation
};

template<typename Image>
Statistics<Image> make_Statistics(Image const& img, ///< Image (or MaskedImage) whose properties we want
                                  int const flags   ///< Describe what we want to calculate
                                 ) {
    return Statistics<Image>(img, flags);
}

}}}

#endif

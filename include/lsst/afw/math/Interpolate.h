// -*- LSST-C++ -*-
#if !defined(LSST_AFW_MATH_INTERPOLATE_H)
#define LSST_AFW_MATH_INTERPOLATE_H

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
#include "lsst/base.h"

namespace lsst {
namespace afw {
namespace math {

 /**
 * @brief Interpolate values for a set of x,y vector<>s
 * @ingroup afw
 * @author Steve Bickerton
 */
class Interpolate {
public:
    enum Style {
        UNKNOWN = -1,
        CONSTANT = 0,
        LINEAR = 1,
        NATURAL_SPLINE = 2,
        CUBIC_SPLINE = 3,
        CUBIC_SPLINE_PERIODIC = 4,
        AKIMA_SPLINE = 5,
        AKIMA_SPLINE_PERIODIC = 6,
        NUM_STYLES
    };

    friend PTR(Interpolate) makeInterpolate(std::vector<double> const &x, std::vector<double> const &y,
                                            Interpolate::Style const style);
    
    virtual ~Interpolate() {}
    virtual double interpolate(double const x) const = 0;
protected:
    /**
     * Base class ctor
     */
    Interpolate(std::vector<double> const &x, ///< the ordinates of points
                std::vector<double> const &y, ///< the values at x[]
                Interpolate::Style const style=UNKNOWN ///< desired interpolator
               ) : _x(x), _y(y), _style(style) {}
    Interpolate(std::pair<std::vector<double>, std::vector<double> > const xy,
                Interpolate::Style const style=UNKNOWN);

    std::vector<double> const _x;
    std::vector<double> const _y;
    Interpolate::Style const _style;
private:
    Interpolate(Interpolate const&);
    Interpolate& operator=(Interpolate const&);
};

PTR(Interpolate) makeInterpolate(std::vector<double> const &x, std::vector<double> const &y,
                                 Interpolate::Style const style=Interpolate::AKIMA_SPLINE);
    
Interpolate::Style stringToInterpStyle(std::string const &style);
Interpolate::Style lookupMaxInterpStyle(int const n);
int lookupMinInterpPoints(Interpolate::Style const style);
        
}}}
                     
#endif // LSST_AFW_MATH_INTERPOLATE_H

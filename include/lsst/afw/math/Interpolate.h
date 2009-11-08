// -*- LSST-C++ -*-
#if !defined(LSST_AFW_MATH_INTERPOLATE_H)
#define LSST_AFW_MATH_INTERPOLATE_H
/**
 * @file Interpolate.h
 * @brief Wrap GSL to interpolate values for a set of x,y vector<>s
 * @ingroup afw
 * @author Steve Bickerton
 */
#include <limits>
#include <map>
#include "gsl/gsl_interp.h"
#include "gsl/gsl_spline.h"
#include "boost/shared_ptr.hpp"

#include "lsst/pex/exceptions.h"

namespace lsst {
namespace afw {
namespace math {

namespace Interp {
    enum Style {
        CONSTANT = 0,
        LINEAR = 1,
        NATURAL_SPLINE = 2,
        CUBIC_SPLINE = 3,
        CUBIC_SPLINE_PERIODIC = 4,
        AKIMA_SPLINE = 5,
        AKIMA_SPLINE_PERIODIC = 6,
        NUM_STYLES
    };
}
    
::gsl_interp_type const *styleToGslInterpType(Interp::Style const style);
::gsl_interp_type const *stringToGslInterpType(std::string const style);
Interp::Style stringToInterpStyle(std::string const style);
    
class Interpolate {
public:

    Interpolate(std::vector<double> const &x, std::vector<double> const &y,
                   ::gsl_interp_type const *gslInterpType = ::gsl_interp_akima) :
        _x(x), _y(y) {
        initialize(_x, _y, gslInterpType);
    }

    Interpolate(std::vector<double> const &x, std::vector<double> const &y,
                Interp::Style const style) :
        _x(x), _y(y) {
        if (style == Interp::CONSTANT) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              "CONSTANT interpolation not supported.");
        }
        initialize(_x, _y, math::styleToGslInterpType(style));
    }

    Interpolate(std::vector<double> const &x, std::vector<double> const &y,
                std::string style) :
        _x(x), _y(y) {
        if (style == "CONSTANT") {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              "CONSTANT interpolation not supported.");
        }
        initialize(_x, _y, math::stringToGslInterpType(style));
    }
    
    void initialize(std::vector<double> const &x, std::vector<double> const &y,
                    ::gsl_interp_type const *gslInterpType) {
        _acc    = ::gsl_interp_accel_alloc();
        _interp = ::gsl_interp_alloc(gslInterpType, y.size());
        ::gsl_interp_init(_interp, &x[0], &y[0], y.size());
    }
    
    virtual ~Interpolate() {
        ::gsl_interp_free(_interp);
        ::gsl_interp_accel_free(_acc);
    }

    double interpolate(double const x) {
        return ::gsl_interp_eval(_interp, &_x[0], &_y[0], x, _acc);
    }
    
private:
    std::vector<double> const &_x;
    std::vector<double> const &_y;
    ::gsl_interp_accel *_acc;
    ::gsl_interp *_interp;
};


    
/**
 * @brief Conversion function to switch an Interp::Style to a gsl_interp_type.
 *
 */
::gsl_interp_type const *styleToGslInterpType(Interp::Style const style) {
    ::gsl_interp_type const* gslInterpTypeStyles[7];
    gslInterpTypeStyles[Interp::CONSTANT]                 = ::gsl_interp_linear;           
    gslInterpTypeStyles[Interp::LINEAR]                   = ::gsl_interp_linear;           
    gslInterpTypeStyles[Interp::CUBIC_SPLINE]             = ::gsl_interp_cspline;          
    gslInterpTypeStyles[Interp::NATURAL_SPLINE]           = ::gsl_interp_cspline;          
    gslInterpTypeStyles[Interp::CUBIC_SPLINE_PERIODIC]    = ::gsl_interp_cspline_periodic; 
    gslInterpTypeStyles[Interp::AKIMA_SPLINE]             = ::gsl_interp_akima;            
    gslInterpTypeStyles[Interp::AKIMA_SPLINE_PERIODIC]    = ::gsl_interp_akima_periodic;
    return gslInterpTypeStyles[style];
}
    
/**
 * @brief Conversion function to switch a string to an Interp::Style.
 *
 */
Interp::Style stringToInterpStyle(std::string const style) {
    std::map<std::string, Interp::Style> gslInterpTypeStrings;
    gslInterpTypeStrings["CONSTANT"]              = Interp::CONSTANT;
    gslInterpTypeStrings["LINEAR"]                = Interp::LINEAR;               
    gslInterpTypeStrings["CUBIC_SPLINE"]          = Interp::CUBIC_SPLINE;         
    gslInterpTypeStrings["NATURAL_SPLINE"]        = Interp::NATURAL_SPLINE;      
    gslInterpTypeStrings["CUBIC_SPLINE_PERIODIC"] = Interp::CUBIC_SPLINE_PERIODIC;
    gslInterpTypeStrings["AKIMA_SPLINE"]          = Interp::AKIMA_SPLINE;  
    gslInterpTypeStrings["AKIMA_SPLINE_PERIODIC"] = Interp::AKIMA_SPLINE_PERIODIC;
    return gslInterpTypeStrings[style];
}

/**
 * @brief Conversion function to switch a string to a gsl_interp_type.
 *
 */
::gsl_interp_type const *stringToGslInterpType(std::string const style) {
    return styleToGslInterpType(stringToInterpStyle(style));
}
    
    
/**
 * @brief Get the highest order Interpolation::Style available for 'n' points.
 *
 */
Interp::Style lookupMaxInterpStyle(int const n) {
    if (n < 1) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, "nx,ny must be greater than 0");
    }
    if (n > 4) {
        return Interp::AKIMA_SPLINE;
    }

    std::vector<Interp::Style> styles(4);
    styles[0] = Interp::CONSTANT;
    styles[1] = Interp::LINEAR;
    styles[2] = Interp::CUBIC_SPLINE;
    styles[3] = Interp::CUBIC_SPLINE;
    return styles[n - 1];
}

    
/**
 * @brief Get the minimum number of points needed to use the requested interpolation style
 *
 */
int lookupMinInterpPoints(Interp::Style const style) {
    std::vector<int> minPoints(Interp::NUM_STYLES);
    minPoints[Interp::CONSTANT]               = 1;
    minPoints[Interp::LINEAR]                 = 2;
    minPoints[Interp::NATURAL_SPLINE]         = 3;
    minPoints[Interp::CUBIC_SPLINE]           = 3;
    minPoints[Interp::CUBIC_SPLINE_PERIODIC]  = 3;
    minPoints[Interp::AKIMA_SPLINE]           = 5;
    minPoints[Interp::AKIMA_SPLINE_PERIODIC]  = 5;
    return minPoints[style];
}

/**
 * @brief Get the minimum number of points needed to use the requested interpolation style
 *
 * Overload of lookupMinInterpPoints() which takes a string
 *
 */
int lookupMinInterpPoints(std::string const style) {
    return lookupMinInterpPoints(stringToInterpStyle(style));
}
    
        
}}}
                     
#endif // LSST_AFW_MATH_INTERPOLATE_H

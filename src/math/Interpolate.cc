// -*- LSST-C++ -*-
/**
 * @file Interpolate.cc
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
#include "lsst/afw/math/Interpolate.h"

namespace math = lsst::afw::math;
namespace ex = lsst::pex::exceptions;

math::Interpolate::Interpolate(std::vector<double> const &x, std::vector<double> const &y,
                               ::gsl_interp_type const *gslInterpType) : _x(x), _y(y) {
    initialize(_x, _y, gslInterpType);
}

math::Interpolate::Interpolate(std::vector<double> const &x, std::vector<double> const &y,
                               Interpolate::Style const style) :  _x(x), _y(y) {
    if (style == Interpolate::CONSTANT) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "CONSTANT interpolation not supported.");
    }
    initialize(_x, _y, math::styleToGslInterpType(style));
}

math::Interpolate::Interpolate(std::vector<double> const &x, std::vector<double> const &y,
                               std::string style) :  _x(x), _y(y) {
    if (style == "CONSTANT") {
        throw LSST_EXCEPT(ex::InvalidParameterException, "CONSTANT interpolation not supported.");
    }
    initialize(_x, _y, math::stringToGslInterpType(style));
}
    
void math::Interpolate::initialize(std::vector<double> const &x, std::vector<double> const &y,
                                   ::gsl_interp_type const *gslInterpType) {
    _acc    = ::gsl_interp_accel_alloc();
    _interp = ::gsl_interp_alloc(gslInterpType, y.size());
    ::gsl_interp_init(_interp, &x[0], &y[0], y.size());
}

math::Interpolate::~Interpolate() {
    ::gsl_interp_free(_interp);
    ::gsl_interp_accel_free(_acc);
}

double math::Interpolate::interpolate(double const x) {
    return ::gsl_interp_eval(_interp, &_x[0], &_y[0], x, _acc);
}

/**
 * @brief Conversion function to switch an Interpolate::Style to a gsl_interp_type.
 *
 */
::gsl_interp_type const *math::styleToGslInterpType(Interpolate::Style const style) {
    ::gsl_interp_type const* gslInterpTypeStyles[7];
    gslInterpTypeStyles[Interpolate::CONSTANT]                 = ::gsl_interp_linear;           
    gslInterpTypeStyles[Interpolate::LINEAR]                   = ::gsl_interp_linear;           
    gslInterpTypeStyles[Interpolate::CUBIC_SPLINE]             = ::gsl_interp_cspline;          
    gslInterpTypeStyles[Interpolate::NATURAL_SPLINE]           = ::gsl_interp_cspline;          
    gslInterpTypeStyles[Interpolate::CUBIC_SPLINE_PERIODIC]    = ::gsl_interp_cspline_periodic; 
    gslInterpTypeStyles[Interpolate::AKIMA_SPLINE]             = ::gsl_interp_akima;            
    gslInterpTypeStyles[Interpolate::AKIMA_SPLINE_PERIODIC]    = ::gsl_interp_akima_periodic;
    return gslInterpTypeStyles[style];
}
    
/**
 * @brief Conversion function to switch a string to an Interpolate::Style.
 *
 */
math::Interpolate::Style math::stringToInterpStyle(std::string const style) {
    std::map<std::string, Interpolate::Style> gslInterpTypeStrings;
    gslInterpTypeStrings["CONSTANT"]              = Interpolate::CONSTANT;
    gslInterpTypeStrings["LINEAR"]                = Interpolate::LINEAR;               
    gslInterpTypeStrings["CUBIC_SPLINE"]          = Interpolate::CUBIC_SPLINE;         
    gslInterpTypeStrings["NATURAL_SPLINE"]        = Interpolate::NATURAL_SPLINE;      
    gslInterpTypeStrings["CUBIC_SPLINE_PERIODIC"] = Interpolate::CUBIC_SPLINE_PERIODIC;
    gslInterpTypeStrings["AKIMA_SPLINE"]          = Interpolate::AKIMA_SPLINE;  
    gslInterpTypeStrings["AKIMA_SPLINE_PERIODIC"] = Interpolate::AKIMA_SPLINE_PERIODIC;
    return gslInterpTypeStrings[style];
}

/**
 * @brief Conversion function to switch a string to a gsl_interp_type.
 *
 */
::gsl_interp_type const *math::stringToGslInterpType(std::string const style) {
    return styleToGslInterpType(stringToInterpStyle(style));
}
    
    
/**
 * @brief Get the highest order Interpolation::Style available for 'n' points.
 *
 */
math::Interpolate::Style math::lookupMaxInterpStyle(int const n) {
    if (n < 1) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "nx,ny must be greater than 0");
    }
    if (n > 4) {
        return math::Interpolate::AKIMA_SPLINE;
    }
    
    std::vector<math::Interpolate::Style> styles(4);
    styles[0] = math::Interpolate::CONSTANT;
    styles[1] = math::Interpolate::LINEAR;
    styles[2] = math::Interpolate::CUBIC_SPLINE;
    styles[3] = math::Interpolate::CUBIC_SPLINE;
    return styles[n - 1];
}

    
/**
 * @brief Get the minimum number of points needed to use the requested interpolation style
 *
 */
int math::lookupMinInterpPoints(math::Interpolate::Style const style) {
    std::vector<int> minPoints(math::Interpolate::NUM_STYLES);
    minPoints[math::Interpolate::CONSTANT]               = 1;
    minPoints[math::Interpolate::LINEAR]                 = 2;
    minPoints[math::Interpolate::NATURAL_SPLINE]         = 3;
    minPoints[math::Interpolate::CUBIC_SPLINE]           = 3;
    minPoints[math::Interpolate::CUBIC_SPLINE_PERIODIC]  = 3;
    minPoints[math::Interpolate::AKIMA_SPLINE]           = 5;
    minPoints[math::Interpolate::AKIMA_SPLINE_PERIODIC]  = 5;
    return minPoints[style];
}

/**
 * @brief Get the minimum number of points needed to use the requested interpolation style
 *
 * Overload of lookupMinInterpPoints() which takes a string
 *
 */
int math::lookupMinInterpPoints(std::string const style) {
    return lookupMinInterpPoints(stringToInterpStyle(style));
}


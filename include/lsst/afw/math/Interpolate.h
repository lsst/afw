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

namespace {
}
    
    
class Interpolate {
public:

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

    Interpolate(std::vector<double> const &x, std::vector<double> const &y,
                   ::gsl_interp_type const *gslInterpType = ::gsl_interp_akima) :
        _x(x), _y(y) {
        initialize(_x, _y, gslInterpType);
    }

    Interpolate(std::vector<double> const &x, std::vector<double> const &y,
                Style const style) :
        _x(x), _y(y) {
        if (style == CONSTANT) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              "CONSTANT interpolation not supported.");
        }
        _initGslInterpTypeStyles();
        initialize(_x, _y, _gslInterpTypeStyles[style]);
    }

    Interpolate(std::vector<double> const &x, std::vector<double> const &y,
                std::string style) :
        _x(x), _y(y) {
        if (style == "CONSTANT") {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              "CONSTANT interpolation not supported.");
        }
        _initGslInterpTypeStrings();
        initialize(_x, _y, _gslInterpTypeStrings[style]);
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
    ::gsl_interp_type const* _gslInterpTypeStyles[7];
    std::map<std::string, const ::gsl_interp_type*> _gslInterpTypeStrings;
    
    void _initGslInterpTypeStyles() {
        _gslInterpTypeStyles[CONSTANT]                = ::gsl_interp_linear;           
        _gslInterpTypeStyles[LINEAR]                  = ::gsl_interp_linear;           
        _gslInterpTypeStyles[CUBIC_SPLINE]            = ::gsl_interp_cspline;          
        _gslInterpTypeStyles[NATURAL_SPLINE]          = ::gsl_interp_cspline;          
        _gslInterpTypeStyles[CUBIC_SPLINE_PERIODIC]   = ::gsl_interp_cspline_periodic; 
        _gslInterpTypeStyles[AKIMA_SPLINE]            = ::gsl_interp_akima;            
        _gslInterpTypeStyles[AKIMA_SPLINE_PERIODIC]   = ::gsl_interp_akima_periodic;   
    }
    
    void _initGslInterpTypeStrings() {
        _gslInterpTypeStrings["CONSTANT"]              = ::gsl_interp_linear;           
        _gslInterpTypeStrings["LINEAR"]                = ::gsl_interp_linear;           
        _gslInterpTypeStrings["CUBIC_SPLINE"]          = ::gsl_interp_cspline;          
        _gslInterpTypeStrings["NATURAL_SPLINE"]        = ::gsl_interp_cspline;          
        _gslInterpTypeStrings["CUBIC_SPLINE_PERIODIC"] = ::gsl_interp_cspline_periodic; 
        _gslInterpTypeStrings["AKIMA_SPLINE"]          = ::gsl_interp_akima;            
        _gslInterpTypeStrings["AKIMA_SPLINE_PERIODIC"] = ::gsl_interp_akima_periodic;
    }
    
};

        
}}}
                     
#endif // LSST_AFW_MATH_INTERPOLATE_H

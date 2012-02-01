// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
/**
 * @file Interpolate.cc
 * @brief Wrap GSL to interpolate values for a set of x,y vector<>s
 * @ingroup afw
 * @author Steve Bickerton
 */
#include <limits>
#include <map>
#include "boost/format.hpp"
#include "boost/shared_ptr.hpp"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_interp.h"
#include "gsl/gsl_spline.h"

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
    if (!_acc) {
        throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException, "gsl_interp_accel_alloc failed");
    }
    
    _interp = ::gsl_interp_alloc(gslInterpType, y.size());
    if (!_interp) {
        throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException,
                          (boost::format("Failed to initialise spline for type %s, length %d")
                           % gslInterpType->name % y.size()).str());

    }
    // Note, "x" and "y" are vector<double>; gsl_inter_init requires double[].
    // The &(x[0]) here is valid because std::vector guarantees that the values are
    // stored contiguously in memory (for types other than bool); C++0X 23.3.6.1 for
    // those of you reading along.
    int const status = ::gsl_interp_init(_interp, &(x[0]), &(y[0]), y.size());
    if (status != 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          str(boost::format("gsl_interp_init failed: %s [%d]")
                              % ::gsl_strerror(status) % status));
    }
}

math::Interpolate::~Interpolate() {
    ::gsl_interp_free(_interp);
    ::gsl_interp_accel_free(_acc);
}

double math::Interpolate::interpolate(double const x) {
    // New GSL versions refuse to extrapolate.
    // gsl_interp_init() requires x to be ordered, so can just check
    // the array endpoints for out-of-bounds.
    if ((x < _x.front() || (x > _x.back()))) {
        // do our own quadratic extrapolation.
        // (GSL only provides first and second derivative functions)
        /* could also just fail via:
         throw LSST_EXCEPT(
         lsst::pex::exceptions::InvalidParameterException,
         (boost::format("Interpolation point %f outside range [%f, %f]")
         % x % _x.front() % _x.back()).str()
         );
         */
        double x0, y0;
        if (x < _x.front()) {
            x0 = _x.front();
            y0 = _y.front();
        } else {
            x0 = _x.back();
            y0 = _y.back();
        }
        // first derivative at endpoint
        double d = ::gsl_interp_eval_deriv(_interp, &_x[0], &_y[0], x0, _acc);
        // second derivative at endpoint
        double d2 = ::gsl_interp_eval_deriv2(_interp, &_x[0], &_y[0], x0, _acc);
        return y0 + (x - x0)*d + (x - x0)*(x - x0)*d2;
    }
    assert(x >= _x.front());
    assert(x <= _x.back());
    return ::gsl_interp_eval(_interp, &_x[0], &_y[0], x, _acc);
}

/**
 * @brief Conversion function to switch an Interpolate::Style to a gsl_interp_type.
 *
 */
::gsl_interp_type const *math::styleToGslInterpType(Interpolate::Style const style) {
    switch (style) {
      case Interpolate::CONSTANT:
        return ::gsl_interp_linear;
      case Interpolate::LINEAR:
        return ::gsl_interp_linear;
      case Interpolate::CUBIC_SPLINE:
        return ::gsl_interp_cspline;
      case Interpolate::NATURAL_SPLINE:
        return ::gsl_interp_cspline;
      case Interpolate::CUBIC_SPLINE_PERIODIC:
        return ::gsl_interp_cspline_periodic;
      case Interpolate::AKIMA_SPLINE:
        return ::gsl_interp_akima;
      case Interpolate::AKIMA_SPLINE_PERIODIC:
        return ::gsl_interp_akima_periodic;
      case Interpolate::NUM_STYLES:
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicErrorException,
                          str(boost::format("You can't get here: style == %") % style));
    }
}
    
/**
 * @brief Conversion function to switch a string to an Interpolate::Style.
 *
 */
math::Interpolate::Style math::stringToInterpStyle(std::string const style) {
    static std::map<std::string, Interpolate::Style> gslInterpTypeStrings;
    if (gslInterpTypeStrings.empty()) {
        gslInterpTypeStrings["CONSTANT"]              = Interpolate::CONSTANT;
        gslInterpTypeStrings["LINEAR"]                = Interpolate::LINEAR;               
        gslInterpTypeStrings["CUBIC_SPLINE"]          = Interpolate::CUBIC_SPLINE;         
        gslInterpTypeStrings["NATURAL_SPLINE"]        = Interpolate::NATURAL_SPLINE;      
        gslInterpTypeStrings["CUBIC_SPLINE_PERIODIC"] = Interpolate::CUBIC_SPLINE_PERIODIC;
        gslInterpTypeStrings["AKIMA_SPLINE"]          = Interpolate::AKIMA_SPLINE;  
        gslInterpTypeStrings["AKIMA_SPLINE_PERIODIC"] = Interpolate::AKIMA_SPLINE_PERIODIC;
    }
    
    if ( gslInterpTypeStrings.find(style) == gslInterpTypeStrings.end()) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Interp style not found: "+style);
    }
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


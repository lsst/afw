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
 * @brief Interpolate values for a set of x,y vector<>s
 * @ingroup afw
 * @author Steve Bickerton
 */
#include <limits>
#include <algorithm>
#include <map>
#include "boost/format.hpp"
#include "boost/shared_ptr.hpp"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_interp.h"
#include "gsl/gsl_spline.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Interpolate.h"

namespace lsst {
namespace afw {
namespace math {

/************************************************************************************************************/
    
namespace {
    std::pair<std::vector<double>, std::vector<double> >
    recenter(std::vector<double> const &x,
             std::vector<double> const &y)
    {
        if (x.size() != y.size()) {
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterException,
                              str(boost::format("Dimensions of x and y must match; %ul != %ul")
                                  % x.size() % y.size()));
        }
        unsigned int const len = x.size();
        if (len == 0) {
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterException,
                              "You must provide at least 1 point");
        } else if (len == 1) {
            return std::make_pair(x, y);
        }

        std::vector<double> recentered_x(len + 1);
        std::vector<double> recentered_y(len + 1);

        int j = 0; 
        recentered_x[j] = 0.5*(3*x[0] - x[1]);
        recentered_y[j] = y[0];

        for (unsigned int i = 0; i < x.size(); ++i) {
            ++j;
            recentered_x[j] = 0.5*(x[i] + x[i + 1]);
            recentered_y[j] = 0.5*(y[i] + y[i + 1]);
        }
        recentered_x[j] = 0.5*(3*x[len - 1] - x[len - 2]);
        recentered_y[j] = y[len - 1];

        return std::make_pair(recentered_x, recentered_y);        
    }
}

class InterpolateConstant : public Interpolate {
    friend PTR(Interpolate) makeInterpolate(std::vector<double> const &x, std::vector<double> const &y,
                                            Interpolate::Style const style);
public:
    virtual ~InterpolateConstant() {}
    virtual double interpolate(double const x) const;
private:
    InterpolateConstant(std::vector<double> const &x, ///< the x-values of points
                        std::vector<double> const &y, ///< the values at x[]
                        Interpolate::Style const style ///< desired interpolator
                       ) :
        Interpolate(recenter(x, y)), _old(_x.begin()) {}
    mutable std::vector<double>::const_iterator _old; // last position we found xInterp at
};
    
    
/// Interpolate a constant to the point \c xInterp
double InterpolateConstant::interpolate(double const xInterp // the value we want to interpolate to
                                       ) const
{
    //
    // Look for the interval wherein lies xInterp.  We could naively use std::upper_bound, but that requires a
    // logarithmic time lookup so we'll cache the previous answer in _old -- this is a good idea if people
    // usually call this routine repeatedly for a range of x
    //
    // We start by searching up from _old
    //
    if (xInterp < *_old) {              // We're to the left of the cache
        if (_old == _x.begin()) {       // ... actually off the array
            return _y[0];
        }
        _old = _x.begin();              // reset the cached point to the start of the array
    } else {                            // see if we're still in the same interval
        if (_old < _x.end() - 1 and xInterp < *(_old + 1)) { // we are, so we're done
            return _y[_old - _x.begin()];
        }
    }
    // We're to the right of the cached point and not in the same inverval, so search up from _old
    std::vector<double>::const_iterator low = std::upper_bound(_old, _x.end(), xInterp);
    //
    // Did that work?
    if (low == _old && _old != _x.begin()) {
        // No.  Sigh.  Search the entire range.
        low = std::upper_bound(_x.begin(), low + 1, xInterp);
    }
    //
    // OK, we've found the right interval.  Return the desired value, being careful at the ends
    //
    if (low == _x.end()) {
        return _y[_y.size() - 1];
    } else if (low == _x.begin()) {
        return _y[0];
    } else {
        --low;
        _old = low;
        return _y[low - _x.begin()];
    }
}

/************************************************************************************************************/
namespace {
/*
 * Conversion function to switch an Interpolate::Style to a gsl_interp_type.
 */
::gsl_interp_type const *
styleToGslInterpType(Interpolate::Style const style)
{
    switch (style) {
      case Interpolate::CONSTANT:
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterException, "CONSTANT interpolation not supported.");
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
      case Interpolate::UNKNOWN:
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterException,
                          "I am unable to make an interpolator of type UNKNOWN");
      case Interpolate::NUM_STYLES:
        throw LSST_EXCEPT(pex::exceptions::LogicErrorException,
                          str(boost::format("You can't get here: style == %") % style));
    }
}
}
    
class InterpolateGsl : public Interpolate {
    friend PTR(Interpolate) makeInterpolate(std::vector<double> const &x, std::vector<double> const &y,
                                            Interpolate::Style const style);
public:
    virtual ~InterpolateGsl();
    virtual double interpolate(double const x) const;
private:
    InterpolateGsl(std::vector<double> const &x, std::vector<double> const &y, Interpolate::Style const style);

    ::gsl_interp_type const *_interpType;
    ::gsl_interp_accel *_acc;
    ::gsl_interp *_interp;
};

InterpolateGsl::InterpolateGsl(std::vector<double> const &x, ///< the x-values of points
                               std::vector<double> const &y, ///< the values at x[]
                               Interpolate::Style const style ///< desired interpolator
                              ) :
    Interpolate(x, y), _interpType(styleToGslInterpType(style))
{
    _acc = ::gsl_interp_accel_alloc();
    if (!_acc) {
        throw LSST_EXCEPT(pex::exceptions::MemoryException, "gsl_interp_accel_alloc failed");
    }
    
    _interp = ::gsl_interp_alloc(_interpType, _y.size());
    if (!_interp) {
        throw LSST_EXCEPT(pex::exceptions::MemoryException,
                          str(boost::format("Failed to initialise spline for type %s, length %d")
                              % _interpType->name % _y.size()));

    }
    // Note, "x" and "y" are vector<double>; gsl_inter_init requires double[].
    // The &(x[0]) here is valid because std::vector guarantees that the values are
    // stored contiguously in memory (for types other than bool); C++0X 23.3.6.1 for
    // those of you reading along.
    int const status = ::gsl_interp_init(_interp, &x[0], &y[0], _y.size());
    if (status != 0) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeErrorException,
                          str(boost::format("gsl_interp_init failed: %s [%d]")
                              % ::gsl_strerror(status) % status));
    }
}

InterpolateGsl::~InterpolateGsl() {
    ::gsl_interp_free(_interp);
    ::gsl_interp_accel_free(_acc);
}

double InterpolateGsl::interpolate(double const xInterp) const
{
    // New GSL versions refuse to extrapolate.
    // gsl_interp_init() requires x to be ordered, so can just check
    // the array endpoints for out-of-bounds.
    if ((xInterp < _x.front() || (xInterp > _x.back()))) {
        // do our own quadratic extrapolation.
        // (GSL only provides first and second derivative functions)
        /* could also just fail via:
         throw LSST_EXCEPT(
         pex::exceptions::InvalidParameterException,
         (boost::format("Interpolation point %f outside range [%f, %f]")
         % x % _x.front() % _x.back()).str()
         );
         */
        double x0, y0;
        if (xInterp < _x.front()) {
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
        return y0 + (xInterp - x0)*d + 0.5*(xInterp - x0)*(xInterp - x0)*d2;
    }
    assert(xInterp >= _x.front());
    assert(xInterp <= _x.back());
    return ::gsl_interp_eval(_interp, &_x[0], &_y[0], xInterp, _acc);
}

/************************************************************************************************************/
/**
 * @brief Conversion function to switch a string to an Interpolate::Style.
 *
 */
Interpolate::Style stringToInterpStyle(std::string const &style ///< desired type of interpolation
                                      )
{
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
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterException, "Interp style not found: "+style);
    }
    return gslInterpTypeStrings[style];
}
    
/**
 * @brief Get the highest order Interpolation::Style available for 'n' points.
 */
Interpolate::Style lookupMaxInterpStyle(int const n ///< Number of points
                                       ) {
    if (n < 1) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterException, "n must be greater than 0");
    } else if (n > 4) {
        return Interpolate::AKIMA_SPLINE;
    } else {
        static std::vector<Interpolate::Style> styles;
        if (styles.empty()) {
            styles.resize(5);
            
            styles[0] = Interpolate::UNKNOWN; // impossible to reach as we check for n < 1
            styles[1] = Interpolate::CONSTANT;
            styles[2] = Interpolate::LINEAR;
            styles[3] = Interpolate::CUBIC_SPLINE;
            styles[4] = Interpolate::CUBIC_SPLINE;
        }
        return styles[n];
    }
}

/**
 * @brief Get the minimum number of points needed to use the requested interpolation style
 */
int lookupMinInterpPoints(Interpolate::Style const style ///< The style in question
                         ) {
    static std::vector<int> minPoints;
    if (minPoints.empty()) {
        minPoints.resize(Interpolate::NUM_STYLES);
        minPoints[Interpolate::CONSTANT]               = 1;
        minPoints[Interpolate::LINEAR]                 = 2;
        minPoints[Interpolate::NATURAL_SPLINE]         = 3;
        minPoints[Interpolate::CUBIC_SPLINE]           = 3;
        minPoints[Interpolate::CUBIC_SPLINE_PERIODIC]  = 3;
        minPoints[Interpolate::AKIMA_SPLINE]           = 5;
        minPoints[Interpolate::AKIMA_SPLINE_PERIODIC]  = 5;
    }

    if (style >= 0 && style < Interpolate::NUM_STYLES) {
        return minPoints[style];
    } else {
        throw LSST_EXCEPT(pex::exceptions::OutOfRangeException,
                          str(boost::format("Style %d is out of range 0..%d")
                              % style % (Interpolate::NUM_STYLES - 1)));
    }
}

/************************************************************************************************************/
/**
 * Base class ctor.  Note that we should use rvalue references when
 * available as the vectors in xy will typically be movable (although the
 * returned-value-optimisation might suffice for the cases we care about)
 *
 * \note this is here, not in the .h file, so as to permit the compiler
 * to avoid copying those vectors
 */
Interpolate::Interpolate(
        std::pair<std::vector<double>, std::vector<double> > const xy, ///< pair (x,y) where
        /// x are the ordinates of points and y are the values at x[]
        Interpolate::Style const style ///< desired interpolator
                        ) : _x(xy.first), _y(xy.second), _style(style)
{
    ;
}

/**
 * A factory function to make Interpolate objects
 */
PTR(Interpolate) makeInterpolate(std::vector<double> const &x, ///< the x-values of points
                                 std::vector<double> const &y, ///< the values at x[]
                                 Interpolate::Style const style ///< desired interpolator
                                )
{
    switch (style) {
      case Interpolate::CONSTANT:
        return PTR(Interpolate)(new InterpolateConstant(x, y, style));
      default:                            // use GSL
        return PTR(Interpolate)(new InterpolateGsl(x, y, style));
    }
}

}}}

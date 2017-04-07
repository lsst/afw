// -*- LSST-C++ -*-
#if !defined(LSST_AFW_MATH_INTERPOLATE_H)
#define LSST_AFW_MATH_INTERPOLATE_H

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
#include "lsst/base.h"
#include "ndarray_fwd.h"

namespace lsst {
namespace afw {
namespace math {

 /*
 * Interpolate values for a set of x,y vector<>s
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
    std::vector<double> interpolate(std::vector<double> const& x) const;
    ndarray::Array<double, 1> interpolate(ndarray::Array<double const, 1> const& x) const;
protected:
    /**
     * Base class ctor
     */
    Interpolate(std::vector<double> const &x, ///< the ordinates of points
                std::vector<double> const &y, ///< the values at x[]
                Interpolate::Style const style=UNKNOWN ///< desired interpolator
               ) : _x(x), _y(y), _style(style) {}
    /**
     * Base class ctor.  Note that we should use rvalue references when
     * available as the vectors in xy will typically be movable (although the
     * returned-value-optimisation might suffice for the cases we care about)
     *
     * @param xy pair (x,y) where x are the ordinates of points and y are the values at x[]
     * @param style desired interpolator
     */
    Interpolate(std::pair<std::vector<double>, std::vector<double> > const xy,
                Interpolate::Style const style=UNKNOWN);

    std::vector<double> const _x;
    std::vector<double> const _y;
    Interpolate::Style const _style;
private:
    Interpolate(Interpolate const&);
    Interpolate& operator=(Interpolate const&);
};

/**
 * A factory function to make Interpolate objects
 *
 * @param x the x-values of points
 * @param y the values at x[]
 * @param style desired interpolator
 */
PTR(Interpolate) makeInterpolate(std::vector<double> const &x, std::vector<double> const &y,
                                 Interpolate::Style const style=Interpolate::AKIMA_SPLINE);
PTR(Interpolate) makeInterpolate(ndarray::Array<double const, 1> const &x,
                                 ndarray::Array<double const, 1> const &y,
                                 Interpolate::Style const style=Interpolate::AKIMA_SPLINE);
/**
 * Conversion function to switch a string to an Interpolate::Style.
 *
 * @param style desired type of interpolation
 */
Interpolate::Style stringToInterpStyle(std::string const &style);
/**
 * Get the highest order Interpolation::Style available for 'n' points.
 *
 * @param n Number of points
 */
Interpolate::Style lookupMaxInterpStyle(int const n);
/**
 * Get the minimum number of points needed to use the requested interpolation style
 *
 * @param style The style in question
 */
int lookupMinInterpPoints(Interpolate::Style const style);

}}}

#endif // LSST_AFW_MATH_INTERPOLATE_H

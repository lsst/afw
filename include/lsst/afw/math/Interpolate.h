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
 
/**
 * @brief Interpolate values for a set of x,y vector<>s
 * @ingroup afw
 * @author Steve Bickerton
 */
#include "lsst/base.h"

namespace lsst {
namespace afw {
namespace math {

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
    
    Interpolate(std::vector<double> const &x, std::vector<double> const &y,
                Interpolate::Style const style=AKIMA_SPLINE);
    
    virtual ~Interpolate();
    double interpolate(double const x);
    
private:
    struct InterpolateGslImpl;
    PTR(InterpolateGslImpl) _gslImpl;
};
    
Interpolate::Style stringToInterpStyle(std::string const &style);

/**
 * @brief Get the highest order Interpolation::Style available for 'n' points.
 */
Interpolate::Style lookupMaxInterpStyle(int const n);
    
/**
 * @brief Get the minimum number of points needed to use the requested interpolation style
 */
int lookupMinInterpPoints(Interpolate::Style const style);
    
/**
 * @brief Get the minimum number of points needed to use the requested interpolation style
 * Overload of lookupMinInterpPoints() which takes a string
 */
int lookupMinInterpPoints(std::string const style);
        
}}}
                     
#endif // LSST_AFW_MATH_INTERPOLATE_H

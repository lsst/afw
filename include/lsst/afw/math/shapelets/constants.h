// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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
 
#ifndef LSST_AFW_MATH_SHAPELETS_CONSTANTS_H
#define LSST_AFW_MATH_SHAPELETS_CONSTANTS_H

/**
 * @file
 *
 * @brief Constants, typedefs, and general-purpose functions for shapelets library.
 *
 * @todo
 *
 * @author Jim Bosch
 */

#include "lsst/afw/geom/ellipses.h"
#include "lsst/ndarray.h"
#include "lsst/pex/exceptions.h"

namespace lsst {
namespace afw {
namespace math {
namespace shapelets {

typedef double Pixel;

/**
 *  @brief An enum that sets whether to use real-valued polar shapelets or Cartesian shapelets.
 *
 *  The conversion between the two bases is theoretically exact, but of course subject to
 *  round-off error here.
 */
enum BasisTypeEnum {
    HERMITE, /**< 
              *   Cartesian shapelets or Gauss-Hermite functions, as defined in 
              *   Refregier, 2003.  That is,
              *   @f$ \psi(x, y)_{n_x, n_y} 
              *          = \frac{H_{n_x}(x) H_{n_y}(y) e^{-\frac{x^2 + y^2}{2}}}
              *                 {\sigma 2^{n_x + n_y} \sqrt{\pi n_x! n_y!}}
              *   @f$
              *   where @f$H_n(x)@f$ is a Hermite polynomial.
              *
              *   The ordering of coefficients [n_x, n_y] is (row-major packed):
              *   [0,0],
              *   [0,1], [1,0],
              *   [0,2], [1,1], [2,0],
              *   [0,3], [1,2], [2,1], [3,0],
              *   [0,4], [1,3], [2,2], [3,1], [4,0]
              *   etc.
              */

    LAGUERRE, /**< 
               *   Polar shapelets or Gauss-Laguerre functions, as defined in 
               *   Bernstein and Jarvis, 2002.  That is,
               *   @f$ \psi(x, y, \sigma)_{p, q}
               *         = (-1)^q \sqrt{\frac{q!}{p!}} (x + i y)^{p-q}
               *                e^{-\frac{x^2 + y^2}{2}} L^{(p-q)}_q(x^2 + y^2)
               *   @f$
               *   where @f$L^{(m)}_n(r)@f$ is an associated Laguerre polynomial.
               *
               *   The ordering of coefficients [p, q] is (row-major packed):
               *   [0,0],
               *   Re([1,0]), Im([1,0]),
               *   Re([2,0]), Im([2,0]), [1,1],
               *   Re([3,0]), Im([3,0], Re([2,1]), Im([2,1]),
               *   Re([4,0]), Im([4,0], Re([3,1]), Im([3,1]), [2,2]
               *   etc.
               *   
               *   Elements with p < q are redundant in representing real-valued functions, 
               *   while those with p == q are inherently real.
               */
};

/// @brief Return the offset of the given order in a coefficient vector.
inline int computeOffset(int order) { return order * (order + 1) / 2; }

/// @brief Return the size of the coefficient vector for the given order.
inline int computeSize(int order) { return computeOffset(order + 1); }

}}}}   // lsst::afw::math::shapelets

#endif // !defined(LSST_AFW_MATH_SHAPELETS_CONSTANTS_H)

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
 
#ifndef LSST_AFW_MATH_ELLIPTICALSHAPELET_H
#define LSST_AFW_MATH_ELLIPTICALSHAPELET_H
/**
 * @file
 *
 * @brief Basic shapelet expansion, mostly for converting between different shapelet basis sets.
 *
 * @todo
 *
 * @author Jim Bosch
 */

#include "lsst/afw/geom/ellipses.h"

namespace lsst {
namespace afw {
namespace math {

class EllipticalShapelet {
public:

    enum BasisEnum {
        HERMITE, /**< 
                  *   Cartesian shapelets or Gauss-Hermite functions, as defined in 
                  *   Refregier, 2003.  Specifically,
                  *   @f$ \psi(x, y, \sigma)_{n_x, n_y} 
                  *          = \frac{H_{n_x}(x/\sigma) H_{n_y}(y/\sigma) e^{-\frac{x^2+y^2}{2\sigma^2}}
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

        LAGUERRE_PQ, /**< 
                      *   Polar shapelets or Gauss-Laguerre functions, (almost) as defined in 
                      *   Bernstein and Jarvis, 2002.  That is,
                      *   @f$ \psi(x, y, \sigma)_{p, q}
                      *         = \frac{(-1)^q}{\sigma} \sqrt{\frac{q!}{p!}} 
                      *                \left(\frac{x + i y}{\sigma}\right)^{p-q}
                      *                e^{-\frac{x^2+y^2}{2\sigma^2}} 
                      *                L^{(p-q)}_q\left(\frac{x^2 + y^2}{\sigma^2}\right)
                      *   @f$
                      *   where @f$L^{(m)}_n(r)@f$ is an associated Laguerre polynomial.
                      *   The only difference from BJ02 is one less factor of @f$\sigma@f$ in
                      *   the denominator; this ensures that the basis functions are orthonormal,
                      *   just as the Cartesian form is.
                      *
                      *   The ordering of coefficients [n_x, n_y] is (row-major packed):
                      *   [0,0],
                      *   Re([1,0]), Im([1,0]),
                      *   Re([2,0]), Im([2,0]), [1,1],
                      *   Re([3,0]), Im([3,0], Re([2,1]), Im([2,1]),
                      *   Re([4,0]), Im([4,0], Re([3,1]), Im([3,1]), [2,2]
                      *   etc.
                      *   
                      *   Elements with p < q are redundant in representing real-valued functions, 
                      *   and those with p == q are inherently real.
                      */

        LAGUERRE_NM,
    };

};

}}}   // lsst::afw::math

#endif // !defined(LSST_AFW_MATH_ELLIPTICALSHAPELET_H)

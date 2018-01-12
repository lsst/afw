// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2015 LSST Corporation.
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

#include <cmath>
#include <iostream>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/Functor.h"

namespace lsst {
namespace afw {
namespace geom {

Functor::Functor(std::string const& name) : daf::base::Citizen(typeid(this)), _name(name) {}

Functor::Functor(Functor const &) = default;
Functor::Functor(Functor &&) = default;
Functor &Functor::operator=(Functor const &) = default;
Functor &Functor::operator=(Functor &&) = default;

double Functor::inverse(double y, double tol, unsigned int maxiter) const {
    // Sanity checks for tol and maxiter.
    if (tol > 1 || tol <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "tol out-of-range, tol <=0 or tol > 1");
    }
    if (maxiter < 1) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "maxiter out-of-range, maxiter < 1");
    }
    // Use Newton-Raphson method to find the inverse.
    double x = y;
    for (unsigned int iter = 0; iter < maxiter; iter++) {
        double dx = y - operator()(x);
        if (std::fabs(dx) < tol) {
            return x;
        }
        x += dx / derivative(x);
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                      "max iteration count exceeded for subclass " + _name);
    return 0;
}

}  // namespace geom
}  // namespace afw
}  // namespace lsst

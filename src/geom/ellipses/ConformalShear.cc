// -*- lsst-c++ -*-

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
#include <cmath>

#include "lsst/afw/geom/ellipses/ConformalShear.h"
#include "lsst/afw/geom/ellipses/ReducedShear.h"
#include "lsst/afw/geom/ellipses/Distortion.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

double ConformalShear::getAxisRatio() const {
    double e = getE();
    return std::exp(-e);
}

ConformalShear& ConformalShear::operator=(Distortion const& other) {
    double delta = other.getE();
    if (delta < 1E-8) {
        _complex = other.getComplex() * (1.0 + delta * delta / 3.0);
    } else {
        double eta = std::atanh(delta);
        _complex = other.getComplex() * eta / delta;
    }
    return *this;
}

ConformalShear& ConformalShear::operator=(ReducedShear const& other) {
    double g = other.getE();
    if (g < 1E-8) {
        _complex = other.getComplex() * 2.0 * (1.0 + g * g / 3.0);
    } else {
        double eta = 2.0 * std::atanh(g);
        _complex = other.getComplex() * eta / g;
    }
    return *this;
}

detail::EllipticityBase::Jacobian ConformalShear::dAssign(Distortion const& other) {
    Jacobian result = Jacobian::Zero();
    double delta = other.getE();
    double alpha, beta;
    if (delta < 1E-8) {
        alpha = 1.0 + delta * delta / 3.0;
        beta = 2.0 / 3.0;
    } else {
        double eta = std::atanh(delta);
        alpha = eta / delta;
        beta = (1.0 / (1.0 - delta * delta) - alpha) / (delta * delta);
    }
    _complex = other.getComplex() * alpha;
    result(0, 0) = alpha + other.getE1() * other.getE1() * beta;
    result(1, 1) = alpha + other.getE2() * other.getE2() * beta;
    result(1, 0) = result(0, 1) = other.getE1() * other.getE2() * beta;
    return result;
}

detail::EllipticityBase::Jacobian ConformalShear::dAssign(ReducedShear const& other) {
    Jacobian result = Jacobian::Zero();
    double g = other.getE();
    double alpha, beta;
    if (g < 1E-8) {
        alpha = 2.0 * (1.0 + g * g / 3.0);
        beta = 4.0 / 3.0;
    } else {
        double eta = 2.0 * std::atanh(g);
        alpha = eta / g;
        beta = 1.0 * (2.0 / (1.0 - g * g) - alpha) / (g * g);
    }
    _complex = other.getComplex() * alpha;
    result(0, 0) = alpha + other.getE1() * other.getE1() * beta;
    result(1, 1) = alpha + other.getE2() * other.getE2() * beta;
    result(1, 0) = result(0, 1) = other.getE1() * other.getE2() * beta;
    return result;
}
}  // namespace ellipses
}  // namespace geom
}  // namespace afw
}  // namespace lsst

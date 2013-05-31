// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2013 LSST Corporation.
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
#include "lsst/afw/geom/ellipses/ReducedShear.h"
#include "lsst/afw/geom/ellipses/ConformalShear.h"
#include "lsst/afw/geom/ellipses/Distortion.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

double ReducedShear::getAxisRatio() const {
    double e = getE();
    return (1.0 - e) / (1.0 + e);
}

void ReducedShear::normalize() {
    if (!(getE() <= 1.0)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            "ReducedShear magnitude cannot be greater than one."
        );
    }
}

ReducedShear & ReducedShear::operator=(Distortion const & other) {
    double delta = other.getE();
    if (delta < 1E-8) {
        _complex = other.getComplex() * (0.5 + 0.125 * delta * delta);
    } else if (delta >= 1.0) {
        _complex = other.getComplex();
    } else {
        double g = (1.0 - std::sqrt(1.0 - delta * delta)) / delta;
        _complex = other.getComplex() * g / delta;
    }
    return *this;
}

ReducedShear & ReducedShear::operator=(ConformalShear const & other) {
    double eta = other.getE();
    if (eta < 1E-8) {
        _complex = other.getComplex() * (0.5 - eta * eta / 12.0);
    } else {
        double g = std::tanh(0.5 * eta);
        _complex = other.getComplex() * g / eta;
    }
    return *this;
}

detail::EllipticityBase::Jacobian ReducedShear::dAssign(Distortion const & other) {
    Jacobian result = Jacobian::Zero();
    double delta = other.getE();
    double s = std::sqrt(1.0 - delta * delta);
    double alpha, beta;
    if (delta < 1E-8) {
        alpha = 0.5 + 0.125 * delta * delta;
        beta = 0.25;
    } else if (delta >= 1.0) {
        alpha = 1.0;
        beta = 0.0;
    } else {
        alpha = (1.0 - s) / (delta * delta);
        beta = (2.0 * alpha - 1.0) / (delta * delta * s);
    }
    _complex = other.getComplex() * alpha;
    result(0, 0) = alpha + other.getE1() * other.getE1() * beta;
    result(1, 1) = alpha + other.getE2() * other.getE2() * beta;
    result(1, 0) = result(0, 1) = other.getE1() * other.getE2() * beta;
    return result;
}

detail::EllipticityBase::Jacobian ReducedShear::dAssign(ConformalShear const & other) {
    Jacobian result = Jacobian::Zero();
    double eta = other.getE();
    double alpha, beta;
    if (eta < 1E-8) {
        alpha = 0.5 - eta * eta / 24.0;
        beta = -1.0 / 12;
    } else {
        double g = std::tanh(0.5 * eta);
        alpha = g / eta;
        beta = (0.5 * (1.0 - g * g) - alpha) / (eta * eta);
    }
    _complex = other.getComplex() * alpha;
    result(0, 0) = alpha + other.getE1() * other.getE1() * beta;
    result(1, 1) = alpha + other.getE2() * other.getE2() * beta;
    result(1, 0) = result(0, 1) = other.getE1() * other.getE2() * beta;
    return result;
}

void ReducedShear::_assignToQuadrupole(double r, double & ixx, double & iyy, double & ixy) const {
    Distortion delta(*this);
    delta._assignToQuadrupole(r, ixx, iyy, ixy);
}
void ReducedShear::_assignFromQuadrupole(double & r, double ixx, double iyy, double ixy) {
    Distortion delta;
    delta._assignFromQuadrupole(r, ixx, iyy, ixy);
    *this = delta;
}

EllipseCore::Jacobian ReducedShear::_dAssignToQuadrupole(
    double r, double & ixx, double & iyy, double & ixy
) const {
    Distortion delta;
    EllipseCore::Jacobian j1 = EllipseCore::Jacobian::Identity();
    j1.block<2,2>(0,0) = delta.dAssign(*this);
    EllipseCore::Jacobian j2 = delta._dAssignToQuadrupole(r, ixx, iyy, ixy);
    return j2 * j1;
}

EllipseCore::Jacobian ReducedShear::_dAssignFromQuadrupole(
    double & r, double ixx, double iyy, double ixy
) {
    Distortion delta;
    EllipseCore::Jacobian j1 = delta._dAssignFromQuadrupole(r, ixx, iyy, ixy);
    EllipseCore::Jacobian j2 = EllipseCore::Jacobian::Identity();
    j2.block<2,2>(0,0) = dAssign(delta);
    return j2 * j1;
}

}}}} // namespace lsst::afw::geom::ellipses

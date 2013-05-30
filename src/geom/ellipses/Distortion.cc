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

#include "lsst/utils/ieee.h"
#include "lsst/afw/geom/ellipses/Distortion.h"
#include "lsst/afw/geom/ellipses/ReducedShear.h"
#include "lsst/afw/geom/ellipses/ConformalShear.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

namespace {

void fixNaN(std::complex<double> & c) {
    double re = c.real();
    double im = c.imag();
    if (utils::isnan(re)) {
        re = 0.0;
    }
    if (utils::isnan(im)) {
        im = 0.0;
    }
    c = std::complex<double>(re, im);
}

} // anonymous

double Distortion::getAxisRatio() const {
    double e = getE();
    return std::sqrt((1.0 - e) / (1.0 + e));
}

Distortion & Distortion::operator=(ConformalShear const & other) {
    double eta = other.getE();
    if (eta < 1E-8) {
        _complex = other.getComplex() * (1.0 - eta * eta / 3.0);
    } else {
        double delta = std::tanh(eta);
        _complex = other.getComplex() * delta / eta;
    }
    return *this;
}

Distortion & Distortion::operator=(ReducedShear const & other) {
    double g = other.getE();
    _complex = other.getComplex() * 2.0 / (1 + g * g);
    return *this;
}

detail::EllipticityBase::Jacobian Distortion::dAssign(ConformalShear const & other) {
    Jacobian result = Jacobian::Zero();
    double eta = other.getE();
    double alpha, beta;
    if (eta < 1E-8) {
        alpha = (1.0 - eta * eta / 3.0);
        beta = -2.0 / 3.0;
    } else {
        double delta = std::tanh(eta);
        alpha = delta / eta;
        beta = (1.0 - delta * delta - alpha) / (eta * eta);
    }
    _complex = other.getComplex() * alpha;
    result(0, 0) = alpha + other.getE1() * other.getE1() * beta;
    result(1, 1) = alpha + other.getE2() * other.getE2() * beta;
    result(1, 0) = result(0, 1) = other.getE1() * other.getE2() * beta;
    return result;
}

detail::EllipticityBase::Jacobian Distortion::dAssign(ReducedShear const & other) {
    Jacobian result = Jacobian::Zero();
    double g = other.getE();
    double alpha = 2.0 / (1 + g * g);
    double beta = -alpha * alpha;
    _complex = other.getComplex() * alpha;
    result(0, 0) = alpha + other.getE1() * other.getE1() * beta;
    result(1, 1) = alpha + other.getE2() * other.getE2() * beta;
    result(1, 0) = result(0, 1) = other.getE1() * other.getE2() * beta;
    return result;
}

void Distortion::normalize() {
    if (!(getE() <= 1.0)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            "Distortion magnitude cannot be greater than one."
        );
    }
}

void Distortion::_assignToQuadrupole(double r, double & ixx, double & iyy, double & ixy) const {
    double r2 = r*r;
    ixx = r2*(1.0 + getE1());
    iyy = r2*(1.0 - getE1());
    ixy = r2*getE2();
}

EllipseCore::Jacobian Distortion::_dAssignToQuadrupole(
    double r, double & ixx, double & iyy, double & ixy
) const {
    double r2 = r * r;
    ixx = r2*(1.0 + getE1());
    iyy = r2*(1.0 - getE1());
    ixy = r2*getE2();
    EllipseCore::Jacobian result;
    result <<
        // e1        e2            r
         r2,        0.0,    2.0*r*(1.0 + getE1()),    // ixx
        -r2,        0.0,    2.0*r*(1.0 - getE1()),    // iyy
        0.0,         r2,            2.0*r*getE2();    // ixy
    return result;
}

void Distortion::_assignFromQuadrupole(double & r, double ixx, double iyy, double ixy) {
    double tr = ixx + iyy;
    r = std::sqrt(0.5*tr);
    _complex = std::complex<double>((ixx - iyy)/tr, 2.0*ixy/tr);
    fixNaN(_complex);
}

EllipseCore::Jacobian Distortion::_dAssignFromQuadrupole(double & r, double ixx, double iyy, double ixy) {
    double tr = ixx + iyy;
    r = std::sqrt(0.5*tr);
    EllipseCore::Jacobian result;
    _complex = std::complex<double>((ixx - iyy)/tr, 2.0*ixy/tr);
    fixNaN(_complex);
    double tr2 = tr*tr;
    result <<
        //   ixx              iyy           ixy
         2.0*iyy/tr2,    -2.0*ixx/tr2,       0.0,  // e1
        -2.0*ixy/tr2,    -2.0*ixy/tr2,    2.0/tr,  // e2
              0.25/r,          0.25/r,       0.0;  // r
    return result;
}

}}}} // namespace lsst::afw::geom::ellipses

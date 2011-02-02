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
#include "lsst/afw/geom/ellipses/Distortion.h"
#include "lsst/afw/geom/ellipses/LogShear.h"
#include "lsst/afw/geom/ellipses/BaseCore.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

double Distortion::getAxisRatio() const {
    double e = getE();
    return std::sqrt((1.0 - e) / (1.0 + e));
}

Distortion & Distortion::operator=(LogShear const & other) {
    double gamma = other.getE();
    if (gamma < 1E-8) {
        _complex = 2.0 * other.getComplex();
    } else {
        double delta = std::tanh(2.0 * gamma);
        _complex = other.getComplex() * delta / gamma;
    }
    return *this;
}

detail::EllipticityBase::Jacobian Distortion::dAssign(LogShear const & other) {
    Jacobian result = Jacobian::Zero();
    double gamma = other.getE();
    if (gamma < 1E-8) {
        _complex = 2.0 * other.getComplex();
        result(0, 0) = 2.0 - 48.0 * other.getE1() * other.getE1();
        result(1, 1) = 2.0 - 48.0 * other.getE2() * other.getE2();
    } else {
        double delta = std::tanh(2.0 * gamma);
        _complex = other.getComplex() * delta / gamma;
        std::complex<double> tmp0 = other.getComplex() / gamma;
        double tmp1 = delta + 2.0 * gamma * (delta + 1.0) * (delta - 1.0);
        result(0, 0) = (delta - tmp0.real() * tmp0.real() * tmp1) / gamma;
        result(1, 0) = result(0, 1) = -tmp0.real() * tmp0.imag() * tmp1 / gamma;
        result(1, 1) = (delta - tmp0.imag() * tmp0.imag() * tmp1) / gamma;
    }
    return result;
}

void Distortion::normalize() {
    if (getE() > 1.0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            "Distortion magnitude cannot be greater than one."
        );
    }
}

}}}} // namespace lsst::afw::geom::ellipses

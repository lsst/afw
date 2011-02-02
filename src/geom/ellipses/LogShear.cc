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
#include "lsst/afw/geom/ellipses/LogShear.h"
#include "lsst/afw/geom/ellipses/Distortion.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

double LogShear::getAxisRatio() const {
    double e = getE();
    return std::exp(-2.0 * e);
}

LogShear & LogShear::operator=(Distortion const & other) {
    double delta = other.getE();
    if (delta < 1E-8) {
        _complex = 0.5 * other.getComplex();
    } else {
        double gamma = 0.25 * std::log((1.0 + delta) / (1.0 - delta));
        _complex = other.getComplex() * gamma / delta;
    }
    return *this;
}

detail::EllipticityBase::Jacobian LogShear::dAssign(Distortion const & other) {
    Jacobian result = Jacobian::Zero();
    double delta = other.getE();
    if (delta < 1E-8) {
        _complex = 0.5 * other.getComplex();
        result(0, 0) = 0.5 + 3.0 * other.getE1() * other.getE1();
        result(1, 1) = 0.5 + 3.0 * other.getE2() * other.getE2();
    } else {
        double gamma = 0.25 * std::log((1.0 + delta) / (1.0 - delta));
        _complex = other.getComplex() * gamma / delta;
        std::complex<double> tmp0 = 0.5 * other.getComplex() / delta;
        double tmp1 = (delta - 1.0) * (delta + 1.0) / delta;
        double tmp2 = -(2.0 / tmp1 + 4.0 * gamma) / delta;
        result(0, 0) = result(1, 1) = gamma / delta;
        result(0, 0) += tmp0.real() * tmp0.real() * tmp2;
        result(1, 1) += tmp0.imag() * tmp0.imag() * tmp2;
        result(0, 1) = result(1, 0) = tmp0.real() * tmp0.imag() * tmp2;
    }
    return result;
}

}}}} // namespace lsst::afw::geom::ellipses

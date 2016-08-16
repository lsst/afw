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

#ifndef LSST_AFW_GEOM_ELLIPSES_ReducedShear_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_ReducedShear_h_INCLUDED

#include "lsst/afw/geom/ellipses/EllipticityBase.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

class Distortion;

class ConformalShear;

/**
 *  @brief A complex ellipticity with magnitude @f$|e| = \frac{a-b}{a+b} @f$.
 *
 *  For a more complete definition, see Bernstein and Jarvis (2002); this the same as their
 *  reduced shear @f$g@f$ (eq. 2.8).
 */
class ReducedShear : public detail::EllipticityBase {
public:

    explicit ReducedShear(std::complex<double> const & complex) : detail::EllipticityBase(complex) {}

    explicit ReducedShear(double e1=0.0, double e2=0.0) : detail::EllipticityBase(e1, e2) {}

    ReducedShear(ReducedShear const & other) : detail::EllipticityBase(other.getComplex()) {}

    explicit ReducedShear(Distortion const & other) { this->operator=(other); }

    explicit ReducedShear(ConformalShear const & other) { this->operator=(other); }

    ReducedShear & operator=(ReducedShear const & other) {
        _complex = other._complex;
        return *this;
    }

    ReducedShear & operator=(Distortion const & other);

    ReducedShear & operator=(ConformalShear const & other);

    Jacobian dAssign(ReducedShear const & other) {
        _complex = other._complex;
        return Jacobian::Identity();
    }

    Jacobian dAssign(Distortion const & other);

    Jacobian dAssign(ConformalShear const & other);

    double getAxisRatio() const;

    void normalize() {}

    static std::string getName() { return "ReducedShear"; }

};

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_ReducedShear_h_INCLUDED

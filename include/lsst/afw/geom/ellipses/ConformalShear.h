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

#ifndef LSST_AFW_GEOM_ELLIPSES_ConformalShear_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_ConformalShear_h_INCLUDED

#include "lsst/afw/geom/ellipses/EllipticityBase.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

class Distortion;

class ReducedShear;

/**
 *  A logarithmic complex ellipticity with magnitude @f$|e| = \ln (a/b) @f$.
 *
 *  For a more complete definition, see Bernstein and Jarvis (2002); this the same as their
 *  conformal shear @f$\eta@f$ (eq. 2.3-2.6).
 */
class ConformalShear : public detail::EllipticityBase {
public:
    explicit ConformalShear(std::complex<double> const& complex) : detail::EllipticityBase(complex) {}

    explicit ConformalShear(double e1 = 0.0, double e2 = 0.0) : detail::EllipticityBase(e1, e2) {}

    ConformalShear(ConformalShear const& other) : detail::EllipticityBase(other.getComplex()) {}
    // Delegate to copy-constructor for backwards compatibility
    ConformalShear(ConformalShear&& other) : ConformalShear(other) {}
    ~ConformalShear() = default;

    explicit ConformalShear(Distortion const& other) { this->operator=(other); }

    explicit ConformalShear(ReducedShear const& other) { this->operator=(other); }

    ConformalShear& operator=(ConformalShear const& other) {
        _complex = other._complex;
        return *this;
    }
    // Delegate to copy-assignment for backwards compatibility
    ConformalShear& operator=(ConformalShear&& other) { return *this = other; }

    ConformalShear& operator=(Distortion const& other);

    ConformalShear& operator=(ReducedShear const& other);

    Jacobian dAssign(ConformalShear const& other) {
        _complex = other._complex;
        return Jacobian::Identity();
    }

    Jacobian dAssign(Distortion const& other);

    Jacobian dAssign(ReducedShear const& other);

    double getAxisRatio() const;

    void normalize() {}

    static std::string getName() { return "ConformalShear"; }
};
}
}
}
}  // namespace lsst::afw::geom::ellipses

#endif  // !LSST_AFW_GEOM_ELLIPSES_ConformalShear_h_INCLUDED

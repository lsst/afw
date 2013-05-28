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

#ifndef LSST_AFW_GEOM_ELLIPSES_radii_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_radii_h_INCLUDED

#include "lsst/afw/geom/ellipses/EllipseCore.h"
#include "lsst/afw/geom/ellipses/Distortion.h"
#include <complex>

namespace lsst { namespace afw { namespace geom {
namespace ellipses {

#ifndef SWIG
template <typename Ellipticity_, typename Radius_> class Separable;
#endif

class DeterminantRadius;
class TraceRadius;

/**
 * @brief The radius defined as the 4th root of the determinant of the quadrupole matrix.
 *
 * The determinant radius is equal to the standard radius for a circle, and
 * \f$\pi R_{det}^2\f$ is the area of the ellipse.
 *
 * @sa Separable
 */
class DeterminantRadius {
public:

    void normalize() {
        if (_value < 0)
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              "Ellipse radius cannot be negative.");
    }

    static std::string getName() { return "DeterminantRadius"; }

    explicit DeterminantRadius(double value=1.0) : _value(value) {}

    operator double const & () const { return _value; }

    operator double & () { return _value; }

    DeterminantRadius & operator=(double value) { _value = value; return *this; }

private:

    template <typename T1, typename T2> friend class Separable;

    /// Undefined and disabled; conversion between trace and determinant radii requires ellipticity.
    void operator=(TraceRadius const &);

    void assignFromQuadrupole(
        double ixx, double iyy, double ixy,
        Distortion & distortion
    );

    EllipseCore::Jacobian dAssignFromQuadrupole(
        double ixx, double iyy, double ixy,
        Distortion & distortion
    );

     void assignToQuadrupole(
        Distortion const & distortion,
        double & ixx, double & iyy, double & ixy
    ) const;

     EllipseCore::Jacobian dAssignToQuadrupole(
        Distortion const & distortion,
        double & ixx, double & iyy, double & ixy
    ) const;

    double _value;
};

/**
 * @brief The radius defined as \f$\sqrt{0.5(I_{xx} + I_{yy})}\f$
 *
 * The trace radius is equal to the standard radius for a circle
 *
 * @sa Separable
 */
class TraceRadius {
public:

    void normalize() {
        if (_value < 0)
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              "Ellipse radius cannot be negative.");
    }

    static std::string getName() { return "TraceRadius"; }

    explicit TraceRadius(double value=1.0) : _value(value) {}

    operator double const & () const { return _value; }

    operator double & () { return _value; }

    TraceRadius & operator=(double value) { _value = value; return *this; }

private:

    template <typename T1, typename T2> friend class Separable;

    /// Undefined and disabled; conversion between trace and determinant radii requires ellipticity.
    void operator=(DeterminantRadius const &);

    void assignFromQuadrupole(
        double ixx, double iyy, double ixy,
        Distortion & distortion
    );

    EllipseCore::Jacobian dAssignFromQuadrupole(
        double ixx, double iyy, double ixy,
        Distortion & distortion
    );

     void assignToQuadrupole(
        Distortion const & distortion,
        double & ixx, double & iyy, double & ixy
    ) const;

     EllipseCore::Jacobian dAssignToQuadrupole(
        Distortion const & distortion,
        double & ixx, double & iyy, double & ixy
    ) const;

    double _value;
};

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_radii_h_INCLUDED

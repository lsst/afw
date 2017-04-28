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

#ifndef LSST_AFW_GEOM_ELLIPSES_radii_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_radii_h_INCLUDED

/*
 *  Helper classes defining radii for Separable core.
 *
 *  Note: do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/BaseCore.h"
#include "lsst/afw/geom/ellipses/Distortion.h"
#include <complex>

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

template <typename Ellipticity_, typename Radius_>
class Separable;

class DeterminantRadius;
class TraceRadius;
class LogDeterminantRadius;
class LogTraceRadius;

/**
 * The radius defined as the 4th root of the determinant of the quadrupole matrix.
 *
 * The determinant radius is equal to the standard radius for a circle, and
 * @f$\pi R_{det}^2@f$ is the area of the ellipse.
 */
class DeterminantRadius {
public:
    void normalize() {
        if (_value < 0)
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              "Ellipse radius cannot be negative.");
    }

    static std::string getName() { return "DeterminantRadius"; }

    explicit DeterminantRadius(double value = 1.0) : _value(value) {}

    explicit DeterminantRadius(LogDeterminantRadius const &other);

    operator double const &() const { return _value; }

    operator double &() { return _value; }

    DeterminantRadius &operator=(double value) {
        _value = value;
        return *this;
    }

    DeterminantRadius &operator=(LogDeterminantRadius const &other);

private:
    template <typename T1, typename T2>
    friend class Separable;

    /// Undefined and disabled; conversion between trace and determinant radii requires ellipticity.
    void operator=(TraceRadius const &);
    /// Undefined and disabled; conversion between trace and determinant radii requires ellipticity.
    void operator=(LogTraceRadius const &);

    void assignFromQuadrupole(double ixx, double iyy, double ixy, Distortion &distortion);

    BaseCore::Jacobian dAssignFromQuadrupole(double ixx, double iyy, double ixy, Distortion &distortion);

    void assignToQuadrupole(Distortion const &distortion, double &ixx, double &iyy, double &ixy) const;

    BaseCore::Jacobian dAssignToQuadrupole(Distortion const &distortion, double &ixx, double &iyy,
                                           double &ixy) const;

    double _value;
};

/**
 * The radius defined as @f$\sqrt{0.5(I_{xx} + I_{yy})}@f$
 *
 * The trace radius is equal to the standard radius for a circle
 */
class TraceRadius {
public:
    void normalize() {
        if (_value < 0)
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              "Ellipse radius cannot be negative.");
    }

    static std::string getName() { return "TraceRadius"; }

    explicit TraceRadius(double value = 1.0) : _value(value) {}

    explicit TraceRadius(LogTraceRadius const &other);

    operator double const &() const { return _value; }

    operator double &() { return _value; }

    TraceRadius &operator=(double value) {
        _value = value;
        return *this;
    }

    TraceRadius &operator=(LogTraceRadius const &other);

private:
    template <typename T1, typename T2>
    friend class Separable;

    /// Undefined and disabled; conversion between trace and determinant radii requires ellipticity.
    void operator=(DeterminantRadius const &);
    /// Undefined and disabled; conversion between trace and determinant radii requires ellipticity.
    void operator=(LogDeterminantRadius const &);

    void assignFromQuadrupole(double ixx, double iyy, double ixy, Distortion &distortion);

    BaseCore::Jacobian dAssignFromQuadrupole(double ixx, double iyy, double ixy, Distortion &distortion);

    void assignToQuadrupole(Distortion const &distortion, double &ixx, double &iyy, double &ixy) const;

    BaseCore::Jacobian dAssignToQuadrupole(Distortion const &distortion, double &ixx, double &iyy,
                                           double &ixy) const;

    double _value;
};

/**
 * The natural logarithm of the DeterminantRadius
 */
class LogDeterminantRadius {
public:
    void normalize() {}

    static std::string getName() { return "LogDeterminantRadius"; }

    explicit LogDeterminantRadius(double value = 0.0) : _value(value) {}

    explicit LogDeterminantRadius(DeterminantRadius const &other);

    operator double const &() const { return _value; }

    operator double &() { return _value; }

    LogDeterminantRadius &operator=(double value) {
        _value = value;
        return *this;
    }

    LogDeterminantRadius &operator=(DeterminantRadius const &other);

private:
    template <typename T1, typename T2>
    friend class Separable;

    /// Undefined and disabled; conversion between trace and determinant radii requires ellipticity.
    void operator=(TraceRadius const &);
    /// Undefined and disabled; conversion between trace and determinant radii requires ellipticity.
    void operator=(LogTraceRadius const &);

    void assignFromQuadrupole(double ixx, double iyy, double ixy, Distortion &distortion);

    BaseCore::Jacobian dAssignFromQuadrupole(double ixx, double iyy, double ixy, Distortion &distortion);

    void assignToQuadrupole(Distortion const &distortion, double &ixx, double &iyy, double &ixy) const;

    BaseCore::Jacobian dAssignToQuadrupole(Distortion const &distortion, double &ixx, double &iyy,
                                           double &ixy) const;

    double _value;
};

/**
 * The natural logarithm of the TraceRadius
 */
class LogTraceRadius {
public:
    void normalize() {}

    static std::string getName() { return "LogTraceRadius"; }

    explicit LogTraceRadius(double value = 0.0) : _value(value) {}

    explicit LogTraceRadius(TraceRadius const &other);

    operator double const &() const { return _value; }

    operator double &() { return _value; }

    LogTraceRadius &operator=(double value) {
        _value = value;
        return *this;
    }

    LogTraceRadius &operator=(TraceRadius const &value);

private:
    template <typename T1, typename T2>
    friend class Separable;

    /// Undefined and disabled; conversion between trace and determinant radii requires ellipticity.
    void operator=(DeterminantRadius const &);
    /// Undefined and disabled; conversion between trace and determinant radii requires ellipticity.
    void operator=(LogDeterminantRadius const &);

    void assignFromQuadrupole(double ixx, double iyy, double ixy, Distortion &distortion);

    BaseCore::Jacobian dAssignFromQuadrupole(double ixx, double iyy, double ixy, Distortion &distortion);

    void assignToQuadrupole(Distortion const &distortion, double &ixx, double &iyy, double &ixy) const;

    BaseCore::Jacobian dAssignToQuadrupole(Distortion const &distortion, double &ixx, double &iyy,
                                           double &ixy) const;

    double _value;
};

inline DeterminantRadius::DeterminantRadius(LogDeterminantRadius const &other) : _value(std::exp(other)) {}
inline TraceRadius::TraceRadius(LogTraceRadius const &other) : _value(std::exp(other)) {}
inline LogDeterminantRadius::LogDeterminantRadius(DeterminantRadius const &other) : _value(std::log(other)) {}
inline LogTraceRadius::LogTraceRadius(TraceRadius const &other) : _value(std::log(other)) {}

inline DeterminantRadius &DeterminantRadius::operator=(LogDeterminantRadius const &other) {
    _value = std::exp(other);
    return *this;
}

inline TraceRadius &TraceRadius::operator=(LogTraceRadius const &other) {
    _value = std::exp(other);
    return *this;
}

inline LogDeterminantRadius &LogDeterminantRadius::operator=(DeterminantRadius const &other) {
    _value = std::log(other);
    return *this;
}

inline LogTraceRadius &LogTraceRadius::operator=(TraceRadius const &other) {
    _value = std::log(other);
    return *this;
}
}
}
}
}  // namespace lsst::afw::geom::ellipses

#endif  // !LSST_AFW_GEOM_ELLIPSES_radii_h_INCLUDED

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

/**
 *  \file
 *  @brief Helper classes defining radii for Separable core.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/BaseCore.h"
#include "lsst/afw/geom/ellipses/Distortion.h"
#include <complex>

namespace lsst { namespace afw { namespace geom {
namespace ellipses {

#ifndef SWIG
template <typename Ellipticity_, typename Radius_> class Separable;
#endif

class GeometricRadius;
class ArithmeticRadius;
class LogGeometricRadius;
class LogArithmeticRadius;

class GeometricRadius {
public:

    void normalize() {
        if (_value < 0)
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, 
                              "Ellipse radius cannot be negative.");
    }

    static std::string getName() { return "GeometricRadius"; }

    explicit GeometricRadius(double value=1.0) : _value(value) {}

    explicit GeometricRadius(LogGeometricRadius const & other);

    operator double const & () const { return _value; }

    operator double & () { return _value; }

    GeometricRadius & operator=(double value) { _value = value; return *this; }

    GeometricRadius & operator=(LogGeometricRadius const & other);

private:
    
    template <typename T1, typename T2> friend class Separable;

    /// Undefined and disabled; conversion between arithmetic and geometric radii requires ellipticity.
    void operator=(ArithmeticRadius const &);
    /// Undefined and disabled; conversion between arithmetic and geometric radii requires ellipticity.
    void operator=(LogArithmeticRadius const &);

    void assignFromQuadrupole(
        double ixx, double iyy, double ixy,
        Distortion & distortion
    );

    BaseCore::Jacobian dAssignFromQuadrupole(
        double ixx, double iyy, double ixy, 
        Distortion & distortion
    );

     void assignToQuadrupole(
        Distortion const & distortion,
        double & ixx, double & iyy, double & ixy
    ) const;

     BaseCore::Jacobian dAssignToQuadrupole(
        Distortion const & distortion,
        double & ixx, double & iyy, double & ixy
    ) const;

    double _value;
};

class ArithmeticRadius {
public:

    void normalize() {
        if (_value < 0)
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              "Ellipse radius cannot be negative.");
    }

    static std::string getName() { return "ArithmeticRadius"; }

    explicit ArithmeticRadius(double value=1.0) : _value(value) {}

    explicit ArithmeticRadius(LogArithmeticRadius const & other);

    operator double const & () const { return _value; }

    operator double & () { return _value; }

    ArithmeticRadius & operator=(double value) { _value = value; return *this; }

    ArithmeticRadius & operator=(LogArithmeticRadius const & other);

private:
    
    template <typename T1, typename T2> friend class Separable;

    /// Undefined and disabled; conversion between arithmetic and geometric radii requires ellipticity.
    void operator=(GeometricRadius const &);
    /// Undefined and disabled; conversion between arithmetic and geometric radii requires ellipticity.
    void operator=(LogGeometricRadius const &);

    void assignFromQuadrupole(
        double ixx, double iyy, double ixy,
        Distortion & distortion
    );

    BaseCore::Jacobian dAssignFromQuadrupole(
        double ixx, double iyy, double ixy, 
        Distortion & distortion
    );

     void assignToQuadrupole(
        Distortion const & distortion,
        double & ixx, double & iyy, double & ixy
    ) const;

     BaseCore::Jacobian dAssignToQuadrupole(
        Distortion const & distortion,
        double & ixx, double & iyy, double & ixy
    ) const;

    double _value;
};

class LogGeometricRadius {
public:

    void normalize() {}

    static std::string getName() { return "LogGeometricRadius"; }

    explicit LogGeometricRadius(double value=0.0) : _value(value) {}

    explicit LogGeometricRadius(GeometricRadius const & other);

    operator double const & () const { return _value; }

    operator double & () { return _value; }

    LogGeometricRadius & operator=(double value) { _value = value; return *this; }

    LogGeometricRadius & operator=(GeometricRadius const & other);

private:
    
    template <typename T1, typename T2> friend class Separable;

    /// Undefined and disabled; conversion between arithmetic and geometric radii requires ellipticity.
    void operator=(ArithmeticRadius const &);
    /// Undefined and disabled; conversion between arithmetic and geometric radii requires ellipticity.
    void operator=(LogArithmeticRadius const &);

    void assignFromQuadrupole(
        double ixx, double iyy, double ixy,
        Distortion & distortion
    );

    BaseCore::Jacobian dAssignFromQuadrupole(
        double ixx, double iyy, double ixy, 
        Distortion & distortion
    );

     void assignToQuadrupole(
        Distortion const & distortion,
        double & ixx, double & iyy, double & ixy
    ) const;

     BaseCore::Jacobian dAssignToQuadrupole(
        Distortion const & distortion,
        double & ixx, double & iyy, double & ixy
    ) const;

    double _value;
};

class LogArithmeticRadius {
public:
    
    void normalize() {}

    static std::string getName() { return "LogArithmeticRadius"; }

    explicit LogArithmeticRadius(double value=0.0) : _value(value) {}

    explicit LogArithmeticRadius(ArithmeticRadius const & other);

    operator double const & () const { return _value; }

    operator double & () { return _value; }

    LogArithmeticRadius & operator=(double value) { _value = value; return *this; }

    LogArithmeticRadius & operator=(ArithmeticRadius const & value);

private:
    
    template <typename T1, typename T2> friend class Separable;

    /// Undefined and disabled; conversion between arithmetic and geometric radii requires ellipticity.
    void operator=(GeometricRadius const &);
    /// Undefined and disabled; conversion between arithmetic and geometric radii requires ellipticity.
    void operator=(LogGeometricRadius const &);

    void assignFromQuadrupole(
        double ixx, double iyy, double ixy,
        Distortion & distortion
    );

    BaseCore::Jacobian dAssignFromQuadrupole(
        double ixx, double iyy, double ixy, 
        Distortion & distortion
    );

     void assignToQuadrupole(
        Distortion const & distortion,
        double & ixx, double & iyy, double & ixy
    ) const;

     BaseCore::Jacobian dAssignToQuadrupole(
        Distortion const & distortion,
        double & ixx, double & iyy, double & ixy
    ) const;

    double _value;
};

inline GeometricRadius::GeometricRadius(LogGeometricRadius const & other) : _value(std::exp(other)) {}
inline ArithmeticRadius::ArithmeticRadius(LogArithmeticRadius const & other) : _value(std::exp(other)) {}
inline LogGeometricRadius::LogGeometricRadius(GeometricRadius const & other) : _value(std::log(other)) {}
inline LogArithmeticRadius::LogArithmeticRadius(ArithmeticRadius const & other) : _value(std::log(other)) {}

inline GeometricRadius & GeometricRadius::operator=(LogGeometricRadius const & other) {
    _value = std::exp(other);
    return *this;
}

inline ArithmeticRadius & ArithmeticRadius::operator=(LogArithmeticRadius const & other) {
    _value = std::exp(other);
    return *this;
}

inline LogGeometricRadius & LogGeometricRadius::operator=(GeometricRadius const & other) {
    _value = std::log(other);
    return *this;
}

inline LogArithmeticRadius & LogArithmeticRadius::operator=(ArithmeticRadius const & other) {
    _value = std::log(other);
    return *this;
}

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_radii_h_INCLUDED

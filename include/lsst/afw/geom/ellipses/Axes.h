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

#ifndef LSST_AFW_GEOM_ELLIPSES_Axes_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_Axes_h_INCLUDED

/**
 *  \file
 *  @brief Definitions and inlines for Axes.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/ellipses/EllipseCore.h"
#include "lsst/afw/geom/ellipses/Convolution.h"
#include "lsst/afw/geom/ellipses/Transformer.h"
#include "lsst/afw/geom/ellipses/GridTransform.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

/**
 *  @brief An ellipse core for the semimajor/semiminor axis and position angle parametrization (a,b,theta).
 */
class Axes : public EllipseCore {
public:

    typedef boost::shared_ptr<Axes> Ptr;
    typedef boost::shared_ptr<Axes const> ConstPtr;

    enum ParameterEnum { A=0, B=1, THETA=2 }; ///< Definitions for elements of a core vector.

    double const getA() const { return _vector[A]; }
    void setA(double a) { _vector[A] = a; }

    double const getB() const { return _vector[B]; }
    void setB(double b) { _vector[B] = b; }

    Angle const getTheta() const { return _vector[THETA] * radians; }
    void setTheta(Angle theta) { _vector[THETA] = theta.asRadians(); }

    /// @brief Deep copy the ellipse core.
    Ptr clone() const { return boost::static_pointer_cast<Axes>(_clone()); }

    /// Return a string that identifies this parametrization.
    virtual std::string getName() const;

    /**
     *  @brief Put the parameters into a "standard form", if possible, and throw InvalidEllipseParameters
     *         if they cannot be normalized.
     */
    virtual void normalize();

    virtual void readParameters(double const * iter);

    virtual void writeParameters(double * iter) const;

    /// @brief Standard assignment.
    Axes & operator=(Axes const & other) { _vector = other._vector; return *this; }

    /// @brief Converting assignment.
    Axes & operator=(EllipseCore const & other) { EllipseCore::operator=(other); return *this; }

    /// @brief Construct from parameter values
    explicit Axes(double a=1.0, double b=1.0, Angle theta=0.0*radians, bool normalize=false) :
        _vector(a, b, theta.asRadians()) { if (normalize) this->normalize(); }

    /// @brief Construct from a parameter vector.
    explicit Axes(EllipseCore::ParameterVector const & vector, bool normalize=false) :
        _vector(vector) { if (normalize) this->normalize(); }

    /// @brief Copy constructor.
    Axes(Axes const & other) : _vector(other._vector) {}

    /// @brief Converting copy constructor.
    Axes(EllipseCore const & other) { *this = other; }

#ifndef SWIG
    /// @brief Converting copy constructor.
    Axes(EllipseCore::Transformer const & transformer) {
        transformer.apply(*this);
    }

    /// @brief Converting copy constructor.
    Axes(EllipseCore::Convolution const & convolution) {
        convolution.apply(*this);
    }
#endif
protected:

    virtual EllipseCore::Ptr _clone() const { return boost::make_shared<Axes>(*this); }

    virtual void _assignToQuadrupole(double & ixx, double & iyy, double & ixy) const;
    virtual void _assignFromQuadrupole(double ixx, double iyy, double ixy);

    virtual void _assignToAxes(double & a, double & b, double & theta) const;
    virtual void _assignFromAxes(double a, double b, double theta);

    virtual Jacobian _dAssignToQuadrupole(double & ixx, double & iyy, double & ixy) const;
    virtual Jacobian _dAssignFromQuadrupole(double ixx, double iyy, double ixy);

    virtual Jacobian _dAssignToAxes(double & a, double & b, double & theta) const;
    virtual Jacobian _dAssignFromAxes(double a, double b, double theta);

private:
    static Registrar<Axes> registrar;

    ParameterVector _vector;
};

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_Axes_h_INCLUDED

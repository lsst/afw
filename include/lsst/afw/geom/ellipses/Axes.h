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

/*
 * Definitions and inlines for Axes.
 *
 * Note: do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/BaseCore.h"
#include "lsst/afw/geom/ellipses/Convolution.h"
#include "lsst/afw/geom/ellipses/Transformer.h"
#include "lsst/afw/geom/ellipses/GridTransform.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

/**
 * An ellipse core for the semimajor/semiminor axis and position angle parametrization (a,b,theta).
 */
class Axes : public BaseCore {
public:

    enum ParameterEnum { A=0, B=1, THETA=2 }; ///< Definitions for elements of a core vector.

    double const getA() const { return _vector[A]; }
    void setA(double a) { _vector[A] = a; }

    double const getB() const { return _vector[B]; }
    void setB(double b) { _vector[B] = b; }

    double const getTheta() const { return _vector[THETA]; }
    void setTheta(double theta) { _vector[THETA] = theta; }

    /// Deep copy the ellipse core.
    std::shared_ptr<Axes> clone() const { return std::static_pointer_cast<Axes>(_clone()); }

    /// Return a string that identifies this parametrization.
    virtual std::string getName() const;

    /**
     *  @brief Put the parameters into a "standard form", if possible, and throw InvalidParameterError
     *         if they cannot be normalized.
     */
    virtual void normalize();

    virtual void readParameters(double const * iter);

    virtual void writeParameters(double * iter) const;

    /// Standard assignment.
    Axes & operator=(Axes const & other) { _vector = other._vector; return *this; }

    /// Converting assignment.
    Axes & operator=(BaseCore const & other) { BaseCore::operator=(other); return *this; }

    /// Construct from parameter values
    explicit Axes(double a=1.0, double b=1.0, double theta=0.0, bool normalize=false) :
        _vector(a, b, theta) { if (normalize) this->normalize(); }

    /// Construct from a parameter vector.
    explicit Axes(BaseCore::ParameterVector const & vector, bool normalize=false) :
        _vector(vector) { if (normalize) this->normalize(); }

    /// Copy constructor.
    Axes(Axes const & other) : _vector(other._vector) {}

    /// Converting copy constructor.
    Axes(BaseCore const & other) { *this = other; }

    /// Converting copy constructor.
    Axes(BaseCore::Transformer const & transformer) {
        transformer.apply(*this);
    }

    /// Converting copy constructor.
    Axes(BaseCore::Convolution const & convolution) {
        convolution.apply(*this);
    }
protected:

    virtual std::shared_ptr<BaseCore> _clone() const { return std::make_shared<Axes>(*this); }

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

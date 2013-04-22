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

#ifndef LSST_AFW_GEOM_ELLIPSES_Axes_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_Axes_h_INCLUDED

#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/ellipses/EllipseCore.h"
#include "lsst/afw/geom/ellipses/Convolution.h"
#include "lsst/afw/geom/ellipses/Transformer.h"
#include "lsst/afw/geom/ellipses/GridTransform.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

/**
 *  @brief An EllipeCore for the semimajor/semiminor axis and position angle parametrization.
 *
 *  The parameters of an Axes are:
 *   - @f$a@f$ - the semimajor axis of the ellipse (largest radius)
 *   - @f$b@f$ - the semiminor axis of the ellipse (smallest radius)
 *   - @f$\theta@f$ - angle of the semimajor axis, measured counterclockwise from the x-axis.  In
 *                    radians when the Angle class cannot be used.
 */
class Axes : public EllipseCore {
public:

    enum ParameterEnum { A=0, B=1, THETA=2 }; ///< Enum used to index the elements of a parameter vector.

    //@{
    /// Basic getters and setters for parameters; see Axes class documentation for more information.
    double const getA() const { return _vector[A]; }
    void setA(double a) { _vector[A] = a; }

    double const getB() const { return _vector[B]; }
    void setB(double b) { _vector[B] = b; }

    Angle const getTheta() const { return _vector[THETA] * radians; }
    void setTheta(Angle theta) { _vector[THETA] = theta.asRadians(); }
    //@}

    /// Polymorphic deep copy.
    PTR(Axes) clone() const { return boost::static_pointer_cast<Axes>(_clone()); }

    /// Return a string that identifies this parametrization ("Axes").
    virtual std::string getName() const;

    /**
     *  @brief Check parameters and put them into standard form.
     *
     *  This will swap @f$a@f$ and @f$b@f$ and adjust @f$\theta@f$ appropriately if @f$b > a@f$, and ensure
     *  that @f$-\pi/2 \lt \theta \le \pi/2@f$.
     *
     *  @throw lsst::pex::exception::InvalidParamterException if @f$a<0@f$ or @f$b<0@f$.
     */
    virtual void normalize();

    /// Standard assignment.
    Axes & operator=(Axes const & other) { _vector = other._vector; return *this; }

    /// Converting assignment.
    Axes & operator=(EllipseCore const & other) { EllipseCore::operator=(other); return *this; }

    /**
     *  @brief Construct from parameter values, and optionally normalize.
     *
     *  For more precise parameter value definitions, see the Axes class documentation.
     *  For more information about normalization, see normalize().
     */
    Axes(double a, double b, Angle theta=0.0*radians, bool normalize=false) :
        _vector(a, b, theta.asRadians()) { if (normalize) this->normalize(); }

    /// Construct a circle with the given radius.
    explicit Axes(double radius=1.0) : _vector(radius, radius, 0.0) {}

    /**
     *  @brief Construct from a parameter vector of (a, b, theta).
     *
     *  @copydetails Axes::Axes
     */
    explicit Axes(EllipseCore::ParameterVector const & vector, bool normalize=false) :
        _vector(vector) { if (normalize) this->normalize(); }

    /// Copy constructor.
    Axes(Axes const & other) : _vector(other._vector) {}

    /// Converting copy constructor.
    Axes(EllipseCore const & other) { *this = other; }

#ifndef SWIG
    /// Implicit construction from a Transformer expression temporary.
    Axes(EllipseCore::Transformer const & transformer) {
        transformer.apply(*this);
    }

    /// Implicit construction from a Convolution expression temporary.
    Axes(EllipseCore::Convolution const & convolution) {
        convolution.apply(*this);
    }
#endif
protected:

    virtual PTR(EllipseCore) _clone() const { return boost::make_shared<Axes>(*this); }

    virtual void readParameters(double const * iter);
    virtual void writeParameters(double * iter) const;

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

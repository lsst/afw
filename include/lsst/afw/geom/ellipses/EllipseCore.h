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

#ifndef LSST_AFW_GEOM_ELLIPSES_EllipseCore_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_EllipseCore_h_INCLUDED

#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
#include <memory>
#include "Eigen/Core"

#include "lsst/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/LinearTransform.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

class Parametric;

/**
 *  @brief A base class for parametrizations of the 3 numbers that define a translation-free ellipse.
 *
 *  A subclass of EllipseCore provides a particular interpretation of the three values that
 *  define an ellipse's size and ellipticity (including position angle).  All EllipseCore subclasses
 *  are implicitly convertible and can be assigned to from any other EllipseCore.
 */
class EllipseCore {
public:
#ifndef SWIG
    class Transformer;
    class GridTransform;
    class Convolution;
#endif

    typedef Eigen::Vector3d ParameterVector;  ///< Type used by getParameterVector and setParameterVector
    typedef Eigen::Matrix3d Jacobian; ///< Matrix type used in derivatives of parametrization conversions

    /// Construct a unit-circle EllipseCore with the parametrization specified by the given name.
    static PTR(EllipseCore) make(std::string const & name);

    /**
     *  @brief Construct an EllipseCore with parameters defined by the given vector.
     *
     *  The interpretation of the vector is specific to the EllipseCore subclass indicated by the given name.
     */
    static PTR(EllipseCore) make(std::string const & name, ParameterVector const & parameters);

    /**
     *  @brief Construct an EllipseCore with parameters defined by the given values.
     *
     *  The interpretation of the vector is specific to the EllipseCore subclass indicated by the given name.
     */
    static PTR(EllipseCore) make(std::string const & name, double v1, double v2, double v3);

    /// Construct an EllipseCore by converting to the parametrization defined by the given name.
    static PTR(EllipseCore) make(std::string const & name, EllipseCore const & other);

#ifndef SWIG
    /// Construct an EllipseCore from a Transformer expression temporary.
    static PTR(EllipseCore) make(std::string const & name, Transformer const & other);

    /// Construct an EllipseCore from a Convolution expression temporary.
    static PTR(EllipseCore) make(std::string const & name, Convolution const & other);
#endif

    /// Return a string that identifies this parametrization.
    virtual std::string getName() const = 0;

    /// Polymorphic deep-copy.
    PTR(EllipseCore) clone() const { return _clone(); }

    /**
     *  @brief Put the parameters into a "standard form".
     *
     *  See the documentation for each EllipseCore subclass for information on what normalization means.
     *
     *  @throw InvalidParameterException if they cannot be normalized.
     */
    virtual void normalize() = 0;

    /// Grow the EllipseCore in-place by adding 'buffer' to its semimajor and semiminor axes
    void grow(double buffer);

    /// Scale the EllipseCore in-place by multiplying its semimajor and semiminor axes by 'factor'.
    void scale(double factor);

    /// Return the area of the EllipseCore.
    double getArea() const;

    /**
     *  @brief Return the radius defined as the 4th root of the determinant of the quadrupole matrix.
     *
     *  The determinant radius is equal to the standard radius for a circle,
     *  and its square times pi is the area of the ellipse.
     */
    double getDeterminantRadius() const;

    /**
     *  @brief Return the radius defined as the square root of one half the trace of the quadrupole matrix.
     *
     *  The trace radius is equal to the standard radius for a circle.
     */
    double getTraceRadius() const;

    //@{
    /**
     *  @brief Coordinate transforms
     *
     *  These member functions transform the ellipse by the given LinearTransform.
     *  The transform can be done in-place by calling inPlace() on the returned
     *  expression object, or returned as a new shared_ptr by calling copy().
     *
     *  The expression object is also implicitly convertible to any EllipseCore, so
     *  you can transform and convert in a single step:
     *  @code
     *  Quadrupole q = Axes(3.0, 2.0).transform(LinearTransform::makeRotation(2.3*radians));
     *  @endcode
     *
     *  In Python, an EllipseCore of the same type as this is returned.
     */
    Transformer transform(LinearTransform const & transform);
    Transformer const transform(LinearTransform const & transform) const;
    //@}

    /**
     *  @brief Return the transform that maps the ellipse to the unit circle.
     *
     *  The returned temporary expression object is implicitly convertible to LinearTransform
     *  and also supports differentiation.
     *
     *  In Python, a LinearTransform object is returned directly.
     */
    GridTransform const getGridTransform() const;

    //@{
    /**
     *  @brief Convolve two bivariate Gaussians defined by their 1-sigma ellipses.
     *
     *  As with transform, the convolution can be done in-place by calling inPlace() on
     *  the returned expression object, or returned as a new shared_ptr by calling copy().
     *
     *  The expression object is also implicitly convertible to any EllipseCore, so
     *  you can convolve and convert in a single step:
     *  @code
     *  Quadrupole q = Axes(3.0, 2.0).convolve(Axes(1.0));
     *  @endcode
     *
     *  In Python, an EllipseCore of the same type as this is returned.
     */
    Convolution convolve(EllipseCore const & other);
    Convolution const convolve(EllipseCore const & other) const;
    //@}

    /// Return the size of the bounding box for the ellipse core.
    Extent2D computeDimensions() const;

    /// Return the parameters of the EllipseCore as a vector.
    ParameterVector const getParameterVector() const;

    /// Set the parameters of the EllipseCore from a vector.
    void setParameterVector(ParameterVector const & vector);

    /**
     *  @brief Compare two EllipseCores for equality.
     *
     *  EllipseCores are only equal if they have the same type and exactly the same parameters;
     *  there is no special handling for approximate floating-point equality.
     */
    bool operator==(EllipseCore const & other) const;

    /**
     *  @brief Compare two ellipse cores for inequality.
     *
     *  EllipseCores are only equal if they have the same type and exactly the same parameters;
     *  there is no special handling for approximate floating-point equality.
     */
    bool operator!=(EllipseCore const & other) const { return !operator==(other); }

    /**
     *  @brief Set the parameters of this ellipse core from another.
     *
     *  This does not change the parametrization of the EllipseCore being assigned to.
     *
     *  Exposed as EllipseCore.assign in Python.
     */
    EllipseCore & operator=(EllipseCore const & other);

    /// Assign other to this and return the derivative of the conversion, d(this)/d(other).
    Jacobian dAssign(EllipseCore const & other);

    /// Return a new EllipseCore equivalent to this with type specified as a template parameter.
#ifndef SWIG
    template <typename Output>
    Output as() const {
        Output r;
        r = *this;
        return r;
    }
#endif

    /**
     *  @brief Return a new PTR(EllipseCore) equivalent to this with type specified by a string name.
     *
     *  Wrapped as "as_" in Python, because "as" is a Python keyword.  In Python, the type classes
     *  of EllipseCore subclasses may be passed instead of the name.
     */
    PTR(EllipseCore) as(std::string const & name) const { return EllipseCore::make(name, *this); }

    virtual ~EllipseCore() {}

protected:
#ifndef SWIG
    friend class Parametric;

    static void registerSubclass(PTR(EllipseCore) const & example);

    template <typename T>
    struct Registrar {
        Registrar() { registerSubclass(boost::make_shared<T>()); }
    };

    virtual PTR(EllipseCore) _clone() const = 0;

    virtual void readParameters(double const * iter) = 0;
    virtual void writeParameters(double * iter) const = 0;

    static void _assignQuadrupoleToAxes(
        double ixx, double iyy, double ixy,
        double & a, double & b, double & theta
    );

    static Jacobian _dAssignQuadrupoleToAxes(
        double ixx, double iyy, double ixy,
        double & a, double & b, double & theta
    );

    static void _assignAxesToQuadrupole(
        double a, double b, double theta,
        double & ixx, double & iyy, double & ixy
    );

    static Jacobian _dAssignAxesToQuadrupole(
        double a, double b, double theta,
        double & ixx, double & iyy, double & ixy
    );

    virtual void _assignToQuadrupole(double & ixx, double & iyy, double & ixy) const = 0;
    virtual void _assignFromQuadrupole(double ixx, double iyy, double ixy) = 0;

    virtual void _assignToAxes(double & a, double & b, double & theta) const = 0;
    virtual void _assignFromAxes(double a, double b, double theta) = 0;

    virtual Jacobian _dAssignToQuadrupole(double & ixx, double & iyy, double & ixy) const = 0;
    virtual Jacobian _dAssignFromQuadrupole(double ixx, double iyy, double ixy) = 0;

    virtual Jacobian _dAssignToAxes(double & a, double & b, double & theta) const = 0;
    virtual Jacobian _dAssignFromAxes(double a, double b, double theta) = 0;

#endif
};

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_EllipseCore_h_INCLUDED

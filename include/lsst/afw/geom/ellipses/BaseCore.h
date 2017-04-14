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

#ifndef LSST_AFW_GEOM_ELLIPSES_BaseCore_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_BaseCore_h_INCLUDED

/*
 *  Forward declarations, typedefs, and definitions for BaseCore.
 *
 *  Note: do not include directly; use the main ellipse header file.
 */

#include <memory>

#include "Eigen/Core"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/LinearTransform.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

class Parametric;

/**
 *  A base class for parametrizations of the "core" of an ellipse - the ellipticity and size.
 *
 *  A subclass of BaseCore provides a particular interpretation of the three pointing point values that
 *  define an ellipse's size and ellipticity (including position angle).  All core subclasses
 *  are implicitly convertible and can be assigned to from any other core.
 */
class BaseCore {
public:
    class Transformer;
    class GridTransform;
    class Convolution;
    template <typename Output> struct Converter;

    typedef std::shared_ptr<BaseCore> Ptr;
    typedef std::shared_ptr<BaseCore const> ConstPtr;

    typedef Eigen::Vector3d ParameterVector;  ///< Parameter vector type.
    typedef Eigen::Matrix3d Jacobian; ///< Parameter Jacobian matrix type.

    static Ptr make(std::string const & name);

    static Ptr make(std::string const & name, ParameterVector const & parameters);

    static Ptr make(std::string const & name, double v1, double v2, double v3);

    static Ptr make(std::string const & name, BaseCore const & other);

    static Ptr make(std::string const & name, Transformer const & other);

    static Ptr make(std::string const & name, Convolution const & other);

    /// Return a string that identifies this parametrization.
    virtual std::string getName() const = 0;

    /// Deep-copy the Core.
    Ptr clone() const { return _clone(); }

    /**
     *  @brief Put the parameters into a "standard form", and throw InvalidParameterError
     *         if they cannot be normalized.
     */
    virtual void normalize() = 0;

    /// Increase the major and minor radii of the ellipse core by the given buffer.
    void grow(double buffer);

    /// Scale the size of the ellipse core by the given factor.
    void scale(double factor);

    /// Return the area of the ellipse core.
    double getArea() const;

    /**
     *  Return the radius defined as the 4th root of the determinant of the quadrupole matrix.
     *
     *  The determinant radius is equal to the standard radius for a circle,
     *  and its square times pi is the area of the ellipse.
     */
    double getDeterminantRadius() const;

    /**
     *  Return the radius defined as the square root of one half the trace of the quadrupole matrix.
     *
     *  The trace radius is equal to the standard radius for a circle.
     */
    double getTraceRadius() const;

    /**
     *  @name Coordinate transforms
     *
     *  These member functions transform the ellipse by the given LinearTransform.
     *  The transform can be done in-place by calling inPlace() on the returned
     *  expression object, or returned as a new shared_ptr by calling copy().
     */
    //@{
    Transformer transform(LinearTransform const & transform);
    Transformer const transform(LinearTransform const & transform) const;
    //@}

    /**
     *  Return the transform that maps the ellipse to the unit circle.
     *
     *  The returned proxy object is implicitly convertible to LinearTransform
     *  and also supports differentiation.
     */
    GridTransform const getGridTransform() const;

    /**
     *  @name Convolve two bivariate Gaussians defined by their 1-sigma ellipses.
     */
    //@{
    Convolution convolve(BaseCore const & other);
    Convolution const convolve(BaseCore const & other) const;
    //@}

    /// Return the size of the bounding box for the ellipse core.
    Extent2D computeDimensions() const;

    virtual void readParameters(double const * iter) = 0;

    virtual void writeParameters(double * iter) const = 0;

    /// Return the core parameters as a vector.
    ParameterVector const getParameterVector() const;

    /// Set the core parameters from a vector.
    void setParameterVector(ParameterVector const & vector);

    /**
     *  Compare two ellipse cores for equality.
     *
     *  Ellipse cores are only equal if they have the same type.
     */
    bool operator==(BaseCore const & other) const;

    /**
     *  Compare two ellipse cores for inequality.
     *
     *  Ellipses are only equal if they have the same type.
     */
    bool operator!=(BaseCore const & other) const { return !operator==(other); }

    /**
     *  Set the parameters of this ellipse core from another.
     *
     *  This does not change the parametrization of the ellipse core.
     */
    BaseCore & operator=(BaseCore const & other);

    /// Assign other to this and return the derivative of the conversion, d(this)/d(other).
    Jacobian dAssign(BaseCore const & other);

    /**
     *  Convert this to the core type specified as a template parameter.
     */
    template <typename Output> Converter<Output> as() const;

    virtual ~BaseCore() {}

protected:
    friend class Parametric;

    static void registerSubclass(Ptr const & example);

    template <typename T>
    struct Registrar {
        Registrar() { registerSubclass(std::make_shared<T>()); }
    };

    virtual BaseCore::Ptr _clone() const = 0;

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

};

template <typename Output>
struct BaseCore::Converter {
    BaseCore const & input;

    explicit Converter(BaseCore const & input_) : input(input_) {}

    operator Output() const { return Output(input); }
    std::shared_ptr<Output> copy() const { return std::shared_ptr<Output>(new Output(input)); }
};

template <typename Output>
inline BaseCore::Converter<Output> BaseCore::as() const {
    return Converter<Output>(*this);
}

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_BaseCore_h_INCLUDED

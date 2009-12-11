// -*- lsst-c++ -*-
#ifndef LSST_AFW_GEOM_ELLIPSES_BASEELLIPSE_H
#define LSST_AFW_GEOM_ELLIPSES_BASEELLIPSE_H

/**
 *  \file
 *  \brief Forward declarations, typedefs, and definitions for BaseEllipse and BaseCore.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <memory>

#include "lsst/afw/geom/config.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/AffineTransform.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

namespace detail {
template <typename DerivedCore, typename DerivedEllipse> class CoreImpl;
template <typename DerivedCore, typename DerivedEllipse> class EllipseImpl;
} // namespace lsst::afw::geom::ellipses::detail

class BaseEllipse;
class BaseCore;
class Quadrupole;
class Axes;
class Distortion;
class LogShear;

/**
 *  This typedef is expected to be used more often than the true class name, 
 *  and exists so that the Ellipse typedef within core classes merely hides
 *  another typedef instead of a true class name.
 */
typedef BaseEllipse Ellipse;

/**
 *  This typedef is expected to be used more often than the true class name, 
 *  and exists so that the Core typedef within ellipse classes merely hides
 *  another typedef instead of a true class name.
 */
typedef BaseCore Core;

/**
 *  \brief A base class for ellipse geometries.
 *
 *  An ellipse is composed of its center coordinate and its Core - a parametrization of the
 *  ellipticity and size of the ellipse.  A subclass of BaseEllipse is defined for each concrete
 *  subclass of BaseCore, and is typedef'd as Core within the core class.  As a result, setting
 *  the core of an ellipse never changes the type of the contained core, it merely sets the parameters
 *  of that core by converting the parametrization.
 *
 *  \ingroup EllipseGroup
 */
class BaseEllipse {
protected:
    class Transformer; ///< Proxy return type for BaseEllipse::transform().
public:

    typedef boost::shared_ptr<BaseEllipse> Ptr;
    typedef boost::shared_ptr<BaseEllipse const> ConstPtr;

    typedef BoxD Envelope; ///< Bounding box type.
    typedef Eigen::Matrix<double,5,1> ParameterVector; ///< Parameter vector type.

    typedef BaseEllipse Ellipse;
    typedef BaseCore Core;

    class RadialFraction;

    enum Parameters { X=0, Y=1 }; ///< Definitions for elements of an ellipse vector.

    /// \brief Deep copy the BaseEllipse.
    std::auto_ptr<BaseEllipse> clone() const { return std::auto_ptr<BaseEllipse>(_clone()); }

    PointD const & getCenter() const { return _center; } ///< \brief Return the center point.
    PointD & getCenter() { return _center; }             ///< \brief Return the center point.
    void setCenter(PointD const & center) { _center = center; } ///< \brief Set the center point.

    BaseCore const & getCore() const; ///< \brief Return the core object.
    BaseCore & getCore();             ///< \brief Return the core object.
    void setCore(Core const & core); ///< \brief Set the core object.

    double & operator[](int n); ///< \brief Access the nth ellipse parameter.
    double const operator[](int n) const; ///< \brief Access the nth ellipse parameter.

    /// \brief Put the parameters in a standard form, and return false if they are invalid.
    bool normalize();

    /// \brief Increase the major and minor radii of the ellipse by the given buffer.
    void grow(double buffer);

    /// \brief Scale the size of the ellipse by the given factor.
    void scale(double factor);

    /// \brief Move the ellipse center by the given offset.
    void shift(ExtentD const & offset);

    /// \brief Transform the ellipse by the given AffineTransform.
    Transformer transform(AffineTransform const & transform);
    Transformer const transform(AffineTransform const & transform) const; ///< \copybrief transform

    /// \brief Return the AffineTransform that transforms the unit circle at the origin into this.
    AffineTransform getGenerator() const;

    /// \brief Return the bounding box of the ellipse.
    Envelope computeEnvelope() const;

    /**
     *  \brief Set the parameters of this ellipse from another.
     *
     *  This does not change the parametrization of the ellipse.
     */
    BaseEllipse & operator=(BaseEllipse const & other);

     /// \brief Return the ellipse parameters as a vector.
    ParameterVector const getVector() const;

    /// \brief Set the ellipse parameters from a vector.
    void setVector(ParameterVector const & vector);

    virtual ~BaseEllipse() {}

protected:

    virtual BaseEllipse * _clone() const = 0;

    explicit BaseEllipse(BaseCore const & core, PointD const & center);

    explicit BaseEllipse(BaseCore * core, PointD const & center);

    PointD _center;
    boost::scoped_ptr<BaseCore> _core;
};

/**
 *  \brief A base class for parametrizations of the "core" of an ellipse - the ellipticity and size.
 *
 *  A subclass of BaseCore provides a particular interpretation of the three pointing point values that
 *  define an ellipse's size and ellipticity (including position angle).  All core subclasses
 *  are implicitly convertible and can be assigned to from any other core.
 *
 *  \ingroup EllipseGroup
 */
class BaseCore {
protected:
    class Transformer;
public:

    typedef boost::shared_ptr<BaseCore> Ptr;
    typedef boost::shared_ptr<BaseCore const> ConstPtr;

    typedef Eigen::Vector3d ParameterVector;  ///< Parameter vector type.
    typedef Eigen::Matrix3d Jacobian; ///< Parameter Jacobian matrix type.

    typedef BaseEllipse Ellipse;
    typedef BaseCore Core;

    class RadialFraction;

    /// \brief Return a string that identifies this parametrization.
    virtual char const * getName() const = 0;

    /// \brief Deep-copy the Core.
    std::auto_ptr<BaseCore> clone() const { return std::auto_ptr<BaseCore>(_clone()); }

    /// \brief Construct an ellipse of the appropriate subclass from this and the given center.
    std::auto_ptr<BaseEllipse> makeEllipse(PointD const & center = PointD()) const {
        return std::auto_ptr<BaseEllipse>(_makeEllipse(center));
    }

    double & operator[](int n) { return _vector[n]; } ///< \brief Access the nth core parameter.
    double const operator[](int n) const { return _vector[n]; } ///< \brief Access the nth core parameter.

    /**
     *  \brief Put the parameters into a "standard form", if possible, and return
     *         false if they are entirely invalid.
     */
    virtual bool normalize() = 0;

    /// \brief Increase the major and minor radii of the ellipse core by the given buffer.
    virtual void grow(double buffer);

    /// \brief Scale the size of the ellipse core by the given factor.
    virtual void scale(double factor) = 0;

    /// \brief Transform the ellipse core by the given AffineTransform.
    Transformer transform(AffineTransform const & transform);
    Transformer const transform(AffineTransform const & transform) const; ///< \copybrief transform

    /// \brief Return the AffineTransform that transforms the unit circle into this.
    virtual AffineTransform getGenerator() const;

    /// \brief Return the size of the bounding box for the ellipse core.
    ExtentD computeDimensions() const;

    /// \brief Return the core parameters as a vector.
    ParameterVector const & getVector() const { return _vector; }

    /// \brief Set the core parameters from a vector.
    void setVector(ParameterVector const & vector) { _vector = vector; }

    /**
     *  \brief Set the parameters of this ellipse core from another.
     *
     *  This does not change the parametrization of the ellipse core.
     */
    virtual BaseCore & operator=(BaseCore const & other) = 0;
    
    /// \brief Assign other to this and return the derivative of the conversion, d(this)/d(other).
    virtual Jacobian dAssign(BaseCore const & other) = 0;

    virtual ~BaseCore() {}

protected:

    template <typename DerivedCore, typename DerivedEllipse> friend class detail::CoreImpl;

    virtual BaseCore * _clone() const = 0;
    virtual BaseEllipse * _makeEllipse(PointD const & center) const = 0;

    explicit BaseCore(ParameterVector const & vector) : _vector(vector) {}

    explicit BaseCore(double v1=0, double v2=0, double v3=0) : _vector(v1,v2,v3) {}

    virtual void _assignTo(Quadrupole & other) const = 0;
    virtual void _assignTo(Axes & other) const = 0;
    virtual void _assignTo(Distortion & other) const = 0;
    virtual void _assignTo(LogShear & other) const = 0;

    virtual Jacobian _dAssignTo(Quadrupole & other) const = 0;
    virtual Jacobian _dAssignTo(Axes & other) const = 0;
    virtual Jacobian _dAssignTo(Distortion & other) const = 0;
    virtual Jacobian _dAssignTo(LogShear & other) const = 0;

    ParameterVector _vector;

    friend class Quadrupole;
    friend class Axes;
    friend class Distortion;
    friend class LogShear;
};

inline BaseCore const & BaseEllipse::getCore() const { return *_core; }
inline BaseCore & BaseEllipse::getCore() { return *_core; }
inline void BaseEllipse::setCore(BaseCore const & core) { *_core = core; }
inline bool BaseEllipse::normalize() { return _core->normalize(); }
inline void BaseEllipse::grow(double buffer) { getCore().grow(buffer); }
inline void BaseEllipse::scale(double factor) { getCore().scale(factor); }
inline void BaseEllipse::shift(ExtentD const & offset) { _center += offset; }

inline double & BaseEllipse::operator[](int i) { return (i<2) ? _center[i] : (*_core)[i-2]; }
inline double const BaseEllipse::operator[](int i) const { return (i<2) ? _center[i] : (*_core)[i-2]; }

inline BaseEllipse::BaseEllipse(Core const & core, PointD const & center) : 
    _center(center), _core(core.clone().release()) {}

inline BaseEllipse::BaseEllipse(Core * core, PointD const & center) :
    _center(center), _core(core) {}

} // namespace lsst::afw::geom::ellipses
}}} // namespace lsst::afw::geom
#endif // !LSST_AFW_GEOM_ELLIPSES_BASEELLIPSE_H

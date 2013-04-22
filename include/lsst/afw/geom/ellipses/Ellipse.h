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

#ifndef LSST_AFW_GEOM_ELLIPSES_Ellipse_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_Ellipse_h_INCLUDED

/**
 *  @file
 *  @brief Forward declarations, typedefs, and definitions for Ellipse.
 *
 *  @note Do not include directly; use the main ellipse header file.
 */


#include "lsst/afw/geom/ellipses/EllipseCore.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/AffineTransform.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

/**
 *  @brief An ellipse defined by an arbitrary EllipseCore and a center point.
 *
 *  An ellipse is composed of its center coordinate and its Core - a parametrization of the
 *  ellipticity and size of the ellipse.  Setting the core of an ellipse never changes the 
 *  type of the contained core, it merely sets the parameters of that core by converting
 *  the parameters.
 */
class Ellipse {
public:
#ifndef SWIG
    class Transformer; ///< Proxy return type for Ellipse::transform().
    class GridTransform; ///< Proxy return type for Ellipse::getGridTransform().
    class Convolution; ///< Proxy return type for Ellipse::convolve().
#endif

    typedef Eigen::Matrix<double,5,1> ParameterVector; ///< Parameter vector type.

    typedef boost::shared_ptr<Ellipse> Ptr;
    typedef boost::shared_ptr<Ellipse const> ConstPtr;

    enum ParameterEnum { X=3, Y=4 }; ///< Definitions for elements of an ellipse vector.

    /// @brief Return the center point.
    Point2D const & getCenter() const { return _center; }

    /// @brief Return the center point.
    Point2D & getCenter() { return _center; }

    /// @brief Set the center point.
    void setCenter(Point2D const & center) { _center = center; }

    /// @brief Return the ellipse core.
    EllipseCore const & getCore() const { return *_core; }

    /// @brief Return the ellipse core.
    EllipseCore & getCore() { return *_core; }

    /// @brief Return the ellipse core.
    EllipseCore::ConstPtr getCorePtr() const { return _core; }

    /// @brief Return the ellipse core.
    EllipseCore::Ptr getCorePtr() { return _core; }

    /// @brief Set the ellipse core; the type of the core is not changed.
    void setCore(EllipseCore const & core) { *_core = core; }

    /// @brief Put the parameters in a standard form.
    void normalize() { _core->normalize(); }

    /// @brief Increase the major and minor radii of the ellipse by the given buffer.
    void grow(double buffer) { _core->grow(buffer); }

    /// @brief Scale the size of the ellipse by the given factor.
    void scale(double factor) { _core->scale(factor); }

    /// @brief Move the ellipse center by the given offset.
    void shift(Extent2D const & offset) { _center += offset; }

     /// @brief Return the ellipse parameters as a vector.
    ParameterVector const getParameterVector() const;

    /// @brief Set the ellipse parameters from a vector.
    void setParameterVector(ParameterVector const & vector);

    void readParameters(double const * iter);

    void writeParameters(double * iter) const;

    /**
     *  @name Coordinate transforms
     *  
     *  These member functions transform the ellipse by the given AffineTransform.
     *  The transform can be done in-place by calling inPlace() on the returned
     *  expression object, or returned as a new shared_ptr by calling copy().
     */
    //@{
    Transformer transform(AffineTransform const & transform);
    Transformer const transform(AffineTransform const & transform) const;
    //@}

    /**
     *  @name Convolve two bivariate Gaussians defined by their 1-sigma ellipses.
     */
    //@{
    Convolution convolve(Ellipse const & other);
    Convolution const convolve(Ellipse const & other) const;
    //@}

    /**
     *  @brief Return the transform that maps the ellipse to the unit circle.
     *
     *  The returned proxy object is implicitly convertible to AffineTransform
     *  and also supports differentiation.
     */
    GridTransform const getGridTransform() const;

    /// @brief Return the bounding box of the ellipse.
    Box2D computeBBox() const;

    /**
     *  @brief Set the parameters of this ellipse from another.
     *
     *  This does not change the parametrization of the ellipse.
     */
    Ellipse & operator=(Ellipse const & other);

    /**
     *  @brief Compare two ellipses for equality.
     *
     *  Ellipses are only equal if they have the same Core types.
     */
    bool operator==(Ellipse const & other) const {
        return getCenter() == other.getCenter() && getCore() == other.getCore();
    }

    /**
     *  @brief Compare two ellipses for inequality.
     *
     *  Ellipses are only equal if they have the same Core types.
     */
    bool operator!=(Ellipse const & other) const { return !operator==(other); }

    virtual ~Ellipse() {}

    explicit Ellipse(EllipseCore const & core, Point2D const & center = Point2D()) :
        _core(core.clone()), _center(center) {}

    explicit Ellipse(EllipseCore::ConstPtr const & core, Point2D const & center = Point2D()) :
        _core(core->clone()), _center(center) {}

#ifndef SWIG
    Ellipse(Transformer const & other);
    Ellipse(Convolution const & other);
#endif

    Ellipse(Ellipse const & other) :
        _core(other.getCore().clone()), _center(other.getCenter()) {}

private:
    EllipseCore::Ptr _core;
    Point2D _center;
};

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_Ellipse_h_INCLUDED

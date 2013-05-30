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

#ifndef LSST_AFW_GEOM_ELLIPSES_Ellipse_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_Ellipse_h_INCLUDED

#include "lsst/base.h"
#include "lsst/afw/geom/ellipses/EllipseCore.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/AffineTransform.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

/**
 *  @brief An ellipse defined by an arbitrary EllipseCore and a center point.
 *
 *  An ellipse is composed of its center coordinate and its ElliseCore - a parametrization of the
 *  ellipticity and size of the ellipse.  Setting the EllipseCore never changes the
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

    enum ParameterEnum { X=3, Y=4 }; ///< Definitions for elements of an ellipse vector.

    /// Return the center point
    Point2D const & getCenter() const { return _center; }

    /// Return the center point
    Point2D & getCenter() { return _center; }

    /// Set the center point
    void setCenter(Point2D const & center) { _center = center; }

    /// Return the EllipseCore
    EllipseCore const & getCore() const { return *_core; }

    /// Return the ellipse core
    EllipseCore & getCore() { return *_core; }

    /// Return the ellipse core
    PTR(EllipseCore const) getCorePtr() const { return _core; }

    /// Return the ellipse core
    PTR(EllipseCore) getCorePtr() { return _core; }

    /// Set the ellipse core; the type of the core is not changed.
    void setCore(EllipseCore const & core) { *_core = core; }

    /// Put the parameters in a standard form.
    void normalize() { _core->normalize(); }

    /// Increase the major and minor radii of the ellipse by the given buffer.
    void grow(double buffer) { _core->grow(buffer); }

    /// Scale the size of the ellipse by the given factor.
    void scale(double factor) { _core->scale(factor); }

    /// Move the ellipse center by the given offset.
    void shift(Extent2D const & offset) { _center += offset; }

    /// Return the ellipse parameters as a vector.
    ParameterVector const getParameterVector() const;

    /// Set the ellipse parameters from a vector.
    void setParameterVector(ParameterVector const & vector);

    //@{
    /**
     *  @brief Coordinate transforms
     *
     *  These member functions transform the ellipse by the given AffineTransform.
     *  The transform can be done in-place by calling inPlace() on the returned
     *  expression object, or returned as a new shared_ptr by calling copy().
     */
    Transformer transform(AffineTransform const & transform);
    Transformer const transform(AffineTransform const & transform) const;
    //@}

    //@{
    /**
     *  @brief Convolve two bivariate Gaussians defined by their 1-sigma ellipses.
     */
    Convolution convolve(Ellipse const & other);
    Convolution const convolve(Ellipse const & other) const;
    //@}

    /// @copydoc EllipseCore::getGridTransform
    GridTransform const getGridTransform() const;

    /// Return the bounding box of the ellipse.
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
     *  Ellipses are only equal if they have the same EllipseCore types.
     */
    bool operator==(Ellipse const & other) const {
        return getCenter() == other.getCenter() && getCore() == other.getCore();
    }

    /**
     *  @brief Compare two ellipses for inequality.
     *
     *  Ellipses are only equal if they have the same EllipseCore types.
     */
    bool operator!=(Ellipse const & other) const { return !operator==(other); }

    friend std::ostream & operator<<(std::ostream & os, Ellipse const & ellipse) {
        return os << "(" << ellipse.getCore() << ", " << ellipse.getCenter() << ")";
    }

    /// Construct an Ellipse from an EllipseCore and center
    explicit Ellipse(EllipseCore const & core, Point2D const & center = Point2D()) :
        _core(core.clone()), _center(center) {}

#ifndef SWIG

    Ellipse(Transformer const & other);
    Ellipse(Convolution const & other);
#endif

    /// Deep-copy an Ellipse (clones the EllipseCore)
    Ellipse(Ellipse const & other) :
        _core(other.getCore().clone()), _center(other.getCenter()) {}

private:
    PTR(EllipseCore) _core;
    Point2D _center;
};

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_Ellipse_h_INCLUDED

// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
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

#ifndef LSST_AFW_GEOM_BOX_H
#define LSST_AFW_GEOM_BOX_H

#include <vector>
#include "boost/format.hpp"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"
#include "ndarray.h"

namespace lsst { namespace afw { namespace geom {

class Box2D;

/**
 *  An integer coordinate rectangle.
 *
 *  Box2I is an inclusive box that represents a rectangular region of pixels.  A box
 *  never has negative dimensions; the empty box is defined to have zero-size dimensions,
 *  and is treated as though it does not have a well-defined position (regardless of the
 *  return value of getMin() or getMax() for an empty box).
 *
 *  @internal
 *
 *  Box2I internally stores its minimum point and dimensions, because we expect
 *  these will be the most commonly accessed quantities.
 *
 *  Box2I sets the minimum point to the origin for an empty box, and returns -1 for both
 *  elements of the maximum point in that case.
 */
class Box2I {
public:

    typedef Point2I Point;
    typedef Extent2I Extent;

    enum EdgeHandlingEnum { EXPAND, SHRINK };

    /// Construct an empty box.
    Box2I() : _minimum(0), _dimensions(0) {}

    /**
     *  Construct a box from its minimum and maximum points.
     *
     *  @param[in] minimum   Minimum (lower left) coordinate (inclusive).
     *  @param[in] maximum   Maximum (upper right) coordinate (inclusive).
     *  @param[in] invert    If true (default), swap the minimum and maximum coordinates if
     *                       minimum > maximum instead of creating an empty box.
     */
    Box2I(Point2I const & minimum, Point2I const & maximum, bool invert=true);

    /**
     *  Construct a box from its minimum point and dimensions.
     *
     *  @param[in] minimum    Minimum (lower left) coordinate.
     *  @param[in] dimensions Box dimensions.  If either dimension coordinate is 0, the box will be empty.
     *  @param[in] invert     If true (default), invert any negative dimensions instead of creating
     *                        an empty box.
     */
    Box2I(Point2I const & minimum, Extent2I const & dimensions, bool invert=true);


    /**
     *  Construct an integer box from a floating-point box.
     *
     *  Floating-point to integer box conversion is based on the concept that a pixel
     *  is not an infinitesimal point but rather a square of unit size centered on
     *  integer-valued coordinates.  Converting a floating-point box to an integer box
     *  thus requires a choice on how to handle pixels which are only partially contained
     *  by the input floating-point box.
     *
     *  @param[in] other          A floating-point box to convert.
     *  @param[in] edgeHandling   If EXPAND, the integer box will contain any pixels that
     *                            overlap the floating-point box.  If SHRINK, the integer
     *                            box will contain only pixels completely contained by
     *                            the floating-point box.
     */
    explicit Box2I(Box2D const & other, EdgeHandlingEnum edgeHandling=EXPAND);

    /// Standard copy constructor.
    Box2I(Box2I const & other) : _minimum(other._minimum), _dimensions(other._dimensions) {}

    void swap(Box2I & other) {
        _minimum.swap(other._minimum);
        _dimensions.swap(other._dimensions);
    }

    /// Standard assignment operator.
    Box2I & operator=(Box2I const & other) {
        _minimum = other._minimum;
        _dimensions = other._dimensions;
        return *this;
    }

    /**
     *  @name Min/Max Accessors
     *
     *  Return the minimum and maximum coordinates of the box (inclusive).
     */
    //@{
    Point2I const getMin() const { return _minimum; }
    int getMinX() const { return _minimum.getX(); }
    int getMinY() const { return _minimum.getY(); }

    Point2I const getMax() const { return _minimum + _dimensions - Extent2I(1); }
    int getMaxX() const { return _minimum.getX() + _dimensions.getX() - 1; }
    int getMaxY() const { return _minimum.getY() + _dimensions.getY() - 1; }
    //@}

    /**
     *  @name Begin/End Accessors
     *
     *  Return STL-style begin (inclusive) and end (exclusive) coordinates for the box.
     */
    //@{
    Point2I const getBegin() const { return _minimum; }
    int getBeginX() const { return _minimum.getX(); }
    int getBeginY() const { return _minimum.getY(); }

    Point2I const getEnd() const { return _minimum + _dimensions; }
    int getEndX() const { return _minimum.getX() + _dimensions.getX(); }
    int getEndY() const { return _minimum.getY() + _dimensions.getY(); }
    //@}

    /**
     *  @name Size Accessors
     *
     *  Return the size of the box in pixels.
     */
    //@{
    Extent2I const getDimensions() const { return _dimensions; }
    int getWidth() const { return _dimensions.getX(); }
    int getHeight() const { return _dimensions.getY(); }
    int getArea() const { return getWidth() * getHeight(); }
    //@}

    /// Return slices to extract the box's region from an ndarray::Array.
    ndarray::View<
        boost::fusion::vector2< ndarray::index::Range, ndarray::index::Range >
        >
    getSlices() const;

    /// Return true if the box contains no points.
    bool isEmpty() const {
        return _dimensions.getX() == 0 && _dimensions.getY() == 0;
    }

    /// Return true if the box contains the point.
    bool contains(Point2I const & point) const;

    /**
     *  Return true if all points contained by other are also contained by this.
     *
     *  An empty box is contained by every other box, including other empty boxes.
     */
    bool contains(Box2I const & other) const;

    /**
     *  Return true if any points in other are also in this.
     *
     *  Any overlap operation involving an empty box returns false.
     */
    bool overlaps(Box2I const & other) const;

    /**
     *  Increase the size of the box by the given buffer amount in all directions.
     *
     *  If a negative buffer is passed and the final size of the box is less than or
     *  equal to zero, the box will be made empty.
     */
    void grow(int buffer) { grow(Extent2I(buffer)); }

    /**
     *  Increase the size of the box by the given buffer amount in each direction.
     *
     *  If a negative buffer is passed and the final size of the box is less than or
     *  equal to zero, the box will be made empty.
     */
    void grow(Extent2I const & buffer);

    /// Shift the position of the box by the given offset.
    void shift(Extent2I const & offset);

    /// Flip a bounding box about the y-axis given a parent box of extent (xExtent).
    void flipLR(int xExtent);

    /// Flip a bounding box about the x-axis given a parent box of extent (yExtent).
    void flipTB(int yExtent);

    /// Expand this to ensure that this->contains(point).
    void include(Point2I const & point);

    /// Expand this to ensure that this->contains(other).
    void include(Box2I const & other);

    /// Shrink this to ensure that other.contains(*this).
    void clip(Box2I const & other);

    /**
     *  Compare two boxes for equality.
     *
     *  All empty boxes are equal.
     */
    bool operator==(Box2I const & other) const;

    /**
     *  Compare two boxes for equality.
     *
     *  All empty boxes are equal.
     */
    bool operator!=(Box2I const & other) const;

    /**
     * Get the corner points
     *
     * The order is counterclockise, starting from the lower left corner, i.e.:
     *   (minX, minY), (maxX, maxY), (maxX, maxX), (minX, maxY)
     */
    std::vector<Point2I> getCorners() const;

    std::string toString() const {
        return (boost::format("Box2I(%s,%s)") % _minimum.toString() % _dimensions.toString()).str();
    }

private:
    Point2I _minimum;
    Extent2I _dimensions;
};

/**
 *  A floating-point coordinate rectangle geometry.
 *
 *  Box2D is a half-open (minimum is inclusive, maximum is exclusive) box. A box
 *  never has negative dimensions; the empty box is defined to zero-size dimensions
 *  and its minimum and maximum values set to NaN.  Only the empty box may have
 *  zero-size dimensions.
 *
 *  @internal
 *
 *  Box2D internally stores its minimum point and maximum point, instead of
 *  minimum point and dimensions, to ensure roundoff error does not affect
 *  whether points are contained by the box.
 *
 *  Despite some recommendations to the contrary, Box2D sets the minimum and maximum
 *  points to NaN for an empty box.  In almost every case, special checks for
 *  emptiness would have been necessary anyhow, so there was little to gain in
 *  using the minimum > maximum condition to denote an empty box, as was used in Box2I.
 */
class Box2D {
public:

    typedef Point2D Point;
    typedef Extent2D Extent;

    /**
     *  Value the maximum coordinate is multiplied by to increase it by the smallest
     *  possible amount.
     */
    static double const EPSILON;

    /// Value used to specify undefined coordinate values.
    static double const INVALID;

    /// Construct an empty box.
    Box2D();

    /**
     *  Construct a box from its minimum and maximum points.
     *
     *  If any(minimum == maximum), the box will always be empty (even if invert==true).
     *
     *  @param[in] minimum   Minimum (lower left) coordinate (inclusive).
     *  @param[in] maximum   Maximum (upper right) coordinate (exclusive).
     *  @param[in] invert    If true (default), swap the minimum and maximum coordinates if
     *                       minimum > maximum instead of creating an empty box.
     */
    Box2D(Point2D const & minimum, Point2D const & maximum, bool invert=true);

    /**
     *  Construct a box from its minimum point and dimensions.
     *
     *  @param[in] minimum    Minimum (lower left) coordinate (inclusive).
     *  @param[in] dimensions Box dimensions.  If either dimension coordinate is 0, the box will be empty.
     *  @param[in] invert     If true (default), invert any negative dimensions instead of creating
     *                        an empty box.
     */
    Box2D(Point2D const & minimum, Extent2D const & dimensions, bool invert=true);

    /**
     *  Construct a floating-point box from an integer box.
     *
     *  Integer to floating-point box conversion is based on the concept that a pixel
     *  is not an infinitesimal point but rather a square of unit size centered on
     *  integer-valued coordinates.  While the output floating-point box thus has
     *  the same dimensions as the input integer box, its minimum/maximum coordinates
     *  are 0.5 smaller/greater.
     */
    explicit Box2D(Box2I const & other);

    /// Standard copy constructor.
    Box2D(Box2D const & other) : _minimum(other._minimum), _maximum(other._maximum) {}

    void swap(Box2D & other) {
        _minimum.swap(other._minimum);
        _maximum.swap(other._maximum);
    }

    /// Standard assignment operator.
    Box2D & operator=(Box2D const & other) {
        _minimum = other._minimum;
        _maximum = other._maximum;
        return *this;
    }

    /**
     *  @name Min/Max Accessors
     *
     *  Return the minimum (inclusive) and maximum (exclusive) coordinates of the box.
     */
    //@{
    Point2D const getMin() const { return _minimum; }
    double getMinX() const { return _minimum.getX(); }
    double getMinY() const { return _minimum.getY(); }

    Point2D const getMax() const { return _maximum; }
    double getMaxX() const { return _maximum.getX(); }
    double getMaxY() const { return _maximum.getY(); }
    //@}

    /**
     *  @name Size Accessors
     *
     *  Return the size of the box.
     */
    //@{
    Extent2D const getDimensions() const { return isEmpty() ? Extent2D(0.0) : _maximum - _minimum; }
    double getWidth() const { return isEmpty() ? 0 : _maximum.getX() - _minimum.getX(); }
    double getHeight() const { return isEmpty() ? 0 : _maximum.getY() - _minimum.getY(); }
    double getArea() const {
        Extent2D dim(getDimensions());
        return dim.getX() * dim.getY();
    }
    //@}

    /**
     *  @name Center Accessors
     *
     *  Return the center coordinate of the box.
     */
    //@{
    Point2D const getCenter() const { return Point2D((_minimum.asEigen() + _maximum.asEigen())*0.5); }
    double getCenterX() const { return (_minimum.getX() + _maximum.getX())*0.5; }
    double getCenterY() const { return (_minimum.getY() + _maximum.getY())*0.5; }
    //@}

    /// Return true if the box contains no points.
    bool isEmpty() const { return _minimum.getX() != _minimum.getX(); }

    /// Return true if the box contains the point.
    bool contains(Point2D const & point) const;

    /**
     *  Return true if all points contained by other are also contained by this.
     *
     *  An empty box is contained by every other box, including other empty boxes.
     */
    bool contains(Box2D const & other) const;

    /**
     *  Return true if any points in other are also in this.
     *
     *  Any overlap operation involving an empty box returns false.
     */
    bool overlaps(Box2D const & other) const;

    /**
     *  Increase the size of the box by the given buffer amount in all directions.
     *
     *  If a negative buffer is passed and the final size of the box is less than or
     *  equal to zero, the box will be made empty.
     */
    void grow(double buffer) { grow(Extent2D(buffer)); }

    /**
     *  Increase the size of the box by the given buffer amount in each direction.
     *
     *  If a negative buffer is passed and the final size of the box is less than or
     *  equal to zero, the box will be made empty.
     */
    void grow(Extent2D const & buffer);

    /// Shift the position of the box by the given offset.
    void shift(Extent2D const & offset);

    /// Flip a bounding box about the y-axis given a parent box of extent (xExtent).
    void flipLR(float xExtent);

    /// Flip a bounding box about the x-axis given a parent box of extent (yExtent).
    void flipTB(float yExtent);

    /**
     *  Expand this to ensure that this->contains(point).
     *
     *  If the point sets a new maximum value for the box, the maximum coordinate will
     *  be adjusted to ensure the point is actually contained
     *  by the box instead of sitting on its exclusive upper edge.
     */
    void include(Point2D const & point);

    /// Expand this to ensure that this->contains(other).
    void include(Box2D const & other);

    /// Shrink this to ensure that other.contains(*this).
    void clip(Box2D const & other);

    /**
     *  Compare two boxes for equality.
     *
     *  All empty boxes are equal.
     */
    bool operator==(Box2D const & other) const;

    /**
     *  Compare two boxes for equality.
     *
     *  All empty boxes are equal.
     */
    bool operator!=(Box2D const & other) const;

    /**
     * Get the corner points
     *
     * The order is counterclockise, starting from the lower left corner, i.e.:
     *   (minX, minY), (maxX, maxY), (maxX, maxX), (minX, maxY)
     */
    std::vector<Point2D> getCorners() const;

    std::string toString() const {
        return (boost::format("Box2D(%s,%s)") % _minimum.toString() % _maximum.toString()).str();
    }

private:
    void _tweakMax(int n) {
        if (_maximum[n] < 0.0) {
            _maximum[n] *= (1.0 - EPSILON);
        } else if (_maximum[n] > 0.0) {
            _maximum[n] *= (1.0 + EPSILON);
        } else {
            _maximum[n] = EPSILON;
        }
    }
    Point2D _minimum;
    Point2D _maximum;
};

typedef Box2D BoxD;
typedef Box2I BoxI;

std::ostream & operator<<(std::ostream & os, Box2I const & box);

std::ostream & operator<<(std::ostream & os, Box2D const & box);

}}}

#endif

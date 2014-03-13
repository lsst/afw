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
 
/**
 * \file
 * \brief A coordinate class intended to represent absolute positions.
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
 *  \brief An integer coordinate rectangle.
 *
 *  Box2I is an inclusive box that represents a rectangular region of pixels.  A box 
 *  never has negative dimensions; the empty box is defined to have zero-size dimensions,
 *  and is treated as though it does not have a well-defined position (regardless of the
 *  return value of getMin() or getMax() for an empty box).
 *
 *  \internal
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

    /// \brief Construct an empty box.
    Box2I() : _minimum(0), _dimensions(0) {}

    Box2I(Point2I const & minimum, Point2I const & maximum, bool invert=true);
    Box2I(Point2I const & minimum, Extent2I const & dimensions, bool invert=true);

    explicit Box2I(Box2D const & other, EdgeHandlingEnum edgeHandling=EXPAND);

    /// \brief Standard copy constructor.
    Box2I(Box2I const & other) : _minimum(other._minimum), _dimensions(other._dimensions) {}

    void swap(Box2I & other) {
        _minimum.swap(other._minimum);
        _dimensions.swap(other._dimensions);
    }
    /// \brief Standard assignment operator.
    Box2I & operator=(Box2I const & other) {
        _minimum = other._minimum;
        _dimensions = other._dimensions;
        return *this;
    }

    /**
     *  @name Min/Max Accessors
     *
     *  @brief Return the minimum and maximum coordinates of the box (inclusive).
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
     *  \brief Return STL-style begin (inclusive) and end (exclusive) coordinates for the box.
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
     *  \brief Return the size of the box in pixels.
     */
    //@{
    Extent2I const getDimensions() const { return _dimensions; }
    int getWidth() const { return _dimensions.getX(); }
    int getHeight() const { return _dimensions.getY(); }
    int getArea() const { return getWidth() * getHeight(); }
    //@}

    ndarray::View< 
        boost::fusion::vector2< ndarray::index::Range, ndarray::index::Range > 
        >
    getSlices() const;

    /// \brief Return true if the box contains no points.
    bool isEmpty() const { 
        return _dimensions.getX() == 0 && _dimensions.getY() == 0; 
    }

    bool contains(Point2I const & point) const;
    bool contains(Box2I const & other) const;
    bool overlaps(Box2I const & other) const;

    void grow(int buffer) { grow(Extent2I(buffer)); }
    void grow(Extent2I const & buffer);
    void shift(Extent2I const & offset);
    void flipLR(int xextent);
    void flipTB(int yextent);
    void include(Point2I const & point);
    void include(Box2I const & other);
    void clip(Box2I const & other);

    bool operator==(Box2I const & other) const;
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
 *  \brief A floating-point coordinate rectangle geometry.
 *
 *  Box2D is a half-open (minimum is inclusive, maximum is exclusive) box. A box 
 *  never has negative dimensions; the empty box is defined to zero-size dimensions
 *  and its minimum and maximum values set to NaN.  Only the empty box may have
 *  zero-size dimensions.
 *
 *  \internal
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

    /// \brief Construct an empty box.
    Box2D();

    Box2D(Point2D const & minimum, Point2D const & maximum, bool invert=true);    
    Box2D(Point2D const & minimum, Extent2D const & dimensions, bool invert=true);
    
    explicit Box2D(Box2I const & other);
    
    /// \brief Standard copy constructor.
    Box2D(Box2D const & other) : _minimum(other._minimum), _maximum(other._maximum) {}
    
    void swap(Box2D & other) {
        _minimum.swap(other._minimum);
        _maximum.swap(other._maximum);
    }
    /// \brief Standard assignment operator.
    Box2D & operator=(Box2D const & other) {
        _minimum = other._minimum;
        _maximum = other._maximum;
        return *this;
    }

    /**
     *  @name Min/Max Accessors
     *
     *  @brief Return the minimum (inclusive) and maximum (exclusive) coordinates of the box.
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
     *  \brief Return the size of the box.
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
     *  \brief Return the center coordinate of the box.
     */
    //@{
    Point2D const getCenter() const { return Point2D((_minimum.asEigen() + _maximum.asEigen())*0.5); }
    double getCenterX() const { return (_minimum.getX() + _maximum.getX())*0.5; }
    double getCenterY() const { return (_minimum.getY() + _maximum.getY())*0.5; }
    //@}

    /// \brief Return true if the box contains no points.
    bool isEmpty() const { return _minimum.getX() != _minimum.getX(); }

    bool contains(Point2D const & point) const;
    bool contains(Box2D const & other) const;
    bool overlaps(Box2D const & other) const;

    void grow(double buffer) { grow(Extent2D(buffer)); }
    void grow(Extent2D const & buffer);
    void shift(Extent2D const & offset);
    void flipLR(float xextent);
    void flipTB(float yextent);
    void include(Point2D const & point);
    void include(Box2D const & other);
    void clip(Box2D const & other);

    bool operator==(Box2D const & other) const;
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

// -*- lsst-c++ -*-
/**
 * \file
 * \brief An axis-aligned rectangle class.
 */
#ifndef LSST_AFW_GEOM_BOX_H
#define LSST_AFW_GEOM_BOX_H

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

namespace lsst { namespace afw { namespace geom {

/**
 *  \brief An axis-alligned rectangle class, intended mostly to represent the spatial bounds of
 *  other objects.
 *
 *  Dealing with boxes that may be empty is a really unpleasant business.  In this design, operations
 *  that individually get/set the minimum and/or maximum points will have undefined behavior on empty
 *  boxes.
 */
template<typename T, int N>
class Box {
public:

    /**
     *  \brief Construct a Box from a pair of points.
     */
    Box(
        Point<T,N> const & min, ///< Minimum point.
        Point<T,N> const & max, ///< Maximum point.
        bool ordered=true ///< If false, swap min and max as necessary to ensure the box is not empty.
    );

    /**
     *  \brief Construct a Box from the minimum point and the dimensions.
     */
    Box(
        Point<T,N> const & min, ///< Minimum point.
        Extent<T,N> const & dimensions, ///< Dimensions.  If negative, the box will be empty.
    );

    /**
     *  \brief Conversion constructor.
     *
     *  Converting from integer to floating-point creates a floating point box with minimum values
     *  0.5 less than the integer minimum and maximum values 0.5 greater than the integer maximum.
     *
     *  Converting from floating-point to integer uses floor and ceil such that any pixel touched
     *  by the floating-point box is included in the integer box.
     *
     *  Note that in both of these conversions, a bounding box can remain the same size or grow,
     *  but will never shrink.
     */
    template <typename U>
    explicit Box(Box<U,N> const & other);

    /**
     *  \brief Return the minimum point (formerly getLLC() for 'lower left corner').
     *
     *  The returned value is undefined if getEmpty()==true.
     */
    Point<T,N> getMin() const;
    T getMinX() const;
    T getMinY() const;

    /**
     *  \brief Set the minimum point.
     *
     *  A Box will become empty if the given point is greater than or equal to the maximum point,
     *  but setting this on an empty Box results in undefined behavior.
     */
    void setMin(Point<T,N> const & point);
    void setMinX(T x);
    void setMinY(T y);

    /**
     *  \brief Return the maximum point (formerly getURC() for 'upper right corner').
     *
     *  The returned value is undefined if getEmpty()==true.
     */
    Point<T,N> getMax() const;
    T getMaxX() const;
    T getMaxY() const;

    /**
     *  \brief Set the minimum point.
     *
     *  A Box will become empty if the given point is less than or equal to the maximum point,
     *  but setting this on an empty Box results in undefined behavior.
     */
    void setMax(Point<T,N> const & point);
    void setMaxX(T x);
    void setMaxY(T y);

    /**
     *  \brief Return the dimensions of the box.
     *
     *  The returned value will always contain zeros for an empty Box, and will never be negative.
     *
     *  For a floating-point box:
     *  \code
     *  getDimensions() == isEmpty() ? 0 : getMax() - getMin();
     *  \endcode
     *  while for an integer box:
     *  \code
     *  getDimensions() == isEmpty() ? 0 : Extent(1) + getMax() - getMin();
     *  \endcode
     */
    Extent<T,N> getDimensions() const;
    T getWidth() const;
    T getHeight() const;

    /// \brief Return getWidth() * getHeight().
    T getArea() const;

    /// \brief Return true if the Box is empty.
    bool isEmpty() const;

    /// \brief Return true if the Box contains the given point.
    bool contains(Point<T,N> const & point) const;

    /// \brief Return true if the Box completely contains another Box.
    bool contains(Box const & other) const;

    /// \brief Return true if the Box overlaps another Box.
    bool overlaps(Box const & other) const;

    /// \brief Return a new Box that contains all points in both this and other.
    Box makeIntersection(Box const & other);

    /// \brief Shrink this to include only points in both this and other (afw::image::BBox uses "clip" here).
    Box & setIntersection(Box const & other);

    /// \brief Return a new Box that contains all points in either this and other.
    Box makeHull(Box const & other);

    /// \brief Expand this to include all points in other.
    Box & setHull(Box const & other);

    /**
     *  \brief Return a new Box that contains all points in this as well as the given point.
     *
     *  Could also name this "expand" or "grow", but I like having a noun so we can provide
     *  both "make" and "set".
     */
    Box makeHull(Point<T,N> const & point);

    /**
     *  \brief Expand this to include the given point.
     *
     *  Could also name this "expand" or "grow", but I like having a noun so we can provide
     *  both "make" and "set" (afw::image::BBox uses "grow" here).
     */
    Box & setHull(Point<T,N> const & point);

    /// \brief Compare two boxes for equality.
    bool operator==(Box const & other) const;
    bool operator!=(Box const & other) const;

    /// \brief Increase the size of the box uniformly by the given buffer amount in all dimensions.
    void expand(T buffer);

    /// \brief Shift the box by the given offset.
    void shift(Extent<T,N> const & offset);

};

typedef Box<int,2> Box2I;
typedef Box<int,3> Box3I;
typedef Box<double,2> Box2D;
typedef Box<double,3> Box3D;

}}}

#endif

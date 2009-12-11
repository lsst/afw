// -*- lsst-c++ -*-
/**
 * \file
 * \brief A coordinate class intended to represent absolute positions.
 */
#ifndef LSST_AFW_GEOM_POINT_H
#define LSST_AFW_GEOM_POINT_H

#include "lsst/afw/geom/CoordinateBase.h"

namespace lsst { namespace afw { namespace geom {

template <typename T, int N> class Extent;

/**
 *  \brief A coordinate class intended to represent absolute positions.
 *
 *  Much of the functionality of Point is provided by its CRTP base class, CoordinateBase.
 */
template<typename T, int N>
class Point : public CoordinateBase<Point<T,N>,T,N> {
public:

    /**
     *  \brief Standard coordinate constructors
     *
     *  See the CoordinateBase constructors for more discussion.
     */
    //@{
    explicit Point(T val=static_cast<T>(0));

    template <typename Vector>
    explicit Point(Eigen::MatrixBase<Vector> const & vector);
    //@}

    /// \brief Explicit constructor from Extent.
    explicit Point(Extent<T,N> const & other);

    /**
     *  \brief Explicit converting constructor.
     *
     *  Converting from floating point to integer rounds to the nearest integer instead of truncating.
     *  This ensures that a floating-point pixel coordinate converts to the coordinate of the pixel
     *  it lies on (assuming the floating point origin is the center of the first pixel).
     */
    template <typename U>
    explicit Point(Point<U,N> const & other);

    /**
     *  @name General comparison operators
     *
     *  Interopability with scalars for these operators, if desired, should probably be provided by a
     *  non-explicit constructor from scalar, since that's really what operator interopability
     *  implies.
     *
     *  Note that these return CoordinateExpr, not bool.
     */
    //@{
    CoordinateExpr operator<(Point const & other) const;
    CoordinateExpr operator<=(Point const & other) const;
    CoordinateExpr operator>(Point const & other) const;
    CoordinateExpr operator>=(Point const & other) const;
    //@}

    /**
     *  @name Arithmetic operators
     *
     *  Interopability with scalars for these operators, if desired, should probably be provided by a
     *  non-explicit constructor from scalar, since that's really what operator interopability
     *  implies.
     */
    //@{
    Extent<T,N> operator-(Point const & other) const;
    Point operator+(Extent<T,N> const & other) const;
    Point operator-(Extent<T,N> const & other) const;
    Point & operator+=(Extent<T,N> const & other);
    Point & operator-=(Extent<T,N> const & other);    
    //@}

    /// \brief Shift the point by the given offset.  Redundant with += Extent, but still worth having.
    void shift(Extent<T,N> const & offset);

};

typedef Point<int,2> Point2I;
typedef Point<int,3> Point3I;
typedef Point<double,2> Point2D;
typedef Point<double,3> Point3D;


}}}

#endif

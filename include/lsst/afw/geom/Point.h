// -*- lsst-c++ -*-
/**
 * \file
 * \brief A coordinate class intended to represent absolute positions.
 */
#ifndef LSST_AFW_GEOM_POINT_H
#define LSST_AFW_GEOM_POINT_H

#include "lsst/afw/geom/CoordinateBase.h"
#include "lsst/afw/geom/CoordinateExpr.h"
#include "lsst/afw/geom/Extent.h"

namespace lsst { namespace afw { namespace geom {

/**
 *  \brief A coordinate class intended to represent absolute positions.
 *
 *  Much of the functionality of Point is provided by its CRTP base class, CoordinateBase.
 */
template<typename T, int N>
class Point : public CoordinateBase<Point<T,N>,T,N> {
    typedef CoordinateBase<Point<T,N>,T,N> Super;
public:

    /**
     *  \brief Standard coordinate constructors
     *
     *  See the CoordinateBase constructors for more discussion.
     */
    //@{
    explicit Point(T val=static_cast<T>(0)) : Super(val) {}

    template <typename Vector>
    explicit Point(Eigen::MatrixBase<Vector> const & vector) : Super(vector) {}
    //@}

    /// \brief Explicit constructor from Extent.
    explicit Point(Extent<T,N> const & other) : Super(other.asVector()) {}

    /**
     *  \brief Explicit converting constructor.
     *
     *  Converting from floating point to integer rounds to the nearest integer instead of truncating.
     *  This ensures that a floating-point pixel coordinate converts to the coordinate of the pixel
     *  it lies on (assuming the floating point origin is the center of the first pixel).
     */
    template <typename U>
    explicit Point(Point<U,N> const & other) : Super(other.asVector().template cast<T>()) {}

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
    CoordinateExpr<N> operator<(Point const & other) const;
    CoordinateExpr<N> operator<=(Point const & other) const;
    CoordinateExpr<N> operator>(Point const & other) const;
    CoordinateExpr<N> operator>=(Point const & other) const;
    //@}

    /**
     *  @name Arithmetic operators
     *
     *  Interopability with scalars for these operators, if desired, should probably be provided by a
     *  non-explicit constructor from scalar, since that's really what operator interopability
     *  implies.
     */
    //@{
    Extent<T,N> operator-(Point const & other) const { return Extent<T,N>(this->_vector - other._vector); }
    Point operator+(Extent<T,N> const & other) const { return Point(this->_vector + other.asVector()); }
    Point operator-(Extent<T,N> const & other) const { return Point(this->_vector - other.asVector()); }
    Point & operator+=(Extent<T,N> const & other) { this->_vector += other.asVector(); return *this; }
    Point & operator-=(Extent<T,N> const & other) { this->_vector -= other.asVector(); return *this; }
    //@}

    /// \brief Shift the point by the given offset.
    void shift(Extent<T,N> const & offset) { this->_vector += offset.asVector(); }

};

typedef Point<int,2> PointI;
typedef Point<int,2> Point2I;
typedef Point<int,3> Point3I;
typedef Point<double,2> PointD;
typedef Point<double,2> Point2D;
typedef Point<double,3> Point3D;


}}}

#endif

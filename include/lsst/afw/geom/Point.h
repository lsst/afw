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

    /// \brief Construct a Point with all elements set to the same scalar value.
    explicit Point(T val=static_cast<T>(0)) : Super(val) {}

    /// \brief Construct a Point from an Eigen vector.
    template <typename Vector>
    explicit Point(Eigen::MatrixBase<Vector> const & vector) : Super(vector) {}

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
    explicit Point(Point<U,N> const & other);

    /**
     *  @name Comparison operators
     *
     *  Note that these return CoordinateExpr, not bool.
     *
     *  Unlike most arithmetic and assignment operators, scalar interoperability is provided
     *  for comparison operators; expressions like 
     *  \code
     *    if (all(extent >= 0)) ...
     *  \endcode
     *  are both ubiquitous and easy to interpret.
     */
    //@{
    CoordinateExpr<N> operator==(Point const & other) const;
    CoordinateExpr<N> operator!=(Point const & other) const;
    CoordinateExpr<N> operator<(Point const & other) const;
    CoordinateExpr<N> operator<=(Point const & other) const;
    CoordinateExpr<N> operator>(Point const & other) const;
    CoordinateExpr<N> operator>=(Point const & other) const;
    CoordinateExpr<N> operator==(T scalar) const { return *this == Point(scalar); }
    CoordinateExpr<N> operator!=(T scalar) const { return *this != Point(scalar); }
    CoordinateExpr<N> operator<(T scalar) const { return *this < Point(scalar); }
    CoordinateExpr<N> operator<=(T scalar) const { return *this <= Point(scalar); }
    CoordinateExpr<N> operator>(T scalar) const { return *this > Point(scalar); }
    CoordinateExpr<N> operator>=(T scalar) const { return *this >= Point(scalar); }
    friend CoordinateExpr<N> operator==(T scalar, Point const & other) { Point(scalar) == other; }
    friend CoordinateExpr<N> operator!=(T scalar, Point const & other) { Point(scalar) != other; }
    friend CoordinateExpr<N> operator<(T scalar, Point const & other) { Point(scalar) < other; }
    friend CoordinateExpr<N> operator<=(T scalar, Point const & other) { Point(scalar) <= other; }
    friend CoordinateExpr<N> operator>(T scalar, Point const & other) { Point(scalar) > other; }
    friend CoordinateExpr<N> operator>=(T scalar, Point const & other) { Point(scalar) >= other; }
    //@}

    /**
     *  @name Arithmetic operators
     *
     *  No scalar interoperability is provided for Point arithmetic operations.
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

// -*- lsst-c++ -*-
/**
 * \file
 * \brief A coordinate class intended to represent offsets and dimensions.
 */
#ifndef LSST_AFW_GEOM_EXTENT_H
#define LSST_AFW_GEOM_EXTENT_H

#include "lsst/afw/geom/CoordinateBase.h"

namespace lsst { namespace afw { namespace geom {

/**
 *  \brief A coordinate class intended to represent offsets and dimensions.
 *
 *  Much of the functionality of Extent is provided by its CRTP base class, CoordinateBase.
 *
 *  Unlike Point, Extent does not have a type-converting constructor, because the rounding semantics
 *  are not as clear.  In most cases, conversions between integer and floating point dimensions are
 *  best handled by Box objects, where the rounding semantics make more sense.
 *
 *  Extent does not have specialized [get/set]Width() and [get/set]Height() accessors, as this would
 *  require lots of partial specialization and break the symmetry with the other coordinate classes.
 */
template<typename T, int N>
class Extent : public CoordinateBase<Extent<T,N>,T,N> {
    typedef CoordinateBase<Extent<T,N>,T,N> Super;
public:

    /// \brief Construct an Extent with all elements set to the same scalar value.
    explicit Extent(T val=static_cast<T>(0)) : Super(val) {}

    /// \brief Construct an Extent from an Eigen vector.
    template <typename Vector>
    explicit Extent(Eigen::MatrixBase<Vector> const & vector) : Super(vector) {}

    /// \brief Explicit constructor from Point.
    explicit Extent(Point<T,N> const & other);

    /// \brief Return the squared L2 norm of the Extent (x^2 + y^2 + ...).
    T computeSquaredNorm() const { return this->asVector().squaredNorm(); }

    /// \brief Return the L2 norm of the Extent (sqrt(x^2 + y^2 + ...)).
    T computeNorm() const { return this->asVector().norm(); }

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
    CoordinateExpr<N> operator==(Extent const & other) const;
    CoordinateExpr<N> operator!=(Extent const & other) const;
    CoordinateExpr<N> operator<(Extent const & other) const;
    CoordinateExpr<N> operator<=(Extent const & other) const;
    CoordinateExpr<N> operator>(Extent const & other) const;
    CoordinateExpr<N> operator>=(Extent const & other) const;
    CoordinateExpr<N> operator==(T scalar) const { return *this == Extent(scalar); }
    CoordinateExpr<N> operator!=(T scalar) const { return *this != Extent(scalar); }
    CoordinateExpr<N> operator<(T scalar) const { return *this < Extent(scalar); }
    CoordinateExpr<N> operator<=(T scalar) const { return *this <= Extent(scalar); }
    CoordinateExpr<N> operator>(T scalar) const { return *this > Extent(scalar); }
    CoordinateExpr<N> operator>=(T scalar) const { return *this >= Extent(scalar); }
    friend CoordinateExpr<N> operator==(T scalar, Extent const & other) { return Extent(scalar) == other; }
    friend CoordinateExpr<N> operator!=(T scalar, Extent const & other) { return Extent(scalar) != other; }
    friend CoordinateExpr<N> operator<(T scalar, Extent const & other) { return Extent(scalar) < other; }
    friend CoordinateExpr<N> operator<=(T scalar, Extent const & other) { return Extent(scalar) <= other; }
    friend CoordinateExpr<N> operator>(T scalar, Extent const & other) { return Extent(scalar) > other; }
    friend CoordinateExpr<N> operator>=(T scalar, Extent const & other) { return Extent(scalar) >= other; }
    //@}

    /**
     *  @name Additive arithmetic operators
     *
     *  No scalar interoperability is provided for Extent additive arithmetic operations.
     */
    //@{
    Point<T,N> operator+(Point<T,N> const & other) const;
    Extent operator+(Extent const & other) const { return Extent(this->_vector + other._vector); }
    Extent operator-(Extent const & other) const { return Extent(this->_vector - other._vector); }
    Extent & operator+=(Extent const & other) { this->_vector += other._vector; return *this; }
    Extent & operator-=(Extent const & other) { this->_vector -= other._vector; return *this; }
    Extent operator+() const { return *this; }
    Extent operator-() const { return Extent(-this->_vector); }
    //@}

    /**
     *  @name Multiplicative arithmetic operators
     *
     *  As usual with matrices and vectors, Extent can be multiplied or divided by a scalar.
     */
    //@{
    Extent operator*(T scalar) const { return Extent(this->_vector * scalar); }
    Extent & operator*=(T scalar) { this->_vector *= scalar; return *this; }
    Extent operator/(T scalar) const { return Extent(this->_vector / scalar); }
    Extent & operator/=(T scalar) { this->_vector /= scalar; return *this; }
    //@}

};

typedef Extent<int,2> ExtentI;
typedef Extent<int,2> Extent2I;
typedef Extent<int,3> Extent3I;
typedef Extent<double,2> ExtentD;
typedef Extent<double,2> Extent2D;
typedef Extent<double,3> Extent3D;

}}}

#endif

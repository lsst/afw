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
 *  best handled by Box objects, where the rounding semantics are more clear.
 *
 *  Extent does not have specialized [get/set]Width() and [get/set]Height() accessors, as this would
 *  require lots of partial specialization and break the symmetry with the other coordinate classes.
 */
template<typename T, int N>
class Extent : public CoordinateBase<Extent<T,N>,T,N> {
    typedef CoordinateBase<Extent<T,N>,T,N> Super;
public:

    /**
     *  \brief Standard coordinate constructors
     *
     *  See the CoordinateBase constructors for more discussion.
     */
    //@{
    explicit Extent(T val=static_cast<T>(0)) : Super(val) {}

    template <typename Vector>
    explicit Extent(Eigen::MatrixBase<Vector> const & vector) : Super(vector) {}
    //@}

    /// \brief Explicit constructor from Point.
    explicit Extent(Point<T,N> const & other);

    /// \brief Return the squared L2 norm of the Extent (x^2 + y^2 + ...)
    T computeSquaredNorm() const { return this->asVector().squaredNorm(); }

    /// \brief Return the L2 norm of the Extent.
    T computeNorm() const { return this->asVector().norm(); }
    
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
    CoordinateExpr<N> operator<(Extent const & other) const;
    CoordinateExpr<N> operator<=(Extent const & other) const;
    CoordinateExpr<N> operator>(Extent const & other) const;
    CoordinateExpr<N> operator>=(Extent const & other) const;
    //@}

    /**
     *  @name Arithmetic operators
     *
     *  Interopability with scalars for these operators, if desired, should probably be provided by a
     *  non-explicit constructor from scalar, since that's really what operator interopability
     *  implies.
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
     *  @name Multiplicative operators
     *
     *  I've currently left out the division operators; integer division bothers me in this context, and
     *  multiplication should be all that's really necessary for convenient scaling.
     */
    //@{
    Extent operator*(T scalar) const { return Extent(this->_vector * scalar); }
    Extent & operator*=(T scalar) { this->_vector *= scalar; return *this; }
    //@}

};

template <typename T, int N>
inline Extent<T,N> abs(Extent<T,N> const & extent) {
    return Extent<T,N>(extent.asVector().cwise().abs());
}

typedef Extent<int,2> ExtentI;
typedef Extent<int,2> Extent2I;
typedef Extent<int,3> Extent3I;
typedef Extent<double,2> ExtentD;
typedef Extent<double,2> Extent2D;
typedef Extent<double,3> Extent3D;

}}}

#endif

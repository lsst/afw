// -*- lsst-c++ -*-
/**
 * \file
 * \brief A coordinate class intended to represent offsets and dimensions.
 */
#ifndef LSST_AFW_GEOM_EXTENT_H
#define LSST_AFW_GEOM_EXTENT_H

#include "lsst/afw/geom/CoordinateBase.h"

namespace lsst { namespace afw { namespace geom {

template <typename T, int N> class Point;

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
public:

    /**
     *  \brief Standard coordinate constructors
     *
     *  See the CoordinateBase constructors for more discussion.
     */
    //@{
    explicit Extent(T val=static_cast<T>(0));

    template <typename Vector>
    explicit Extent(Eigen::MatrixBase<Vector> const & vector);
    //@}

    /// \brief Explicit constructor from Point.
    explicit Extent(Point<T,N> const & other);

    /**
     *  \brief Return an Extent with each element the absolute value of the correspond element of this.
     */
    Extent makePositive() const;

    /// \brief Return the squared L2 norm of the Extent (x^2 + y^2 + ...)
    T computeSquaredNorm() const;

    /// \brief Return the L2 norm of the Extent.
    T computeNorm() const { return std::sqrt(computeSquaredNorm()); }
    
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
    CoordinateExpr operator<(Extent const & other) const;
    CoordinateExpr operator<=(Extent const & other) const;
    CoordinateExpr operator>(Extent const & other) const;
    CoordinateExpr operator>=(Extent const & other) const;
    //@}

    /**
     *  @name Arithmetic operators
     *
     *  Interopability with scalars for these operators, if desired, should probably be provided by a
     *  non-explicit constructor from scalar, since that's really what operator interopability
     *  implies.
     */
    //@{
    Point<T,N> operator+(Point<T,N> const & other) const; ///< Debatable whether this should exist.
    Extent operator+(Extent const & other) const;
    Extent operator-(Extent const & other) const;
    Extent & operator+=(Extent const & other);
    Extent & operator-=(Extent const & other);
    Extent operator+() const;
    Extent operator-() const;
    //@}

    /**
     *  @name Multiplicative operators
     *
     *  I've currently left out the division operators; integer division bothers me in this context, and
     *  multiplication should be all that's really necessary for convenient scaling.
     */
    //@{
    Extent operator*(T scalar) const;
    Extent & operator*=(T scalar);
    //@}

};

}}}

#endif

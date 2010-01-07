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
     *  @name Named comparison functions
     *
     *  Note that these return CoordinateExpr, not bool.
     *
     *  Unlike most arithmetic and assignment operators, scalar interoperability is provided
     *  for comparisons; expressions like 
     *  \code
     *    if (all(extent.gt(0))) ...
     *  \endcode
     *  are both ubiquitous and easy to interpret.
     */
    //@{
    bool operator==(Extent const & other) const { return all(this->eq(other)); }
    bool operator!=(Extent const & other) const { return any(this->ne(other)); }
    CoordinateExpr<N> eq(Extent const & other) const;
    CoordinateExpr<N> ne(Extent const & other) const;
    CoordinateExpr<N> lt(Extent const & other) const;
    CoordinateExpr<N> le(Extent const & other) const;
    CoordinateExpr<N> gt(Extent const & other) const;
    CoordinateExpr<N> ge(Extent const & other) const;
    CoordinateExpr<N> eq(T scalar) const { return this->eq(Extent(scalar)); }
    CoordinateExpr<N> ne(T scalar) const { return this->ne(Extent(scalar)); }
    CoordinateExpr<N> lt(T scalar) const { return this->lt(Extent(scalar)); }
    CoordinateExpr<N> le(T scalar) const { return this->le(Extent(scalar)); }
    CoordinateExpr<N> gt(T scalar) const { return this->gt(Extent(scalar)); }
    CoordinateExpr<N> ge(T scalar) const { return this->ge(Extent(scalar)); }
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

inline Extent2I makeExtentI(int x, int y) { return Extent2I::make(x,y); }
inline Extent3I makeExtentI(int x, int y, int z) { return Extent3I::make(x,y,z); }
inline Extent2D makeExtentD(double x, double y) { return Extent2D::make(x,y); }
inline Extent3D makeExtentD(double x, double y, double z) { return Extent3D::make(x,y,z); }

}}}

#endif

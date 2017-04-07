// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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

/*
 * A coordinate class intended to represent offsets and dimensions.
 */
#ifndef LSST_AFW_GEOM_EXTENT_H
#define LSST_AFW_GEOM_EXTENT_H

#include <tuple>
#include <type_traits>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/CoordinateExpr.h"

namespace lsst { namespace afw { namespace geom {

// These are present to avoid a static assertion for instantiating computeNorm() on integer types.
namespace detail {

template <int N>
double computeExtentNorm(Extent<double,N> const & s) {
    return s.asEigen().norm();
}

template <int N>
int computeExtentNorm(Extent<int,N> const & s) {
    throw LSST_EXCEPT(
        pex::exceptions::LogicError,
        "Cannot compute norm of integer extent"
    );
#if 1                                   // make compilers happy in non-void function
    return -1;
#endif
}

} // namespace detail


template<typename T, int N>
class ExtentBase : public CoordinateBase<Extent<T,N>,T,N> {
    typedef CoordinateBase<Extent<T,N>,T,N> Super;
public:

    /// Return the squared L2 norm of the Extent (x^2 + y^2 + ...).
    T computeSquaredNorm() const { return this->asEigen().squaredNorm(); }

    /// Return the L2 norm of the Extent (sqrt(x^2 + y^2 + ...)).
    T computeNorm() const { return detail::computeExtentNorm(static_cast<Extent<T,N> const &>(*this)); }

    /**
     *  Standard equality comparison.
     *
     *  Returns true iff all(this->eq(other));
     */
    bool operator==(Extent<T,N> const & other) const { return all(this->eq(other)); }

    /**
     *  Standard inequality comparison.
     *
     *  Returns true iff any(this->ne(other));
     */
    bool operator!=(Extent<T,N> const & other) const { return any(this->ne(other)); }

    /**
     *  @name Named comparison functions
     *
     *  Note that these return CoordinateExpr, not bool.
     *
     *  Unlike most arithmetic and assignment operators, scalar interoperability is provided
     *  for comparisons; expressions like
     *
     *      if (all(extent.gt(0))) ...
     *
     *  are both ubiquitous and easy to interpret.
     */
    //@{
    CoordinateExpr<N> eq(Extent<T,N> const & other) const;
    CoordinateExpr<N> ne(Extent<T,N> const & other) const;
    CoordinateExpr<N> lt(Extent<T,N> const & other) const;
    CoordinateExpr<N> le(Extent<T,N> const & other) const;
    CoordinateExpr<N> gt(Extent<T,N> const & other) const;
    CoordinateExpr<N> ge(Extent<T,N> const & other) const;
    CoordinateExpr<N> eq(T scalar) const { return this->eq(Extent<T,N>(scalar)); }
    CoordinateExpr<N> ne(T scalar) const { return this->ne(Extent<T,N>(scalar)); }
    CoordinateExpr<N> lt(T scalar) const { return this->lt(Extent<T,N>(scalar)); }
    CoordinateExpr<N> le(T scalar) const { return this->le(Extent<T,N>(scalar)); }
    CoordinateExpr<N> gt(T scalar) const { return this->gt(Extent<T,N>(scalar)); }
    CoordinateExpr<N> ge(T scalar) const { return this->ge(Extent<T,N>(scalar)); }
    //@}

    /**
     *  @name Additive arithmetic operators
     *
     *  No scalar interoperability is provided for Extent additive arithmetic operations.
     */
    //@{
    Point<T,N> operator+(Point<T,N> const & other) const;
    Extent<T,N> operator+(Extent<T,N> const & other) const {
        return Extent<T,N>(this->_vector + other._vector);
    }
    Extent<T,N> operator-(Extent<T,N> const & other) const {
        return Extent<T,N>(this->_vector - other._vector);
    }
    Extent<T,N> & operator+=(Extent<T,N> const & other) {
        this->_vector += other._vector;
        return static_cast<Extent<T,N>&>(*this);
    }
    Extent<T,N> & operator-=(Extent<T,N> const & other) {
        this->_vector -= other._vector;
        return static_cast<Extent<T,N>&>(*this);
    }
    Extent<T,N> operator+() const { return static_cast<Extent<T,N> const &>(*this); }
    Extent<T,N> operator-() const { return Extent<T,N>(-this->_vector); }
    //@}

    /**
     *  @name Multiplicative arithmetic operators
     *
     *  As usual with matrices and vectors, Extent can be multiplied or divided by a scalar.
     */
    //@{
    Extent<T,N> operator*(T scalar) const { return Extent<T,N>(this->_vector * scalar); }
    Extent<T,N> & operator*=(T scalar) { this->_vector *= scalar; return static_cast<Extent<T,N>&>(*this); }
    Extent<T,N> operator/(T scalar) const { return Extent<T,N>(this->_vector / scalar); }
    Extent<T,N> & operator/=(T scalar) { this->_vector /= scalar; return static_cast<Extent<T,N>&>(*this); }
    //@}

    /// Cast this object to an Extent of the same numeric type and dimensionality.
    Point<T,N> asPoint() const;

    std::string toString() const {
        std::stringstream out;
        out << "Extent(";
        for (size_t i = 0; i < N; ++i) {
            if (i != 0) {
                out << ",";
            }
            out << (*this)[i];
        }
        out << ")";
        return out.str();
    }

protected:

    /// Construct an Extent<T,N> with all elements set to the same scalar value.
    explicit ExtentBase(T val=static_cast<T>(0)) : Super(val) {}

    /// Construct an Extent from an Eigen vector.
    template <typename Vector>
    explicit ExtentBase(Eigen::MatrixBase<Vector> const & vector) : Super(vector) {}

};

/**
 *  A coordinate class intended to represent offsets and dimensions.
 *
 *  Much of the functionality of Extent is provided by its CRTP base class, ExtentBase.
 *
 *  See @ref afwGeomOps for mathematical operators on Extent.
 */
template<typename T, int N>
class Extent : public ExtentBase<T,N> {
    typedef ExtentBase<T,N> Super;
public:
    typedef typename Super::EigenVector EigenVector;

    /// Construct an Extent with all elements set to the same scalar value.
    explicit Extent(T val=static_cast<T>(0)) : Super(val) {}

    /// Construct an Extent from an Eigen vector.
    explicit Extent(EigenVector const & vector) : Super(vector) {}

    /// Explicit constructor from Point.
    explicit Extent(Point<T,N> const & other);

    /// Explicit constructor from Extent of different type (if allowed)
    template<typename U>
    explicit Extent(Extent<U,N> const & other);
    template<typename U>
    explicit Extent(Point<U,N> const & other);

    /// Return the squared L2 norm of the Extent (x^2 + y^2 + ...).
    T computeSquaredNorm() const { return this->asEigen().squaredNorm(); }

    /// Return the L2 norm of the Extent (sqrt(x^2 + y^2 + ...)).
    T computeNorm() const { return this->asEigen().norm(); }

    void swap(Extent & other) { this->_swap(other); }
};

/**
 *  A coordinate class intended to represent offsets and dimensions (2-d specialization).
 *
 *  See @ref afwGeomOps for mathematical operators on Extent.
 */
template<typename T>
class Extent<T,2> : public ExtentBase<T,2> {
    typedef ExtentBase<T,2> Super;
public:
    typedef typename Super::EigenVector EigenVector;

    /// Construct an Extent with all elements set to the same scalar value.
    explicit Extent(T val=static_cast<T>(0)) : Super(val) {}

    /// Construct an Extent from an Eigen vector.
    explicit Extent(EigenVector const & vector) : Super(vector) {}

    /// Explicit constructor from Point.
    explicit Extent(Point<T,2> const & other);

    /// Explicit constructor from Extent of different type (if allowed)
    template<typename U>
    explicit Extent(Extent<U,2> const & other);
    template<typename U>
    explicit Extent(Point<U,2> const & other);

    /// Construct from two scalars.
    explicit Extent(T x, T y) : Super(EigenVector(x, y)) {}

    /// Construct from a two-element array.
    explicit Extent(T const xy[2]) : Super(EigenVector(xy[0], xy[1])) {}

    /// Construct from a std::pair.
    explicit Extent(std::pair<T,T> const & xy) : Super(EigenVector(xy.first, xy.second)) {}

    /// Construct from std::tuple.
    explicit Extent(std::tuple<T,T> const & xy) :
        Super(EigenVector(std::get<0>(xy), std::get<1>(xy))) {}

#ifdef SWIG
    T getX() const;
    T getY() const;
    void setX(T x);
    void setY(T y);
#endif

    void swap(Extent & other) { this->_swap(other); }
};

/**
 *  A coordinate class intended to represent offsets and dimensions (3-d specialization).
 *
 *  See @ref afwGeomOps for mathematical operators on Extent.
 */
template<typename T>
class Extent<T,3> : public ExtentBase<T,3> {
    typedef ExtentBase<T,3> Super;
public:
    typedef typename Super::EigenVector EigenVector;

    /// Construct an Extent with all elements set to the same scalar value.
    explicit Extent(T val=static_cast<T>(0)) : Super(val) {}

    /// Construct an Extent from an Eigen vector.
    explicit Extent(EigenVector const & vector) : Super(vector) {}

    /// Explicit constructor from Point.
    explicit Extent(Point<T,3> const & other);

    /// Explicit constructor from Extent of different type (if allowed)
    template<typename U>
    explicit Extent(Extent<U,3> const & other);
    template<typename U>
    explicit Extent(Point<U,3> const & other);

    /// Construct from three scalars.
    explicit Extent(T x, T y, T z) : Super(EigenVector(x, y, z)) {}

    /// Construct from a two-element array.
    explicit Extent(T const xyz[3]) : Super(EigenVector(xyz[0], xyz[1], xyz[2])) {}

    /// Construct from std::tuple.
    explicit Extent(std::tuple<T,T,T> const & xyz) :
        Super(EigenVector(std::get<0>(xyz), std::get<1>(xyz), std::get<2>(xyz))) {}

#ifdef SWIG
    T getX() const;
    T getY() const;
    T getZ() const;
    void setX(T x);
    void setY(T y);
    void setZ(T z);
#endif

    void swap(Extent & other) { this->_swap(other); }
};

// Constructor for any 2D type from 2I type
template<typename T>
template<typename U>
Extent<T, 2>::Extent(Extent<U, 2> const & other)
{
    static_assert((!std::is_same<T,U>::value && std::is_integral<U>::value),
            "can only construct from Extent of different but integral type");
    this->setX(static_cast<T>(other.getX()));
    this->setY(static_cast<T>(other.getY()));
};

template<typename T>
template<typename U>
Extent<T, 2>::Extent(Point<U, 2> const & other)
{
    static_assert((!std::is_same<T,U>::value && std::is_integral<U>::value),
            "can only construct from Extent of different but integral type");
    this->setX(static_cast<T>(other.getX()));
    this->setY(static_cast<T>(other.getY()));
};

// Constructor for any 3D type from 3I type
template<typename T>
template<typename U>
Extent<T, 3>::Extent(Extent<U, 3> const & other)
{
    static_assert((!std::is_same<T,U>::value && std::is_integral<U>::value),
            "can only construct from Extent of different but integral type");
    this->setX(static_cast<T>(other.getX()));
    this->setY(static_cast<T>(other.getY()));
    this->setZ(static_cast<T>(other.getZ()));
};

// Constructor for any 3D type from 3I type
template<typename T>
template<typename U>
Extent<T, 3>::Extent(Point<U, 3> const & other)
{
    static_assert((!std::is_same<T,U>::value && std::is_integral<U>::value),
            "can only construct from Extent of different but integral type");
    this->setX(static_cast<T>(other.getX()));
    this->setY(static_cast<T>(other.getY()));
    this->setZ(static_cast<T>(other.getZ()));
};

typedef Extent<int,2> ExtentI;
typedef Extent<int,2> Extent2I;
typedef Extent<int,3> Extent3I;
typedef Extent<double,2> ExtentD;
typedef Extent<double,2> Extent2D;
typedef Extent<double,3> Extent3D;

/**
 *  Return the component-wise truncation (round towards zero).
 *
 *  In Python, this is available as both a free function and a method on ExtentD.
 */
template <int N>
Extent<int,N> truncate(Extent<double,N> const & input);

/**
 *  Return the component-wise floor (round towards more negative).
 *
 *  In Python, this is available as both a free function and a method on ExtentD.
 */
template <int N>
Extent<int,N> floor(Extent<double,N> const & input);

/**
 *  Return the component-wise ceil (round towards more positive).
 *
 *  In Python, this is available as both a free function and a method on ExtentD.
 */
template <int N>
Extent<int,N> ceil(Extent<double,N> const & input);

#ifndef SWIG

// Some operators below need to take ExtentBase arguments rather than Extent to
// avoid ambiguous overloads (since some competing operators are defined as member
// functions on ExtentBase).

template <typename T, int N>
Extent<T,N> operator*(T scalar, ExtentBase<T,N> const & rhs) {
    return rhs * scalar;
}

template <int N>
Extent<double,N> operator*(ExtentBase<int,N> const & lhs, double rhs) {
    return Extent<double,N>(static_cast<Extent<int,N> const &>(lhs)) * rhs;
}

template <int N>
void operator*=(ExtentBase<int,N> & lhs, double rhs) {
    // use "N < 0" so assertion is dependent on template instantiation, instead of triggering all the time
    static_assert(N < 0, "In-place multiplication of Extent<int,N> by double would truncate.");
}

template <int N>
Extent<double,N> operator/(ExtentBase<int,N> const & lhs, double rhs) {
    return Extent<double,N>(static_cast<Extent<int,N> const &>(lhs)) / rhs;
}

template <int N>
void operator/=(ExtentBase<int,N> & lhs, double rhs) {
    // use "N < 0" so assertion is dependent on template instantiation, instead of triggering all the time
    static_assert(N < 0, "In-place division of Extent<int,N> by double would truncate.");
}

template <int N>
Extent<double,N> operator*(double lhs, ExtentBase<int,N> const & rhs) {
    return lhs * Extent<double,N>(static_cast<Extent<int,N> const &>(rhs));
}

template <int N>
Extent<double,N> operator+(Extent<double,N> const & lhs, Extent<int,N> const & rhs) {
    return lhs + Extent<double,N>(rhs);
}

template <int N>
Extent<double,N> & operator+=(Extent<double,N> & lhs, Extent<int,N> const & rhs) {
    return lhs += Extent<double,N>(rhs);
}

template <int N>
Extent<double,N> operator-(Extent<double,N> const & lhs, Extent<int,N> const & rhs) {
    return lhs - Extent<double,N>(rhs);
}

template <int N>
Extent<double,N> & operator-=(Extent<double,N> & lhs, Extent<int,N> const & rhs) {
    return lhs -= Extent<double,N>(rhs);
}

template <int N>
Extent<double,N> operator+(Extent<int,N> const & lhs, Extent<double,N> const & rhs) {
    return Extent<double,N>(lhs) + rhs;
}

template <int N>
Extent<double,N> operator-(Extent<int,N> const & lhs, Extent<double,N> const & rhs) {
    return Extent<double,N>(lhs) - rhs;
}

#endif // !SWIG

}}}

#endif

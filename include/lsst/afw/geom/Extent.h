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
 * \brief A coordinate class intended to represent offsets and dimensions.
 */
#ifndef LSST_AFW_GEOM_EXTENT_H
#define LSST_AFW_GEOM_EXTENT_H

#include "lsst/afw/geom/CoordinateExpr.h"

namespace lsst { namespace afw { namespace geom {

template<typename T, int N>
class ExtentBase : public CoordinateBase<Extent<T,N>,T,N> {
    typedef CoordinateBase<Extent<T,N>,T,N> Super;
public:

    /// \brief Return the squared L2 norm of the Extent (x^2 + y^2 + ...).
    T computeSquaredNorm() const { return this->asEigen().squaredNorm(); }

    /// \brief Return the L2 norm of the Extent (sqrt(x^2 + y^2 + ...)).
    T computeNorm() const { return this->asEigen().norm(); }

    /**
     *  @brief Standard equality comparison.
     *
     *  Returns true iff all(this->eq(other));
     */
    bool operator==(Extent<T,N> const & other) const { return all(this->eq(other)); }

    /**
     *  @brief Standard inequality comparison.
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
     *  \code
     *    if (all(extent.gt(0))) ...
     *  \endcode
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

protected:

    /// \brief Construct an Extent<T,N> with all elements set to the same scalar value.
    explicit ExtentBase(T val=static_cast<T>(0)) : Super(val) {}

    /// \brief Construct an Extent from an Eigen vector.
    template <typename Vector>
    explicit ExtentBase(Eigen::MatrixBase<Vector> const & vector) : Super(vector) {}

};

/**
 *  \brief A coordinate class intended to represent offsets and dimensions.
 *
 *  Much of the functionality of Extent is provided by its CRTP base class, CoordinateBase.
 *
 *  Unlike Point, Extent does not have a type-converting constructor, because the rounding semantics
 *  are not as clear.  In most cases, conversions between integer and floating point dimensions are
 *  best handled by Box objects, where the rounding semantics make more sense.
 */
template<typename T, int N>
class Extent : public ExtentBase<T,N> {
    typedef ExtentBase<T,N> Super;
public:
    typedef typename Super::EigenVector EigenVector;

    /// \brief Construct an Extent with all elements set to the same scalar value.
    explicit Extent(T val=static_cast<T>(0)) : Super(val) {}

    /// \brief Construct an Extent from an Eigen vector.
    explicit Extent(EigenVector const & vector) : Super(vector) {}

    /// \brief Explicit constructor from Point.
    explicit Extent(Point<T,N> const & other);

    /// \brief Return the squared L2 norm of the Extent (x^2 + y^2 + ...).
    T computeSquaredNorm() const { return this->asEigen().squaredNorm(); }

    /// \brief Return the L2 norm of the Extent (sqrt(x^2 + y^2 + ...)).
    T computeNorm() const { return this->asEigen().norm(); }
    
    void swap(Extent & other) { this->_swap(other); }
};

/**
 *  \brief A coordinate class intended to represent offsets and dimensions (2-d specialization).
 */
template<typename T>
class Extent<T,2> : public ExtentBase<T,2> {
    typedef ExtentBase<T,2> Super;
public:
    typedef typename Super::EigenVector EigenVector;

    /// \brief Construct an Extent with all elements set to the same scalar value.
    explicit Extent(T val=static_cast<T>(0)) : Super(val) {}

    /// \brief Construct an Extent from an Eigen vector.
    explicit Extent(EigenVector const & vector) : Super(vector) {}

    /// \brief Explicit constructor from Point.
    explicit Extent(Point<T,2> const & other);

    /// @brief Construct from two scalars.
    explicit Extent(T x, T y) : Super(EigenVector(x, y)) {}

    /// @brief Construct from a two-element array.
    explicit Extent(T const xy[2]) : Super(EigenVector(xy[0], xy[1])) {}

    /// @brief Construct from a std::pair.
    explicit Extent(std::pair<T,T> const & xy) : Super(EigenVector(xy.first, xy.second)) {}

    /// @brief Construct from boost::tuple.
    explicit Extent(boost::tuple<T,T> const & xy) : 
        Super(EigenVector(xy.template get<0>(), xy.template get<1>())) {}

    T getX() const { return this->_vector.x(); }
    T getY() const { return this->_vector.y(); }
    void setX(T x) { this->_vector.x() = x; }
    void setY(T y) { this->_vector.y() = y; }

    /// @brief Return a std::pair representation of the coordinate object.
    std::pair<T,T> asPair() const { return std::make_pair(this->_vector.x(),this->_vector.y()); }

    /// @brief Return a boost::tuple representation of the coordinate object.
    boost::tuple<T,T> asTuple() const { return boost::make_tuple(this->_vector.x(),this->_vector.y()); }

    void swap(Extent & other) { this->_swap(other); }
};

/**
 *  \brief A coordinate class intended to represent offsets and dimensions (3-d specialization).
 */
template<typename T>
class Extent<T,3> : public ExtentBase<T,3> {
    typedef ExtentBase<T,3> Super;
public:
    typedef typename Super::EigenVector EigenVector;

    /// \brief Construct an Extent with all elements set to the same scalar value.
    explicit Extent(T val=static_cast<T>(0)) : Super(val) {}

    /// \brief Construct an Extent from an Eigen vector.
    explicit Extent(EigenVector const & vector) : Super(vector) {}

    /// \brief Explicit constructor from Point.
    explicit Extent(Point<T,3> const & other);

    /// @brief Construct from three scalars.
    explicit Extent(T x, T y, T z) : Super(EigenVector(x, y, z)) {}

    /// @brief Construct from a two-element array.
    explicit Extent(T const xyz[3]) : Super(EigenVector(xyz[0], xyz[1], xyz[2])) {}

    /// @brief Construct from boost::tuple.
    explicit Extent(boost::tuple<T,T,T> const & xyz) : 
        Super(EigenVector(xyz.template get<0>(), xyz.template get<1>(), xyz.template get<2>())) {}

    T getX() const { return this->_vector.x(); }
    T getY() const { return this->_vector.y(); }
    T getZ() const { return this->_vector.z(); }
    void setX(T x) { this->_vector.x() = x; }
    void setY(T y) { this->_vector.y() = y; }
    void setZ(T z) { this->_vector.z() = z; }

    /// @brief Return a boost::tuple representation of the coordinate object.
    boost::tuple<T,T,T> asTuple() const {
        return boost::make_tuple(this->_vector.x(), this->_vector.y(), this->_vector.z());
    }

    void swap(Extent & other) { this->_swap(other); }
};

typedef Extent<int,2> ExtentI;
typedef Extent<int,2> Extent2I;
typedef Extent<int,3> Extent3I;
typedef Extent<double,2> ExtentD;
typedef Extent<double,2> Extent2D;
typedef Extent<double,3> Extent3D;

}}}

#endif

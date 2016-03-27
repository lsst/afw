// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
/**
 * \file
 * @brief A coordinate class intended to represent absolute positions.
 */
#ifndef LSST_AFW_GEOM_POINT_H
#define LSST_AFW_GEOM_POINT_H

#include "lsst/afw/geom/CoordinateBase.h"
#include "lsst/afw/geom/CoordinateExpr.h"
#include "lsst/afw/geom/Extent.h"

namespace lsst { namespace afw { namespace geom {

template <typename T, int N>
class PointBase : public CoordinateBase<Point<T,N>,T,N> {
    typedef CoordinateBase<Point<T,N>,T,N> Super;
public:

    /**
     *  @brief Standard equality comparison.
     *
     *  Returns true iff all(this->eq(other));
     */
    bool operator==(Point<T,N> const & other) const { return all(this->eq(other)); }

    /**
     *  @brief Standard inequality comparison.
     *
     *  Returns true iff any(this->ne(other));
     */
    bool operator!=(Point<T,N> const & other) const { return any(this->ne(other)); }

    /**
     *  @name Named vectorized comparison functions
     *
     *  Note that these return CoordinateExpr, not bool.
     *
     *  Unlike most arithmetic and assignment operators, scalar interoperability is provided
     *  for comparisons; expressions like 
     *  \code
     *    if (all(point.gt(0))) ...
     *  \endcode
     *  are both ubiquitous and easy to interpret.
     */
    //@{
    CoordinateExpr<N> eq(Point<T,N> const & other) const;
    CoordinateExpr<N> ne(Point<T,N> const & other) const;
    CoordinateExpr<N> lt(Point<T,N> const & other) const;
    CoordinateExpr<N> le(Point<T,N> const & other) const;
    CoordinateExpr<N> gt(Point<T,N> const & other) const;
    CoordinateExpr<N> ge(Point<T,N> const & other) const;
    CoordinateExpr<N> eq(T scalar) const { return this->eq(Point<T,N>(scalar)); }
    CoordinateExpr<N> ne(T scalar) const { return this->ne(Point<T,N>(scalar)); }
    CoordinateExpr<N> lt(T scalar) const { return this->lt(Point<T,N>(scalar)); }
    CoordinateExpr<N> le(T scalar) const { return this->le(Point<T,N>(scalar)); }
    CoordinateExpr<N> gt(T scalar) const { return this->gt(Point<T,N>(scalar)); }
    CoordinateExpr<N> ge(T scalar) const { return this->ge(Point<T,N>(scalar)); }
    //@}

    /**
     *  @name Arithmetic operators
     *
     *  No scalar interoperability is provided for Point arithmetic operations.
     */
    //@{
    Extent<T,N> operator-(Point<T,N> const & other) const {
        return Extent<T,N>(this->_vector - other._vector);
    }
    Point<T,N> operator+(Extent<T,N> const & other) const {
        return Point<T,N>(this->_vector + other.asEigen());
    }
    Point<T,N> operator-(Extent<T,N> const & other) const {
        return Point<T,N>(this->_vector - other.asEigen());
    }
    Point<T,N> & operator+=(Extent<T,N> const & other) {
        this->_vector += other.asEigen();
        return static_cast<Point<T,N> &>(*this); 
    }
    Point<T,N> & operator-=(Extent<T,N> const & other) {
        this->_vector -= other.asEigen();
        return static_cast<Point<T,N> &>(*this); 
    }
    //@}

    /// Cast this object to an Extent of the same numeric type and dimensionality.
    Extent<T,N> asExtent() const { return Extent<T,N>(static_cast<Point<T,N> const &>(*this)); }

    /// @brief Shift the point by the given offset.
    void shift(Extent<T,N> const & offset) { this->_vector += offset.asEigen(); }


    void scale(double factor) { this->_vector *= factor; }

    double distanceSquared(PointBase<T,N> const & other) const {
        // the cast to double is lame but Eigen seems to require they be the same type
        return (this->asEigen() - other.asEigen()).squaredNorm();
    }
    
    std::string toString() const {
        std::stringstream out;
        out << "Point(";
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

    explicit PointBase(T val = static_cast<T>(0)) : Super(val) {}

    template <typename Vector>
    explicit PointBase(Eigen::MatrixBase<Vector> const & vector) : Super(vector) {}
};

/**
 *  @brief A coordinate class intended to represent absolute positions.
 *
 *  See @ref afwGeomOps for mathematical operators on Point.
 */
template<typename T, int N>
class Point : public PointBase<T,N> {
    typedef PointBase<T,N> Super;
public:
    typedef typename Super::EigenVector EigenVector;

    /// @brief Construct a Point with all elements set to the same scalar value.
    explicit Point(T val=static_cast<T>(0)) : Super(val) {}

    /**
     *  @brief Explicit converting constructor.
     *
     *  Converting from floating point to integer rounds to the nearest integer instead of truncating.
     *  This ensures that a floating-point pixel coordinate converts to the coordinate of the pixel
     *  it lies on (assuming the floating point origin is the center of the first pixel).
     */
    template <typename U>
    explicit Point(Point<U,N> const & other);

    /// @brief Construct a Point from an Eigen vector.
    explicit Point(EigenVector const & vector) : Super(vector) {}

    /// @brief Explicit constructor from Extent.
    explicit Point(Extent<T,N> const & other) : Super(other.asEigen()) {}          

    void swap(Point & other) { this->_swap(other); }
};

/**
 *  @brief A coordinate class intended to represent absolute positions (2-d specialization).
 *
 *  See @ref afwGeomOps for mathematical operators on Point.
 */
template<typename T>
class Point<T,2> : public PointBase<T,2> {
    typedef PointBase<T,2> Super;
public:
    typedef typename Super::EigenVector EigenVector;

    /// @brief Construct a Point with all elements set to the same scalar value.
    explicit Point(T val=static_cast<T>(0)) : Super(val) {}

    /**
     *  @brief Explicit converting constructor.
     *
     *  Converting from floating point to integer rounds to the nearest integer instead of truncating.
     *  This ensures that a floating-point pixel coordinate converts to the coordinate of the pixel
     *  it lies on (assuming the floating point origin is the center of the first pixel).
     */
    template <typename U>
    explicit Point(Point<U,2> const & other);

    /// @brief Construct a Point from an Eigen vector.
    explicit Point(EigenVector const & vector) : Super(vector) {}

    /// @brief Explicit constructor from Extent.
    explicit Point(Extent<T,2> const & other) : Super(other.asEigen()) {}          

    /// @brief Explicit constructor from a pair of doubles.
    explicit Point(T x, T y) : Super(EigenVector(x,y)) {}

    /// @brief Construct from a two-element array.
    explicit Point(T const xy[2]) : Super(EigenVector(xy[0], xy[1])) {}

    /// @brief Construct from a std::pair.
    explicit Point(std::pair<T,T> const & xy) : Super(EigenVector(xy.first, xy.second)) {}

    /// @brief Construct from boost::tuple.
    explicit Point(boost::tuple<T,T> const & xy) : 
        Super(EigenVector(xy.template get<0>(), xy.template get<1>())) {}

#ifdef SWIG
    T getX() const;
    T getY() const;
    void setX(T x);
    void setY(T y);
#endif

    void swap(Point & other) { this->_swap(other); }
};

/**
 *  @brief A coordinate class intended to represent absolute positions (3-d specialization).
 *
 *  See @ref afwGeomOps for mathematical operators on Point.
 */
template<typename T>
class Point<T,3> : public PointBase<T,3> {
    typedef PointBase<T,3> Super;
public:
    typedef typename Super::EigenVector EigenVector;

    /// @brief Construct a Point with all elements set to the same scalar value.
    explicit Point(T val=static_cast<T>(0)) : Super(val) {}

    /**
     *  @brief Explicit converting constructor.
     *
     *  Converting from floating point to integer rounds to the nearest integer instead of truncating.
     *  This ensures that a floating-point pixel coordinate converts to the coordinate of the pixel
     *  it lies on (assuming the floating point origin is the center of the first pixel).
     */
    template <typename U>
    explicit Point(Point<U,3> const & other);

    /// @brief Construct a Point from an Eigen vector.
    explicit Point(EigenVector const & vector) : Super(vector) {}

    /// @brief Explicit constructor from Extent.
    explicit Point(Extent<T,3> const & other) : Super(other.asEigen()) {}          

    /// @brief Explicit constructor from a sequence of doubles.
    explicit Point(T x, T y, T z) : Super(EigenVector(x,y,z)) {}

    /// @brief Construct from a two-element array.
    explicit Point(T const xyz[3]) : Super(EigenVector(xyz[0], xyz[1], xyz[2])) {}

    /// @brief Construct from boost::tuple.
    explicit Point(boost::tuple<T,T,T> const & xyz) : 
        Super(EigenVector(xyz.template get<0>(), xyz.template get<1>(), xyz.template get<2>())) {}

#ifdef SWIG
    T getX() const;
    T getY() const;
    T getZ() const;
    void setX(T x);
    void setY(T y);
    void setZ(T z);
#endif

    void swap(Point & other) { this->_swap(other); }
};

typedef Point<int,2> PointI;
typedef Point<int,2> Point2I;
typedef Point<int,3> Point3I;
typedef Point<double,2> PointD;
typedef Point<double,2> Point2D;
typedef Point<double,3> Point3D;

template <int N>
Point<double,N> operator+(Point<double,N> const & lhs, Extent<int,N> const & rhs) {
    return lhs + Extent<double,N>(rhs);
}

template <int N>
Point<double,N> operator+(Extent<int,N> const & rhs, Point<double,N> const & lhs) {
    return Point<double,N>(lhs) + rhs;
}

template <int N>
Point<double,N> & operator+=(Point<double,N> & lhs, Extent<int,N> const & rhs) {
    return lhs += Extent<double,N>(rhs);
}

template <int N>
Point<double,N> operator+(Point<int,N> const & lhs, Extent<double,N> const & rhs) {
    return Point<double,N>(lhs) + rhs;
}

template <int N>
Point<double,N> operator-(Point<double,N> const & lhs, Extent<int,N> const & rhs) {
    return lhs - Extent<double,N>(rhs);
}

template <int N>
Point<double,N> & operator-=(Point<double,N> & lhs, Extent<int,N> const & rhs) {
    return lhs -= Extent<double,N>(rhs);
}

template <int N>
Point<double,N> operator-(Point<int,N> const & lhs, Extent<double,N> const & rhs) {
    return Point<double,N>(lhs) - rhs;
}

template <int N>
Extent<double,N> operator-(Point<double,N> const & lhs, Point<int,N> const & rhs) {
    return lhs - Point<double,N>(rhs);
}

template <int N>
Extent<double,N> operator-(Point<int,N> const & lhs, Point<double,N> const & rhs) {
    return Point<double,N>(lhs) - rhs;
}

}}}

#endif

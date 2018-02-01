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

/*
 * A CRTP base class for coordinate objects, providing partial specializations for 2D and 3D.
 */
#ifndef LSST_AFW_GEOM_COORDINATEBASE_H
#define LSST_AFW_GEOM_COORDINATEBASE_H

#include <iostream>
#include <tuple>
#include <utility>

#include "Eigen/Core"

namespace lsst {
namespace afw {
namespace geom {

template <typename T, int N = 2>
class Point;
template <typename T, int N = 2>
class Extent;

/**
 *  A CRTP base class for coordinate objects.
 *
 *  CoordinateBase has partial specializations for 2 and 3 dimensions so its subclasses don't have to.
 */
template <typename Derived, typename T, int N>
class CoordinateBase {
public:
    typedef T Element;
    static int const dimensions = N;
    typedef Eigen::Matrix<T, N, 1, Eigen::DontAlign> EigenVector;

    CoordinateBase(CoordinateBase const&) = default;
    CoordinateBase(CoordinateBase&&) = default;
    CoordinateBase& operator=(CoordinateBase const&) = default;
    CoordinateBase& operator=(CoordinateBase&&) = default;
    ~CoordinateBase() = default;

    T& operator[](int n) { return _vector[n]; }
    T const& operator[](int n) const { return const_cast<EigenVector&>(_vector)[n]; }
    T& coeffRef(int n) { return _vector.coeffRef(n); }
    T const& coeffRef(int n) const { return const_cast<EigenVector&>(_vector).coeffRef(n); }

    /**
     *  Return a fixed-size Eigen representation of the coordinate object.
     *
     *  The fact that this returns by const reference rather than by value should not be considered
     *  part of the API; this is merely an optimization enabled by the implementation.
     */
    EigenVector const& asEigen() const { return _vector; }

protected:
    /**
     *  Initialize all elements to a scalar.
     *
     *  A public constructor with the same signature is expected for subclasses.
     */
    explicit CoordinateBase(T val = static_cast<T>(0)) : _vector(EigenVector::Constant(val)) {}

    /**
     *  Initialize all elements from an N-d Eigen vector.
     *
     *  A public constructor with the same signature is expected for subclasses.
     */
    template <typename Vector>
    explicit CoordinateBase(Eigen::MatrixBase<Vector> const& vector) : _vector(vector) {}

    void _swap(CoordinateBase& other) { _vector.swap(other._vector); }
    EigenVector _vector;
};

/**
 *  Floating-point comparison with tolerance.
 *
 *  Interface, naming, and default tolerances matches Numpy.
 *
 *  @relatesalso CoordinateBase
 */
template <typename Derived, typename T, int N>
bool allclose(CoordinateBase<Derived, T, N> const& a, CoordinateBase<Derived, T, N> const& b,
              T rtol = static_cast<T>(1E-5), T atol = static_cast<T>(1E-8));

/**
 *  Specialization of CoordinateBase for 2 dimensions.
 */
template <typename Derived, typename T>
class CoordinateBase<Derived, T, 2> {
public:
    typedef T Element;
    static int const dimensions = 2;
    typedef Eigen::Matrix<T, 2, 1, Eigen::DontAlign> EigenVector;

    CoordinateBase(CoordinateBase const&) = default;
    CoordinateBase(CoordinateBase&&) = default;
    CoordinateBase& operator=(CoordinateBase const&) = default;
    CoordinateBase& operator=(CoordinateBase&&) = default;
    ~CoordinateBase() = default;

    T& operator[](int n) { return _vector[n]; }
    T const& operator[](int n) const { return const_cast<EigenVector&>(_vector)[n]; }
    T& coeffRef(int n) { return _vector.coeffRef(n); }
    T const& coeffRef(int n) const { return const_cast<EigenVector&>(_vector).coeffRef(n); }

    /**
     *  Return a fixed-size Eigen representation of the coordinate object.
     *
     *  The fact that this returns by const reference rather than by value should not be considered
     *  part of the API; this is merely an optimization enabled by the implementation.
     */
    EigenVector const& asEigen() const { return _vector; }

    T const& getX() const { return _vector.x(); }
    T const& getY() const { return _vector.y(); }
    T& getX() { return _vector.x(); }
    T& getY() { return _vector.y(); }
    void setX(T x) { _vector.x() = x; }
    void setY(T y) { _vector.y() = y; }

    /// Return a std::pair representation of the coordinate object.
    std::pair<T, T> asPair() const { return std::make_pair(_vector.x(), _vector.y()); }

    /// Return a std::tuple representation of the coordinate object.
    std::tuple<T, T> asTuple() const { return std::make_tuple(_vector.x(), _vector.y()); }

protected:
    explicit CoordinateBase(T val = static_cast<T>(0)) : _vector(EigenVector::Constant(val)) {}

    template <typename Vector>
    explicit CoordinateBase(Eigen::MatrixBase<Vector> const& vector) : _vector(vector) {}
    void _swap(CoordinateBase& other) { _vector.swap(other._vector); }
    EigenVector _vector;
};

/**
 *  Specialization of CoordinateBase for 3 dimensions.
 */
template <typename Derived, typename T>
class CoordinateBase<Derived, T, 3> {
public:
    typedef T Element;
    static int const dimensions = 3;
    typedef Eigen::Matrix<T, 3, 1, Eigen::DontAlign> EigenVector;

    CoordinateBase(CoordinateBase const&) = default;
    CoordinateBase(CoordinateBase&&) = default;
    CoordinateBase& operator=(CoordinateBase const&) = default;
    CoordinateBase& operator=(CoordinateBase&&) = default;
    ~CoordinateBase() = default;

    T& operator[](int n) { return _vector[n]; }
    T const& operator[](int n) const { return const_cast<EigenVector&>(_vector)[n]; }
    T& coeffRef(int n) { return _vector.coeffRef(n); }
    T const& coeffRef(int n) const { return const_cast<EigenVector&>(_vector).coeffRef(n); }

    /**
     *  Return a fixed-size Eigen representation of the coordinate object.
     *
     *  The fact that this returns by const reference rather than by value should not be considered
     *  part of the API; this is merely an optimization enabled by the implementation.
     */
    EigenVector const& asEigen() const { return _vector; }

    T const& getX() const { return _vector.x(); }
    T const& getY() const { return _vector.y(); }
    T const& getZ() const { return _vector.z(); }
    T& getX() { return _vector.x(); }
    T& getY() { return _vector.y(); }
    T& getZ() { return _vector.z(); }
    void setX(T x) { _vector.x() = x; }
    void setY(T y) { _vector.y() = y; }
    void setZ(T z) { _vector.z() = z; }

    /// Return a std::tuple representation of the coordinate object.
    std::tuple<T, T, T> asTuple() const { return std::make_tuple(_vector.x(), _vector.y(), _vector.z()); }

protected:
    explicit CoordinateBase(T val = static_cast<T>(0)) : _vector(EigenVector::Constant(val)) {}

    template <typename Vector>
    explicit CoordinateBase(Eigen::MatrixBase<Vector> const& vector) : _vector(vector) {}
    void _swap(CoordinateBase& other) { _vector.swap(other._vector); }
    EigenVector _vector;
};

template <typename Derived, typename T, int N>
std::ostream& operator<<(std::ostream& os, CoordinateBase<Derived, T, N> const& coordinate) {
    os << "(" << coordinate[0];
    for (int n = 1; n < N; ++n) os << ", " << coordinate[n];
    return os << ")";
}
}
}
}

#endif

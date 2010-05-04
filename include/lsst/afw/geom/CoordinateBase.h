// -*- lsst-c++ -*-
/**
 * \file
 * \brief A CRTP base class for coordinate objects, providing partial specializations for 2D and 3D.
 */
#ifndef LSST_AFW_GEOM_COORDINATEBASE_H
#define LSST_AFW_GEOM_COORDINATEBASE_H

#include <utility>

#include <Eigen/Core>
#include <Eigen/Array>
#include <iostream>
#include "boost/tuple/tuple.hpp"

namespace lsst { namespace afw { namespace geom {

template <typename T, int N=2> class Point;
template <typename T, int N=2> class Extent;

/**
 *  \brief A CRTP base class for coordinate objects.
 *
 *  CoordinateBase has partial specializations for 2 and 3 dimensions so its subclasses don't have to.
 */
template <typename Derived, typename T, int N>
class CoordinateBase {
public:
    typedef T Element;
    static int const dimensions = N;
    typedef Eigen::Matrix<T,N,1,Eigen::DontAlign> EigenVector;

#ifndef SWIG
    T & operator[](int n) { return _vector[n]; }
    T const & operator[](int n) const { return const_cast<EigenVector&>(_vector)[n]; }
#endif

    /**
     *  \brief Return a fixed-size Eigen representation of the coordinate object.
     *
     *  The fact that this returns by const reference rather than by value should not be considered
     *  part of the API; this is merely an optimization enabled by the implementation.
     */
    EigenVector const & asVector() const { return _vector; }

protected:

    /**
     *  \brief Initialize all elements to a scalar.
     *  
     *  A public constructor with the same signature is expected for subclasses.
     */
    explicit CoordinateBase(T val = static_cast<T>(0)) : _vector(EigenVector::Constant(val)) {}

    /**
     *  \brief Initialize all elements from an N-d Eigen vector.
     *
     *  A public constructor with the same signature is expected for subclasses.
     */
    template <typename Vector> 
    explicit CoordinateBase(Eigen::MatrixBase<Vector> const & vector) : _vector(vector) {}

    EigenVector _vector;
};

template <typename Derived, typename T, int N>
bool allclose(
    CoordinateBase<Derived,T,N> const & a, CoordinateBase<Derived,T,N> const & b, 
    T rtol = static_cast<T>(1E-5),
    T atol = static_cast<T>(1E-8)
);

template <typename Derived, typename T, int N>
std::ostream & operator<<(std::ostream & os, CoordinateBase<Derived,T,N> const & coordinate) {
    os << "(" << coordinate[0];
    for (int n=1; n<N; ++n) os << ", " << coordinate[n];
    return os << ")";
}

}}}

#endif

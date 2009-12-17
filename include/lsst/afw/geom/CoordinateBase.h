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

/**
 *  \brief Specialization of CoordinateBase for 2 dimensions.
 */
template <typename Derived, typename T>
class CoordinateBase<Derived,T,2> {
public:
    typedef T Element;
    static int const dimensions = 2;
    typedef Eigen::Matrix<T,2,1,Eigen::DontAlign> EigenVector;

#ifndef SWIG
    T & operator[](int n) { return _vector[n]; }
    T const & operator[](int n) const { return const_cast<EigenVector&>(_vector)[n]; }
#endif

    T getX() const { return _vector.x(); }
    T getY() const { return _vector.y(); }
    void setX(T x) { _vector.x() = x; }
    void setY(T y) { _vector.y() = y; }

    /**
     *  \brief Return a fixed-size Eigen representation of the coordinate object.
     *
     *  The fact that this returns by const reference rather than by value should not be considered
     *  part of the API; this is merely an optimization enabled by the implementation.
     */
    EigenVector const & asVector() const { return _vector; }

    /// \brief Return a std::pair representation of the coordinate object.
    std::pair<T,T> asPairXY() const { return std::make_pair(_vector.x(),_vector.y()); }

    /// \brief Return a boost::tuple representation of the coordinate object.
    boost::tuple<T,T> asTupleXY() const { return boost::make_tuple(_vector.x(),_vector.y()); }

    /**
     *  @name Named constructors
     *
     *  While it might be nice to make these true constructors of CoordinateBase subclasses,
     *  that would require either partial specialization of all those subclasses, or the
     *  presence of constructors that would be invalid for certain dimensions.
     *
     *  And while it is slightly more verbose, having a named constructor also explicitly states
     *  that the arguments are (x,y) rather than (y,x).
     */
    //@{
    static Derived makeXY(T x, T y) { return Derived(EigenVector(x, y)); }
    static Derived makeXY(T const xy[2]) { return Derived(EigenVector(xy[0], xy[1])); }
    static Derived makeXY(std::pair<T,T> const & xy) { return Derived(EigenVector(xy.first, xy.second)); }
    static Derived makeXY(boost::tuple<T,T> const & xy) {
        return Derived(EigenVector(xy.template get<0>(), xy.template get<1>()));
    }
    //@}

protected:

    explicit CoordinateBase(T val = static_cast<T>(0)) : _vector(EigenVector::Constant(val)) {}

    template <typename Vector>
    explicit CoordinateBase(Eigen::MatrixBase<Vector> const & vector) : _vector(vector) {}

    EigenVector _vector;
};

/**
 *  \brief Specialization of CoordinateBase for 2 dimensions.
 */
template <typename Derived, typename T>
class CoordinateBase<Derived,T,3> {
public:
    typedef T Element;
    static int const dimensions = 3;
    typedef Eigen::Matrix<T,3,1,Eigen::DontAlign> EigenVector;

#ifndef SWIG
    T & operator[](int n) { return _vector[n]; }
    T const & operator[](int n) const { return const_cast<EigenVector&>(_vector)[n]; }
#endif

    T getX() const { return _vector.x(); }
    T getY() const { return _vector.y(); }
    T getZ() const { return _vector.z(); }
    void setX(T x) { _vector.x() = x; }
    void setY(T y) { _vector.y() = y; }
    void setZ(T z) { _vector.z() = z; }

    /**
     *  \brief Return a fixed-size Eigen representation of the coordinate object.
     *
     *  The fact that this returns by const reference rather than by value should not be considered
     *  part of the API; this is merely an optimization enabled by the implementation.
     */
    EigenVector const & asVector() const { return _vector; }

    /// \brief Return a boost::tuple representation of the coordinate object.
    boost::tuple<T,T,T> asTupleXYZ() const {
        return boost::make_tuple(_vector.x(), _vector.y(), _vector.z()); 
    }

    /**
     *  @name Named constructors
     *
     *  While it might be nice to make these true constructors of CoordinateBase subclasses,
     *  that would require either partial specialization of all those subclasses, or the
     *  presence of constructors that would be invalid for certain dimensions.
     *
     *  And while it is slightly more verbose, having a named constructor also explicitly states
     *  that the arguments are (x,y,z) rather than (z,y,x).
     */
    //@{
    static Derived makeXYZ(T x, T y, T z) { return Derived(EigenVector(x, y, z)); }
    static Derived makeXYZ(T const xyz[3]) { return Derived(EigenVector(xyz[0], xyz[1], xyz[2])); }
    static Derived makeXYZ(boost::tuple<T,T,T> const & xyz) {
        return Derived(EigenVector(xyz.template get<0>(), xyz.template get<1>(), xyz.template get<2>()));
    }
    //@}

protected:

    explicit CoordinateBase(T val = static_cast<T>(0)) : _vector(EigenVector::Constant(val)) {}

    template <typename Vector>
    explicit CoordinateBase(Eigen::MatrixBase<Vector> const & vector) : _vector(vector) {}

    EigenVector _vector;
};

}}}

#endif

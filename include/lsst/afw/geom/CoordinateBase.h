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
    const static int dimensions = N;
    typedef Eigen::Matrix<T,N,1,Eigen::DontAlign> EigenVector;

    T & operator[](int n) { return _vector[n]; }
    T const & operator[](int n) const { return const_cast<EigenVector&>(_vector)[n]; }

    /// Deep copy, or possibly a const reference to an internal.
    EigenVector const & asVector() const { return _vector; }

protected:

    /**
     *  \brief Initialize all elements to a scalar.
     *  
     *  A public constructor with the same signature is expected for subclasses.
     *
     *  It is debatable whether this should be explicit or not (mostly a question for subclasses, since
     *  it is protected here).
     *  The original afw::image::Point object had a non-explicit constructor from scalar.
     */
    explicit CoordinateBase(T val = static_cast<T>(0)) : _vector(EigenVector::Constant(val)) {}

    /**
     *  \brief Initialize all elements from an N-d Eigen vector.
     *
     *  A public constructor with the same signature is expected for subclasses.
     *
     *  It may be prudent to use boost::enable_if here to disable matches with
     *  vectors of the wrong size (or to remove the templating and accept an additional
     *  deep copy).
     */
    template <typename Vector> 
    explicit CoordinateBase(Eigen::MatrixBase<Vector> const & vector) : _vector(vector) {}

    EigenVector _vector;
};

/**
 *  \brief Equality comparison for CoordinateBase.
 *
 *  Unlike other boolean operators, I propose returning a scalar here; testing for equality
 *  generally implies testing for complete equality, not element-wise equality.
 *
 *  Note that while this provides equality comparison for all subclasses of CoordinateBase,
 *  it only allows comparison between objects of the same type, as desired.
 *
 *  \relatesalso CoordinateBase
 */
template <typename Derived, typename T, int N>
inline bool operator==(CoordinateBase<Derived,T,N> const & a, CoordinateBase<Derived,T,N> const & b) {
    return a.asVector() == b.asVector();
}

/**
 *  \brief Inequality comparison for CoordinateBase.
 *
 *  \relatesalso CoordinateBase
 */
template <typename Derived, typename T, int N>
inline bool operator!=(CoordinateBase<Derived,T,N> const & a, CoordinateBase<Derived,T,N> const & b) {
    return a.asVector() == b.asVector();
}

/**
 *  \brief Floating-point comparison with tolerance.
 *  
 *  Interface, naming, and default tolerances matches Numpy; I'd be happy with anything that
 *  accomplishes the same task.
 *
 *  \relatesalso CoordinateBase
 */
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
    const static int dimensions = 2;
    typedef Eigen::Matrix<T,2,1,Eigen::DontAlign> EigenVector;

    T & operator[](int n) { return _vector[n]; }
    T const & operator[](int n) const { return const_cast<EigenVector&>(_vector)[n]; }
    
    T getX() const { return _vector.x(); }
    T getY() const { return _vector.y(); }
    void setX(T x) { _vector.x() = x; }
    void setY(T y) { _vector.y() = y; }

    EigenVector const & asVector() const { return _vector; }
    std::pair<T,T> asPairXY() const { return std::make_pair(_vector.x(),_vector.y()); }
    boost::tuple<T,T> asTupleXY() const { return boost::make_tuple(_vector.x(),_vector.y()); }

    /**
     *  @name Named constructors
     *
     *  While it might be nice to make these true constructors of CoordinateBase subclasses,
     *  that would require either partial specialization of all those subclasses, or the
     *  presence of constructors that would be invalid for certain dimensions.
     *
     *  And while it is slightly more verbose, having a named constructor also explicitly states
     *  that the arguments are (x,y) rather than (y,x), which I consider valuable.
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
    const static int dimensions = 3;
    typedef Eigen::Matrix<T,3,1,Eigen::DontAlign> EigenVector;

    T & operator[](int n) { return _vector[n]; }
    T const & operator[](int n) const { return const_cast<EigenVector&>(_vector)[n]; }
    
    T getX() const { return _vector.x(); }
    T getY() const { return _vector.y(); }
    T getZ() const { return _vector.z(); }
    void setX(T x) { _vector.x() = x; }
    void setY(T y) { _vector.y() = y; }
    void setZ(T z) { _vector.z() = z; }

    EigenVector const & asVector() const { return _vector; }
    boost::tuple<T,T,T> asTupleXYZ() const {
        return boost::make_tuple(_vector.x(), _vector.y(), _vector.z()); 
    }

    static Derived makeXYZ(T x, T y, T z) { return Derived(EigenVector(x, y, z)); }
    static Derived makeXYZ(T const xyz[3]) { return Derived(EigenVector(xyz[0], xyz[1], xyz[2])); }
    static Derived makeXYZ(boost::tuple<T,T,T> const & xyz) {
        return Derived(EigenVector(xyz.template get<0>(), xyz.template get<1>(), xyz.template get<2>()));
    }

protected:

    explicit CoordinateBase(T val = static_cast<T>(0)) : _vector(EigenVector::Constant(val)) {}

    template <typename Vector>
    explicit CoordinateBase(Eigen::MatrixBase<Vector> const & vector) : _vector(vector) {}

    EigenVector _vector;
};

}}}

#endif

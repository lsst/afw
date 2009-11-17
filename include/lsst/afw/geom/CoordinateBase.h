// -*- lsst-c++ -*-
/**
 * \file
 * \brief A CRTP base class for coordinate objects, providing partial specializations for 2D and 3D.
 */
#ifndef LSST_AFW_GEOM_COORDINATEBASE_H
#define LSST_AFW_GEOM_COORDINATEBASE_H

#include <utility>

#include "Eigen/Core"
#include "boost/tuple.hpp"

namespace lsst { namespace afw { namespace geom {

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
    typedef Eigen::Matrix<T,N,1> EigenVector;

    T & operator[](int n);
    T const & operator[](int n) const;

    EigenVector asVector() const; ///< Deep copy, or possibly a const reference to an internal.

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
    explicit CoordinateBase(T val = static_cast<T>(0));

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
    explicit CoordinateBase(Eigen::MatrixBase<Vector> const & vector);

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
template <typename Derived>
bool operator==(CoordinateBase<Derived> const & a, CoordinateBase<Derived> const & b) const;

/**
 *  \brief Inequality comparison for CoordinateBase.
 *
 *  \relatesalso CoordinateBase
 */
template <typename Derived>
bool operator!=(CoordinateBase<Derived> const & a, CoordinateBase<Derived> const & b) const;

/**
 *  \brief Floating-point comparison with tolerance.
 *  
 *  Interface, naming, and default tolerances matches Numpy; I'd be happy with anything that
 *  accomplishes the same task.
 *
 *  \relatesalso CoordinateBase
 */
template <typename Derived>
bool allclose(
    CoordinateBase<Derived> const & a, CoordinateBase<Derived> const & b, 
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
    typedef Eigen::Matrix<T,2,1> EigenVector;

    T & operator[](int n);
    T const & operator[](int n) const;
    
    T getX() const;
    T getY() const;
    void setX(T x);
    void setY(T y);

    EigenVector asVector() const;
    std::pair<T,T> asPairXY() const;
    boost::tuple<T,T> asTupleXY() const;

    /**
     *  @name Named constructors
     *
     *  While it might be nice to make these true constructors of CoordinateBase subclasses,
     *  that would require either partial specialization of all those subclasses, or the
     *  presence of constructors that would be invalid for certain dimensions.
     *
     *  While it is slightly more verbose, having a named constructor also explicitly states
     *  that the arguments are (x,y) rather than (y,x), which I consider valuable.
     */
    //@{
    static Derived makeXY(T x, T y);
    static Derived makeXY(T const xy[2]);
    static Derived makeXY(std::pair<T,T> const & xy);
    static Derived makeXY(boost::tuple<T,T> const & xy);
    //@}

protected:

    explicit CoordinateBase(T val = static_cast<T>(0));

    template <typename Vector>
    explicit CoordinateBase(Eigen::MatrixBase<Vector> const & vector);

};

/**
 *  \brief Specialization of CoordinateBase for 2 dimensions.
 */
template <typename Derived, typename T>
class CoordinateBase<Derived,T,3> {
public:
    typedef T Element;
    const static int dimensions = 3;
    typedef Eigen::Matrix<T,3,1> EigenVector;

    T & operator[](int n);
    T const & operator[](int n) const;
    
    T getX() const;
    T getY() const;
    T getZ() const;
    void setX(T x);
    void setY(T y);
    void setZ(T z);

    EigenVector asVector() const;
    boost::tuple<T,T,T> asTupleXYZ() const;

    static Derived makeXYZ(T x, T y, T z);
    static Derived makeXYZ(T const xyz[3]);
    static Derived makeXYZ(boost::tuple<T,T,T> const & xyz);

protected:

    explicit CoordinateBase(T val = static_cast<T>(0));

    template <typename Vector>
    explicit CoordinateBase(Eigen::MatrixBase<Vector> const & vector);

};

}}}

#endif

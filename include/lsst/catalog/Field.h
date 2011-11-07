// -*- c++ -*-
#ifndef CATALOG_Field_h_INCLUDED
#define CATALOG_Field_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include <cstring>
#include <iostream>

#include "boost/type_traits/is_same.hpp"
#include "boost/mpl/if.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/preprocessor/punctuation/paren.hpp"
#include "Eigen/Core"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/catalog/FieldBase.h"
#include "lsst/catalog/FieldDescription.h"
#include "lsst/catalog/Covariance.h"

#define CATALOG_SCALAR_FIELD_TYPE_N 3
#define CATALOG_SCALAR_FIELD_TYPES              \
    int, float, double
#define CATALOG_SCALAR_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() CATALOG_SCALAR_FIELD_TYPES BOOST_PP_RPAREN()

#define CATALOG_FIELD_TYPE_N 16
#define CATALOG_FIELD_TYPES                     \
    CATALOG_SCALAR_FIELD_TYPES,                 \
    Point<int>, Point<float>, Point<double>,    \
    Shape<float>, Shape<double>,                \
    Array<float>, Array<double>,                \
    Covariance<float>, Covariance<double>,                      \
    Covariance< Point<float> >, Covariance< Point<double> >,    \
    Covariance< Shape<float> >, Covariance< Shape<double> >
#define CATALOG_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() CATALOG_FIELD_TYPES BOOST_PP_RPAREN()

#define FIELD_SIMPLE_PUBLIC_INTERFACE(SIZE)                             \
    /** @brief Standard constructor for a field with static size. */                       \
    Field(char const * name, char const * doc, NullEnum canBeNull=ALLOW_NULL) \
        : FieldBase(name, doc, canBeNull) {}                            \
    /** @brief Construct a typed field from a non-typed FieldBase. */   \
    explicit Field(FieldBase const & base) : FieldBase(base) {}         \
    /** @brief Return the number of elements in a compound or array field. */    \
    int getElementCount() const { return SIZE; }                        \
    /** @brief Return a non-template struct that describes the Field, including its type. */ \
    FieldDescription describe() const {                                 \
        return FieldDescription(this->name, this->doc, this->getTypeString()); \
    }                                                                   \
    /** @brief Return a string description of the field type. */        \
    std::string getTypeString() const

#define FIELD_SIZED_PUBLIC_INTERFACE(SIZE)                              \
    /** @brief Standard constructor for a field with dynamic size. */   \
    Field(int size_, char const * name, char const * doc, NullEnum canBeNull=ALLOW_NULL) \
        : FieldBase(name, doc, canBeNull), size(size_) {}               \
    /** @brief Construct a typed field from a size and a non-typed FieldBase. */   \
    Field(int size_, FieldBase const & base) : FieldBase(base), size(size_) {} \
    /** @brief Return the number of elements in a compound or array field. */    \
    int getElementCount() const { return SIZE; }                        \
    /** @brief Return a non-template struct that describes the Field, including its type. */ \
    FieldDescription describe() const {                                 \
        return FieldDescription(this->name, this->doc, this->getTypeString()); \
    }                                                                   \
    /** @brief Return a string description of the field type. */        \
    std::string getTypeString() const


namespace lsst { namespace catalog {

template <typename T> class Point;
template <typename T> class Shape;
template <typename T> class Array;
template <typename T> class Covariance;

namespace detail {

struct FieldAccess;

} // namespace detail


/**
 *  @brief A simple class that defines and documents a field in a table.
 *
 *  The default Field template should be used for scalar numeric POD types.
 */
template <typename T>
struct Field : public FieldBase {

    /// @brief The type returned for an individual record.
    typedef T Value;

    /// @brief The type of a single element if this were a compound field (which it isn't).
    typedef T Element;

    FIELD_SIMPLE_PUBLIC_INTERFACE(1);

private:

    friend class detail::FieldAccess;

    void setDefault(char * buf) const;

    Value getValue(char * buf) const { return *reinterpret_cast<T*>(buf); }

    void setValue(char * buf, T value) const { *reinterpret_cast<T*>(buf) = value; }
};

/**
 *  @brief A simple class that defines and documents a field in a table.
 *
 *  This specialization is for compound point fields.  It uses lsst::afw::geom::Point2I or Point2D
 *  as a value type.  Note that we use Point2D even with Point<float>, so precision may be lost when
 *  setting a Point<float> field with a Point2D.
 */
template <typename U>
struct Field< Point<U> > : public FieldBase {

    /// @brief The type returned for an individual record.
    typedef afw::geom::Point<
        typename boost::mpl::if_<
            boost::is_same<U,float>,
            double, U
            >::type
        > Value;

    /// @brief The type of a single element of the compound field.
    typedef U Element;

    FIELD_SIMPLE_PUBLIC_INTERFACE(2);

private:

    friend class detail::FieldAccess;

    void setDefault(char * buf) const;

    Value getValue(char * buf) const {
        return Value(*reinterpret_cast<U*>(buf), *(reinterpret_cast<U*>(buf) + 1));
    }

    void setValue(char * buf, Value const & value) const {
        reinterpret_cast<U*>(buf)[0] = value.getX();
        reinterpret_cast<U*>(buf)[1] = value.getY();
    }
};

/**
 *  @brief A simple class that defines and documents a field in a table.
 *
 *  This specialization is for compound shape fields.  It uses lsst::afw::ellipses::Quadrupole
 *  as a value type.  Note that precision may be lost when setting a Shape<float> field with
 *  a Quadrupole.
 */
template <typename U>
struct Field< Shape<U> > : public FieldBase {

    /// @brief The type returned for an individual record.
    typedef afw::geom::ellipses::Quadrupole Value;

    /// @brief The type of a single element of the compound field.
    typedef U Element;

    FIELD_SIMPLE_PUBLIC_INTERFACE(3);

private:

    friend class detail::FieldAccess;

    void setDefault(char * buf) const;

    Value getValue(char * buf) const {
        return Value(
            *reinterpret_cast<U*>(buf), *(reinterpret_cast<U*>(buf) + 1), *(reinterpret_cast<U*>(buf) + 2)
        );
    }

    void setValue(char * buf, Value const & value) const {
        reinterpret_cast<U*>(buf)[0] = value.getIXX();
        reinterpret_cast<U*>(buf)[1] = value.getIYY();
        reinterpret_cast<U*>(buf)[2] = value.getIXY();
    }
};

/**
 *  @brief A simple class that defines and documents a field in a table.
 *
 *  This specialization is for array fields.  The array size is part of the field itself;
 *  different records may not have different array sizes (but the size does not need to be
 *  known at compile time).  Array values for individual records are Eigen::Array objects
 *  (or as an optimization, a const Eigen::Map behaving like and Eigen::Array).  Array
 *  fields can be set with any dense 1-d Eigen object.
 */
template <typename U> 
struct Field< Array<U> > : public FieldBase {

    /// @brief The type returned for an individual record.
    typedef Eigen::Map< const Eigen::Array<U,Eigen::Dynamic,1> > Value;

    /// @brief The type of a single element of the array.
    typedef U Element;

    FIELD_SIZED_PUBLIC_INTERFACE(size);

    /// @brief Number of elements in the array.
    int size;

private:

    friend class detail::FieldAccess;

    void setDefault(char * buf) const;

    Value getValue(char * buf) const { return Value(reinterpret_cast<U*>(buf), size); }

    template <typename Derived>
    void setValue(char * buf, Eigen::DenseBase<Derived> const & value) const {
        BOOST_STATIC_ASSERT( Derived::IsVectorAtCompileTime );
        if (value.size() != size) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthErrorException,
                "Incorrect size in array field assignment."
            );
        }
        for (int i = 0; i < size; ++i) {
             reinterpret_cast<U*>(buf)[i] = value[i];
        }
    }
};

/**
 *  @brief A simple class that defines and documents a field in a table.
 *
 *  This specialization is for covariance matrices associated with array fields.  The size
 *  is part of the field itself; different records may not have different sizes (but the
 *  size does not need to be known at compile time).
 *
 *  The covariance matrix is symmetric, and stored as a pack ed array with
 *  (size * (size+1) / 2) elements.
 *
 *  Covariance values for individual records are Eigen::Matrix objects.  If Eigen
 *  adds support for symmetric packed storage, we may return a Map instead.  Any Eigen
 *  matrix expression can be used to set the field.
 */
template <typename U>
struct Field< Covariance<U> > : public FieldBase {

    /// @brief The type returned for an individual record.
    typedef Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic> Value;

    /// @brief The type of a single element of the covariance matrix.
    typedef U Element;

    FIELD_SIZED_PUBLIC_INTERFACE(detail::computeCovarianceSize(size));

    /// @brief Number of rows or columns of the (square) covariance matrix.
    int size;

private:

    friend class detail::FieldAccess;

    void setDefault(char * buf) const;

    Value getValue(char * buf) const;

    template <typename Derived>
    void setValue(char * buf, Eigen::DenseBase<Derived> const & value) const {
        if (value.rows() != size || value.cols() != size) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthErrorException,
                "Incorrect size in covariance field assignment."
            );
        }
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) { 
                reinterpret_cast<U*>(buf)[detail::indexCovariance(i, j)] = value(i, j);
            }
        }
    }
};

/**
 *  @brief A simple class that defines and documents a field in a table.
 *
 *  This specialization is for covariance matrices associated with point fields.
 *
 *  The covariance matrix is symmetric, and stored as a packed 3-element array.
 *
 *  Covariance values for individual records are Eigen::Matrix objects.  If Eigen
 *  adds support for symmetric packed storage, we may return a Map instead.  Any Eigen
 *  matrix expression can be used to set the field.
 */
template <typename U>
struct Field< Covariance< Point<U> > > : public FieldBase {

    /// @brief The type returned for an individual record.
    typedef Eigen::Matrix<U,2,2> Value;

    /// @brief The type of a single element of the covariance matrix.
    typedef U Element;

    FIELD_SIMPLE_PUBLIC_INTERFACE(3);

private:

    friend class detail::FieldAccess;

    void setDefault(char * buf) const;
;
    Value getValue(char * buf) const;

    template <typename Derived>
    void setValue(char * buf, Eigen::DenseBase<Derived> const & value) const {
        BOOST_STATIC_ASSERT( Derived::RowsAtCompileTime == 2 && Derived::ColsAtCompileTime == 2);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) { 
                reinterpret_cast<U*>(buf)[detail::indexCovariance(i, j)] = value(i, j);
            }
        }
    }
};

/**
 *  @brief A simple class that defines and documents a field in a table.
 *
 *  This specialization is for covariance matrices associated with point fields.
 *
 *  The covariance matrix is symmetric, and stored as a packed 6-element array.
 *
 *  Covariance values for individual records are Eigen::Matrix objects.  If Eigen
 *  adds support for symmetric packed storage, we may return a Map instead.  Any Eigen
 *  matrix expression can be used to set the field.
 */
template <typename U>
struct Field< Covariance< Shape<U> > > : public FieldBase {

    /// @brief The type returned for an individual record.
    typedef Eigen::Matrix<U,3,3> Value;

    /// @brief The type of a single element of the covariance matrix.
    typedef U Element;

    FIELD_SIMPLE_PUBLIC_INTERFACE(6);

private:

    friend class detail::FieldAccess;

    void setDefault(char * buf) const;
;
    Value getValue(char * buf) const;

    template <typename Derived>
    void setValue(char * buf, Eigen::DenseBase<Derived> const & value) const {
        BOOST_STATIC_ASSERT( Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 3);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) { 
                reinterpret_cast<U*>(buf)[detail::indexCovariance(i, j)] = value(i, j);
            }
        }
    }
};

namespace detail {

typedef boost::mpl::vector< CATALOG_SCALAR_FIELD_TYPES > ScalarFieldTypes;
typedef boost::mpl::vector< CATALOG_FIELD_TYPES > FieldTypes;

} // namespace detail
}} // namespace lsst::catalog

#endif // !CATALOG_Field_h_INCLUDED

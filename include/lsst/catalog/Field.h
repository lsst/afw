// -*- c++ -*-
#ifndef CATALOG_Field_h_INCLUDED
#define CATALOG_Field_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include <cstring>
#include <iostream>

#include "boost/mpl/vector.hpp"
#include "boost/preprocessor/punctuation/paren.hpp"
#include "Eigen/Core"

#include "lsst/pex/exceptions.h"
#include "lsst/catalog/FieldBase.h"
#include "lsst/catalog/FieldDescription.h"
#include "lsst/catalog/Point.h"
#include "lsst/catalog/Shape.h"
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
    Field(char const * name, char const * doc, NullEnum canBeNull=ALLOW_NULL) \
        : FieldBase(name, doc, canBeNull) {}                            \
    explicit Field(FieldBase const & base) : FieldBase(base) {}         \
    int getElementCount() const { return SIZE; }                        \
    FieldDescription describe() const {                                 \
        return FieldDescription(this->name, this->doc, this->getTypeString()); \
    }                                                                   \
    std::string getTypeString() const

#define FIELD_SIZED_PUBLIC_INTERFACE(SIZE)                              \
    Field(int size_, char const * name, char const * doc, NullEnum canBeNull=ALLOW_NULL) \
        : FieldBase(name, doc, canBeNull), size(size_) {}               \
    Field(int size_, FieldBase const & base) : FieldBase(base), size(size_) {} \
    int getElementCount() const { return SIZE; }                        \
    FieldDescription describe() const {                                 \
        return FieldDescription(this->name, this->doc, this->getTypeString()); \
    }                                                                   \
    std::string getTypeString() const;                                  \
    int size


namespace lsst { namespace catalog {

template <typename T> class Array;
template <typename T> class Covariance;

namespace detail {

struct FieldAccess;

} // namespace detail

struct NoFieldData {};

template <typename T>
struct Field : public FieldBase {
    typedef T Value;
    typedef T Element;

    FIELD_SIMPLE_PUBLIC_INTERFACE(1);

private:

    friend class detail::FieldAccess;

    void setDefault(char * buf) const;

    Value getValue(char * buf) const { return *reinterpret_cast<T*>(buf); }

    void setValue(char * buf, T value) const { *reinterpret_cast<T*>(buf) = value; }
};

template <typename U>
struct Field< Point<U> > : public FieldBase {
    typedef Point<U> Value;
    typedef U Element;

    FIELD_SIMPLE_PUBLIC_INTERFACE(2);

private:

    friend class detail::FieldAccess;

    void setDefault(char * buf) const;

    Value getValue(char * buf) const {
        return Value(*reinterpret_cast<U*>(buf), *(reinterpret_cast<U*>(buf) + 1));
    }

    void setValue(char * buf, Point<U> const & value) const {
        reinterpret_cast<U*>(buf)[0] = value.x;
        reinterpret_cast<U*>(buf)[1] = value.y;
    }
};

template <typename U>
struct Field< Shape<U> > : public FieldBase {
    typedef Shape<U> Value;
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

    void setValue(char * buf, Shape<U> const & value) const {
        reinterpret_cast<U*>(buf)[0] = value.xx;
        reinterpret_cast<U*>(buf)[1] = value.yy;
        reinterpret_cast<U*>(buf)[2] = value.xy;
    }
};

template <typename U> 
struct Field< Array<U> > : public FieldBase {
    typedef Eigen::Map< const Eigen::Array<U,Eigen::Dynamic,1> > Value;
    typedef U Element;

    FIELD_SIZED_PUBLIC_INTERFACE(size);

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

template <typename U>
struct Field< Covariance<U> > : public FieldBase {
    typedef Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic> Value;
    typedef U Element;

    FIELD_SIZED_PUBLIC_INTERFACE(detail::computeCovarianceSize(size));

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

template <typename U>
struct Field< Covariance< Point<U> > > : public FieldBase {
    typedef Eigen::Matrix<U,2,2> Value;
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

template <typename U>
struct Field< Covariance< Shape<U> > > : public FieldBase {
    typedef Eigen::Matrix<U,3,3> Value;
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

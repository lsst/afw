changecom(`###')dnl
define(`FIELD_BODY_NO_DATA',
`
    typedef $1 Column;

    Field(char const * name, char const * doc, NullEnum canBeNull=ALLOW_NULL)
        : FieldBase(name, doc, canBeNull) {}

    int getByteSize() const { return $2; }
    int getByteAlign() const { return $3; }

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, this->getTypeString());
    }

    std::string getTypeString() const;

private:

    friend class detail::FieldAccess;

    Column getColumn(
        char * buf, int recordCount, int recordSize,
        ndarray::Manager::Ptr const & manager
    ) const;

    void setDefault(char * buf) const;
')dnl
define(`FIELD_BODY_SIZED',
`
    typedef $1 Column;

    Field(int size_, char const * name, char const * doc, NullEnum canBeNull)
        : FieldBase(name, doc, canBeNull), size(size_) {}

    int getByteSize() const { return $2; }
    int getByteAlign() const { return $3; }

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, this->getTypeString());
    }

    std::string getTypeString() const;

    int size;

private:

    friend class detail::FieldAccess;

    Column getColumn(
        char * buf, int recordCount, int recordSize,
        ndarray::Manager::Ptr const & manager
    ) const;

    void setDefault(char * buf) const;
')dnl
#ifndef CATALOG_Field_h_INCLUDED
#define CATALOG_Field_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include <cstring>
#include <iostream>

#include "boost/mpl/vector.hpp"
#include "boost/preprocessor/punctuation/paren.hpp"

#include "lsst/catalog/FieldBase.h"
#include "lsst/catalog/FieldDescription.h"
#include "lsst/catalog/Point.h"
#include "lsst/catalog/Shape.h"
#include "lsst/catalog/Covariance.h"

#define CATALOG_FIELD_TYPE_N 16
#define CATALOG_FIELD_TYPES                     \
    int, float, double,                         \
    Point<int>, Point<float>, Point<double>,    \
    Shape<float>, Shape<double>,                \
    Array<float>, Array<double>,                \
    Covariance<float>, Covariance<double>,                      \
    Covariance< Point<float> >, Covariance< Point<double> >,    \
    Covariance< Shape<float> >, Covariance< Shape<double> >
#define CATALOG_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() CATALOG_FIELD_TYPES BOOST_PP_RPAREN()

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
FIELD_BODY_NO_DATA(`lsst::ndarray::Array<T,1>', `sizeof(T)', `sizeof(T)')
    Value getValue(char * buf) const { return *reinterpret_cast<T*>(buf); }

    void setValue(char * buf, T value) const { *reinterpret_cast<T*>(buf) = value; }
};

template <typename U>
struct Field< Point<U> > : public FieldBase {
    typedef Point<U> Value;
FIELD_BODY_NO_DATA(`Point< lsst::ndarray::Array<U,1> >', `sizeof(U)*2', `sizeof(U)*2')
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
FIELD_BODY_NO_DATA(`Shape< lsst::ndarray::Array<U,1> >', `sizeof(U)*3', `sizeof(U)*2')
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
FIELD_BODY_SIZED(`lsst::ndarray::Array<U,2,1>', `sizeof(U)*size', `sizeof(U)')
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
FIELD_BODY_SIZED(`CovarianceColumn<U>', `sizeof(U)*detail::computePackedSize(size)', `sizeof(U)')
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
FIELD_BODY_NO_DATA(`CovarianceColumn< Point<U> >', `sizeof(U)*3', `sizeof(U)*2');
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
FIELD_BODY_NO_DATA(`CovarianceColumn< Shape<U> >', `sizeof(U)*6', `sizeof(U)*2');
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

typedef boost::mpl::vector< CATALOG_FIELD_TYPES > FieldTypes;

} // namespace detail
}} // namespace lsst::catalog

#endif // !CATALOG_Field_h_INCLUDED

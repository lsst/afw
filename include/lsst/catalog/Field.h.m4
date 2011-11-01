changecom(`###')dnl
define(`FIELD_BODY_NO_DATA',
`
    typedef $1 Column;

    typedef NoFieldData FieldData;

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    int getByteSize() const { return $2; }
    int getByteAlign() const { return $3; }

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, this->getTypeString());
    }

    std::string getTypeString() const;

private:

    template <typename OtherT> friend class Key;
    friend class ColumnView;

    static Column makeColumn(
        void * buf, ndarray::DataOrderEnum order, int recordCount, int recordSize,
        ndarray::Manager::Ptr const & manager, NoFieldData const &
    );

    FieldData getFieldData() const { return FieldData(); }
')dnl
define(`FIELD_BODY_SIZED',
`
    typedef $1 Column;

    typedef int FieldData;

    Field(char const * name, char const * doc, int size_) : FieldBase(name, doc), size(size_) {}

    int getByteSize() const { return $2; }
    int getByteAlign() const { return $3; }

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, this->getTypeString());
    }

    std::string getTypeString() const;

    int size;

private:

    template <typename OtherT> friend class Key;
    friend class ColumnView;

    static Column makeColumn(
        void * buf, ndarray::DataOrderEnum order, int recordCount, int recordSize,
        ndarray::Manager::Ptr const & manager, int size
    );

    FieldData getFieldData() const { return size; }
')dnl
#ifndef CATALOG_Field_h_INCLUDED
#define CATALOG_Field_h_INCLUDED

#include <cstring>
#include <iostream>

#include "boost/mpl/vector.hpp"
#include "boost/preprocessor/punctuation/paren.hpp"

#include "lsst/catalog/Point.h"
#include "lsst/catalog/Shape.h"
#include "lsst/catalog/Array.h"
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

namespace detail {
    class FieldInstantiator;
}

template <typename T> class Key;
class ColumnView;

struct NoFieldData {};

struct FieldDescription {
    char const * name;
    char const * doc;
    std::string type;

    bool operator<(FieldDescription const & other) const {
        return std::strcmp(name, other.name) < 0;
    }

    bool operator==(FieldDescription const & other) const {
        return name == other.name; // okay because these are all string literals
    }

    bool operator!=(FieldDescription const & other) const {
        return name != other.name; // okay because these are all string literals
    }

    friend std::ostream & operator<<(std::ostream & os, FieldDescription const & d) {
        return os << d.name << ": " << d.type << " (" << d.doc << ")";
    }

    FieldDescription(char const * name_, char const * doc_, std::string const & type_) :
        name(name_), doc(doc_), type(type_)
    {}
};

struct FieldBase {

    FieldBase(char const * name_, char const * doc_) : name(name_), doc(doc_) {}

    char const * name;
    char const * doc;
};

template <typename T>
struct Field : public FieldBase {
FIELD_BODY_NO_DATA(`lsst::ndarray::Array<T,1>', `sizeof(T)', `sizeof(T)')
};

template <typename U>
struct Field< Point<U> > : public FieldBase {
FIELD_BODY_NO_DATA(`Point< lsst::ndarray::Array<U,1> >', `sizeof(U)*2', `sizeof(U)*2')
};

template <typename U>
struct Field< Shape<U> > : public FieldBase {
FIELD_BODY_NO_DATA(`Shape< lsst::ndarray::Array<U,1> >', `sizeof(U)*3', `sizeof(U)*2')
};

template <typename U> 
struct Field< Array<U> > : public FieldBase {
FIELD_BODY_SIZED(`lsst::ndarray::Array<U,2,1>', `sizeof(U)*size', `sizeof(U)')
};

template <typename U>
struct Field< Covariance<U> > : public FieldBase {
FIELD_BODY_SIZED(`CovarianceColumn<U>', `sizeof(U)*detail::computePackedSize(size)', `sizeof(U)')
};

template <typename U>
struct Field< Covariance< Point<U> > > : public FieldBase {
FIELD_BODY_NO_DATA(`CovarianceColumn< Point<U> >', `sizeof(U)*3', `sizeof(U)*2');
};

template <typename U>
struct Field< Covariance< Shape<U> > > : public FieldBase {
FIELD_BODY_NO_DATA(`CovarianceColumn< Shape<U> >', `sizeof(U)*6', `sizeof(U)*2');
};

namespace detail {

typedef boost::mpl::vector< CATALOG_FIELD_TYPES > FieldTypes;

} // namespace detail
}} // namespace lsst::catalog

#endif // !CATALOG_Field_h_INCLUDED

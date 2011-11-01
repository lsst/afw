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

    typedef lsst::ndarray::Array<T,1> Column;

    typedef NoFieldData FieldData;

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    int getByteSize() const { return sizeof(T); }
    int getByteAlign() const { return sizeof(T); }

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

};

template <typename U>
struct Field< Point<U> > : public FieldBase {

    typedef Point< lsst::ndarray::Array<U,1> > Column;

    typedef NoFieldData FieldData;

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    int getByteSize() const { return sizeof(U)*2; }
    int getByteAlign() const { return sizeof(U)*2; }

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

};

template <typename U>
struct Field< Shape<U> > : public FieldBase {

    typedef Shape< lsst::ndarray::Array<U,1> > Column;

    typedef NoFieldData FieldData;

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    int getByteSize() const { return sizeof(U)*3; }
    int getByteAlign() const { return sizeof(U)*2; }

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

};

template <typename U> 
struct Field< Array<U> > : public FieldBase {

    typedef lsst::ndarray::Array<U,2,1> Column;

    typedef int FieldData;

    Field(char const * name, char const * doc, int size_) : FieldBase(name, doc), size(size_) {}

    int getByteSize() const { return sizeof(U)*size; }
    int getByteAlign() const { return sizeof(U); }

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

};

template <typename U>
struct Field< Covariance<U> > : public FieldBase {

    typedef CovarianceColumn<U> Column;

    typedef int FieldData;

    Field(char const * name, char const * doc, int size_) : FieldBase(name, doc), size(size_) {}

    int getByteSize() const { return sizeof(U)*detail::computePackedSize(size); }
    int getByteAlign() const { return sizeof(U); }

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

};

template <typename U>
struct Field< Covariance< Point<U> > > : public FieldBase {

    typedef CovarianceColumn< Point<U> > Column;

    typedef NoFieldData FieldData;

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    int getByteSize() const { return sizeof(U)*3; }
    int getByteAlign() const { return sizeof(U)*2; }

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
;
};

template <typename U>
struct Field< Covariance< Shape<U> > > : public FieldBase {

    typedef CovarianceColumn< Shape<U> > Column;

    typedef NoFieldData FieldData;

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    int getByteSize() const { return sizeof(U)*6; }
    int getByteAlign() const { return sizeof(U)*2; }

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
;
};

namespace detail {

typedef boost::mpl::vector< CATALOG_FIELD_TYPES > FieldTypes;

} // namespace detail
}} // namespace lsst::catalog

#endif // !CATALOG_Field_h_INCLUDED

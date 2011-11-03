#ifndef CATALOG_Field_h_INCLUDED
#define CATALOG_Field_h_INCLUDED

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

template <typename T> class Key;
template <typename T> class Array;
template <typename T> class Covariance;
class ColumnView;

struct NoFieldData {};

template <typename T>
struct Field : public FieldBase {
    typedef T Value;

    typedef lsst::ndarray::Array<T,1> Column;

    typedef NoFieldData FieldData;

    Field(char const * name, char const * doc, NullEnum canBeNull=ALLOW_NULL)
        : FieldBase(name, doc, canBeNull) {}

    int getByteSize() const { return sizeof(T); }
    int getByteAlign() const { return sizeof(T); }

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, this->getTypeString());
    }

    std::string getTypeString() const;

private:

    template <typename OtherT> friend class Key;
    friend class ColumnView;

    Field(FieldBase const & base, NoFieldData const &) : FieldBase(base) {}

    static Column makeColumn(
        void * buf, int recordCount, int recordSize,
        ndarray::Manager::Ptr const & manager, NoFieldData const &
    );

    FieldData getFieldData() const { return FieldData(); }

    static Value makeValue(void * buf, NoFieldData const &) {
        return *reinterpret_cast<T*>(buf);
    }
};

template <typename U>
struct Field< Point<U> > : public FieldBase {
    typedef Point<U> Value;

    typedef Point< lsst::ndarray::Array<U,1> > Column;

    typedef NoFieldData FieldData;

    Field(char const * name, char const * doc, NullEnum canBeNull=ALLOW_NULL)
        : FieldBase(name, doc, canBeNull) {}

    int getByteSize() const { return sizeof(U)*2; }
    int getByteAlign() const { return sizeof(U)*2; }

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, this->getTypeString());
    }

    std::string getTypeString() const;

private:

    template <typename OtherT> friend class Key;
    friend class ColumnView;

    Field(FieldBase const & base, NoFieldData const &) : FieldBase(base) {}

    static Column makeColumn(
        void * buf, int recordCount, int recordSize,
        ndarray::Manager::Ptr const & manager, NoFieldData const &
    );

    FieldData getFieldData() const { return FieldData(); }

    static Value makeValue(void * buf, NoFieldData const &) {
        return Value(*reinterpret_cast<U*>(buf), *(reinterpret_cast<U*>(buf) + 1));
    }
};

template <typename U>
struct Field< Shape<U> > : public FieldBase {
    typedef Shape<U> Value;

    typedef Shape< lsst::ndarray::Array<U,1> > Column;

    typedef NoFieldData FieldData;

    Field(char const * name, char const * doc, NullEnum canBeNull=ALLOW_NULL)
        : FieldBase(name, doc, canBeNull) {}

    int getByteSize() const { return sizeof(U)*3; }
    int getByteAlign() const { return sizeof(U)*2; }

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, this->getTypeString());
    }

    std::string getTypeString() const;

private:

    template <typename OtherT> friend class Key;
    friend class ColumnView;

    Field(FieldBase const & base, NoFieldData const &) : FieldBase(base) {}

    static Column makeColumn(
        void * buf, int recordCount, int recordSize,
        ndarray::Manager::Ptr const & manager, NoFieldData const &
    );

    FieldData getFieldData() const { return FieldData(); }

    static Value makeValue(void * buf, NoFieldData const &) {
        return Value(
            *reinterpret_cast<U*>(buf), *(reinterpret_cast<U*>(buf) + 1), *(reinterpret_cast<U*>(buf) + 2)
        );
    }
};

template <typename U> 
struct Field< Array<U> > : public FieldBase {
    typedef Eigen::Map< const Eigen::Array<U,Eigen::Dynamic,1> > Value;

    typedef lsst::ndarray::Array<U,2,1> Column;

    typedef int FieldData;

    Field(int size_, char const * name, char const * doc, NullEnum canBeNull)
        : FieldBase(name, doc, canBeNull), size(size_) {}

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

    Field(FieldBase const & base, int size_) : FieldBase(base), size(size_) {}

    static Column makeColumn(
        void * buf, int recordCount, int recordSize,
        ndarray::Manager::Ptr const & manager, int size
    );

    FieldData getFieldData() const { return size; }

    static Value makeValue(void * buf, int size) {
        return Value(reinterpret_cast<U*>(buf), size);
    }
};

template <typename U>
struct Field< Covariance<U> > : public FieldBase {
    typedef Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic> Value;

    typedef CovarianceColumn<U> Column;

    typedef int FieldData;

    Field(int size_, char const * name, char const * doc, NullEnum canBeNull)
        : FieldBase(name, doc, canBeNull), size(size_) {}

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

    Field(FieldBase const & base, int size_) : FieldBase(base), size(size_) {}

    static Column makeColumn(
        void * buf, int recordCount, int recordSize,
        ndarray::Manager::Ptr const & manager, int size
    );

    FieldData getFieldData() const { return size; }

    static Value makeValue(void * buf, int size);
};

template <typename U>
struct Field< Covariance< Point<U> > > : public FieldBase {
    typedef Eigen::Matrix<U,2,2> Value;

    typedef CovarianceColumn< Point<U> > Column;

    typedef NoFieldData FieldData;

    Field(char const * name, char const * doc, NullEnum canBeNull=ALLOW_NULL)
        : FieldBase(name, doc, canBeNull) {}

    int getByteSize() const { return sizeof(U)*3; }
    int getByteAlign() const { return sizeof(U)*2; }

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, this->getTypeString());
    }

    std::string getTypeString() const;

private:

    template <typename OtherT> friend class Key;
    friend class ColumnView;

    Field(FieldBase const & base, NoFieldData const &) : FieldBase(base) {}

    static Column makeColumn(
        void * buf, int recordCount, int recordSize,
        ndarray::Manager::Ptr const & manager, NoFieldData const &
    );

    FieldData getFieldData() const { return FieldData(); }
;
    static Value makeValue(void * buf, NoFieldData const &);
};

template <typename U>
struct Field< Covariance< Shape<U> > > : public FieldBase {
    typedef Eigen::Matrix<U,3,3> Value;

    typedef CovarianceColumn< Shape<U> > Column;

    typedef NoFieldData FieldData;

    Field(char const * name, char const * doc, NullEnum canBeNull=ALLOW_NULL)
        : FieldBase(name, doc, canBeNull) {}

    int getByteSize() const { return sizeof(U)*6; }
    int getByteAlign() const { return sizeof(U)*2; }

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, this->getTypeString());
    }

    std::string getTypeString() const;

private:

    template <typename OtherT> friend class Key;
    friend class ColumnView;

    Field(FieldBase const & base, NoFieldData const &) : FieldBase(base) {}

    static Column makeColumn(
        void * buf, int recordCount, int recordSize,
        ndarray::Manager::Ptr const & manager, NoFieldData const &
    );

    FieldData getFieldData() const { return FieldData(); }
;
    static Value makeValue(void * buf, NoFieldData const &);
};

namespace detail {

typedef boost::mpl::vector< CATALOG_FIELD_TYPES > FieldTypes;

} // namespace detail
}} // namespace lsst::catalog

#endif // !CATALOG_Field_h_INCLUDED

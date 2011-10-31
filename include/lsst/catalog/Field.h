// -*- c++ -*-
#ifndef CATALOG_Field_h_INCLUDED
#define CATALOG_Field_h_INCLUDED

#include <cstring>
#include <iostream>

#include "boost/mpl/vector.hpp"
#include "boost/format.hpp"

#include "lsst/catalog/Point.h"
#include "lsst/catalog/Shape.h"
#include "lsst/catalog/Array.h"
#include "lsst/catalog/Covariance.h"

namespace lsst { namespace catalog {

template <typename T> class Key;

namespace detail {

template <typename T> struct TypeName;
template <> struct TypeName<int> { static char const * getName() { return "int"; } };
template <> struct TypeName<float> { static char const * getName() { return "float"; } };
template <> struct TypeName<double> { static char const * getName() { return "double"; } };

struct NoFieldData {};

} // namespace detail

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

    typedef lsst::ndarray::Array<T,1,1> Column;

    typedef detail::NoFieldData FieldData;

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, detail::TypeName<T>::getName());
    }

    int getByteSize() const { return sizeof(T); }
    int getByteAlign() const { return sizeof(T); }

private:
    template <typename W> friend class Key;

    static Column makeColumn(
        void * buf, int recordCount, ndarray::Manager::Ptr const & manager, FieldData
    ) {
        return ndarray::detail::ArrayAccess<Column>::construct(
            reinterpret_cast<T*>(buf),
            ndarray::detail::Core<1>::create(
                ndarray::makeVector(recordCount),
                ndarray::makeVector(1),
                manager
            )
        );
    }

    FieldData getFieldData() const { return FieldData(); }
};

template <typename U>
struct Field< Point<U> > : public FieldBase {

    typedef Point< ndarray::Array<U,1> > Column;

    typedef detail::NoFieldData FieldData;

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    FieldDescription describe() const {
        return FieldDescription(
            this->name, this->doc, 
            (boost::format("Point(%s)") % detail::TypeName<U>::getName()).str()
        );
    }

    int getByteSize() const { return sizeof(U) * 2; }
    int getByteAlign() const { return sizeof(U) * 2; }

private:
    template <typename W> friend class Key;

    static Column makeColumn(
        void * buf, int recordCount, ndarray::Manager::Ptr const & manager, FieldData
    ) {
        ndarray::detail::Core<1>::Ptr core = ndarray::detail::Core<1>::create(
            ndarray::makeVector(recordCount),
            ndarray::makeVector(2),
            manager
        );
        return Column(
            ndarray::detail::ArrayAccess< ndarray::Array<U,1> >::construct(reinterpret_cast<U*>(buf), core),
            ndarray::detail::ArrayAccess< ndarray::Array<U,1> >::construct(reinterpret_cast<U*>(buf)+1, core)
        );   
    }

    FieldData getFieldData() const { return FieldData(); }
};

template <typename U>
struct Field< Shape<U> > : public FieldBase {

    typedef Shape< ndarray::Array<U,1> > Column;

    typedef detail::NoFieldData FieldData;

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    FieldDescription describe() const {
        return FieldDescription(
            this->name, this->doc, 
            (boost::format("Shape(%s)") % detail::TypeName<U>::getName()).str()
        );
    }

    int getByteSize() const { return sizeof(U) * 3; }
    int getByteAlign() const { return sizeof(U) * 2; }

private:
    template <typename W> friend class Key;

    static Column makeColumn(
        void * buf, int recordCount, ndarray::Manager::Ptr const & manager, FieldData
    ) {
        ndarray::detail::Core<1>::Ptr core = ndarray::detail::Core<1>::create(
            ndarray::makeVector(recordCount),
            ndarray::makeVector(3),
            manager
        );
        return Column(
            ndarray::detail::ArrayAccess< ndarray::Array<U,1> >::construct(reinterpret_cast<U*>(buf), core),
            ndarray::detail::ArrayAccess< ndarray::Array<U,1> >::construct(reinterpret_cast<U*>(buf)+1, core),
            ndarray::detail::ArrayAccess< ndarray::Array<U,1> >::construct(reinterpret_cast<U*>(buf)+2, core)
        );   
    }

    FieldData getFieldData() const { return FieldData(); }
};

template <typename U> 
struct Field< Array<U> > : public FieldBase {
    
    typedef ndarray::Array<U,2,2> Column;

    typedef int FieldData;

    Field(char const * name, char const * doc, int size_) : FieldBase(name, doc), size(size_) {}

    FieldDescription describe() const {
        return FieldDescription(
            this->name, this->doc, 
            (boost::format("%s[%d]") % detail::TypeName<U>::getName() % size).str()
        );
    }

    int getByteSize() const { return size * sizeof(U); }
    int getByteAlign() const { return sizeof(U) * 2; }

    int size;

private:
    template <typename W> friend class Key;

    static Column makeColumn(
        void * buf, int recordCount, ndarray::Manager::Ptr const & manager, FieldData size
    ) {
        return ndarray::detail::ArrayAccess<Column>::construct(
            reinterpret_cast<U*>(buf),
            ndarray::detail::Core<2>::create(
                ndarray::makeVector(recordCount, size),
                ndarray::makeVector(size, 1),
                manager
            )
        );
    }

    FieldData getFieldData() const { return size; }
};

template <typename U>
struct Field< Covariance<U> > : public FieldBase {

    typedef CovarianceColumn<U> Column;

    typedef int FieldData;

    Field(char const * name, char const * doc, int size_) : FieldBase(name, doc), size(size_) {}

    FieldDescription describe() const {
        return FieldDescription(
            this->name, this->doc, 
            (boost::format("Cov(%s[%d])") % detail::TypeName<U>::getName() % size).str()
        );
    }

    int getByteSize() const { return sizeof(U) * (size * (size + 1)) / 2; }
    int getByteAlign() const { return sizeof(U) * 2; }

    int size;

private:
    template <typename W> friend class Key;

    static Column makeColumn(
        void * buf, int recordCount, ndarray::Manager::Ptr const & manager, FieldData size
    ) {
        return Column(reinterpret_cast<U*>(buf), recordCount, manager, size);
    }

    FieldData getFieldData() const { return size; }
};

template <typename U>
struct Field< Covariance< Point<U> > > : public FieldBase {

    typedef CovarianceColumn< Point<U> > Column;

    typedef detail::NoFieldData FieldData;

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    FieldDescription describe() const {
        return FieldDescription(
            this->name, this->doc,
            (boost::format("Cov(Point(%s))") % detail::TypeName<U>::getName()).str()
        );
    }

    int getByteSize() const { return sizeof(U) * 3; }
    int getByteAlign() const { return sizeof(U) * 2; }

private:
    template <typename W> friend class Key;

    static Column makeColumn(
        void * buf, int recordCount, ndarray::Manager::Ptr const & manager, FieldData
    ) {
        return Column(reinterpret_cast<U*>(buf), recordCount, manager);
    }

    FieldData getFieldData() const { return FieldData(); }
};

template <typename U>
struct Field< Covariance< Shape<U> > > : public FieldBase {

    typedef CovarianceColumn< Shape<U> > Column;

    typedef detail::NoFieldData FieldData;

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    FieldDescription describe() const {
        return FieldDescription(
            this->name, this->doc,
            (boost::format("Cov(Shape(%s))") % detail::TypeName<U>::getName()).str()
        );
    }

    int getByteSize() const { return sizeof(double) * 6; }
    int getByteAlign() const { return sizeof(double) * 2; }

private:
    template <typename W> friend class Key;

    static Column makeColumn(
        void * buf, int recordCount, ndarray::Manager::Ptr const & manager, FieldData
    ) {
        return Column(reinterpret_cast<U*>(buf), recordCount, manager);
    }

    FieldData getFieldData() const { return FieldData(); }
};

namespace detail {

typedef boost::mpl::vector<
    int,
    float,
    double,
    Point<int>,
    Point<float>,
    Point<double>,
    Shape<float>,
    Shape<double>,
    Array<float>,
    Array<double>,
    Covariance<float>,
    Covariance<double>,
    Covariance< Point<float> >,
    Covariance< Point<double> >,
    Covariance< Shape<float> >,
    Covariance< Shape<double> >
> FieldTypes;

} // namespace detail
}} // namespace lsst::catalog

#endif // !CATALOG_Field_h_INCLUDED

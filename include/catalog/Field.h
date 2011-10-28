// -*- c++ -*-
#ifndef CATALOG_Field_h_INCLUDED
#define CATALOG_Field_h_INCLUDED

#include <cstring>
#include <iostream>

#include "boost/mpl/vector.hpp"
#include "boost/format.hpp"

namespace catalog {

namespace detail {

template <typename T> struct TypeName {};
template <> struct TypeName<int> { static char const * getName() { return "int"; } };
template <> struct TypeName<float> { static char const * getName() { return "float"; } };
template <> struct TypeName<double> { static char const * getName() { return "double"; } };

} // namespace detail

struct Point {
    double x;
    double y;
};

struct Shape {
    double xx;
    double yy;
    double xy;
};

template <typename T> struct Vector;
template <typename T> struct Covariance;

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
        os << d.name << ": " << d.type << " (" << d.doc << ")";
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

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, detail::TypeName<T>::getName());
    }

    int getByteSize() const { return sizeof(T); }
    int getByteAlign() const { return sizeof(T); }
};

template <>
struct Field< Point > : public FieldBase {

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, "Point");
    }

    int getByteSize() const { return sizeof(double) * 2; }
    int getByteAlign() const { return sizeof(double) * 2; }
};

template <>
struct Field< Shape > : public FieldBase {

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, "Shape");
    }

    int getByteSize() const { return sizeof(double) * 3; }
    int getByteAlign() const { return sizeof(double) * 2; }
};

template <typename U> 
struct Field< Vector<U> > : public FieldBase {
    
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
};

template <typename U>
struct Field< Covariance<U> > : public FieldBase {
    
    Field(char const * name, char const * doc, int size_) : FieldBase(name, doc), size(size_) {}

    FieldDescription describe() const {
        return FieldDescription(
            this->name, this->doc, 
            (boost::format("Cov(%s[%d])") % detail::TypeName<U>::getName() % size).str()
        );
    }

    int getByteSize() const { return sizeof(U) * (size * (size - 1)) / 2; }
    int getByteAlign() const { return sizeof(U) * 2; }

    int size;
};

template <>
struct Field< Covariance<Point> > : public FieldBase {

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, "Cov(Point)");
    }

    int getByteSize() const { return sizeof(double) * 3; }
    int getByteAlign() const { return sizeof(double) * 2; }
};

template <>
struct Field< Covariance<Shape> > : public FieldBase {

    Field(char const * name, char const * doc) : FieldBase(name, doc) {}

    FieldDescription describe() const {
        return FieldDescription(this->name, this->doc, "Cov(Shape)");
    }

    int getByteSize() const { return sizeof(double) * 6; }
    int getByteAlign() const { return sizeof(double) * 2; }
};

namespace detail {

typedef boost::mpl::vector<
    int,
    float,
    double,
    Point,
    Shape,
    Vector<float>,
    Vector<double>,
    Covariance<float>,
    Covariance<double>,
    Covariance<Point>,
    Covariance<Shape>
> FieldTypes;

} // namespace detail
} // namespace catalog

#endif // !CATALOG_Field_h_INCLUDED

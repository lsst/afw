#ifndef CATALOG_FieldDescription_h_INCLUDED
#define CATALOG_FieldDescription_h_INCLUDED

#include <cstring>
#include <iostream>

namespace lsst { namespace catalog {

struct FieldDescription {
    char const * name;
    char const * doc;
    std::string type;

    bool operator<(FieldDescription const & other) const {
        return std::strcmp(name, other.name) < 0;
    }

    bool operator==(FieldDescription const & other) const {
        return name == other.name; // okay because these are always string literals
    }

    bool operator!=(FieldDescription const & other) const {
        return name != other.name; // okay because these are always string literals
    }

    friend std::ostream & operator<<(std::ostream & os, FieldDescription const & d) {
        return os << d.name << ": " << d.type << " (" << d.doc << ")";
    }

    FieldDescription(char const * name_, char const * doc_, std::string const & type_) :
        name(name_), doc(doc_), type(type_)
    {}
};

}} // namespace lsst::catalog

#endif // !CATALOG_FieldBase_h_INCLUDED

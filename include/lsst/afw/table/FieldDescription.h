// -*- lsst-c++ -*-
#ifndef AFW_TABLE_FieldDescription_h_INCLUDED
#define AFW_TABLE_FieldDescription_h_INCLUDED

#include <cstring>
#include <iostream>

namespace lsst { namespace afw { namespace table {

struct FieldDescription {
    std::string name;
    std::string doc;
    std::string type;

    bool operator<(FieldDescription const & other) const { return name < other.name; }

    bool operator==(FieldDescription const & other) const {
        return name == other.name;
    }

    bool operator!=(FieldDescription const & other) const {
        return name != other.name;
    }

    friend std::ostream & operator<<(std::ostream & os, FieldDescription const & d) {
        return os << d.name << ": " << d.type << " (" << d.doc << ")";
    }

    FieldDescription(std::string name_, std::string doc_, std::string const & type_) :
        name(name_), doc(doc_), type(type_)
    {}

};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_FieldDescription_h_INCLUDED

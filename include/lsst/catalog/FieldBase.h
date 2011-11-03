// -*- c++ -*-
#ifndef CATALOG_FieldBase_h_INCLUDED
#define CATALOG_FieldBase_h_INCLUDED

namespace lsst { namespace catalog {

enum NullEnum { NOT_NULL=0, ALLOW_NULL=1 };

struct FieldBase {

    FieldBase(char const * name_, char const * doc_, NullEnum canBeNull_) : 
        name(name_), doc(doc_), canBeNull(canBeNull_) {}

    char const * name;
    char const * doc;
    NullEnum canBeNull;
};

}} // namespace lsst::catalog

#endif // !CATALOG_FieldBase_h_INCLUDED

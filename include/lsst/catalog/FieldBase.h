#ifndef CATALOG_FieldBase_h_INCLUDED
#define CATALOG_FieldBase_h_INCLUDED

namespace lsst { namespace catalog {

struct FieldBase {

    FieldBase(char const * name_, char const * doc_) : name(name_), doc(doc_) {}

    char const * name;
    char const * doc;
    bool notNull;
};

}} // namespace lsst::catalog

#endif // !CATALOG_FieldBase_h_INCLUDED

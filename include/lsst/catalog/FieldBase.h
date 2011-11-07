// -*- c++ -*-
#ifndef CATALOG_FieldBase_h_INCLUDED
#define CATALOG_FieldBase_h_INCLUDED

namespace lsst { namespace catalog {

/// @brief Enum used to declare whether a field may be null or not.
enum NullEnum { NOT_NULL=0, ALLOW_NULL=1 };

/**
 *  @brief A non-template base class for Field that actually holds most or all of the data for the Field.
 *
 *  It is expected that the name and doc data members will point to string literals, which is why we
 *  haven't used std::string (it's nice for Field to be a relatively lightweight object).
 */
struct FieldBase {

    FieldBase(char const * name_, char const * doc_, NullEnum canBeNull_) : 
        name(name_), doc(doc_), canBeNull(canBeNull_) {}

    char const * name;
    char const * doc;
    NullEnum canBeNull;
};

}} // namespace lsst::catalog

#endif // !CATALOG_FieldBase_h_INCLUDED

// -*- c++ -*-
#ifndef CATALOG_DETAIL_KeyData_h_INCLUDED
#define CATALOG_DETAIL_KeyData_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include "lsst/catalog/Field.h"

namespace lsst { namespace catalog {

template <typename T> class Key;

namespace detail {

template <typename T>
struct KeyData {
    
    Field<T> field;
    int offset;
    int nullOffset;
    int nullMask;

    explicit KeyData(Field<T> const & field_) : field(field_), offset(0), nullOffset(0), nullMask(0) {}
};

struct KeyAccess;

}}} // namespace lsst::catalog::detail

#endif // !CATALOG_DETAIL_KeyData_h_INCLUDED

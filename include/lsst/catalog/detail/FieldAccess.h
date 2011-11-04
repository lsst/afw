// -*- c++ -*-
#ifndef CATALOG_DETAIL_FieldAccess_h_INCLUDED
#define CATALOG_DETAIL_FieldAccess_h_INCLUDED

#include "lsst/catalog/Field.h"

namespace lsst { namespace catalog { namespace detail {

struct FieldAccess {

    template <typename T>
    static typename Field<T>::Column getColumn(
        Field<T> const & field,
        char * buf, int recordCount, int recordSize, 
        ndarray::Manager::Ptr const & manager
    ) {
        return field.getColumn(buf, recordCount, recordSize, manager);
    }

    template <typename T>
    static typename Field<T>::Value getValue(
        Field<T> const & field, char * buf
    ) {
        return field.getValue(buf);
    }

    template <typename T, typename U>
    static void setValue(
        Field<T> const & field, char * buf, U value
    ) {
        return field.setValue(buf, value);
    }

    template <typename T, typename U>
    static void setDefault(
        Field<T> const & field, char * buf
    ) {
        return field.setDefault(buf);
    }

};

}}} // namespace lsst::catalog::detail

#endif // !CATALOG_DETAIL_FieldAccess_h_INCLUDED

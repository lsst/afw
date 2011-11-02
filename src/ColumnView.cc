#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/catalog/ColumnView.h"

namespace lsst { namespace catalog {

template <typename T>
ColumnView::IsNullColumn ColumnView::isNull(Key<T> const & key) const {
    return Field<int>::makeColumn(
        reinterpret_cast<int *>(_buf) + key._nullOffset,
        _recordCount, _layout.getRecordSize(), _manager, NoFieldData()
    ) & key._nullMask;
}

template <typename T>
typename Field<T>::Column ColumnView::operator[](Key<T> const & key) const {
    return Field<T>::makeColumn(
        reinterpret_cast<char *>(_buf) + key._data.first(),
        _recordCount, _layout.getRecordSize(), _manager, key._data.second()
    );
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_COLUMNVIEW(r, data, elem)                       \
    template ColumnView::IsNullColumn ColumnView::isNull(Key< elem > const &) const; \
    template Field< elem >::Column ColumnView::operator[](Key< elem > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_COLUMNVIEW, _,
    BOOST_PP_TUPLE_TO_SEQ(CATALOG_FIELD_TYPE_N, CATALOG_FIELD_TYPE_TUPLE)
)

}} // namespace lsst::catalog

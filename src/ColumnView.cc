#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/catalog/ColumnView.h"
#include "lsst/catalog/detail/KeyAccess.h"
#include "lsst/catalog/detail/FieldAccess.h"

namespace lsst { namespace catalog {

template <typename T>
ColumnView::IsNullColumn ColumnView::isNull(Key<T> const & key) const {
    return ndarray::detail::ArrayAccess< ndarray::Array<int,1> >::construct(
        reinterpret_cast<int *>(_buf) + detail::KeyAccess::getData(key).nullOffset,
        _intCore
    ) & detail::KeyAccess::getData(key).nullMask;
}

template <typename T>
typename Field<T>::Column ColumnView::operator[](Key<T> const & key) const {
    return detail::FieldAccess::getColumn(
        detail::KeyAccess::getData(key).field,
        reinterpret_cast<char *>(_buf) + detail::KeyAccess::getData(key).offset,
        _recordCount, _layout.getRecordSize(), _intCore->getManager()
    );
}

ColumnView::ColumnView(
    Layout const & layout, int recordCount, char * buf, ndarray::Manager::Ptr const & manager
) : _recordCount(recordCount), _buf(buf), _layout(layout),
    _intCore(
        ndarray::detail::Core<1>::create(
            ndarray::makeVector(recordCount),
            ndarray::makeVector(static_cast<int>(layout.getRecordSize() / sizeof(int))),
            manager
        )
    )
{}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_COLUMNVIEW(r, data, elem)                       \
    template ColumnView::IsNullColumn ColumnView::isNull(Key< elem > const &) const; \
    template Field< elem >::Column ColumnView::operator[](Key< elem > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_COLUMNVIEW, _,
    BOOST_PP_TUPLE_TO_SEQ(CATALOG_FIELD_TYPE_N, CATALOG_FIELD_TYPE_TUPLE)
)

}} // namespace lsst::catalog

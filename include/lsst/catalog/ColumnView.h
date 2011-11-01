// -*- c++ -*-
#ifndef CATALOG_ColumnView_h_INCLUDED
#define CATALOG_ColumnView_h_INCLUDED

#include "lsst/catalog/Layout.h"

namespace lsst { namespace catalog {

class ColumnView {
public:
    
    Layout const & getLayout() const { return _layout; }

    template <typename T>
    typename Field<T>::Column operator[](Key<T> const & key) const {
        return Field<T>::makeColumn(
            reinterpret_cast<char *>(_buf) + key._data.first() * _colStride,
            _order, _recordCount, _recordSize, _manager, key._data.second()
        );
    }

    ColumnView copy() const;
    ColumnView copy(ndarraY::DataOrderEnum order) const;

    ndarray::DataOrderEnum getOrder() const;

    static ColumnView allocate(Layout const & layout, int recordCount);

private:
    ndarray::DataOrderEnum _order;
    int _recordCount;
    int _recordSize;
    ndarray::Manager::Ptr _manager;
    void * _buf;
    Layout _layout;
};

}} // namespace lsst::catalog

#endif // !CATALOG_ColumnView_h_INCLUDED

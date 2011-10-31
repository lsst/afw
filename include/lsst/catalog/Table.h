// -*- c++ -*-
#ifndef TABLE_Table_h_INCLUDED
#define TABLE_Table_h_INCLUDED

#include "lsst/catalog/Layout.h"

namespace lsst { namespace catalog {

class Table {
public:
    
    Layout const & getLayout() const { return _layout; }

    template <typename T>
    typename Field<T>::Column operator[](Key<T> const & key) const {
        return key.makeColumn(_buf, _recordCount, _manager);
    }

private:
    Layout _layout;
    int _recordSize;
    int _recordCount;
    ndarray::Manager::Ptr _manager;
    void * _buf;
};

}} // namespace lsst::catalog

#endif // !TABLE_Table_h_INCLUDED

// -*- c++ -*-
#ifndef TABLE_Layout_h_INCLUDED
#define TABLE_Layout_h_INCLUDED

#include "boost/shared_array.hpp"

#include "lsst/catalog/Layout.h"

namespace lsst { namespace catalog {

class Table {
public:
    
    Layout const & getLayout() const { return _layout; }

private:
    Layout _layout;
    int _recordSize;
    int _recordCount;
    ndarray::Manager::Ptr _manager;
    char * _data;
};

}} // namespace lsst::catalog

#endif // !TABLE_Layout_h_INCLUDED

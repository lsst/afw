// -*- c++ -*-
#ifndef CATALOG_Table_h_INCLUDED
#define CATALOG_Table_h_INCLUDED

#include "lsst/catalog/Layout.h"
#include "lsst/catalog/ColumnView.h"

namespace lsst { namespace catalog {

class RecordBase {
    
private:
    void * _buf;
};

class TableBase {
public:
    
    Layout getLayout() const;

    ColumnView consolidate();

private:

    struct Impl;

    boost::shared_ptr<Impl> _storage;

};

}} // namespace lsst::catalog

#endif // !CATALOG_Table_h_INCLUDED

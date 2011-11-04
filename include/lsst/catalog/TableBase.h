// -*- c++ -*-
#ifndef CATALOG_TableBase_h_INCLUDED
#define CATALOG_TableBase_h_INCLUDED

#include "lsst/catalog/Layout.h"
#include "lsst/catalog/ColumnView.h"
#include "lsst/catalog/RecordBase.h"

namespace lsst { namespace catalog {

class TableBase {
public:

    Layout getLayout() const;

    bool isConsolidated() const;

    ColumnView consolidate();

    int getRecordCount() const;

    RecordBase operator[](int index) const;

    void erase(int index);

    RecordBase append(Aux::Ptr const & aux = Aux::Ptr());

    TableBase(Layout const & layout, int defaultBlockSize, int capacity=0);

    TableBase(TableBase const & other) : _storage(other._storage) {}

    ~TableBase() {}

private:
    boost::shared_ptr<detail::TableStorage> _storage;
};

}} // namespace lsst::catalog

#endif // !CATALOG_TableBase_h_INCLUDED

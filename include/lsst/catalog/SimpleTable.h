// -*- c++ -*-
#ifndef CATALOG_SimpleTable_h_INCLUDED
#define CATALOG_SimpleTable_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include "lsst/catalog/Layout.h"
#include "lsst/catalog/ColumnView.h"
#include "lsst/catalog/SimpleRecord.h"

namespace lsst { namespace catalog {

class SimpleTable {
public:

    Layout getLayout() const;

    bool isConsolidated() const;

    ColumnView consolidate();

    int getRecordCount() const;

    SimpleRecord operator[](int index) const;

    void erase(int index);

    SimpleRecord append(Aux::Ptr const & aux = Aux::Ptr());

    SimpleTable(Layout const & layout, int defaultBlockSize, int capacity=0);

    SimpleTable(SimpleTable const & other) : _storage(other._storage) {}

    ~SimpleTable() {}

private:
    boost::shared_ptr<detail::TableStorage> _storage;
};

}} // namespace lsst::catalog

#endif // !CATALOG_SimpleTable_h_INCLUDED

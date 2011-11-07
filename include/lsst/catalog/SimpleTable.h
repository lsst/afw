// -*- c++ -*-
#ifndef CATALOG_SimpleTable_h_INCLUDED
#define CATALOG_SimpleTable_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include "lsst/catalog/Layout.h"
#include "lsst/catalog/ColumnView.h"
#include "lsst/catalog/SimpleRecord.h"

namespace lsst { namespace catalog {

class TableAux {
public:
    typedef boost::shared_ptr<TableAux> Ptr;
    virtual ~TableAux() {}
};

class SimpleTable {
public:

    Layout getLayout() const;

    bool isConsolidated() const;

    ColumnView consolidate();

    int getRecordCount() const;

    SimpleRecord operator[](int index) const;

    void erase(int index);

    SimpleRecord append(RecordAux::Ptr const & aux = RecordAux::Ptr());

    SimpleTable(
        Layout const & layout,
        int defaultBlockSize,
        int capacity,
        TableAux::Ptr const & aux = TableAux::Ptr()
    );

    SimpleTable(
        Layout const & layout,
        int defaultBlockSize,
        TableAux::Ptr const & aux
    );

    SimpleTable(Layout const & layout, int defaultBlockSize);

    SimpleTable(SimpleTable const & other) : _storage(other._storage) {}

    ~SimpleTable() {}

protected:

    TableAux::Ptr getAux() const;

private:
    boost::shared_ptr<detail::TableStorage> _storage;
};

}} // namespace lsst::catalog

#endif // !CATALOG_SimpleTable_h_INCLUDED

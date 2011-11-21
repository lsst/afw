// -*- c++ -*-
#ifndef AFW_TABLE_SimpleTable_h_INCLUDED
#define AFW_TABLE_SimpleTable_h_INCLUDED

#include "lsst/afw/table/detail/fusion_limits.h"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/ColumnView.h"
#include "lsst/afw/table/SimpleRecord.h"

namespace lsst { namespace afw { namespace table {

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

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_SimpleTable_h_INCLUDED

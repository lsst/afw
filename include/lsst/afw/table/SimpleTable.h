// -*- c++ -*-
#ifndef AFW_TABLE_SimpleTable_h_INCLUDED
#define AFW_TABLE_SimpleTable_h_INCLUDED

#include "lsst/afw/table/detail/fusion_limits.h"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/ColumnView.h"
#include "lsst/afw/table/SimpleRecord.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

class TableAux {
public:
    typedef boost::shared_ptr<TableAux> Ptr;
    virtual ~TableAux() {}
};

} // namespace detail

class SimpleTable {
public:

    Layout getLayout() const;

    bool isConsolidated() const;

#if 0
    ColumnView consolidate();
#endif

    int getRecordCount() const;

    SimpleRecord append(detail::RecordAux::Ptr const & aux = detail::RecordAux::Ptr());

    SimpleRecord front() const;

    SimpleRecord back(IteratorTypeEnum iterType=ALL) const;

    SimpleTable(
        Layout const & layout,
        int defaultBlockRecordCount,
        int capacity,
        detail::TableAux::Ptr const & aux = detail::TableAux::Ptr()
    );

    SimpleTable(
        Layout const & layout,
        int defaultBlockRecordCount,
        detail::TableAux::Ptr const & aux
    );

    SimpleTable(Layout const & layout, int defaultBlockRecordCount);

    SimpleTable(SimpleTable const & other) : _storage(other._storage) {}

    ~SimpleTable() {}

protected:

    detail::TableAux::Ptr getAux() const;

private:
    boost::shared_ptr<detail::TableStorage> _storage;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_SimpleTable_h_INCLUDED

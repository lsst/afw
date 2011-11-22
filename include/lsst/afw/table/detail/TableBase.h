// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_TableBase_h_INCLUDED
#define AFW_TABLE_DETAIL_TableBase_h_INCLUDED

#include "lsst/afw/table/detail/fusion_limits.h"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/ColumnView.h"
#include "lsst/afw/table/detail/RecordBase.h"

namespace lsst { namespace afw { namespace table { namespace detail {

class TableAux {
public:
    typedef boost::shared_ptr<TableAux> Ptr;
    virtual ~TableAux() {}
};

class TableBase {
public:

    Layout getLayout() const;

    bool isConsolidated() const;

#if 0
    ColumnView consolidate();
#endif

    int getRecordCount() const;

    ~TableBase() {}

protected:

    TableBase(
        Layout const & layout,
        int defaultBlockRecordCount,
        int capacity,
        TableAux::Ptr const & aux = TableAux::Ptr()
    );

    TableBase(TableBase const & other) : _impl(other._impl) {}

    RecordBase append(RecordAux::Ptr const & aux = RecordAux::Ptr());

    RecordBase front() const;

    RecordBase back(IteratorTypeEnum iterType=ALL_RECORDS) const;

    TableAux::Ptr getAux() const;

private:
    boost::shared_ptr<TableImpl> _impl;
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_TableBase_h_INCLUDED

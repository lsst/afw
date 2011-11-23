// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_TableBase_h_INCLUDED
#define AFW_TABLE_DETAIL_TableBase_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/ColumnView.h"
#include "lsst/afw/table/detail/RecordBase.h"
#include "lsst/afw/table/detail/IteratorBase.h"

namespace lsst { namespace afw { namespace table { namespace detail {

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
        AuxBase::Ptr const & aux = AuxBase::Ptr()
    );

    TableBase(TableBase const & other) : _impl(other._impl) {}

    IteratorBase _begin(IteratorMode mode) const;

    IteratorBase _end(IteratorMode mode) const;

    RecordBase _front() const;

    RecordBase _back(IteratorMode mode) const;

    RecordBase _addRecord(AuxBase::Ptr const & aux = AuxBase::Ptr());

    AuxBase::Ptr getAux() const;

private:
    boost::shared_ptr<TableImpl> _impl;
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_TableBase_h_INCLUDED

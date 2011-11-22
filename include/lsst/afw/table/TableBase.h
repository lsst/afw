// -*- c++ -*-
#ifndef AFW_TABLE_TableBase_h_INCLUDED
#define AFW_TABLE_TableBase_h_INCLUDED

#include "lsst/afw/table/detail/fusion_limits.h"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/ColumnView.h"
#include "lsst/afw/table/RecordBase.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

class TableAux {
public:
    typedef boost::shared_ptr<TableAux> Ptr;
    virtual ~TableAux() {}
};

} // namespace detail

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
        detail::TableAux::Ptr const & aux = detail::TableAux::Ptr()
    );

    TableBase(TableBase const & other) : _impl(other._impl) {}

    RecordBase append(detail::RecordAux::Ptr const & aux = detail::RecordAux::Ptr());

    RecordBase front() const;

    RecordBase back(IteratorTypeEnum iterType=ALL_RECORDS) const;

    detail::TableAux::Ptr getAux() const;

private:
    boost::shared_ptr<detail::TableImpl> _impl;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_TableBase_h_INCLUDED

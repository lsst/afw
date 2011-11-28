// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_TableBase_h_INCLUDED
#define AFW_TABLE_DETAIL_TableBase_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/ColumnView.h"
#include "lsst/afw/table/detail/RecordBase.h"
#include "lsst/afw/table/detail/TreeIteratorBase.h"
#include "lsst/afw/table/detail/SetIteratorBase.h"

namespace lsst { namespace afw { namespace table {

class IdFactory {
public:
    typedef boost::shared_ptr<IdFactory> Ptr;
    virtual RecordId operator()() = 0;
    virtual ~IdFactory() {}
};

namespace detail {

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
        IdFactory::Ptr const & idFactory = IdFactory::Ptr(),
        AuxBase::Ptr const & aux = AuxBase::Ptr()
    );

    TableBase(TableBase const & other) : _impl(other._impl) {}

    SetIteratorBase _erase(SetIteratorBase const & iter);
    TreeIteratorBase _erase(TreeIteratorBase const & iter);
    void _erase(RecordBase const & record) { _erase(record._asSetIterator()); }

    TreeIteratorBase _beginTree(IteratorMode mode) const;
    TreeIteratorBase _endTree(IteratorMode mode) const;

    SetIteratorBase _beginSet() const;
    SetIteratorBase _endSet() const;

    RecordBase _get(RecordId id) const;
    SetIteratorBase _find(RecordId id) const;

    RecordBase _addRecord(AuxBase::Ptr const & aux = AuxBase::Ptr());
    RecordBase _addRecord(RecordId id, AuxBase::Ptr const & aux = AuxBase::Ptr());

    AuxBase::Ptr getAux() const;

private:
    boost::shared_ptr<TableImpl> _impl;
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_TableBase_h_INCLUDED

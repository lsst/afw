// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_TableBase_h_INCLUDED
#define AFW_TABLE_DETAIL_TableBase_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/ColumnView.h"
#include "lsst/afw/table/detail/RecordBase.h"
#include "lsst/afw/table/detail/TreeIteratorBase.h"
#include "lsst/afw/table/detail/IteratorBase.h"
#include "lsst/afw/table/IdFactory.h"

namespace lsst { namespace afw { namespace table { namespace detail {

class TableBase : protected ModificationFlags {
public:

    Layout getLayout() const;

    bool isConsolidated() const;

#if 0
    ColumnView consolidate();
#endif

    int getRecordCount() const;

    void disable(ModificationFlags::Bit n) { unsetBit(n); }
    void makeReadOnly() { unsetAll(); }

    ~TableBase() {}

protected:

    TableBase(
        Layout const & layout,
        int defaultBlockRecordCount,
        int capacity,
        IdFactory::Ptr const & idFactory = IdFactory::Ptr(),
        AuxBase::Ptr const & aux = AuxBase::Ptr(),
        ModificationFlags const & flags = ModificationFlags::all()
    );

    TableBase(TableBase const & other) : ModificationFlags(other), _impl(other._impl) {}

    IteratorBase _unlink(IteratorBase const & iter) const;
    TreeIteratorBase _unlink(TreeIteratorBase const & iter) const;
    void _unlink(RecordBase const & record) const { _unlink(record._asIterator()); }

    TreeIteratorBase _beginTree(TreeMode mode) const;
    TreeIteratorBase _endTree(TreeMode mode) const;

    IteratorBase _begin() const;
    IteratorBase _end() const;

    RecordBase _get(RecordId id) const;
    IteratorBase _find(RecordId id) const;

    RecordBase _addRecord(AuxBase::Ptr const & aux = AuxBase::Ptr()) const;
    RecordBase _addRecord(RecordId id, AuxBase::Ptr const & aux = AuxBase::Ptr()) const;

    AuxBase::Ptr getAux() const;

private:
    boost::shared_ptr<TableImpl> _impl;
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_TableBase_h_INCLUDED

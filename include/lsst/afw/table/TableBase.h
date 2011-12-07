// -*- lsst-c++ -*-
#ifndef AFW_TABLE_TableBase_h_INCLUDED
#define AFW_TABLE_TableBase_h_INCLUDED

#include "lsst/base.h"
#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/ColumnView.h"
#include "lsst/afw/table/RecordBase.h"
#include "lsst/afw/table/TreeIteratorBase.h"
#include "lsst/afw/table/IteratorBase.h"
#include "lsst/afw/table/IdFactory.h"

namespace lsst {

namespace daf { namespace base {

class PropertySet;

}} // namespace daf::base

namespace afw { namespace table {

class LayoutMapper;

class TableBase : protected ModificationFlags {
public:

    Layout getLayout() const;

    bool isConsolidated() const;

    void consolidate(int extraCapacity=0);

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

    void _writeFits(
        std::string const & name,
        CONST_PTR(daf::base::PropertySet) const & metadata = CONST_PTR(daf::base::PropertySet)(),
        std::string const & mode = "w"
    ) const;

    void _writeFits(
        std::string const & name,
        LayoutMapper const & mapper,
        CONST_PTR(daf::base::PropertySet) const & metadata = CONST_PTR(daf::base::PropertySet)(),
        std::string const & mode = "w"
    ) const;

    IteratorBase _unlink(IteratorBase const & iter) const;
    TreeIteratorBase _unlink(TreeIteratorBase const & iter) const;

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
    boost::shared_ptr<detail::TableImpl> _impl;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_TableBase_h_INCLUDED

// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_TreeIteratorBase_h_INCLUDED
#define AFW_TABLE_DETAIL_TreeIteratorBase_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "boost/iterator/iterator_facade.hpp"

#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/detail/RecordData.h"
#include "lsst/afw/table/detail/RecordBase.h"

namespace lsst { namespace afw { namespace table { namespace detail {

struct TableImpl;

class TreeIteratorBase : 
        public boost::iterator_facade<TreeIteratorBase,RecordBase,boost::forward_traversal_tag,RecordBase> 
{
public:

    TreeIteratorBase() : _record(), _mode(ALL_RECORDS) {}

    TreeIteratorBase(
        RecordData * record,
        boost::shared_ptr<TableImpl> const & table, 
        ModificationFlags const & flags,
        TreeMode mode
    ) : _record(record, table, flags), _mode(mode)
    {}

private:

    friend class boost::iterator_core_access;

    RecordBase const & dereference() const {
        assert(_record._data);
        assert(_record._table);
        return _record;
    }

    bool equal(TreeIteratorBase const & other) const { return _record == other._record; }

    void increment();

    RecordBase _record;    
    TreeMode _mode;
};

inline TreeIteratorBase RecordBase::_asTreeIterator(TreeMode mode) const {
    return TreeIteratorBase(_data, _table, *this, mode);
}

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_TreeIteratorBase_h_INCLUDED

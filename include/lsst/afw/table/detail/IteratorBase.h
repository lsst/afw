// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_IteratorBase_h_INCLUDED
#define AFW_TABLE_DETAIL_IteratorBase_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/detail/RecordData.h"
#include "lsst/afw/table/detail/RecordBase.h"

namespace lsst { namespace afw { namespace table { 

enum IteratorMode { NO_CHILDREN, ALL_RECORDS };

namespace detail {

struct TableImpl;

class IteratorBase : 
        public boost::iterator_facade<IteratorBase,RecordBase,boost::forward_traversal_tag,RecordBase> 
{
public:

    IteratorBase() : _record(), _mode(ALL_RECORDS) {}

    IteratorBase(IteratorBase const & other) : _record(other._record), _mode(other._mode) {}

    IteratorBase(RecordData * record, boost::shared_ptr<TableImpl> const & table, IteratorMode mode) : 
        _record(record, table), _mode(mode)
    {}

private:

    void operator=(IteratorBase const & other) { _record = other._record; _mode = other._mode; }

    RecordBase const & dereference() const {
        assert(_record._data);
        assert(_record._table);
        return _record;
    }

    bool equal(IteratorBase const & other) const { return _record == other._record; }

    void increment();

    RecordBase _record;    
    IteratorMode _mode;
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_IteratorBase_h_INCLUDED

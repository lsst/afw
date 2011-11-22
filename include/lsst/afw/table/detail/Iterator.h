// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_Iterator_h_INCLUDED
#define AFW_TABLE_DETAIL_Iterator_h_INCLUDED

#include "lsst/afw/table/detail/fusion_limits.h"

#include "boost/iterator/iterator_facade.hpp"

#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/RecordBase.h"

namespace lsst { namespace afw { namespace table { 

namespace detail {

template <typename RecordT, IteratorTypeEnum type>
class Iterator : public boost::iterator_facade<
    Iterator<RecordT,type>,
    RecordT,
    boost::forward_traversal_tag,
    RecordT
> {
public:

    Iterator(Iterator const & other) : _record(other._record) {}

    template <typename IteratorTypeEnum otherType>
    explicit Iterator(Iterator<RecordT,otherType> const & other) : _record(other._record) {}

    explicit Iterator(RecordT const & record) :
        _record(Access::getRecordData(record)), _table(Access::getRecordTable(record))
    {}

private:

    friend class boost::iterator_core_access;

    template <typename OtherRecordT, IteratorTypeEnum otherType> friend class Iterator;

    RecordT dereference() const {
        return RecordT(RecordBase(_record, _table));
    }

    template <IteratorTypeEnum otherType>
    bool equal(Iterator<RecordT,otherType> const & other) const {
        return other._record == _record && other._table == _table;
    }

    void increment() const {
        if (_record) return;
        if (_record->child && type == ALL_RECORDS) {
            _record = _record->child;
        } else if (_record->sibling || type == NO_CHILDREN) {
            _record = _record->sibling;
        } else if (record->parent != 0) {
            _record = _record->parent->sibling;
        } else {
            _record = 0;
        }
    }

    RecordBase _record;
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_Iterator_h_INCLUDED

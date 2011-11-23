// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_SetIteratorBase_h_INCLUDED
#define AFW_TABLE_DETAIL_SetIteratorBase_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "boost/iterator/iterator_adaptor.hpp"

#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/detail/RecordData.h"
#include "lsst/afw/table/detail/RecordBase.h"

namespace lsst { namespace afw { namespace table { namespace detail {

struct TableImpl;

class SetIteratorBase : 
    public boost::iterator_adaptor<
        SetIteratorBase,     // Derived
        RecordSet::iterator, // Base
        RecordBase,          // Value
        boost::use_default,  // CategoryOrTraversal
        RecordBase           // Reference
    > 
{
public:

    SetIteratorBase() {}

    SetIteratorBase(base_type const & base, boost::shared_ptr<TableImpl> const & table) : 
        SetIteratorBase::iterator_adaptor_(base), _table(table)
    {}

    ~SetIteratorBase();

private:

    friend class boost::iterator_core_access;

    RecordBase dereference() const {
        return RecordBase(&(*base()), _table);
    }

    bool equal(SetIteratorBase const & other) const {
        return base() == other.base() && _table == other._table;
    }

    boost::shared_ptr<TableImpl> _table;
};

inline SetIteratorBase RecordBase::_asSetIterator() const {
    return SetIteratorBase(RecordSet::s_iterator_to(*_data), _table);
}

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_SetIteratorBase_h_INCLUDED

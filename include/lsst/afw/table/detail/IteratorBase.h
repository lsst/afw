// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_IteratorBase_h_INCLUDED
#define AFW_TABLE_DETAIL_IteratorBase_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "boost/iterator/iterator_adaptor.hpp"

#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/detail/RecordData.h"
#include "lsst/afw/table/detail/RecordBase.h"
#include "lsst/afw/table/ModificationFlags.h"

namespace lsst { namespace afw { namespace table { namespace detail {

struct TableImpl;

class IteratorBase : 
    public boost::iterator_adaptor<
        IteratorBase,     // Derived
        RecordSet::iterator, // Base
        RecordBase,          // Value
        boost::use_default,  // CategoryOrTraversal
        RecordBase           // Reference
    >,
    protected ModificationFlags
{
public:

    IteratorBase() {}

    IteratorBase(
        base_type const & base,
        boost::shared_ptr<TableImpl> const & table,
        ModificationFlags const & flags
    ) : IteratorBase::iterator_adaptor_(base), ModificationFlags(flags), _table(table)
    {}

    ~IteratorBase();

private:

    friend class boost::iterator_core_access;
    friend class TableBase;

    RecordBase dereference() const {
        return RecordBase(&(*base()), _table, *this);
    }

    bool equal(IteratorBase const & other) const {
        return base() == other.base() && _table == other._table;
    }

    boost::shared_ptr<TableImpl> _table;
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_IteratorBase_h_INCLUDED

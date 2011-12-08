// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IteratorBase_h_INCLUDED
#define AFW_TABLE_IteratorBase_h_INCLUDED

#include "boost/iterator/iterator_adaptor.hpp"

#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/detail/RecordData.h"
#include "lsst/afw/table/RecordBase.h"
#include "lsst/afw/table/ModificationFlags.h"

namespace lsst { namespace afw { namespace table {

struct TableImpl;

/**
 *  @brief A set-like iterator ordered by record ID.
 *
 *  Because IteratorBase dereferences to RecordBase, it is usually wrapped
 *  with the boost::transform_iterator adapter to return a final record class
 *  in the iterator-returning methods of a final table class.
 */
class IteratorBase : 
    public boost::iterator_adaptor<
        IteratorBase,     // Derived
        detail::RecordSet::iterator, // Base
        RecordBase,          // Value
        boost::use_default,  // CategoryOrTraversal
        RecordBase           // Reference
    >,
    protected ModificationFlags
{
public:

    IteratorBase() {}

    ~IteratorBase();

private:

    friend class boost::iterator_core_access;
    friend class RecordBase;
    friend class TableBase;

    IteratorBase(
        base_type const & base,
        boost::shared_ptr<detail::TableImpl> const & table,
        ModificationFlags const & flags
    ) : IteratorBase::iterator_adaptor_(base), ModificationFlags(flags), _table(table)
    {}

    RecordBase dereference() const {
        return RecordBase(&(*base()), _table, *this);
    }

    bool equal(IteratorBase const & other) const {
        return base() == other.base() && _table == other._table;
    }

    boost::shared_ptr<detail::TableImpl> _table;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_IteratorBase_h_INCLUDED

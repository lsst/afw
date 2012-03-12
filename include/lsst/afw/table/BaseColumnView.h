// -*- lsst-c++ -*-
#ifndef AFW_TABLE_BaseColumnView_h_INCLUDED
#define AFW_TABLE_BaseColumnView_h_INCLUDED

#include "lsst/afw/table/BaseTable.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

/// Functor to compute a flag bit, used to create an ndarray expression template for flag columns.
struct FlagBitExtractor {
    typedef Field<Flag>::Element argument_type;
    typedef bool result_type;

    result_type operator()(argument_type element) const { return element & _mask; }

    explicit FlagBitExtractor(Key<Flag> const & key) : _mask(argument_type(1) << key.getBit()) {}

private:
    argument_type _mask;
};

} // namespace detail

class BaseTable;

/**
 *  @brief Column-wise view into a sequence of records that have been allocated contiguously.
 *
 *  A BaseColumnView can be created from any iterator range that dereferences to records, as long
 *  as those records' field data is contiguous in memory.  In practice, that means they must
 *  have been created from the same table, and be in the same order they were created (with
 *  no deletions).  It also requires that those records be allocated in the same block,
 *  which can be guaranteed with BaseTable::preallocate().
 *
 *  Geometric (point and shape) fields cannot be accessed through a BaseColumnView, but their
 *  scalar components can be.
 *
 *  BaseColumnViews do not allow table data to be modified.
 */
class BaseColumnView {
public:

    /// @brief Return the table that owns the records.
    PTR(BaseTable) getTable() const;

    /// @brief Return the schema that defines the fields.
    Schema getSchema() const { return getTable()->getSchema(); }

    /// @brief Return a 1-d array corresponding to a scalar field (or subfield).
    template <typename T>
    ndarray::Array<T const,1> operator[](Key<T> const & key) const;

    /// @brief Return a 2-d array corresponding to an array field.
    template <typename T>
    ndarray::Array<T const,2,1> operator[](Key< Array<T> > const & key) const;

    /**
     *  @brief Return a 1-d array expression corresponding to a flag bit.
     *
     *  In C++, the return value is a lazy ndarray expression template that performs the bitwise
     *  & operation on every element when that element is requested.  In Python, the result will
     *  be copied into a bool NumPy array.
     */
    ndarray::result_of::vectorize< detail::FlagBitExtractor,
                                   ndarray::Array< Field<Flag>::Element const,1> >::type
    operator[](Key<Flag> const & key) const;

    /**
     *  @brief Construct a BaseColumnView from an iterator range.
     *
     *  The iterators must dereference to a reference or const reference to a record.
     *  If the record data is not contiguous in memory, throws lsst::pex::exceptions::RuntimeErrorException.
     */
    template <typename TableT, typename InputIterator>
    static BaseColumnView make(PTR(TableT) const & table, InputIterator first, InputIterator last);

    ~BaseColumnView();

protected:

    BaseColumnView(
        PTR(BaseTable) const & table, int recordCount, void * buf, ndarray::Manager::Ptr const & manager
    );

private:

    friend class BaseTable;

    struct Impl;

    boost::shared_ptr<Impl> _impl;
};

template <typename TableT, typename InputIterator>
BaseColumnView BaseColumnView::make(PTR(TableT) const & table, InputIterator first, InputIterator last) {
    if (first == last) {
        return BaseColumnView(table, 0, 0, ndarray::Manager::Ptr());
    }
    Schema schema = table->getSchema();
    std::size_t recordSize = schema.getRecordSize();
    std::size_t recordCount = 1;
    void * buf = first->_data;
    ndarray::Manager::Ptr manager = first->_manager;
    char * expected = reinterpret_cast<char*>(buf) + recordSize;
    for (++first; first != last; ++first, ++recordCount, expected += recordSize) {
        if (first->_data != expected || first->_manager != manager) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeErrorException,
                "Record data is not contiguous in memory."
            );
        }
    }
    return BaseColumnView(table, recordCount, buf, manager);
}

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_BaseColumnView_h_INCLUDED

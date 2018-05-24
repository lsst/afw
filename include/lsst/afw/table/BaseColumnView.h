// -*- lsst-c++ -*-
#ifndef AFW_TABLE_BaseColumnView_h_INCLUDED
#define AFW_TABLE_BaseColumnView_h_INCLUDED

#include <cstdint>

#include "lsst/afw/table/BaseTable.h"

namespace lsst {
namespace afw {
namespace table {

namespace detail {

/// Functor to compute a flag bit, used to create an ndarray expression template for flag columns.
struct FlagExtractor {
    typedef Field<Flag>::Element argument_type;
    typedef bool result_type;

    result_type operator()(argument_type element) const { return element & _mask; }

    explicit FlagExtractor(Key<Flag> const& key) : _mask(argument_type(1) << key.getBit()) {}

private:
    argument_type _mask;
};

}  // namespace detail

class BaseTable;

class BaseColumnView;

/**
 *  A packed representation of a collection of Flag field columns.
 *
 *  The packing of bits here is not necessarily the same as the packing using in the actual
 *  table, as the latter may contain more than 64 bits spread across multiple integers.
 *
 *  A BitsColumn can only be constructed by calling BaseColumnView::getBits().
 */
class BitsColumn {
public:
    typedef std::int64_t IntT;

    ndarray::Array<IntT, 1, 1> getArray() const { return _array; }

    IntT getBit(Key<Flag> const& key) const;
    IntT getBit(std::string const& name) const;

    IntT getMask(Key<Flag> const& key) const { return IntT(1) << getBit(key); }
    IntT getMask(std::string const& name) const { return IntT(1) << getBit(name); }

    std::vector<SchemaItem<Flag> > const& getSchemaItems() const { return _items; }

private:
    friend class BaseColumnView;

    explicit BitsColumn(int size);

    ndarray::Array<IntT, 1, 1> _array;
    std::vector<SchemaItem<Flag> > _items;
};

/**
 *  Column-wise view into a sequence of records that have been allocated contiguously.
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
 *  BaseColumnView and its subclasses are always non-const views into a catalog, and so cannot
 *  be obtained from a catalog-of-const (trying this results in an exception, not a compilation
 *  error).  As a result, all its accessors return arrays of non-const elements, even though
 *  they are themselves const member functions.  This is no different from a shared_ptr<T>'s
 *  get() member function returning a non-const T*, even though get() is a const member function.
 */
class BaseColumnView {
public:
    /// Return the table that owns the records.
    std::shared_ptr<BaseTable> getTable() const;

    /// Return the schema that defines the fields.
    Schema getSchema() const { return getTable()->getSchema(); }

    /// Return a 1-d array corresponding to a scalar field (or subfield).
    template <typename T>
    ndarray::ArrayRef<T, 1> const operator[](Key<T> const& key) const;

    /// Return a 2-d array corresponding to an array field.
    template <typename T>
    ndarray::ArrayRef<T, 2, 1> const operator[](Key<Array<T> > const& key) const;

    /**
     *  Return a 1-d array expression corresponding to a flag bit.
     *
     *  In C++, the return value is a lazy ndarray expression template that performs the bitwise
     *  & operation on every element when that element is requested.  In Python, the result will
     *  be copied into a bool NumPy array.
     */
    ndarray::result_of::vectorize<detail::FlagExtractor, ndarray::Array<Field<Flag>::Element const, 1> >::type
    operator[](Key<Flag> const& key) const;

    /**
     *  Return an integer array with the given Flag fields repacked into individual bits.
     *
     *  The returned object contains both the int64 array and accessors to obtain a mask given
     *  a Key or field name.
     *
     *  @throws pex::exceptions::LengthError if keys.size() > 64
     */
    BitsColumn getBits(std::vector<Key<Flag> > const& keys) const;

    /**
     *  Return an integer array with all Flag fields repacked into individual bits.
     *
     *  The returned object contains both the int64 array and accessors to obtain a mask given
     *  a Key or field name.
     *
     *  @throws pex::exceptions::LengthError if the schema has more than 64 Flag fields.
     */
    BitsColumn getAllBits() const;

    /**
     *  Construct a BaseColumnView from an iterator range.
     *
     *  The iterators must dereference to a reference or const reference to a record.
     *  If the record data is not contiguous in memory, throws lsst::pex::exceptions::RuntimeError.
     */
    template <typename InputIterator>
    static BaseColumnView make(std::shared_ptr<BaseTable> const& table, InputIterator first,
                               InputIterator last);

    /**
     *  @brief Return true if the given record iterator range is continuous and the records all belong
     *         to the given table.
     *
     *  This tests exactly the same requiremetns needed to construct a column view, so if this test
     *  succeeds, BaseColumnView::make should as well.
     */
    template <typename InputIterator>
    static bool isRangeContiguous(std::shared_ptr<BaseTable> const& table, InputIterator first,
                                  InputIterator last);

    BaseColumnView(BaseColumnView const&);
    BaseColumnView(BaseColumnView&&);
    BaseColumnView& operator=(BaseColumnView const&);
    BaseColumnView& operator=(BaseColumnView&&);

    ~BaseColumnView();

protected:
    BaseColumnView(std::shared_ptr<BaseTable> const& table, int recordCount, void* buf,
                   ndarray::Manager::Ptr const& manager);

private:
    friend class BaseTable;

    struct Impl;

    std::shared_ptr<Impl> _impl;
};

template <typename RecordT>
class ColumnViewT : public BaseColumnView {
public:
    typedef RecordT Record;
    typedef typename RecordT::Table Table;

    /// @copydoc BaseColumnView::getTable
    std::shared_ptr<Table> getTable() const {
        return std::static_pointer_cast<Table>(BaseColumnView::getTable());
    }

    /// @copydoc BaseColumnView::make
    template <typename InputIterator>
    static ColumnViewT make(std::shared_ptr<Table> const& table, InputIterator first, InputIterator last) {
        return ColumnViewT(BaseColumnView::make(table, first, last));
    }

    ColumnViewT(ColumnViewT const&) = default;
    ColumnViewT(ColumnViewT&&) = default;
    ColumnViewT& operator=(ColumnViewT const&) = default;
    ColumnViewT& operator=(ColumnViewT&&) = default;
    ~ColumnViewT() = default;

protected:
    explicit ColumnViewT(BaseColumnView const& base) : BaseColumnView(base) {}
};

template <typename InputIterator>
BaseColumnView BaseColumnView::make(std::shared_ptr<BaseTable> const& table, InputIterator first,
                                    InputIterator last) {
    if (first == last) {
        return BaseColumnView(table, 0, 0, ndarray::Manager::Ptr());
    }
    Schema schema = table->getSchema();
    std::size_t recordSize = schema.getRecordSize();
    std::size_t recordCount = 1;
    void* buf = first->_data;
    ndarray::Manager::Ptr manager = first->_manager;
    char* expected = reinterpret_cast<char*>(buf) + recordSize;
    for (++first; first != last; ++first, ++recordCount, expected += recordSize) {
        if (first->_data != expected || first->_manager != manager) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                              "Record data is not contiguous in memory.");
        }
    }
    return BaseColumnView(table, recordCount, buf, manager);
}

template <typename InputIterator>
bool BaseColumnView::isRangeContiguous(std::shared_ptr<BaseTable> const& table, InputIterator first,
                                       InputIterator last) {
    if (first == last) return true;
    Schema schema = table->getSchema();
    std::size_t recordSize = schema.getRecordSize();
    std::size_t recordCount = 1;
    void* buf = first->_data;
    ndarray::Manager::Ptr manager = first->_manager;
    char* expected = reinterpret_cast<char*>(buf) + recordSize;
    for (++first; first != last; ++first, ++recordCount, expected += recordSize) {
        if (first->_data != expected || first->_manager != manager) {
            return false;
        }
    }
    return true;
}
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_BaseColumnView_h_INCLUDED

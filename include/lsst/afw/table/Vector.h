// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Vector_h_INCLUDED
#define AFW_TABLE_Vector_h_INCLUDED

#include <vector>

#include "boost/iterator/iterator_adaptor.hpp"

#include "lsst/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/ColumnView.h"
#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/io/FitsReader.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief Iterator class for VectorT.
 *
 *  Iterators dereference to record references or const references, even though the VectorT container
 *  is based on a vector of shared_ptr internally.  This is usually very convenient (and is one of
 *  the reasons for having a custom container class in the first place).
 *
 *  Sometimes, however, we'd like to get a shared_ptr from an iterator (especially because records
 *  are noncopyable).  With that in mind, VectorIterator is implicitly convertible to the shared_ptr
 *  type it holds internally, and can also be assigned a shared_ptr to set the pointer in the 
 *  underlying container.  This conversion makes sense from the perspective that both iterators
 *  and smart pointers mimic the interface of pointers and provide the same interface to get at
 *  the underlying record.
 */
template <typename BaseT>
class VectorIterator
    : public boost::iterator_adaptor<VectorIterator<BaseT>,BaseT,typename BaseT::value_type::element_type>
{
public:

    VectorIterator() {}

    template <typename OtherBaseT>
    VectorIterator(VectorIterator<OtherBaseT> const & other) :
        VectorIterator::iterator_adaptor_(other.base())
    {}

    explicit VectorIterator(BaseT const & base) : VectorIterator::iterator_adaptor_(base) {}

    template <typename RecordT>
    operator PTR(RecordT) () const { return *this->base(); }

    template <typename RecordT>
    VectorIterator & operator=(PTR(RecordT) const & other) const {
        if (other->getTable() != dereference().getTable()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LogicErrorException,
                "Record to assign must be associated with the container's table."
            );
        }
        *this->base() = other;
        return *this;
    }

private:
    friend class boost::iterator_core_access;
    typename BaseT::value_type::element_type & dereference() const { return **this->base(); }
};

/**
 *  @brief A custom container class for records, based on std::vector.
 *
 *  VectorT wraps a std::vector<PTR(RecordT)> in an interface that looks more like a std::vector<RecordT>;
 *  its iterators and accessors return references or const references, rather than pointers, making them
 *  easier to use.  It also holds a table, and requires that all records in the container be allocated
 *  by that table (this is sufficient to also ensure that all records have the same schema).  Because 
 *  a VectorT is holds shared_ptrs internally, many of its operations can be either shallow or deep,
 *  with new deep copies allocated by the vector's table object.  New records can be also be inserted
 *  by pointer (shallow) or by value (deep).
 *
 *  In the future, we may have additional containers (a Set class that sorts by ID, for instance), and
 *  most of the tables library is designed to work with any container class that, like VectorT, has
 *  a getTable() member function and yields references rather than pointers to records.
 *
 *  The constness of records is determined by the constness of the first template
 *  parameter to VectorT; a container instance is always either const or non-const
 *  in that respect (like smart pointers).  Also like smart pointers, const member
 *  functions (and by extension, const_iterators) do not allow the underlying 
 *  pointers to be changed, while non-const member functions and iterators do.
 *
 *  VectorT does not permit empty (null) pointers as elements.  As a result, VectorT has no resize
 *  member function.
 *
 *  VectorT has a very different interface in Python; it mimics Python's list instead of C++'s std::vector.
 *  It is also considerably simpler, because it doesn't deal with iterator ranges or the distinction
 *  between references and shared_ptrs to records.
 */
template <typename RecordT, typename TableT>
class VectorT {
    typedef std::vector<PTR(RecordT)> Internal;
public:

    typedef RecordT Record;
    typedef TableT Table;

    typedef RecordT value_type;
    typedef RecordT & reference;
    typedef PTR(RecordT) pointer;
    typedef typename Internal::size_type size_type;
    typedef typename Internal::difference_type difference_type;
    typedef VectorIterator<typename Internal::iterator> iterator;
    typedef VectorIterator<typename Internal::const_iterator> const_iterator;

    /// @brief Return the table associated with the vector.
    PTR(TableT) getTable() const { return _table; }

    /// @brief Return the schema associated with the vector's table.
    Schema getSchema() const { return _table->getSchema(); }

    /**
     *  @brief Construct a vector from a table (or nothing).
     *
     *  A vector with no table is considered invalid; a valid table must be assigned to it
     *  before it can be used.
     */
    explicit VectorT(PTR(TableT) const & table = PTR(TableT)()) : _table(table), _internal() {}

    /// @brief Construct a vector from a schema, creating a table with TableT::make(schema).
    explicit VectorT(Schema const & schema) : _table(TableT::make(schema)), _internal() {}

    /**
     *  @brief Construct a vector from a table and an iterator range.
     *
     *  If deep is true, new records will be created using table->copyRecord before being inserted.
     *  If deep is false, records will be not be copied, but they must already be associated with
     *  the given table.  The table itself is never deep-copied.
     *
     *  The iterator must dereference to a record reference or const reference rather than a pointer,
     *  but should be implicitly convertible to a record pointer as well (see VectorIterator).
     *
     *  If InputIterator models RandomAccessIterator (according to std::iterator_traits) and deep
     *  is true, table->preallocate will be used to ensure that the resulting records are
     *  contiguous in memory and can be used with ColumnView.  To ensure this is the case for
     *  other iterator types, the user must preallocate the table manually.
     */
    template <typename InputIterator>
    VectorT(PTR(TableT) const & table, InputIterator first, InputIterator last, bool deep=false) :
        _table(table), _internal()
    {
        insert(first, last, deep);
    }

    /// Shallow copy constructor.
    VectorT(VectorT const & other) : _table(other._table), _internal(other._internal) {}

    /**
     *  @brief Shallow copy constructor from a container containing a related record type.
     *
     *  This conversion only succeeds if OtherRecordT is convertible to RecordT and OtherTableT is
     *  convertible to TableT.
     */
    template <typename OtherRecordT, typename OtherTableT>
    VectorT(VectorT<OtherRecordT,OtherTableT> const & other) :
        _table(other.getTable()), _internal(other.begin().base(), other.end().base())
    {}

    /// Shallow assigment.
    VectorT & operator=(VectorT const & other) {
        if (&other != this) {
            _table = other._table;
            _internal = other._internal;
        }
        return *this;
    }

    /// Write a FITS binary table.
    void writeFits(std::string const & filename) const {
        io::FitsWriter::apply(filename, *this);
    }

    /// Read a FITS binary table.
    static VectorT readFits(std::string const & filename) {
        return io::FitsReader::apply<VectorT>(filename);
    }

    /**
     *  @brief Return a ColumnView of this vector's records.
     *
     *  Will throw RuntimeErrorException if records are not contiguous.
     */
    ColumnView getColumnView() const { return ColumnView::make(begin(), end()); }

    //@{
    /**
     *  Iterator access.
     *
     *  @sa VectorIterator
     */
    iterator begin() { return iterator(_internal.begin()); }
    iterator end() { return iterator(_internal.end()); }
    const_iterator begin() const { return const_iterator(_internal.begin()); }
    const_iterator end() const { return const_iterator(_internal.end()); }
    //@}

    /// Return true if the vector has no records.
    bool empty() const { return _internal.empty(); }

    /// Return the number of elements in the vector.
    size_type size() const { return _internal.size(); }

    /// Return the maximum number of elements allowed in a vector.
    size_type max_size() const { return _internal.max_size(); }

    /// Return the capacity of the internal vector; this is unrelated to the space available in the table.
    size_type capacity() const { return _internal.capacity(); }

    /// Increase the capacity of the internal vector to the given size.  This does not affect the table.
    void reserve(size_type n) { _internal.reserve(n); }

    /// Return the record at index i.
    reference operator[](size_type i) const { return *_internal[i]; }

    /// Return the record at index i (throws std::out_of_range).
    reference at(size_type i) const { return *_internal.at(i); }

    /// Return the first record.
    reference front() const { return *_internal.front(); }

    /// Return the last record.
    reference back() const { return *_internal.back(); }

    /// Return a pointer to the record at index i.
    PTR(RecordT) const get(size_type i) const { return _internal[i]; }

    /// Set the record at index i to a pointer.
    void set(size_type i, PTR(RecordT) const & p) {
        if (p->getTable() != _table) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LogicErrorException,
                "Record to assign must be associated with the container's table."
            );
        }
        _internal[i] = p;
    }

    /**
     *  @brief Replace the contents of the table with an iterator range.
     *
     *  Delegates to insert(); look there for more information.
     */
    template <typename InputIterator>
    void assign(InputIterator first, InputIterator last, bool deep=false) {
        clear();
        insert(end(), first, last, deep);
    }

    /// @brief Add a copy of the given record to the end of the vector.
    void push_back(Record const & r) {
        PTR(RecordT) p = _table->copyRecord(r);
        _internal.push_back(p);
    }

    /// @brief Add the given record to the end of the vector without copying.
    void push_back(PTR(RecordT) const & p) {
        if (p->getTable() != _table) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LogicErrorException,
                "Record to append must be associated with the container's table."
            );
        }
        _internal.push_back(p);
    }

    /// @brief Create a new record, add it to the end of the vector, and return a pointer to it.
    PTR(RecordT) addNew() {
        PTR(RecordT) r = _table->makeRecord();
        _internal.push_back(r);
        return r;
    }

    /// @brief Remove the last record in the vector
    void pop_back() { _internal.pop_back(); }

    /**
     *  @brief Insert an iterator range into the table.
     *
     *  InputIterator must dereference to a record reference that is convertible to the record type
     *  held by the vector, and must be implicitly convertible to a shared_ptr to a record.
     *
     *  If deep is true, new records will be created by calling copyRecord on the vector's table.
     *  If deep is false, the new records will not be copied, but they must have been created
     *  with the vector's table (not that a table may be shared by multiple vectors).
     *
     *  If InputIterator models RandomAccessIterator (according to std::iterator_traits) and deep
     *  is true, table->preallocate will be used to ensure that the resulting records are
     *  contiguous in memory and can be used with ColumnView.  To ensure this is the case for
     *  other iterator types, the user must preallocate the table manually.
     */
    template <typename InputIterator>
    void insert(iterator pos, InputIterator first, InputIterator last, bool deep=false) {
        _insert(pos, first, last, deep, (typename std::iterator_traits<InputIterator>::iterator_category*)0);
    }

    /// Insert a copy of the given record at the given position.
    iterator insert(iterator pos, Record const & r) {
        PTR(RecordT) p = _table->copyRecord(r);
        return iterator(_internal.insert(pos.base(), p));
    }

    /// Insert the given record at the given position without copying.
    iterator insert(iterator pos, PTR(RecordT) const & p) {
        if (p->getTable() != _table) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LogicErrorException,
                "Record to insert must be associated with the container's table."
            );
        }
        return iterator(_internal.insert(pos.base(), p));
    }

    /// Erase the record pointed to by pos, and return an iterator the next record.
    iterator erase(iterator pos) { return iterator(_internal.erase(pos.base())); }

    /// Erase the records in the range [first, last).
    iterator erase(iterator first, iterator last) {
        return iterator(_internal.erase(first.base(), last.base()));
    }

    /// Shallow swap of two vectors.
    void swap(VectorT & other) {
        _table.swap(other._table);
        _internal.swap(other._internal);
    }

    /// Remove all records from the vector.
    void clear() { _internal.clear(); }
    
private:

    template <typename InputIterator>
    void _insert(
        iterator pos, InputIterator first, InputIterator last, bool deep,
        std::random_access_iterator_tag *
    ) {
        _internal.reserve(_internal.size() + last - first);
        _table->preallocate(last - first);
        _insert(pos, first, last, deep, (std::input_iterator_tag *)0);
    }

    template <typename InputIterator>
    void _insert(
        iterator pos, InputIterator first, InputIterator last, bool deep,
        std::input_iterator_tag *
    ) {
        if (deep) {
            while (first != last) {
                pos = insert(pos, *first);
                ++pos;
                ++first;
            }
        } else {
            while (first != last) {
                pos = insert(pos, first);
                ++pos;
                ++first;
            }
        }
    }

    PTR(TableT) _table;
    Internal _internal;
};

typedef VectorT<BaseRecord,BaseTable> BaseVector;
typedef VectorT<BaseRecord const,BaseTable> ConstBaseVector;

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Vector_h_INCLUDED

// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Catalog_h_INCLUDED
#define AFW_TABLE_Catalog_h_INCLUDED

#include <vector>

#include "boost/iterator/iterator_adaptor.hpp"
#include "boost/iterator/transform_iterator.hpp"

#include "lsst/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/io/FitsReader.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/SchemaMapper.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief Iterator class for CatalogT.
 *
 *  Iterators dereference to record references or const references, even though the CatalogT container
 *  is based on a vector of shared_ptr internally.  This is usually very convenient (and is one of
 *  the reasons for having a custom container class in the first place).
 *
 *  Sometimes, however, we'd like to get a shared_ptr from an iterator (especially because records
 *  are noncopyable).  With that in mind, CatalogIterator is implicitly convertible to the shared_ptr
 *  type it holds internally, and can also be assigned a shared_ptr to set the pointer in the 
 *  underlying container.  This conversion makes sense from the perspective that both iterators
 *  and smart pointers mimic the interface of pointers and provide the same interface to get at
 *  the underlying record.
 */
template <typename BaseT>
class CatalogIterator
    : public boost::iterator_adaptor<CatalogIterator<BaseT>,BaseT,typename BaseT::value_type::element_type>
{
public:

    CatalogIterator() {}

    template <typename OtherBaseT>
    CatalogIterator(CatalogIterator<OtherBaseT> const & other) :
        CatalogIterator::iterator_adaptor_(other.base())
    {}

    explicit CatalogIterator(BaseT const & base) : CatalogIterator::iterator_adaptor_(base) {}

    template <typename RecordT>
    operator PTR(RecordT) () const { return *this->base(); }

    template <typename RecordT>
    CatalogIterator & operator=(PTR(RecordT) const & other) const {
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
 *  CatalogT wraps a std::vector<PTR(RecordT)> in an interface that looks more like a std::vector<RecordT>;
 *  its iterators and accessors return references or const references, rather than pointers, making them
 *  easier to use.  It also holds a table, and requires that all records in the container be allocated
 *  by that table (this is sufficient to also ensure that all records have the same schema).  Because 
 *  a CatalogT is holds shared_ptrs internally, many of its operations can be either shallow or deep,
 *  with new deep copies allocated by the catalog's table object.  New records can be also be inserted
 *  by pointer (shallow) or by value (deep).
 *
 *  In the future, we may have additional containers, and
 *  most of the tables library is designed to work with any container class that, like CatalogT, has
 *  a getTable() member function and yields references rather than pointers to records.
 *
 *  The constness of records is determined by the constness of the first template
 *  parameter to CatalogT; a container instance is always either const or non-const
 *  in that respect (like smart pointers).  Also like smart pointers, const member
 *  functions (and by extension, const_iterators) do not allow the underlying 
 *  pointers to be changed, while non-const member functions and iterators do.
 *
 *  CatalogT does not permit empty (null) pointers as elements.  As a result, CatalogT has no resize
 *  member function.
 *
 *  CatalogT has a very different interface in Python; it mimics Python's list instead of C++'s std::vector.
 *  It is also considerably simpler, because it doesn't need to deal with iterator ranges or the distinction
 *  between references and shared_ptrs to records.  See the Python docstring for more information.
 */
template <typename RecordT>
class CatalogT {
    typedef std::vector<PTR(RecordT)> Internal;
public:

    typedef RecordT Record;
    typedef typename Record::Table Table;
    typedef typename Record::ColumnView ColumnView;

    typedef RecordT value_type;
    typedef RecordT & reference;
    typedef PTR(RecordT) pointer;
    typedef typename Internal::size_type size_type;
    typedef typename Internal::difference_type difference_type;
    typedef CatalogIterator<typename Internal::iterator> iterator;
    typedef CatalogIterator<typename Internal::const_iterator> const_iterator;

    /// @brief Return the table associated with the catalog.
    PTR(Table) getTable() const { return _table; }

    /// @brief Return the schema associated with the catalog's table.
    Schema getSchema() const { return _table->getSchema(); }

    /**
     *  @brief Construct a catalog from a table (or nothing).
     *
     *  A catalog with no table is considered invalid; a valid table must be assigned to it
     *  before it can be used.
     */
    explicit CatalogT(PTR(Table) const & table = PTR(Table)()) : _table(table), _internal() {}

    /// @brief Construct a catalog from a schema, creating a table with Table::make(schema).
    explicit CatalogT(Schema const & schema) : _table(Table::make(schema)), _internal() {}

    /**
     *  @brief Construct a catalog from a table and an iterator range.
     *
     *  If deep is true, new records will be created using table->copyRecord before being inserted.
     *  If deep is false, records will be not be copied, but they must already be associated with
     *  the given table.  The table itself is never deep-copied.
     *
     *  The iterator must dereference to a record reference or const reference rather than a pointer,
     *  but should be implicitly convertible to a record pointer as well (see CatalogIterator).
     *
     *  If InputIterator models RandomAccessIterator (according to std::iterator_traits) and deep
     *  is true, table->preallocate will be used to ensure that the resulting records are
     *  contiguous in memory and can be used with ColumnView.  To ensure this is the case for
     *  other iterator types, the user must preallocate the table manually.
     */
    template <typename InputIterator>
    CatalogT(PTR(Table) const & table, InputIterator first, InputIterator last, bool deep=false) :
        _table(table), _internal()
    {
        insert(end(), first, last, deep);
    }

    /// Shallow copy constructor.
    CatalogT(CatalogT const & other) : _table(other._table), _internal(other._internal) {}

    /**
     *  @brief Shallow copy constructor from a container containing a related record type.
     *
     *  This conversion only succeeds if OtherRecordT is convertible to RecordT and OtherTable is
     *  convertible to Table.
     */
    template <typename OtherRecordT>
    CatalogT(CatalogT<OtherRecordT> const & other) :
        _table(other.getTable()), _internal(other.begin().base(), other.end().base())
    {}

    /// Shallow assigment.
    CatalogT & operator=(CatalogT const & other) {
        if (&other != this) {
            _table = other._table;
            _internal = other._internal;
        }
        return *this;
    }

    /**
     * @brief Returns a shallow copy of a subset of this Catalog.  The arguments
     * correspond to python's slice() syntax.
     */
    CatalogT<RecordT> subset(std::ptrdiff_t startd, std::ptrdiff_t stopd, std::ptrdiff_t step) const {
        /* Python's slicing syntax is weird and wonderful.
         
         Both the "start" and "stop" indices can be negative, which means the
         abs() of the index less than the size; [-1] means the last item.
         Moreover, it's possible to have a negative index less than -len(); it
         will get clipped.  That is in fact one way to slice *backward* through
         the array *and* include element 0;

         >>> range(10)[5:-20:-1]
         [5, 4, 3, 2, 1, 0]

         The clipping tests in this function look more complicated than they
         need to be, but that is partly because there are some weird edge cases.

         Also, ptrdiff_t vs size_t introduces some annoying complexity.  Note
         that the args are "startd"/"stopd" (not "start"/"stop").

         There is a fairly complete set of tests in tests/ticket2026.py; if you
         try to simplify this function, be sure they continue to pass.
         */
        size_type S = size();
        size_type start, stop = 0;
        // Python doesn't allow step == 0
        if (step == 0) {
            throw LSST_EXCEPT(
                pex::exceptions::InvalidParameterException,
                "Step cannot be zero"
            );
        }
        // Basic negative indexing rule: first add size
        if (startd < 0) {
            startd += S;
        }
        if (stopd  < 0) {
            stopd  += S;
        }
        // Start gets clipped to zero; stop does not (yet).
        if (startd < 0) {
            startd = 0;
        }
        // Now start is non-negative, so can cast to size_t.
        start = (size_type)startd;
        if (start > S) {
            start = S;
        }
        if (step > 0) {
            // When stepping forward, stop gets clipped at zero,
            // so is non-negative and can get cast to size_t.
            if (stopd < 0) {
                stopd = 0;
            }
            stop = (size_type)stopd;
            if (stop > S) {
                stop = S;
            }
        } else if (step < 0) {
            // When stepping backward, stop gets clipped at -1 so that slices
            // including 0 are possible.
            if (stopd < 0) {
                stopd = -1;
            }
        }

        if (((step > 0) && (start >= stop)) ||
            ((step < 0) && ((std::ptrdiff_t)start <= stopd))) {
            // Empty range
            return CatalogT<RecordT>(getTable(), begin(), begin());
        }

        if (step == 1) {
            // Use the iterator-based constructor for this simple case
            assert(start >= 0);
            assert(stop  >  0);
            assert(start <  S);
            assert(stop  <= S);
            return CatalogT<RecordT>(getTable(), begin()+start, begin()+stop);
        }

        // Build a new CatalogT and copy records into it.
        CatalogT<RecordT> cat(getTable());
        size_type N = 0;
        if (step >= 0) {
            N = (stop - start) / step + (((stop - start) % step) ? 1 : 0);
        } else {
            N = (size_t)((stopd - (std::ptrdiff_t)start) / step +
                         (((stopd - (std::ptrdiff_t)start) % step) ? 1 : 0));
        }
        cat.reserve(N);
        if (step >= 0) {
            for (size_type i=start; i<stop; i+=step) {
                cat.push_back(get(i));
            }
        } else {
            for (std::ptrdiff_t i=(std::ptrdiff_t)start; i>stopd; i+=step) {
                cat.push_back(get(i));
            }
        }
        return cat;
    }

    /// Write a FITS binary table to a regular file.
    void writeFits(std::string const & filename, std::string const & mode="w") const {
        io::FitsWriter::apply(filename, mode, *this);
    }
    /// Write a FITS binary table to a RAM file.
    void writeFits(fits::MemFileManager & manager, std::string const & mode="w") const {
        io::FitsWriter::apply(manager, mode, *this);
    }
    /// Write a FITS binary table to an open file object.
    void writeFits(fits::Fits & fitsfile, PTR(io::OutputArchive) archive = PTR(io::OutputArchive)()) const {
        io::FitsWriter::apply(fitsfile, *this, archive);
    }

    /// @brief Read a FITS binary table from a regular file.
    static CatalogT readFits(std::string const & filename, int hdu=0) {
        return io::FitsReader::apply<CatalogT>(filename, hdu);
    }
    /// @brief Read a FITS binary table from a RAM file.
    static CatalogT readFits(fits::MemFileManager & manager, int hdu=0) {
        return io::FitsReader::apply<CatalogT>(manager, hdu);
    }
    /**
     *  @brief Read a FITS binary table from a file object already at the correct extension.
     *
     *  If non-null, nested Persistables will be read from the given archive instead of loading
     *  a new archive from the HDUs following the catalog.
     */
    static CatalogT readFits(
        fits::Fits & fitsfile,
        PTR(io::InputArchive) archive = PTR(io::InputArchive)()
    ) {
        return io::FitsReader::apply<CatalogT>(fitsfile, archive);
    }

    /**
     *  @brief Return a ColumnView of this catalog's records.
     *
     *  Will throw RuntimeErrorException if records are not contiguous.
     *
     *  Note that this is a 
     */
    ColumnView getColumnView() const {
        if (boost::is_const<RecordT>::value) {
            throw LSST_EXCEPT(
                pex::exceptions::LogicErrorException,
                "Cannot get a column view from a CatalogT<RecordT const> (as column views are always "
                "non-const views)."
            );
        }
        return ColumnView::make(_table, begin(), end());
    }

    /// @brief Return true if all records are contiguous.
    bool isContiguous() const { return ColumnView::isRangeContiguous(_table, begin(), end()); }

    //@{
    /**
     *  Iterator access.
     *
     *  @sa CatalogIterator
     */
    iterator begin() { return iterator(_internal.begin()); }
    iterator end() { return iterator(_internal.end()); }
    const_iterator begin() const { return const_iterator(_internal.begin()); }
    const_iterator end() const { return const_iterator(_internal.end()); }
    //@}

    /// Return true if the catalog has no records.
    bool empty() const { return _internal.empty(); }

    /// Return the number of elements in the catalog.
    size_type size() const { return _internal.size(); }

    /// Return the maximum number of elements allowed in a catalog.
    size_type max_size() const { return _internal.max_size(); }

    /**
     *  @brief Return the capacity of the catalog.
     *
     *  This is computed as the sum of the current size and the unallocated space in the table.  It
     *  does not reflect the size of the internal vector, and hence cannot be used to judge when
     *  iterators may be invalidated.
     */
    size_type capacity() const { return _internal.size() + _table->getBufferSize(); }

    /**
     *  @brief Increase the capacity of the catalog to the given size.
     *
     *  This can be used to guarantee that the catalog will be contiguous, but it only affects
     *  records constructed after reserve().
     */
    void reserve(size_type n) {
        if (n <= _internal.size()) return;
        _table->preallocate(n - _internal.size());
    }

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

    /// @brief Add a copy of the given record to the end of the catalog.
    void push_back(Record const & r) {
        PTR(RecordT) p = _table->copyRecord(r);
        _internal.push_back(p);
    }

    /// @brief Add the given record to the end of the catalog without copying.
    void push_back(PTR(RecordT) const & p) {
        if (p->getTable() != _table) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LogicErrorException,
                "Record to append must be associated with the container's table."
            );
        }
        _internal.push_back(p);
    }

    /// @brief Create a new record, add it to the end of the catalog, and return a pointer to it.
    PTR(RecordT) addNew() {
        PTR(RecordT) r = _table->makeRecord();
        _internal.push_back(r);
        return r;
    }

    /// @brief Remove the last record in the catalog
    void pop_back() { _internal.pop_back(); }

    /// @brief Deep-copy the catalog using a cloned table.
    CatalogT copy() const { return CatalogT(getTable()->clone(), begin(), end(), true); }

    /**
     *  @brief Insert an iterator range into the table.
     *
     *  InputIterator must dereference to a record reference that is convertible to the record type
     *  held by the catalog, and must be implicitly convertible to a shared_ptr to a record.
     *
     *  If deep is true, new records will be created by calling copyRecord on the catalog's table.
     *  If deep is false, the new records will not be copied, but they must have been created
     *  with the catalog's table (note that a table may be shared by multiple catalogs).
     *
     *  If InputIterator models RandomAccessIterator (according to std::iterator_traits) and deep
     *  is true, table->preallocate will be used to ensure that the resulting records are
     *  contiguous in memory and can be used with ColumnView.  To ensure this is the case for
     *  other iterator types, the user must preallocate the table manually.
     */
    template <typename InputIterator>
    void insert(iterator pos, InputIterator first, InputIterator last, bool deep=false) {
        _maybeReserve(
            pos, first, last, deep, (typename std::iterator_traits<InputIterator>::iterator_category*)0
        );
        if (deep) {
            while (first != last) {
                pos = insert(pos, *first);
                ++pos;
                ++first;
            }
        } else {
            while (first != last) {
                pos = insert(pos, first);
                assert(pos != end());
                ++pos;
                ++first;
            }
        }
    }

    /// @brief Insert a range of records into the catalog by copying them with a SchemaMapper.
    template <typename InputIterator>
    void insert(SchemaMapper const & mapper, iterator pos, InputIterator first, InputIterator last) {
        if (!_table->getSchema().contains(mapper.getOutputSchema())) {
            throw LSST_EXCEPT(
                pex::exceptions::InvalidParameterException,
                "SchemaMapper's output schema does not match catalog's schema"
            );
        }
        _maybeReserve(
            pos, first, last, true, (typename std::iterator_traits<InputIterator>::iterator_category*)0
        );
        while (first != last) {
            pos = insert(pos, _table->copyRecord(*first, mapper));
            ++pos;
            ++first;
        }
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

    /// Shallow swap of two catalogs.
    void swap(CatalogT & other) {
        _table.swap(other._table);
        _internal.swap(other._internal);
    }

    /// Remove all records from the catalog.
    void clear() { _internal.clear(); }
    
    /// @brief Return true if the catalog is in ascending order according to the given key.
    template <typename T>
    bool isSorted(Key<T> const & key) const;

    /**
     *  @brief Return true if the catalog is in ascending order according to the given predicate.
     *
     *  cmp(a, b) should return true if record a is less than record b, and false otherwise.
     */
    template <typename Compare>
    bool isSorted(Compare cmp) const;
    
    /// @brief Sort the catalog in-place by the field with the given key.
    template <typename T>
    void sort(Key<T> const & key);
    
    /**
     *  @brief Sort the catalog in-place by the field with the given predicate.
     *
     *  cmp(a, b) should return true if record a is less than record b, and false otherwise.
     */
    template <typename Compare>
    void sort(Compare cmp);

    //@{
    /**
     *  @brief Return an iterator to the record with the given value.
     *
     *  @note The catalog must be sorted in ascending order according to the given key
     *        before calling find (i.e. isSorted(key) must be true).
     *
     *  Returns end() if the Record cannot be found.
     */
    template <typename T>
    iterator find(typename Field<T>::Value const & value, Key<T> const & key);

    template <typename T>
    const_iterator find(typename Field<T>::Value const & value, Key<T> const & key) const;
    //@}

private:

    template <typename InputIterator>
    void _maybeReserve(
        iterator & pos, InputIterator first, InputIterator last, bool deep,
        std::random_access_iterator_tag *
    ) {
        std::ptrdiff_t n = pos - begin();
        _internal.reserve(_internal.size() + last - first);
        pos = begin() + n;
        if (deep) _table->preallocate(last - first);
    }

    template <typename InputIterator>
    void _maybeReserve(
        iterator pos, InputIterator first, InputIterator last, bool deep,
        std::input_iterator_tag *
    ) {}

    PTR(Table) _table;
    Internal _internal;
};

namespace detail {

template <typename RecordT, typename T>
struct KeyComparisonFunctor {

    bool operator()(RecordT const & a, RecordT const & b) const { return a.get(key) < b.get(key); }

    Key<T> key;
};

template <typename RecordT, typename Adaptee>
struct ComparisonAdaptor {

    bool operator()(PTR(RecordT) const & a, PTR(RecordT) const & b) const {
        return adaptee(*a, *b);
    }

    Adaptee adaptee;
};

template <typename RecordT, typename T>
struct KeyExtractionFunctor {

    typedef typename Field<T>::Value result_type;

    result_type operator()(RecordT const & r) const { return r.get(key); }

    Key<T> key;
};

} // namespace detail

template <typename RecordT>
template <typename Compare>
bool CatalogT<RecordT>::isSorted(Compare cmp) const {
    detail::ComparisonAdaptor<RecordT,Compare> f = { cmp };
    if (empty()) return true;
    const_iterator last = this->begin();
    const_iterator i = last; ++i;
    for (; i != this->end(); ++i) {
        if (f(i, last)) return false;
        last = i;
    }
    return true;
}

template <typename RecordT>
template <typename Compare>
void CatalogT<RecordT>::sort(Compare cmp) {
    detail::ComparisonAdaptor<RecordT,Compare> f = { cmp };
    std::sort(_internal.begin(), _internal.end(), f);
}

template <typename RecordT>
template <typename T>
bool CatalogT<RecordT>::isSorted(Key<T> const & key) const {
    detail::KeyComparisonFunctor<RecordT,T> f = { key };
    return isSorted(f);
}

template <typename RecordT>
template <typename T>
void CatalogT<RecordT>::sort(Key<T> const & key) {
    detail::KeyComparisonFunctor<RecordT,T> f = { key };
    return sort(f);
}

template <typename RecordT>
template <typename T>
typename CatalogT<RecordT>::iterator
CatalogT<RecordT>::find(typename Field<T>::Value const & value, Key<T> const & key) {
    detail::KeyExtractionFunctor<RecordT,T> f = { key };
    // Iterator adaptor that makes a CatalogT iterator work like an iterator over field values.
    typedef boost::transform_iterator<detail::KeyExtractionFunctor<RecordT,T>,iterator> SearchIter;
    // Use binary search for log n search; requires sorted table.
    SearchIter i = std::lower_bound(SearchIter(begin(), f), SearchIter(end(), f), value);
    if (i.base() == end() || *i != value) return end();
    return i.base();
}

template <typename RecordT>
template <typename T>
typename CatalogT<RecordT>::const_iterator
CatalogT<RecordT>::find(typename Field<T>::Value const & value, Key<T> const & key) const {
    detail::KeyExtractionFunctor<RecordT,T> f = { key };
    // Iterator adaptor that makes a CatalogT iterator work like an iterator over field values.
    typedef boost::transform_iterator<detail::KeyExtractionFunctor<RecordT,T>,const_iterator> SearchIter;
    // Use binary search for log n search; requires sorted table.
    SearchIter i = std::lower_bound(SearchIter(begin(), f), SearchIter(end(), f), value);
    if (i.base() == end() || *i != value) return end();
    return i.base();
}

typedef CatalogT<BaseRecord> BaseCatalog;
typedef CatalogT<BaseRecord const> ConstBaseCatalog;

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Catalog_h_INCLUDED

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

template <typename RecordT, typename TableT>
class Vector {
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

    PTR(TableT) getTable() const { return _table; }

    Schema const getSchema() const { return _table->getSchema(); }

    explicit Vector(PTR(TableT) const & table = PTR(TableT)()) : _table(table), _internal() {}

    explicit Vector(Schema const & schema) : _table(TableT::make(schema)), _internal() {}

    template <typename InputIterator>
    Vector(PTR(TableT) const & table, InputIterator first, InputIterator last, bool deep=false) :
        _table(table), _internal()
    {
        insert(first, last, deep);
    }

    Vector(Vector const & other) : _table(other._table), _internal(other._internal) {}

    template <typename OtherRecordT, typename OtherTableT>
    Vector(Vector<OtherRecordT,OtherTableT> const & other) :
        _table(other.getTable()), _internal(other.begin().base(), other.end().base())
    {}

    Vector & operator=(Vector const & other) {
        if (&other != this) {
            _table = other._table;
            _internal = other._internal;
        }
        return *this;
    }

    void writeFits(std::string const & filename) const {
        io::FitsWriter::apply(filename, *this);
    }

    static Vector readFits(std::string const & filename) {
        return io::FitsReader::apply<Vector>(filename);
    }

    ColumnView getColumnView() const { return ColumnView::make(begin(), end()); }

    iterator begin() { return iterator(_internal.begin()); }
    iterator end() { return iterator(_internal.end()); }

    const_iterator begin() const { return const_iterator(_internal.begin()); }
    const_iterator end() const { return const_iterator(_internal.end()); }

    bool empty() const { return _internal.empty(); }
    size_type size() const { return _internal.size(); }
    size_type max_size() const { return _internal.max_size(); }

    void resize(size_type n) { _internal.resize(n); }
    size_type capacity() const { return _internal.capacity(); }
    void reserve(size_type n) { _table->preallocate(n - size()); _internal.reserve(n); }

    reference operator[](size_type i) const { return *_internal[i]; }
    reference at(size_type i) const { return *_internal.at(i); }
    reference front() const { return *_internal.front(); }
    reference back() const { return *_internal.back(); }

    PTR(RecordT) const get(size_type i) const { return _internal[i]; }

    void set(size_type i, PTR(RecordT) const & p) {
        if (p->getTable() != _table) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LogicErrorException,
                "Record to assign must be associated with the container's table."
            );
        }
        _internal[i] = p;
    }

    template <typename InputIterator>
    void assign(InputIterator first, InputIterator last, bool deep=false) {
        clear();
        insert(end(), first, last, deep);
    }

    void push_back(Record const & r) {
        PTR(RecordT) p = _table->copyRecord(r);
        _internal.push_back(p);
    }

    void push_back(PTR(RecordT) const & p) {
        if (p->getTable() != _table) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LogicErrorException,
                "Record to append must be associated with the container's table."
            );
        }
        _internal.push_back(p);
    }

    PTR(RecordT) addNew() {
        PTR(RecordT) r = _table->makeRecord();
        _internal.push_back(r);
        return r;
    }

    void pop_back() { _internal.pop_back(); }

    iterator insert(iterator pos, Record const & r) {
        PTR(RecordT) p = _table->copyRecord(r);
        return iterator(_internal.insert(pos.base(), p));
    }

    iterator insert(iterator pos, PTR(RecordT) const & p) {
        if (p->getTable() != _table) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LogicErrorException,
                "Record to insert must be associated with the container's table."
            );
        }
        return iterator(_internal.insert(pos.base(), p));
    }

    template <typename InputIterator>
    void insert(iterator pos, InputIterator first, InputIterator last, bool deep=false) {
        _insert(pos, first, last, deep, (typename std::iterator_traits<InputIterator>::iterator_category*)0);
    }

    iterator erase(iterator pos) { return iterator(_internal.erase(pos.base())); }

    iterator erase(iterator first, iterator last) {
        return iterator(_internal.erase(first.base(), last.base()));
    }

    void swap(Vector & other) {
        _table.swap(other._table);
        _internal.swap(other._internal);
    }

    void clear() { _internal.clear(); }
    
private:

    template <typename InputIterator>
    void _insert(
        iterator pos, InputIterator first, InputIterator last, bool deep,
        std::random_access_iterator_tag *
    ) {
        _internal.reserve(_internal.size() + last - first);
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
                ++first;
            }
        } else {
            while (first != last) {
                pos = insert(pos, first);
                ++first;
            }
        }
    }

    PTR(TableT) _table;
    Internal _internal;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Vector_h_INCLUDED

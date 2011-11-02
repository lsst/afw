// -*- c++ -*-
#ifndef CATALOG_Table_h_INCLUDED
#define CATALOG_Table_h_INCLUDED

#include "lsst/catalog/Layout.h"
#include "lsst/catalog/ColumnView.h"

namespace lsst { namespace catalog {

namespace detail {

class TableStorage;

} // namespace detail

class NoAux {};

template <typename Aux = NoAux>
class Record {
    
    Layout getLayout() const;

    template <typename T>
    bool isNull(Key<T> const & key) const;
    
    template <typename T>
    typename Field<T>::Value get(Key<T> const & key) const;

    template <typename T, typename U>
    void set(Key<T> const & key, U const & value) const;

    template <typename T>
    void unset(Key<T> const & key) const;

    Aux & getAux() const { return _data.second(); }

private:

    template <typename OtherAux> friend class Iterator;
    template <typename OtherAux> friend class Table;

    boost::compressed_pair<void*,Aux> _data;
    boost::shared_ptr<detail::TableStorage> _storage;
};

template <typename Aux = NoAux, typename RecordT = Record<Aux> >
class Table {
public:

    Layout getLayout() const;

    RecordT operator[](int index) const;

    RecordT append();

    ColumnView consolidate();

private:
    
    boost::shared_ptr<detail::TableStorage> _storage;

};

}} // namespace lsst::catalog

#endif // !CATALOG_Table_h_INCLUDED

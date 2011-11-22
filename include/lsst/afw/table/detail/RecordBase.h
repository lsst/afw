// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_RecordBase_h_INCLUDED
#define AFW_TABLE_DETAIL_RecordBase_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/detail/RecordData.h"

namespace lsst { namespace afw { namespace table { namespace detail {

struct TableImpl;
class TableBase;

class RecordBase {
public:

    Layout getLayout() const;

    RecordId getId() const { return _data->id; }

    template <typename T> 
    typename Field<T>::Reference operator[](Key<T> const & key) const {
        return Access::getReference(key, _data);
    }
    
    template <typename T>
    typename Field<T>::Value get(Key<T> const & key) const {
        return Access::getValue(key, _data);
    }

    template <typename T, typename U>
    void set(Key<T> const & key, U const & value) const {
        Access::setValue(key, _data, value);
    }

    bool operator==(RecordBase const & other) const {
        return _data == other._data && _table == other._table;
    }

    bool operator!=(RecordBase const & other) const {
        return !this->operator==(other);
    }

    bool iterateInPlace(IteratorTypeEnum iterType);

    ~RecordBase();

protected:

    RecordAux::Ptr getAux() const { return _data->aux; }

    RecordBase(RecordBase const & other)
        : _data(other._data), _table(other._table) {}

    void operator=(RecordBase const & other) {
        _data = other._data;
        _table = other._table;
    }

private:

    friend class TableBase;
    friend class IteratorBase;

    RecordBase() : _data(0), _table() {}

    RecordBase(
        RecordData * data,
        boost::shared_ptr<TableImpl> const & storage
    ) : _data(data), _table(storage)
    {}

    RecordData * _data;
    boost::shared_ptr<TableImpl> _table;
};
  
}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_RecordBase_h_INCLUDED

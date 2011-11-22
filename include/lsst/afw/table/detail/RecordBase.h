// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_RecordBase_h_INCLUDED
#define AFW_TABLE_DETAIL_RecordBase_h_INCLUDED

#include "lsst/afw/table/detail/fusion_limits.h"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/detail/RecordData.h"

namespace lsst { namespace afw { namespace table { namespace detail {

struct TableImpl;
class TableBase;
template <typename RecordT, IteratorTypeEnum type> class Iterator;

class RecordBase {
public:

    typedef RecordData::IdType IdType;

    Layout getLayout() const;

    IdType getId() const { return _data->id; }

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

    ~RecordBase();

protected:

    RecordBase(RecordBase const & other)
        : _data(other._data), _table(other._table) {}

    RecordBase & operator=(RecordBase const & other) {
        _data = other._data;
        _table = other._table;
        return *this;
    }

    RecordAux::Ptr getAux() const { return _data->aux; }

private:

    friend class TableBase;
    template <typename RecordT, IteratorTypeEnum type> friend class Iterator;

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

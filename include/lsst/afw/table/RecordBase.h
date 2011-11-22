// -*- c++ -*-
#ifndef AFW_TABLE_RecordBase_h_INCLUDED
#define AFW_TABLE_RecordBase_h_INCLUDED

#include "lsst/afw/table/detail/fusion_limits.h"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/detail/RecordData.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

struct TableImpl;
template <typename RecordT, IteratorTypeEnum type> class Iterator;

} // namespace detail

class TableBase;

class RecordBase {
public:

    typedef detail::RecordData::IdType IdType;

    Layout getLayout() const;

    IdType getId() const { return _data->id; }

    template <typename T> 
    typename Field<T>::Reference operator[](Key<T> const & key) const {
        return detail::Access::getReference(key, _data);
    }
    
    template <typename T>
    typename Field<T>::Value get(Key<T> const & key) const {
        return detail::Access::getValue(key, _data);
    }

    template <typename T, typename U>
    void set(Key<T> const & key, U const & value) const {
        detail::Access::setValue(key, _data, value);
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

    detail::RecordAux::Ptr getAux() const { return _data->aux; }

private:

    friend class TableBase;
    template <typename RecordT, IteratorTypeEnum type> friend class detail::Iterator;

    RecordBase(
        detail::RecordData * data,
        boost::shared_ptr<detail::TableImpl> const & storage
    ) : _data(data), _table(storage)
    {}

    detail::RecordData * _data;
    boost::shared_ptr<detail::TableImpl> _table;
};
  
}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_RecordBase_h_INCLUDED

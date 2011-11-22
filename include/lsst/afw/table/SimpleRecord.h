// -*- c++ -*-
#ifndef AFW_TABLE_SimpleRecord_h_INCLUDED
#define AFW_TABLE_SimpleRecord_h_INCLUDED

#include "lsst/afw/table/detail/fusion_limits.h"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

struct TableStorage;

template <typename RecordT> friend class detail::Iterator;

} // namespace detail

class SimpleTable;

class SimpleRecord {
public:

    typedef detail::RecordData::IdType IdType;

    Layout getLayout() const;

    IdType getId() const { return _buf->id; }

    template <typename T> 
    typename Field<T>::Reference operator[](Key<T> const & key) const {
        return detail::Access::getReference(key, _buf);
    }
    
    template <typename T>
    typename Field<T>::Value get(Key<T> const & key) const {
        return detail::Access::getValue(key, _buf);
    }

    template <typename T, typename U>
    void set(Key<T> const & key, U const & value) const {
        detail::Access::setValue(key, _buf, value);
    }

    SimpleRecord(SimpleRecord const & other)
        : _buf(other._buf), _storage(other._storage) {}

    SimpleRecord & operator=(SimpleRecord const & other) {
        _buf = other._buf;
        _storage = other._storage;
        return *this;
    }

    ~SimpleRecord();

protected:

    detail::RecordAux::Ptr getAux() const { return _buf->aux; }

private:

    friend class SimpleTable;
    template <typename RecordT> friend class detail::Iterator;

    SimpleRecord(
        detail::RecordData * buf,
        boost::shared_ptr<detail::TableStorage> const & storage
    ) : _buf(buf), _storage(storage)
    {}

    detail::RecordData * _buf;
    boost::shared_ptr<detail::TableStorage> _storage;
};
  
}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_SimpleRecord_h_INCLUDED

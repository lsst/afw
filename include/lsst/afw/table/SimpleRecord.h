// -*- c++ -*-
#ifndef AFW_TABLE_SimpleRecord_h_INCLUDED
#define AFW_TABLE_SimpleRecord_h_INCLUDED

#include "lsst/afw/table/detail/fusion_limits.h"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

struct TableStorage;

} // namespace detail

class RecordAux {
public:
    typedef boost::shared_ptr<RecordAux> Ptr;
    virtual ~RecordAux() {}
};

class SimpleTable;

class SimpleRecord {
public:

    Layout getLayout() const;

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
        : _buf(other._buf), _aux(other._aux), _storage(other._storage) {}

    SimpleRecord & operator=(SimpleRecord const & other) {
        _buf = other._buf;
        _aux = other._aux;
        _storage = other._storage;
        return *this;
    }

    ~SimpleRecord();

protected:

    RecordAux::Ptr getAux() const { return _aux; }

private:

    friend class SimpleTable;

    SimpleRecord(
        char * buf,
        RecordAux::Ptr const & aux,
        boost::shared_ptr<detail::TableStorage> const & storage
    ) :
        _buf(reinterpret_cast<char*>(buf)), _aux(aux), _storage(storage)
    {}

    void initialize() const;

    char * _buf;
    RecordAux::Ptr _aux;
    boost::shared_ptr<detail::TableStorage> _storage;
};
  
}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_SimpleRecord_h_INCLUDED

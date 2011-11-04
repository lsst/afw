// -*- c++ -*-
#ifndef CATALOG_RecordBase_h_INCLUDED
#define CATALOG_RecordBase_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include "lsst/catalog/Layout.h"
#include "lsst/catalog/detail/KeyAccess.h"
#include "lsst/catalog/detail/FieldAccess.h"

namespace lsst { namespace catalog {

namespace detail {

struct TableStorage;

} // namespace detail

class TableBase;

class Aux {
public:
    typedef boost::shared_ptr<Aux> Ptr;
    virtual ~Aux() {}
};

class RecordBase {
public:

    Layout getLayout() const;

    template <typename T> bool isNull(Key<T> const & key) const;
    
    template <typename T> typename Field<T>::Value get(Key<T> const & key) const;

    template <typename T, typename U> void set(Key<T> const & key, U const & value) const;

    template <typename T> void unset(Key<T> const & key) const;

    RecordBase(RecordBase const & other) : _buf(other._buf), _aux(other._aux), _storage(other._storage) {}

    RecordBase & operator=(RecordBase const & other) {
        _buf = other._buf;
        _aux = other._aux;
        _storage = other._storage;
        return *this;
    }

    Aux::Ptr getAux() const { return _aux; }

    ~RecordBase();

private:

    friend class TableBase;

    RecordBase(char * buf, Aux::Ptr const & aux, boost::shared_ptr<detail::TableStorage> const & storage) :
        _buf(reinterpret_cast<char*>(buf)), _aux(aux), _storage(storage)
    {}

    void initialize() const;

    char * _buf;
    Aux::Ptr _aux;
    boost::shared_ptr<detail::TableStorage> _storage;
};

template <typename T>
inline bool RecordBase::isNull(Key<T> const & key) const {
    return *reinterpret_cast<int*>(_buf + detail::KeyAccess::getData(key).nullOffset)
        & detail::KeyAccess::getData(key).nullMask;
}
    
template <typename T>
inline typename Field<T>::Value RecordBase::get(Key<T> const & key) const {
    return detail::FieldAccess::getValue(
        detail::KeyAccess::getData(key).field,
        _buf + detail::KeyAccess::getData(key).offset
    );
}

template <typename T, typename U>
inline void RecordBase::set(Key<T> const & key, U const & value) const {
    detail::FieldAccess::setValue(
        detail::KeyAccess::getData(key).field,
        _buf + detail::KeyAccess::getData(key).offset,
        value
    );
    if (detail::KeyAccess::getData(key).field.canBeNull) {
        *reinterpret_cast<int*>(_buf + detail::KeyAccess::getData(key).nullOffset)
            &= ~detail::KeyAccess::getData(key).nullMask;
    }
}

template <typename T>
inline void RecordBase::unset(Key<T> const & key) const {
    detail::FieldAccess::setDefault(
        detail::KeyAccess::getData(key).field,
        _buf + detail::KeyAccess::getData(key).offset
    );
    if (detail::KeyAccess::getData(key).field.canBeNull) {
        *reinterpret_cast<int*>(_buf + detail::KeyAccess::getData(key).nullOffset)
            |= detail::KeyAccess::getData(key).nullMask;
    }
}

}} // namespace lsst::catalog

#endif // !CATALOG_RecordBase_h_INCLUDED

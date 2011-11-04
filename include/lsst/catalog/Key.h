// -*- c++ -*-
#ifndef CATALOG_Key_h_INCLUDED
#define CATALOG_Key_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include "boost/shared_ptr.hpp"

#include "lsst/catalog/detail/KeyData.h"

namespace lsst { namespace catalog {

template <typename T>
class Key {
public:

    template <typename U>
    bool operator==(Key<U> const & other) const { return false; }

    template <typename U>
    bool operator!=(Key<U> const & other) const { return true; }

    bool operator==(Key const & other) const { return _data == other._data; }

    bool operator!=(Key const & other) const { return _data == other._data; }

    Field<T> const & getField() const { return _data->field; }

private:

    friend class detail::KeyAccess;

    typedef detail::KeyData<T> Data;

    Key(boost::shared_ptr<Data> const & data) : _data(data) {}

    boost::shared_ptr<Data> _data;
};

}} // namespace lsst::catalog

#endif // !CATALOG_Key_h_INCLUDED

// -*- c++ -*-
#ifndef CATALOG_DETAIL_KeyAccess_h_INCLUDED
#define CATALOG_DETAIL_KeyAccess_h_INCLUDED

#include "boost/make_shared.hpp"

#include "lsst/catalog/detail/fusion_limits.h"
#include "lsst/catalog/Key.h"

namespace lsst { namespace catalog { namespace detail {

struct KeyAccess {

    template <typename T>
    static KeyData<T> & getData(Key<T> const & key) {
        return *key._data;
    }

    template <typename T>
    static Key<T> make(boost::shared_ptr< KeyData<T> > const & data) {
        return Key<T>(data);
    }

    template <typename T>
    static Key<typename Field<T>::Element> extractElement(Key<T> const & key, int offset);

};

}}} // namespace lsst::catalog::detail

#endif // !CATALOG_DETAIL_KeyAccess_h_INCLUDED

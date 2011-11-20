// -*- c++ -*-
#ifndef CATALOG_DETAIL_Access_h_INCLUDED
#define CATALOG_DETAIL_Access_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"
#include "lsst/catalog/detail/FieldBase.h"
#include "lsst/catalog/Layout.h"
#include "lsst/catalog/detail/LayoutData.h"

namespace lsst { namespace catalog { namespace detail {

class Access {
public:

    template <typename T>
    static typename Key<T>::Reference getReference(Key<T> const & key, char * buf) {
        return key.getReference(reinterpret_cast<typename Key<T>::Element*>(buf + key._offset));
    }

    template <typename T>
    static typename Key<T>::Value getValue(Key<T> const & key, char * buf) {
        return key.getValue(reinterpret_cast<typename Key<T>::Element*>(buf + key._offset));
    }

    template <typename T, typename Value>
    static void setValue(Key<T> const & key, char * buf, Value const & value) {
        key.setValue(reinterpret_cast<typename Key<T>::Element*>(buf + key._offset), value);
    }

    template <typename T>
    static Key<typename Key<T>::Element> extractElement(KeyBase<T> const & kb, int n) {
        return Key<typename Key<T>::Element>(
            static_cast<Key<T> const &>(kb)._offset + n * sizeof(typename Key<T>::Element)
        );
    }

    template <typename T>
    static int getOffset(Key<T> const & key) { return key._offset; }

    template <typename T>
    static Key<T> makeKey(Field<T> const & field, int offset) {
        return Key<T>(offset, field);
    }

    static LayoutData const & getData(Layout const & layout) {
        return *layout._data;
    }

};

}}} // namespace lsst::catalog::detail

#endif // !CATALOG_DETAIL_Access_h_INCLUDED

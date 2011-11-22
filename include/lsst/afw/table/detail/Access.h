// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_Access_h_INCLUDED
#define AFW_TABLE_DETAIL_Access_h_INCLUDED

#include "lsst/afw/table/detail/fusion_limits.h"
#include "lsst/afw/table/detail/FieldBase.h"
#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/detail/LayoutData.h"

namespace lsst { namespace afw { namespace table { namespace detail {

class TableImpl;

class Access {
public:

    template <typename T>
    static typename Key<T>::Reference getReference(Key<T> const & key, void * buf) {
        return key.getReference(
            reinterpret_cast<typename Key<T>::Element*>(
                reinterpret_cast<char *>(buf) + key._offset
            )
        );
    }

    template <typename T>
    static typename Key<T>::Value getValue(Key<T> const & key, void * buf) {
        return key.getValue(
            reinterpret_cast<typename Key<T>::Element*>(
                reinterpret_cast<char *>(buf) + key._offset
            )
        );
    }

    template <typename T, typename Value>
    static void setValue(Key<T> const & key, void * buf, Value const & value) {
        key.setValue(
            reinterpret_cast<typename Key<T>::Element*>(
                reinterpret_cast<char *>(buf) + key._offset
            ),
            value
        );
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

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_Access_h_INCLUDED

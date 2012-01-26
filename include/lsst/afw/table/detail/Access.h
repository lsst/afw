// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_Access_h_INCLUDED
#define AFW_TABLE_DETAIL_Access_h_INCLUDED

#include <cstring>

#include "lsst/ndarray/Manager.h"
#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/detail/SchemaImpl.h"

namespace lsst { namespace afw { namespace table {

class RecordBase;
class TableBase;

namespace detail {

class Access {
public:

    template <typename T>
    static Key<typename Key<T>::Element> extractElement(KeyBase<T> const & kb, int n) {
        return Key<typename Key<T>::Element>(
            static_cast<Key<T> const &>(kb)._offset + n * sizeof(typename Key<T>::Element)
        );
    }

    template <typename T>
    static Key<T> makeKey(Field<T> const & field, int offset) {
        return Key<T>(offset, field);
    }

    static Key<Flag> makeKey(int offset, int bit) {
        return Key<Flag>(offset, bit);
    }

    static void padSchema(Schema & schema, int bytes) {
        schema._edit();
        schema._impl->_recordSize += bytes;
    }

    template <typename ContainerT>
    static void readFits(std::string const & filename, ContainerT & container);

    template <typename ContainerT>
    static void writeFits(std::string const & filename, ContainerT const & container);

};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_Access_h_INCLUDED

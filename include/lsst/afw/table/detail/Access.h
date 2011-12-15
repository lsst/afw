// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_Access_h_INCLUDED
#define AFW_TABLE_DETAIL_Access_h_INCLUDED

#include <cstring>

#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/detail/SchemaImpl.h"

namespace lsst { namespace afw { namespace table {

class RecordBase;

namespace detail {

struct RecordData;
class TableImpl;

class Access {
public:

    template <typename T>
    static typename Key<T>::Reference
    getReference(Key<T> const & key, void * buf) {
        return key.getReference(
            reinterpret_cast<typename Key<T>::Element*>(
                reinterpret_cast<char *>(buf) + key._offset
            )
        );
    }

    template <typename T>
    static typename Key<T>::Value getValue(Key<T> const & key, void const * buf) {
        return key.getValue(
            reinterpret_cast<typename Key<T>::Element const*>(
                reinterpret_cast<char const *>(buf) + key._offset
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
    static void copyValue(
        Key<T> const & inputKey, void const * inputBuf,
        Key<T> const & outputKey, void * outputBuf
    ) {
        assert(inputKey.getElementCount() == outputKey.getElementCount());
        std::memcpy(
            reinterpret_cast<char*>(outputBuf) + outputKey._offset,
            reinterpret_cast<char const*>(inputBuf) + inputKey._offset,
            inputKey.getElementCount() * sizeof(typename Key<T>::Element)
        );
    }

    static void copyValue(
        Key<Flag> const & inputKey, void const * inputBuf,
        Key<Flag> const & outputKey, void * outputBuf
    ) {
        setValue(outputKey, outputBuf, getValue(inputKey, inputBuf));
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

    static Key<Flag> makeKey(int offset, int bit) {
        return Key<Flag>(offset, bit);
    }

    static RecordId & getParentId(Schema const & schema, RecordData & record) {
        return schema._impl->getParentId(record);
    }

    static void padSchema(Schema & schema, int bytes) {
        schema._edit();
        schema._impl->_recordSize += bytes;
    }

    template <typename RecordT>
    static RecordT makeRecord(RecordBase const & base) {
        return RecordT(base);
    }

};

template <typename RecordT>
struct RecordConverter {
    typedef RecordBase argument_type;
    typedef RecordT result_type;
    
    result_type operator()(argument_type const & base) const {
        return Access::makeRecord<RecordT>(base);
    }
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_Access_h_INCLUDED

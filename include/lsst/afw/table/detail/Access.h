// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_Access_h_INCLUDED
#define AFW_TABLE_DETAIL_Access_h_INCLUDED

#include <cstring>

#include "ndarray/Manager.h"
#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/detail/SchemaImpl.h"

namespace lsst { namespace afw { namespace table {

class BaseRecord;
class BaseTable;

namespace detail {

/**
 *  @internal
 *
 *  @brief Friendship-aggregation class for afw/table.
 *
 *  Access is a collection of static member functions that provide access to internals of other
 *  classes.  It allows many classes to just declare Access as a friend rather than a long list of
 *  related classes.  This is less secure, but it's obviously not part of the public interface,
 *  and that's good enough.
 */
class Access {
public:

    /// @internal @brief Return a sub-field key corresponding to the nth element.
    template <typename T>
    static Key<typename Key<T>::Element> extractElement(KeyBase<T> const & kb, int n) {
        if (!static_cast<Key<T> const &>(kb).isValid()) {
            throw LSST_EXCEPT(
                pex::exceptions::LogicErrorException,
                (boost::format("Cannot extract subfield key from invalid key of type '%s' "
                              "(most often this is caused by failing to setup centroid or shape slots)")
                 % Key<T>::getTypeString()).str()
            );
        }
        return Key<typename Key<T>::Element>(
            static_cast<Key<T> const &>(kb).getOffset() + n * sizeof(typename Key<T>::Element)
        );
    }

    /// @internal @brief Return a sub-field key corresponding to a range
    template <typename T>
    static Key< Array<T> > extractRange(KeyBase< Array<T> > const & kb, int begin, int end) {
        if (!static_cast<Key< Array<T> > const &>(kb).isValid()) {
            throw LSST_EXCEPT(
                pex::exceptions::LogicErrorException,
                (boost::format("Cannot extract subfield key from invalid key of type '%s' ")
                 % Key<T>::getTypeString()).str()
            );
        }
        return Key< Array<T> >(
            static_cast<Key< Array<T> > const &>(kb).getOffset() + begin * sizeof(typename Key<T>::Element),
            end - begin
        );
    }

    /// @internal @brief Access to the private Key constructor.
    template <typename T>
    static Key<T> makeKey(Field<T> const & field, int offset) {
        return Key<T>(offset, field);
    }

    /// @internal @brief Access to the private Key constructor.
    static Key<Flag> makeKey(int offset, int bit) {
        return Key<Flag>(offset, bit);
    }

    /// @internal @brief Add some padding to a schema without adding a field.
    static void padSchema(Schema & schema, int bytes) {
        schema._edit();
        schema._impl->_recordSize += bytes;
    }

};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_Access_h_INCLUDED

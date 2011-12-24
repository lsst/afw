// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Key_h_INCLUDED
#define AFW_TABLE_Key_h_INCLUDED

#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/KeyBase.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

class Access;

} // namespace detail

/**
 *  @brief A class used as a handle to a particular field in a table.
 *
 *  All access to table data ultimately goes through Key objects, which
 *  know (via an internal offset) how to address and cast the internal
 *  data buffer of a record or table.
 *
 *  Keys can be obtained from a Schema by name, and are also returned
 *  when a new field is added.  Compound and array keys also provide
 *  accessors to retrieve scalar keys to their elements (see the
 *  documentation for the KeyBase specializations), even though these
 *  element keys do not correspond to a field that exists in any Schema.
 *  For example:
 *  @code
 *  Schema schema;
 *  Key< Array<float> > arrayKey = schema.addField("array", "docs for array", 5);
 *  Key< Point<int> > pointKey = schema.addField("point", "docs for point");
 *  Key<float> elementKey = arrayKey[3];
 *  Key<int> xKey = pointKey.getX();
 *  SimpleTable table(schema);
 *  SimpleRecord record = table.addRecord();
 *  assert(&record[arrayKey][3] == &record[elementKey3]);
 *  assert(record.get(pointKey).getX() == record[xKey]);
 *  @endcode
 *
 *  Key inherits from FieldBase to allow a key for a dynamically-sized field
 *  to know its size without needing to specialize Key itself or hold a full
 *  Field object.
 */
template <typename T>
class Key : public KeyBase<T>, public FieldBase<T> {
public:

    //@{
    /**
     *  @brief Equality comparison.
     *
     *  Two keys with different types are never equal.  Keys with the same type
     *  are equal if they point to the same location in a table, regardless of
     *  what Schema they were constructed from (for instance, if a field has a
     *  different name in one Schema than another, but is otherwise the same,
     *  the two keys will be equal).
     */
    template <typename OtherT> bool operator==(Key<OtherT> const & other) const { return false; }
    template <typename OtherT> bool operator!=(Key<OtherT> const & other) const { return true; }

    bool operator==(Key const & other) const { return _offset == other._offset; }
    bool operator!=(Key const & other) const { return _offset == other._offset; }
    //@}

private:

    friend class detail::Access;

    explicit Key(int offset, FieldBase<T> const & fb = FieldBase<T>())
        : FieldBase<T>(fb), _offset(offset) {}

    int _offset;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Key_h_INCLUDED

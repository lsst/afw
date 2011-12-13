// -*- lsst-c++ -*-
#ifndef LSST_AFW_TABLE_Flag_h_INCLUDED
#define LSST_AFW_TABLE_Flag_h_INCLUDED

#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/KeyBase.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

class Access;

} // namespace detail

template <>
struct FieldBase<Flag> {

    typedef bool Value;        ///< @brief the type returned by RecordBase::get
    typedef boost::uint64_t Element;   ///< @brief the actual storage type (shared by multiple flag fields)

    /// @brief Return the number of subfield elements (always one for scalars).
    int getElementCount() const { return 1; }

    /// @brief Return a string description of the field type.
    std::string getTypeString() const { return "Flag"; }

};

/**
 *  @brief A base class for Key that allows the underlying storage field to be extracted.
 */
template <>
class KeyBase< Flag > {
public:
    Key<FieldBase<Flag>::Element> getStorage() const;
};

/**
 *  @brief Key specialization for Flag.
 *
 *  Flag fields are special; their keys need to contain not only the offset to the
 *  integer field they share with other Flag fields, but also their position
 *  in that shared field.
 */
template <>
class Key<Flag> : public KeyBase<Flag>, public FieldBase<Flag> {
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
    friend class Schema;

    Value getValue(Element * p, PTR(detail::TableImpl) const & table) const {
        return (*p) & (Element(1) << _bit);
    }

    void setValue(Element * p, Value v, PTR(detail::TableImpl) const & table) const { 

        if (v) {
            *p |= (Element(1) << _bit);
        } else {
            *p &= ~(Element(1) << _bit);
        }
    }

    explicit Key(int offset, int bit) : _offset(offset), _bit(bit) {}

    int _offset;
    int _bit;
};

}}} // namespace lsst::afw::table

#endif // !LSST_AFW_TABLE_Flag_h_INCLUDED

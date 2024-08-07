// -*- lsst-c++ -*-
#ifndef LSST_AFW_TABLE_Flag_h_INCLUDED
#define LSST_AFW_TABLE_Flag_h_INCLUDED

#include <cstdint>

#include "lsst/cpputils/hashCombine.h"

#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/KeyBase.h"

namespace lsst {
namespace afw {
namespace table {

namespace detail {

class Access;

}  // namespace detail

/**
 *  Specialization for Flag fields.
 *
 *  Flag fields are handled specially in many places, because their keys have both an offset into an
 *  integer element and the bit in that element; while other fields have one or more elements per field,
 *  Flags have multiple fields per element.  This means we can't put all the custom code for Flag in
 *  FieldBase, and because Flags have an explicit Key specialization, we put the record access
 *  implementation in Key.
 */
template <>
struct FieldBase<Flag> {
    using Value = bool;            ///< the type returned by BaseRecord::get
    using Element = std::int64_t;  ///< the actual storage type (shared by multiple flag fields)

    /// Return the number of subfield elements (always one for scalars).
    std::size_t getElementCount() const { return 1; }

    /// Return a string description of the field type.
    static std::string getTypeString() { return "Flag"; }

    // Only the first of these constructors is valid for this specializations, but
    // it's convenient to be able to instantiate both, since the other is used
    // by other specializations.
    FieldBase() = default;
    FieldBase(std::size_t) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,
                          "Constructor disabled (this Field type is not sized).");
    }

    FieldBase(FieldBase const &) = default;
    FieldBase(FieldBase &&) = default;
    FieldBase &operator=(FieldBase const &) = default;
    FieldBase &operator=(FieldBase &&) = default;
    ~FieldBase() = default;

protected:
    /// Defines how fields are printed.
    void stream(std::ostream &os) const {}
};

/**
 *  A base class for Key that allows the underlying storage field to be extracted.
 */
template <>
class KeyBase<Flag> {
public:
    static bool const HAS_NAMED_SUBFIELDS = false;

    /// Return a key corresponding to the integer element where this field's bit is packed.
    Key<FieldBase<Flag>::Element> getStorage() const;

    KeyBase() = default;
    KeyBase(KeyBase const &) = default;
    KeyBase(KeyBase &&) = default;
    KeyBase &operator=(KeyBase const &) = default;
    KeyBase &operator=(KeyBase &&) = default;
    ~KeyBase() = default;
};

/**
 *  Key specialization for Flag.
 *
 *  Flag fields are special; their keys need to contain not only the offset to the
 *  integer element they share with other Flag fields, but also their position
 *  in that shared field.
 *
 *  Flag fields operate mostly like a bool field, but they do not support reference
 *  access, and internally they are packed into an integer shared by multiple fields
 *  so the marginal cost of each Flag field is only one bit.
 */
template <>
class Key<Flag> : public KeyBase<Flag>, public FieldBase<Flag> {
public:
    //@{
    /**
     *  Equality comparison.
     *
     *  Two keys with different types are never equal.  Keys with the same type
     *  are equal if they point to the same location in a table, regardless of
     *  what Schema they were constructed from (for instance, if a field has a
     *  different name in one Schema than another, but is otherwise the same,
     *  the two keys will be equal).
     */
    template <typename OtherT>
    bool operator==(Key<OtherT> const &other) const {
        return false;
    }
    template <typename OtherT>
    bool operator!=(Key<OtherT> const &other) const {
        return true;
    }

    bool operator==(Key const &other) const { return _offset == other._offset && _bit == other._bit; }
    bool operator!=(Key const &other) const { return !this->operator==(other); }
    //@}

    /// Return a hash of this object.
    std::size_t hash_value() const noexcept {
        // Completely arbitrary seed
        return cpputils::hashCombine(17, _offset, _bit);
    }

    /// Return the offset in bytes of the integer element that holds this field's bit.
    std::size_t getOffset() const { return _offset; }

    /// The index of this field's bit within the integer it shares with other Flag fields.
    std::size_t getBit() const { return _bit; }

    /**
     *  Return true if the key was initialized to valid offset.
     *
     *  This does not guarantee that a key is valid with any particular schema, or even
     *  that any schemas still exist in which this key is valid.
     *
     *  A key that is default constructed will always be invalid.
     */
    bool isValid() const { return _valid; }

    /**
     *  Default construct a field.
     *
     *  The new field will be invalid until a valid Key is assigned to it.
     */
    Key() : FieldBase<Flag>(), _offset(0), _bit(0), _valid(false) {}

    Key(Key const &) = default;
    Key(Key &&) = default;
    Key &operator=(Key const &) = default;
    Key &operator=(Key &&) = default;
    ~Key() = default;

    /// Stringification.
    inline friend std::ostream &operator<<(std::ostream &os, Key<Flag> const &key) {
        return os << "Key['" << Key<Flag>::getTypeString() << "'](offset=" << key.getOffset()
                  << ", bit=" << key.getBit() << ")";
    }

private:
    friend class detail::Access;
    friend class BaseRecord;

    /// Used to implement BaseRecord::get.
    Value getValue(Element const *p, ndarray::Manager::Ptr const &) const {
        return (*p) & (Element(1) << _bit);
    }

    /// Used to implement BaseRecord::set.
    void setValue(Element *p, ndarray::Manager::Ptr const &, Value v) const {
        if (v) {
            *p |= (Element(1) << _bit);
        } else {
            *p &= ~(Element(1) << _bit);
        }
    }

    explicit Key(std::size_t offset, std::size_t bit) : _offset(offset), _bit(bit), _valid(true) {}

    std::size_t _offset;
    std::size_t _bit;
    bool _valid;
};
}  // namespace table
}  // namespace afw
}  // namespace lsst

namespace std {
template <>
struct hash<lsst::afw::table::Key<lsst::afw::table::Flag>> {
    using argument_type = lsst::afw::table::Key<lsst::afw::table::Flag>;
    using result_type = size_t;
    size_t operator()(argument_type const &obj) const noexcept { return obj.hash_value(); }
};
}  // namespace std

#endif  // !LSST_AFW_TABLE_Flag_h_INCLUDED

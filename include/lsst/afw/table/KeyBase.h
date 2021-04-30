// -*- lsst-c++ -*-
#ifndef AFW_TABLE_KeyBase_h_INCLUDED
#define AFW_TABLE_KeyBase_h_INCLUDED

#include <vector>

#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/FieldBase.h"

namespace lsst {
namespace afw {
namespace table {

class BaseRecord;

/// A base class for Key that allows subfield keys to be extracted for some field types.
template <typename T>
class KeyBase {
public:
    KeyBase() = default;
    KeyBase(KeyBase const &) = default;
    KeyBase(KeyBase &&) = default;
    KeyBase &operator=(KeyBase const &) = default;
    KeyBase &operator=(KeyBase &&) = default;
    ~KeyBase() = default;
};

/// KeyBase specialization for Arrays.
template <typename U>
class KeyBase<Array<U> > {
public:
    KeyBase() = default;
    KeyBase(KeyBase const &) = default;
    KeyBase(KeyBase &&) = default;
    KeyBase &operator=(KeyBase const &) = default;
    KeyBase &operator=(KeyBase &&) = default;
    ~KeyBase() = default;

    std::vector<U> extractVector(BaseRecord const& record) const;

    void assignVector(BaseRecord& record, std::vector<U> const& values) const;

    Key<U> operator[](int i) const;  ///< Return a subfield key for the i-th element of the array.

    Key<Array<U> > slice(int begin, int end) const;  ///< Return a key for a range of elements
};

/// KeyBase specialization for Flags.
template <>
class KeyBase<Flag> {
public:
    KeyBase() = default;
    KeyBase(KeyBase const &) = default;
    KeyBase(KeyBase &&) = default;
    KeyBase &operator=(KeyBase const &) = default;
    KeyBase &operator=(KeyBase &&) = default;
    ~KeyBase() = default;

    /// Return a key corresponding to the integer element where this field's bit is packed.
    Key<FieldBase<Flag>::Element> getStorage() const;

};

}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_KeyBase_h_INCLUDED

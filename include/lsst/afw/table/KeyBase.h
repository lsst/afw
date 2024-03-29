// -*- lsst-c++ -*-
#ifndef AFW_TABLE_KeyBase_h_INCLUDED
#define AFW_TABLE_KeyBase_h_INCLUDED

#include <vector>

#include "lsst/afw/table/misc.h"

namespace lsst {
namespace afw {
namespace table {

class BaseRecord;

template <typename T>
class Key;

/// A base class for Key that allows subfield keys to be extracted for some field types.
template <typename T>
class KeyBase {
public:
    static bool const HAS_NAMED_SUBFIELDS = false;
};

/// KeyBase specialization for Arrays.
template <typename U>
class KeyBase<Array<U> > {
public:
    static bool const HAS_NAMED_SUBFIELDS = false;

    std::vector<U> extractVector(BaseRecord const& record) const;

    void assignVector(BaseRecord& record, std::vector<U> const& values) const;

    Key<U> operator[](std::size_t i) const;  ///< Return a subfield key for the i-th element of the array.

    Key<Array<U> > slice(std::size_t begin, std::size_t end) const;  ///< Return a key for a range of elements
};
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_KeyBase_h_INCLUDED

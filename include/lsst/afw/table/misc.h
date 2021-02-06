// -*- lsst-c++ -*-
#ifndef AFW_TABLE_misc_h_INCLUDED
#define AFW_TABLE_misc_h_INCLUDED

#include <cstdint>
#include <string>

#include "lsst/geom/Angle.h"
#include "lsst/geom/SpherePoint.h"

namespace lsst {
namespace afw {
namespace table {

/**
 *  Type used for unique IDs for records.
 *
 *  FITS isn't fond of uint64, so we can save a lot of pain by using signed ints here unless
 *  we really need unsigned.
 */
typedef std::int64_t RecordId;

//@{
/**
 *  Tag types used to declare specialized field types.
 *
 *  See the documentation for specializations of FieldBase and KeyBase
 *  for more information.
 */
template <typename T>
class Array;
class Flag;
typedef lsst::geom::Angle Angle;
typedef lsst::geom::SpherePoint SpherePoint;
//@}


// Forward-declare schema primitives, and then forward-declare all
// explicit specializations, to guard against code implicitly instantiating
// the default template for any of those.

template <typename T> class KeyBase;
template <typename T> class FieldBase;
template <typename T> class Key;
template <typename T> class Field;

template <> class KeyBase<Flag>;
template <> class FieldBase<Flag>;
template <> class Key<Flag>;

template <typename U> class KeyBase<Array<U>>;
template <typename U> class FieldBase<Array<U>>;
template <> class FieldBase<std::string>;

}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_misc_h_INCLUDED

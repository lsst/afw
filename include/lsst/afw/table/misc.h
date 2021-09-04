// -*- lsst-c++ -*-
#ifndef AFW_TABLE_misc_h_INCLUDED
#define AFW_TABLE_misc_h_INCLUDED

#include <cstdint>

#include "boost/mpl/if.hpp"

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
using RecordId = std::int64_t;

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
using Angle = lsst::geom::Angle;
using SpherePoint = lsst::geom::SpherePoint;
//@}
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_misc_h_INCLUDED

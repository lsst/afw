// -*- lsst-c++ -*-
#ifndef AFW_TABLE_misc_h_INCLUDED
#define AFW_TABLE_misc_h_INCLUDED

#include <cstdint>

#include "boost/mpl/if.hpp"

#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/coord/Coord.h"

namespace lsst { namespace afw { namespace table {

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
template <typename T> class Array;
class Flag;
typedef lsst::afw::coord::Coord Coord;
typedef lsst::afw::coord::IcrsCoord IcrsCoord;
typedef lsst::afw::geom::Angle Angle;
//@}

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_misc_h_INCLUDED

// -*- lsst-c++ -*-
#ifndef AFW_TABLE_misc_h_INCLUDED
#define AFW_TABLE_misc_h_INCLUDED

#include "boost/cstdint.hpp"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/coord/Coord.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief Type used for unique IDs for records.
 *
 *  FITS isn't fond of uint64, so we can save a lot of pain by using signed ints here unless
 *  we really need unsigned.
 */
typedef boost::int64_t RecordId;

//@{
/**
 *  @brief Tag types used to declare specialized field types.
 *
 *  The documentation for specializations of FieldBase and KeyBase
 *  for more information.
 */
template <typename T> class Point;
template <typename T> class Moments;
template <typename T> class Array;
template <typename T> class Covariance;
class Flag;
typedef lsst::afw::coord::Coord Coord;
typedef lsst::afw::geom::Angle Angle;
//@}

/**
 *  @brief Class used to attach arbitrary extra data members to table and record classes.
 *
 *  Final table and record classes that need to additional data members will generally
 *  create new subclasses of AuxBase that holds these additional members, and then static_cast
 *  the return value of TableBase::getAux and RecordBase::getAux to the subclass type.
 */
class AuxBase {
public:
    virtual ~AuxBase() {}
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_misc_h_INCLUDED

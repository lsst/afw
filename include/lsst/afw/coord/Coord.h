// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#if !defined(LSST_AFW_COORD_COORD_H)
#define LSST_AFW_COORD_COORD_H
/*
 * Functions to handle coordinates
 */
#include <string>

#include "lsst/geom/Angle.h"

namespace lsst {
namespace afw {
namespace coord {

/**
 * Convert a dd:mm:ss string to lsst::geom::Angle
 *
 * @param[in] dms Coord as a string in dd:mm:ss format
 */
lsst::geom::Angle dmsStringToAngle(std::string const dms);
/// Convert a hh:mm:ss string to lsst::geom::Angle
lsst::geom::Angle hmsStringToAngle(std::string const hms);

/**
 * Convert an angle to a string with form dd:mm:ss
 *
 * @param[in] deg  angle to convert
 */
std::string angleToDmsString(lsst::geom::Angle const deg);
/// a function to convert decimal degrees to a string with form hh:mm:ss.s
std::string angleToHmsString(lsst::geom::Angle const deg);

}  // namespace coord
}  // namespace afw
}  // namespace lsst

#endif

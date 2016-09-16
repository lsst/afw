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

/**
 * @file
 * @brief Provide functions to handle dates
 * @ingroup afw
 * @author Steve Bickerton
 *
 */
#include <sstream>
#include <cmath>

#include "lsst/pex/exceptions.h"
#include "boost/format.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/tuple/tuple.hpp"

#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/coord/Observatory.h"

namespace coord        = lsst::afw::coord;
namespace ex           = lsst::pex::exceptions;
namespace afwGeom      = lsst::afw::geom;



/**
 * @brief Constructor for the observatory with lat/long as afwGeom::Angles
 */
coord::Observatory::Observatory(
                                afwGeom::Angle const longitude, ///< observatory longitude (+ve E of Greenwich)
                                afwGeom::Angle const latitude,  ///< observatory latitude
                                double const elevation  ///< observatory elevation
                               ) :
    _latitude(latitude),
    _longitude(longitude),
    _elevation(elevation) {
}



/*
 * @brief Constructor for the observatory with lat/long as strings
 *
 * @note RA is assumed to be in DMS, not HMS!
 *
 */
coord::Observatory::Observatory(
                                std::string const longitude, ///< observatory longitude
                                std::string const latitude,  ///< observatory latitude
                                double const elevation       ///< observatory elevation
                               ) :
    _latitude(dmsStringToAngle(latitude)),
    _longitude(dmsStringToAngle(longitude)),
    _elevation(elevation) {
}




/**
 * @brief The main access method for the longitudinal coordinate
 *
 */
afwGeom::Angle coord::Observatory::getLongitude() const {
    return _longitude;
}

/**
 * @brief The main access method for the longitudinal coordinate
 *
 * @note There's no reason to want a latitude in hours, so that unit will cause
 *       an exception to be thrown
 *
 */
afwGeom::Angle coord::Observatory::getLatitude() const {
    return _latitude;
}


/**
 * @brief Set the latitude
 */
void coord::Observatory::setLatitude(
                 afwGeom::Angle const latitude ///< the latitude
                )   {
    _latitude = latitude;
}

/**
 * @brief Set the longitude
 */
void coord::Observatory::setLongitude(
                                      afwGeom::Angle const longitude ///< the longitude
                                     ) {
    _longitude = longitude;
}


/**
 * @brief Set the Elevation
 */
void coord::Observatory::setElevation(
                                      double const elevation ///< the elevation
                                     ) {
    _elevation = elevation;
}



/**
 * @brief Allow quick access to the longitudinal coordinate as a string
 *
 * @note There's no reason to want a longitude string in radians, so that unit will cause
 *       an exception to be thrown
 *
 */
std::string coord::Observatory::getLongitudeStr() const {
    return angleToDmsString(_longitude);
}
/**
 * @brief Allow quick access to the longitude coordinate as a string
 *
 * @note There's no reason to want a latitude string in radians or hours, so
 *       the units can not be explicitly requested.
 *
 */
std::string coord::Observatory::getLatitudeStr() const {
    return angleToDmsString(_latitude);
}

/**
 * @brief Get string representation
 */
std::string coord::Observatory::toString() const {
    return (boost::format("%gW, %gN  %g")
            % getLatitude().asDegrees()
            % getLongitude().asDegrees()
            % getElevation()).str();
}

/**
 * Print an Observatory to the stream
 */
std::ostream & coord::operator<<(std::ostream &os,             ///< Stream to print to
                                 coord::Observatory const& obs ///< the Observatory to print
                                )
{
    os << obs.toString();
    return os;
}

/************************************************************************************************************/

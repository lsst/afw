// -*- lsst-c++ -*-
/**
 * @file Date.cc
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

#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/coord/Observatory.h"

namespace coord        = lsst::afw::coord;
namespace ex           = lsst::pex::exceptions;



/**
 * @brief Constructor for the observatory with lat/long as doubles
 */
coord::Observatory::Observatory(
                                double const longitude, ///< observatory longitude
                                double const latitude,  ///< observatory latitude 
                                double const elevation  ///< observatory elevation
                               ) :
    _latitudeRad(degToRad*latitude),
    _longitudeRad(degToRad*longitude),
    _elevation(elevation) {
}



/*
 * @brief Constructor for the observatory with lat/long as strings
 *
 */
coord::Observatory::Observatory(
                                std::string const longitude, ///< observatory longitude
                                std::string const latitude,  ///< observatory latitude 
                                double const elevation       ///< observatory elevation
                               ) : 
    _latitudeRad(degToRad*dmsStringToDegrees(latitude)),
    _longitudeRad(degToRad*dmsStringToDegrees(longitude)),
    _elevation(elevation) {
}




/**
 * @brief The main access method for the longitudinal coordinate
 *
 */
double coord::Observatory::getLongitude(
                                        CoordUnit unit  ///< units to return (DEGREES, RADIANS, HOURS)
                                       ) const {
    switch (unit) {
      case DEGREES:
        return radToDeg*_longitudeRad;
        break;
      case RADIANS:
        return _longitudeRad;
        break;
      case HOURS:
        return radToDeg*_longitudeRad/15.0;
        break;
      default:
        throw LSST_EXCEPT(ex::InvalidParameterException, "Units must be DEGREES, RADIANS, or HOURS.");
        break;
    }
}

/**
 * @brief The main access method for the longitudinal coordinate
 *
 * @note There's no reason to want a latitude in hours, so that unit will cause
 *       an exception to be thrown
 *
 */
double coord::Observatory::getLatitude(
                                       CoordUnit unit ///< units to return (DEGREES, RADIANS)
                                      ) const {
    switch (unit) {
      case DEGREES:
        return radToDeg*_latitudeRad;
        break;
      case RADIANS:
        return _latitudeRad;
        break;
      default:
        throw LSST_EXCEPT(ex::InvalidParameterException, "Units must be DEGREES, or RADIANS.");
        break;
    }
}


/**
 * @brief Set the latitude
 */
void coord::Observatory::setLatitude(
                 double const latitude ///< the latitude
                )   {
    _latitudeRad = degToRad*latitude;
}

/**
 * @brief Set the longitude
 */
void coord::Observatory::setLongitude(
                                      double const longitude ///< the longitude
                                     ) {
    _longitudeRad = degToRad*longitude;
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
    return degreesToDmsString(radToDeg*_longitudeRad);
}
/**
 * @brief Allow quick access to the longitude coordinate as a string
 *
 * @note There's no reason to want a latitude string in radians or hours, so
 *       the units can not be explicitly requested.
 *
 */
std::string coord::Observatory::getLatitudeStr() const {
    return degreesToDmsString(radToDeg*_latitudeRad);
}

/**
 * Print an Observatory to the stream
 */
std::ostream & coord::operator<<(std::ostream &os,             ///< Stream to print to
                                 coord::Observatory const& obs ///< the Observatory to print
                                )
{
    return os << (boost::format("%gW, %gN  %g")
                  % obs.getLatitude(coord::DEGREES)
                  % obs.getLongitude(coord::DEGREES)
                  % obs.getElevation()).str();
}

/************************************************************************************************************/

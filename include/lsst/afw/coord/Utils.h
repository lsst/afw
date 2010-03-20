// -*- lsst-c++ -*-
#if !defined(LSST_AFW_COORD_UTILS_H)
#define LSST_AFW_COORD_UTILS_H
/**
 * @file Utils.h
 * @brief Header to declare constants and enums for Coord and Observatory
 * @ingroup afw
 * @author Steve Bickerton
 *
 *
 */ 

namespace lsst {
namespace afw {    
namespace coord {

double const degToRad = M_PI/180.0;
double const radToDeg = 180.0/M_PI;

enum CoordSystem { FK5, ICRS, GALACTIC, ECLIPTIC, ALTAZ, EQUATORIAL };
enum CoordUnit   { DEGREES, RADIANS, HOURS };
    

}}}

#endif

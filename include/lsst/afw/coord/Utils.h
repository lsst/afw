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

#include <cmath>

namespace lsst {
namespace afw {    
namespace coord {

double const degToRad = M_PI/180.0;
double const radToDeg = 180.0/M_PI;

double const arcminToRad = M_PI/(180.0*60.0);
double const radToArcmin = (180.0*60.0)/M_PI;
    
double const arcsecToRad = M_PI/(180.0*3600.0);
double const radToArcsec = (180.0*3600.0)/M_PI;
    
enum CoordUnit   { DEGREES, RADIANS, HOURS };
    

}}}

#endif

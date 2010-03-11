// -*- lsst-c++ -*-
#if !defined(LSST_AFW_COORD_OBSERVATORY_H)
#define LSST_AFW_COORD_OBSERVATORY_H
/**
 * @file Observatory.h
 * @brief Class to hold observatory information
 * @ingroup afw
 * @author Steve Bickerton
 *
 *
 */ 

#include <cmath>
#include "lsst/afw/coord/Utils.h"

namespace lsst {
namespace afw {    
namespace coord {

/**
 * @class Observatory
 * @brief Store information about an observatory ... lat/long, elevation
 */
class Observatory {
public:
    
    Observatory(double const latitude, double const longitude, double const elevation) :
        _latitudeRad(degToRad*latitude), _longitudeRad(degToRad*longitude), _elevation(elevation) {}
    
    Observatory(std::string const latitude, std::string const longitude, double const elevation);
    
    void setLatitude(double const latitude)   { _latitudeRad = degToRad*latitude; }
    void setLongitude(double const longitude) { _longitudeRad = degToRad*longitude; }
    void setElevation(double const elevation) { _elevation = elevation; }
    
    double getLatitude(CoordUnit unit);
    double getLongitude(CoordUnit unit);
    double getElevation() { return _elevation; }
    
    std::string getLatitudeStr();
    std::string getLongitudeStr(CoordUnit unit=DEGREES);

 private:
    double _latitudeRad;
    double _longitudeRad;
    double _elevation;
};


}}}

#endif

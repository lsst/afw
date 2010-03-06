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
        _latitude(latitude), _longitude(longitude), _elevation(elevation) {}
    
    Observatory(std::string const latitude, std::string const longitude, double const elevation);
    
    void setLatitude(double const latitude)   { _latitude = latitude; }
    void setLongitude(double const longitude) { _longitude = longitude; }
    void setElevation(double const elevation) { _elevation = elevation; }
    
    double getLatitude()  { return _latitude; }
    double getLongitude() { return _longitude; }
    double getElevation() { return _elevation; }
    
    double getLatitudeRad()  { return _latitude*M_PI/180.0; }
    double getLongitudeRad() { return _longitude*M_PI/180.0; }

    std::string getLatitudeStr();
    std::string getLongitudeStr();

 private:
    double _latitude;
    double _longitude;
    double _elevation;
};


}}}

#endif

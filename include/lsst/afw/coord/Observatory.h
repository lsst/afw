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
    
    Observatory(double const longitude, double const latitude, double const elevation);
    Observatory(std::string const longitude, std::string const latitude, double const elevation);
    
    void setLatitude(double const latitude);
    void setLongitude(double const longitude);
    void setElevation(double const elevation);
    
    double getLatitude(CoordUnit unit);
    double getLongitude(CoordUnit unit);
    double getElevation() { return _elevation; }
    
    std::string getLatitudeStr();
    std::string getLongitudeStr();

private:
    double _latitudeRad;
    double _longitudeRad;
    double _elevation;
};


}}}

#endif

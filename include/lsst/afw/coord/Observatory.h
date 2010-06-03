// -*- lsst-c++ -*-
#if !defined(LSST_AFW_COORD_OBSERVATORY_H)
#define LSST_AFW_COORD_OBSERVATORY_H
/**
 * @file
 * @brief Class to hold observatory information
 * @ingroup afw
 * @author Steve Bickerton
 *
 *
 */ 

#include <iostream>
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
    
    double getLatitude(CoordUnit unit) const;
    double getLongitude(CoordUnit unit) const;
    double getElevation() const { return _elevation; }
    
    std::string getLatitudeStr() const;
    std::string getLongitudeStr() const;

    bool operator==(Observatory const& rhs) const {
        return
            (_latitudeRad - rhs._latitudeRad) == 0.0 &&
            (_longitudeRad - rhs._longitudeRad) == 0.0 &&
            (_elevation - rhs._elevation) == 0.0;
    }
    bool operator!=(Observatory const& rhs) const {
        return !(*this == rhs);
    }

private:
    double _latitudeRad;
    double _longitudeRad;
    double _elevation;
};

std::ostream & operator<<(std::ostream &os, Observatory const& obs);

}}}

#endif

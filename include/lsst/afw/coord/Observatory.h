// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
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
#include "lsst/afw/geom/Angle.h"

namespace lsst {
namespace afw {    
namespace coord {


/**
 * @class Observatory
 * @brief Store information about an observatory ... lat/long, elevation
 */
class Observatory {
public:
    
    Observatory(lsst::afw::geom::Angle const longitude, lsst::afw::geom::Angle const latitude, double const elevation);
    Observatory(std::string const longitude, std::string const latitude, double const elevation);
    
    void setLatitude(lsst::afw::geom::Angle const latitude);
    void setLongitude(lsst::afw::geom::Angle const longitude);
    void setElevation(double const elevation);
    
    lsst::afw::geom::Angle getLatitude() const;
    lsst::afw::geom::Angle getLongitude() const;
    double getElevation() const { return _elevation; }
    
    std::string getLatitudeStr() const;
    std::string getLongitudeStr() const;

    bool operator==(Observatory const& rhs) const {
        return
            ((_latitude - rhs._latitude) == 0.0) &&
            ((_longitude - rhs._longitude) == 0.0) &&
            ((_elevation - rhs._elevation) == 0.0);
    }
    bool operator!=(Observatory const& rhs) const {
        return !(*this == rhs);
    }

private:
    lsst::afw::geom::Angle _latitude;
    lsst::afw::geom::Angle _longitude;
    double _elevation;
};

std::ostream & operator<<(std::ostream &os, Observatory const& obs);

}}}

#endif

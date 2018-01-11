// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008-2016 LSST Corporation.
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

#if !defined(LSST_AFW_COORD_OBSERVATORY_H)
#define LSST_AFW_COORD_OBSERVATORY_H
/*
 * Class to hold observatory/telescope location
 */

#include <iostream>
#include "lsst/afw/geom/Angle.h"

namespace lsst {
namespace afw {
namespace coord {

/**
 * Hold the location of an observatory
 */
class Observatory {
public:
    /**
     * Construct an Observatory with longitude and latitude specified as lsst::afw::geom::Angle
     *
     * @param[in] longitude  telescope longitude (positive values are E of Greenwich)
     * @param[in] latitude  telescope latitude
     * @param[in] elevation  telescope elevation (meters above reference spheroid)
     */
    Observatory(lsst::afw::geom::Angle const longitude, lsst::afw::geom::Angle const latitude,
                double const elevation);

    /**
     * Construct an Observatory with longitude and latitude specified as sexagesimal strings
     *
     * @param[in] longitude  telescope longitude (dd:mm:ss.s, positive values are E of Greenwich)
     * @param[in] latitude  telescope latitude  (dd:mm:ss.s)
     * @param[in] elevation  telescope elevation (meters above reference spheroid)
     *
     */
    Observatory(std::string const& longitude, std::string const& latitude, double const elevation);

    ~Observatory();
    Observatory(Observatory const&);
    Observatory(Observatory&&);
    Observatory& operator=(Observatory const&);
    Observatory& operator=(Observatory&&);

    /// set telescope longitude
    void setLongitude(lsst::afw::geom::Angle const longitude);
    /// set telescope latitude (positive values are E of Greenwich)
    void setLatitude(lsst::afw::geom::Angle const latitude);
    /// set telescope elevation (meters above reference spheroid)
    void setElevation(double const elevation);

    /// get telescope longitude (positive values are E of Greenwich)
    lsst::afw::geom::Angle getLongitude() const;
    /// get telescope latitude
    lsst::afw::geom::Angle getLatitude() const;
    /// get telescope elevation (meters above reference spheroid)
    double getElevation() const { return _elevation; }

    /// get telescope longitude as a dd:mm:ss.s string (positive values are E of Greenwich)
    std::string getLongitudeStr() const;
    /// get telescope latitude as a dd:mm:ss.s string
    std::string getLatitudeStr() const;
    /// get string representation
    std::string toString() const;

    bool operator==(Observatory const& rhs) const {
        auto deltaLongitude = (_latitude - rhs.getLatitude()).wrapCtr();
        auto deltaLatitude = (_longitude - rhs.getLongitude()).wrapCtr();
        return (deltaLongitude == 0.0 * lsst::afw::geom::degrees) &&
               (deltaLatitude == 0.0 * lsst::afw::geom::degrees) && ((_elevation - rhs._elevation) == 0.0);
    }
    bool operator!=(Observatory const& rhs) const { return !(*this == rhs); }

private:
    lsst::afw::geom::Angle _latitude;
    lsst::afw::geom::Angle _longitude;
    double _elevation;
};

/**
 * Print an Observatory to the stream
 *
 * @param[in, out] os Stream to print to
 * @param[in] obs the Observatory to print
 */
std::ostream& operator<<(std::ostream& os, Observatory const& obs);
}
}
}

#endif

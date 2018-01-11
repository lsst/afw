// -*- LSST-C++ -*- // fixed format comment for emacs
/*
 * LSST Data Management System
 * Copyright 2016 LSST Corporation.
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

#ifndef LSST_AFW_COORD_WEATHER_H_INCLUDED
#define LSST_AFW_COORD_WEATHER_H_INCLUDED

#include <sstream>

namespace lsst {
namespace afw {
namespace coord {

/**
 * Basic weather information sufficient for a simple model for air mass or refraction
 *
 * Weather is immutable.
 */
class Weather {
public:
    /**
     * Construct a Weather
     *
     * @param[in] airTemperature  outside air temperature (C)
     * @param[in] airPressure  outside air pressure (Pascal)
     * @param[in] humidity  outside relative humidity (%)
     * @throws lsst::pex::exceptions::InvalidParameterError if humidity < 0
     */
    explicit Weather(double airTemperature, double airPressure, double humidity);

    ~Weather() = default;

    Weather(Weather const &) = default;
    Weather(Weather &&) = default;
    Weather &operator=(Weather const &) = default;
    Weather &operator=(Weather &&) = default;

    bool operator==(Weather const &other) const;
    bool operator!=(Weather const &other) const { return !(*this == other); }

    /// get outside air temperature (C)
    double getAirTemperature() const { return _airTemperature; };

    /// get outside air pressure (Pascal)
    double getAirPressure() const { return _airPressure; };

    /// get outside relative humidity (%)
    double getHumidity() const { return _humidity; };

private:
    double _airTemperature;  ///< air temperature (C)
    double _airPressure;     ///< air pressure (Pascals)
    double _humidity;        ///< relative humidity (%)

    /**
     * Validate the values
     * @throws lsst::pex::exceptions::InvalidParameterError if humidity < 0
     */
    void validate() const;
};

/// print a Weather to an output stream
std::ostream &operator<<(std::ostream &os, Weather const &weath);
}
}
}  // lsst::afw::coord

#endif  // !LSST_AFW_COORD_WEATHER_H_INCLUDED

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
class Weather final {
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

    ~Weather() noexcept = default;

    Weather(Weather const &) noexcept = default;
    Weather(Weather &&) noexcept = default;
    Weather &operator=(Weather const &) noexcept = default;
    Weather &operator=(Weather &&) noexcept = default;

    bool operator==(Weather const &other) const noexcept;
    bool operator!=(Weather const &other) const noexcept { return !(*this == other); }

    /// Return a hash of this object
    std::size_t hash_value() const noexcept;

    /// get outside air temperature (C)
    double getAirTemperature() const noexcept { return _airTemperature; };

    /// get outside air pressure (Pascal)
    double getAirPressure() const noexcept { return _airPressure; };

    /// get outside relative humidity (%)
    double getHumidity() const noexcept { return _humidity; };

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
}  // namespace coord
}  // namespace afw
}  // namespace lsst

namespace std {
template <>
struct hash<lsst::afw::coord::Weather> {
    using argument_type = lsst::afw::coord::Weather;
    using result_type = size_t;
    size_t operator()(argument_type const &obj) const noexcept { return obj.hash_value(); }
};
}  // namespace std

#endif  // !LSST_AFW_COORD_WEATHER_H_INCLUDED

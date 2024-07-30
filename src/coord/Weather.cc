// -*- lsst-c++ -*-

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

#include <cmath>
#include <sstream>

#include "lsst/cpputils/hashCombine.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/coord/Weather.h"

namespace lsst {
namespace afw {
namespace coord {

Weather::Weather(double airTemperature, double airPressure, double humidity)
        : _airTemperature(airTemperature), _airPressure(airPressure), _humidity(humidity) {
    validate();
}

bool Weather::operator==(Weather const& other) const noexcept {
    // Weather may be initialized to NaN values as a placeholder, or to indicate "unknown".
    bool tempMatch = (std::isnan(_airTemperature) && std::isnan(other.getAirTemperature())) ||
                     _airTemperature == other.getAirTemperature();
    bool presMatch = (std::isnan(_airPressure) && std::isnan(other.getAirPressure())) ||
                     _airPressure == other.getAirPressure();
    bool humiMatch =
            (std::isnan(_humidity) && std::isnan(other.getHumidity())) || _humidity == other.getHumidity();
    return (tempMatch && presMatch && humiMatch);
}

std::size_t Weather::hash_value() const noexcept {
    // Completely arbitrary seed
    return cpputils::hashCombine(17, _airTemperature, _airPressure, _humidity);
}

void Weather::validate() const {
    if (_humidity < 0.0) {  // allow > 100 even though supersaturation is most unlikely
        std::ostringstream os;
        os << "Relative humidity = " << _humidity << " must not be negative";
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, os.str());
    }
}

std::ostream& operator<<(std::ostream& os, Weather const& weath) {
    return os << "Weather(" << weath.getAirTemperature() << ", " << weath.getAirPressure() << ", "
              << weath.getHumidity() << ")";
}
}  // namespace coord
}  // namespace afw
}  // namespace lsst

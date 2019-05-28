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

#include <sstream>
#include <string>

#include "boost/format.hpp"

#include "lsst/geom/Angle.h"
#include "lsst/afw/coord/Observatory.h"

namespace lsst {
namespace afw {
namespace coord {

Observatory::Observatory(lsst::geom::Angle const longitude, lsst::geom::Angle const latitude,
                         double const elevation)
        : _latitude(latitude), _longitude(longitude), _elevation(elevation) {}

Observatory::~Observatory() noexcept = default;

Observatory::Observatory(Observatory const&) noexcept = default;
Observatory::Observatory(Observatory&&) noexcept = default;
Observatory& Observatory::operator=(Observatory const&) noexcept = default;
Observatory& Observatory::operator=(Observatory&&) noexcept = default;

lsst::geom::Angle Observatory::getLongitude() const noexcept { return _longitude; }

lsst::geom::Angle Observatory::getLatitude() const noexcept { return _latitude; }

void Observatory::setLatitude(lsst::geom::Angle const latitude) { _latitude = latitude; }

void Observatory::setLongitude(lsst::geom::Angle const longitude) { _longitude = longitude; }

void Observatory::setElevation(double const elevation) { _elevation = elevation; }

std::string Observatory::toString() const {
    return (boost::format("%gW, %gN  %g") % getLongitude().asDegrees() % getLatitude().asDegrees() %
            getElevation())
            .str();
}

std::ostream& operator<<(std::ostream& os, Observatory const& obs) {
    os << obs.toString();
    return os;
}
}  // namespace coord
}  // namespace afw
}  // namespace lsst

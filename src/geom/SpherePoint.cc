// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See COPYRIGHT file at the top of the source tree.
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program. If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#include <cmath>
#include <string>
#include <sstream>

#include "Eigen/Geometry"

#include "lsst/afw/geom/SpherePoint.h"
#include "lsst/pex/exceptions.h"

namespace pexExcept = lsst::pex::exceptions;
using namespace std;

namespace lsst {
namespace afw {
namespace geom {
/// @internal Static implementation details for SpherePoint.
namespace {
/**
 * @internal Angular distance between two points using the Haversine formula.
 *
 * Besides the differences between the two coordinates, we also input the
 * cosine of the two declinations, so as to be thrifty with the use of
 * trigonometric functions.
 *
 * @param deltaLon Difference between longitudes.
 * @param deltaLat Difference between latitudes. Must be in the same
 *                 sense as `deltaLon`.
 * @param cosLat1, cosLat2 Cosines of the two points' latitudes.
 * @returns the distance between the two points
 */
Angle haversine(Angle const& deltaLon, Angle const& deltaLat, double cosLat1, double cosLat2) {
    double const sinLat = sin(deltaLat / 2.0);
    double const sinLon = sin(deltaLon / 2.0);
    double const havDDelta = sinLat * sinLat;
    double const havDAlpha = sinLon * sinLon;
    double const havD = havDDelta + cosLat1 * cosLat2 * havDAlpha;
    double const sinDHalf = sqrt(havD);
    return (2.0 * asin(sinDHalf)) * radians;
}
}  // end namespace

SpherePoint::SpherePoint(double longitude, double latitude, AngleUnit units)
        : SpherePoint(longitude * units, latitude * units) {}

SpherePoint::SpherePoint(Angle const& longitude, Angle const& latitude)
        : _longitude(longitude.wrap().asRadians()), _latitude(latitude.asRadians()) {
    if (fabs(_latitude) > HALFPI) {
        throw pexExcept::InvalidParameterError("Angle " + to_string(latitude.asDegrees()) +
                                               " is not a valid latitude.");
    }
}

SpherePoint::SpherePoint(Point3D const& vector) {
    double norm = vector.asEigen().norm();
    if (norm <= 0.0) {
        stringstream buffer;
        buffer << "Vector " << vector << " has zero norm and cannot be normalized.";
        throw pexExcept::InvalidParameterError(buffer.str());
    }
    // To avoid unexpected behavior from mixing finite elements and infinite norm
    if (!isfinite(norm)) {
        norm = NAN;
    }

    double const x = vector.getX() / norm;
    double const y = vector.getY() / norm;
    double const z = vector.getZ() / norm;

    _latitude = asin(z);
    if (!atPole()) {
        // Need to convert to Angle, Angle::wrap, and convert back to radians
        //     to handle _longitude = -1e-16 without code duplication
        _longitude = (atan2(y, x) * radians).wrap().asRadians();
    } else {
        _longitude = 0;
    }
}

SpherePoint::SpherePoint(SpherePoint const& other) noexcept = default;

SpherePoint::SpherePoint(SpherePoint&& other) noexcept = default;

SpherePoint& SpherePoint::operator=(SpherePoint const& other) noexcept = default;

SpherePoint& SpherePoint::operator=(SpherePoint&& other) noexcept = default;

SpherePoint::~SpherePoint() = default;

Point3D SpherePoint::getVector() const noexcept {
    return Point3D(cos(_longitude) * cos(_latitude), sin(_longitude) * cos(_latitude), sin(_latitude));
}

Angle SpherePoint::operator[](size_t index) const {
    switch (index) {
        case 0:
            return getLongitude();
        case 1:
            return getLatitude();
        default:
            throw pexExcept::OutOfRangeError("Index " + to_string(index) + " must be 0 or 1.");
    }
}

bool SpherePoint::isFinite() const noexcept { return isfinite(_longitude) && isfinite(_latitude); }

bool SpherePoint::operator==(SpherePoint const& other) const noexcept {
    // Deliberate override of Style Guide 5-12
    // Approximate FP comparison would make object equality intransitive
    return _longitude == other._longitude && _latitude == other._latitude;
}

bool SpherePoint::operator!=(SpherePoint const& other) const noexcept { return !(*this == other); }

Angle SpherePoint::bearingTo(SpherePoint const& other) const {
    Angle const deltaLon = other.getLongitude() - this->getLongitude();

    double const sinDelta1 = sin(getLatitude().asRadians());
    double const sinDelta2 = sin(other.getLatitude().asRadians());
    double const cosDelta1 = cos(getLatitude().asRadians());
    double const cosDelta2 = cos(other.getLatitude().asRadians());

    // Adapted from http://www.movable-type.co.uk/scripts/latlong.html
    double const y = sin(deltaLon) * cosDelta2;
    double const x = cosDelta1 * sinDelta2 - sinDelta1 * cosDelta2 * cos(deltaLon);
    return (90.0 * degrees - atan2(y, x) * radians).wrap();
}

Angle SpherePoint::separation(SpherePoint const& other) const noexcept {
    return haversine(getLongitude() - other.getLongitude(), getLatitude() - other.getLatitude(),
                     cos(getLatitude().asRadians()), cos(other.getLatitude().asRadians()));
}

SpherePoint SpherePoint::rotated(SpherePoint const& axis, Angle const& amount) const noexcept {
    auto const rotation = Eigen::AngleAxisd(amount.asRadians(), axis.getVector().asEigen()).matrix();
    auto const x = getVector().asEigen();
    auto const xprime = rotation * x;
    return SpherePoint(Point3D(xprime));
}

SpherePoint SpherePoint::offset(Angle const& bearing, Angle const& amount) const {
    double const phi = bearing.asRadians();

    // let v = vector in the direction bearing points (tangent to surface of sphere)
    // To do the rotation, use rotate() method.
    // - must provide an axis of rotation: take the cross product r x v to get that axis (pole)

    Eigen::Vector3d r = getVector().asEigen();

    // Get the vector v:
    //  let u = unit vector lying on a parallel of declination
    //  let w = unit vector along line of longitude = r x u
    // the vector v must satisfy the following:
    //  r . v = 0
    //  u . v = cos(phi)
    //  w . v = sin(phi)

    // v is a linear combination of u and w
    // v = cos(phi)*u + sin(phi)*w
    auto u = Eigen::Vector3d(-sin(_longitude), cos(_longitude), 0.0);
    auto w = r.cross(u);
    Eigen::Vector3d v = cos(phi) * u + sin(phi) * w;

    // take r x v to get the axis
    SpherePoint axis = SpherePoint(Point3D(r.cross(v)));

    return rotated(axis, amount);
}

ostream& operator<<(ostream& os, SpherePoint const& point) {
    // Can't provide atomic guarantee anyway for I/O, so ok to be sloppy.
    auto oldFlags = os.setf(ostream::fixed);
    auto oldPrecision = os.precision(6);

    os << "(" << point.getLongitude().asDegrees() << ", ";
    os.setf(ostream::showpos);
    os << point.getLatitude().asDegrees() << ")";

    os.flags(oldFlags);
    os.precision(oldPrecision);
    return os;
}
}
}
} /* namespace lsst::afw::geom */

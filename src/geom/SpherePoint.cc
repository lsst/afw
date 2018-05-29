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
#include "lsst/afw/geom/sphgeomUtils.h"
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

SpherePoint::SpherePoint(sphgeom::Vector3d const& vector) {
    // sphgeom Vector3d has its own normalization,
    // but its behavior is not documented for non-finite values
    double norm = vector.getNorm();
    if (norm <= 0.0) {
        stringstream buffer;
        buffer << "Vector " << vector << " has zero norm and cannot be normalized.";
        throw pexExcept::InvalidParameterError(buffer.str());
    }
    // To avoid unexpected behavior from mixing finite elements and infinite norm
    if (!isfinite(norm)) {
        norm = NAN;
    }
    auto unitVector =
            sphgeom::UnitVector3d::fromNormalized(vector.x() / norm, vector.y() / norm, vector.z() / norm);
    _set(unitVector);
}

SpherePoint::SpherePoint(sphgeom::LonLat const& lonLat)
        : SpherePoint(lonLat.getLon().asRadians(), lonLat.getLat().asRadians(), radians) {}

void SpherePoint::_set(sphgeom::UnitVector3d const& unitVector) {
    _latitude = asin(unitVector.z());
    if (!atPole()) {
        // Need to convert to Angle, Angle::wrap, and convert back to radians
        //     to handle _longitude = -1e-16 without code duplication
        _longitude = (atan2(unitVector.y(), unitVector.x()) * radians).wrap().asRadians();
    } else {
        _longitude = 0;
    }
}

SpherePoint::SpherePoint() : _longitude(nan("")), _latitude(nan("")) {}

SpherePoint::SpherePoint(SpherePoint const& other) noexcept = default;

SpherePoint::SpherePoint(SpherePoint&& other) noexcept = default;

SpherePoint& SpherePoint::operator=(SpherePoint const& other) noexcept = default;

SpherePoint& SpherePoint::operator=(SpherePoint&& other) noexcept = default;

SpherePoint::operator sphgeom::LonLat() const {
    return sphgeom::LonLat::fromRadians(getLongitude().asRadians(), getLatitude().asRadians());
}

SpherePoint::~SpherePoint() = default;

sphgeom::UnitVector3d SpherePoint::getVector() const noexcept {
    return sphgeom::UnitVector3d::fromNormalized(cos(_longitude) * cos(_latitude),
                                                 sin(_longitude) * cos(_latitude), sin(_latitude));
}

Point2D SpherePoint::getPosition(AngleUnit unit) const {
    return Point2D(getLongitude().asAngularUnits(unit), getLatitude().asAngularUnits(unit));
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
    auto const rotation = Eigen::AngleAxisd(amount.asRadians(), asEigen(axis.getVector())).matrix();
    auto const x = asEigen(getVector());
    auto const xprime = rotation * x;
    return SpherePoint(sphgeom::Vector3d(xprime[0], xprime[1], xprime[2]));
}

SpherePoint SpherePoint::offset(Angle const& bearing, Angle const& amount) const {
    double const phi = bearing.asRadians();

    // let v = vector in the direction bearing points (tangent to surface of sphere)
    // To do the rotation, use rotated() method.
    // - must provide an axis of rotation: take the cross product r x v to get that axis (pole)

    auto r = getVector();

    // Get the vector v:
    //  let u = unit vector lying on a parallel of declination
    //  let w = unit vector along line of longitude = r x u
    // the vector v must satisfy the following:
    //  r . v = 0
    //  u . v = cos(phi)
    //  w . v = sin(phi)

    // v is a linear combination of u and w
    // v = cos(phi)*u + sin(phi)*w
    sphgeom::Vector3d const u(-sin(_longitude), cos(_longitude), 0.0);
    auto w = r.cross(u);
    auto v = cos(phi) * u + sin(phi) * w;

    // take r x v to get the axis
    SpherePoint axis = SpherePoint(r.cross(v));

    return rotated(axis, amount);
}

std::pair<geom::Angle, geom::Angle> SpherePoint::getTangentPlaneOffset(SpherePoint const& other) const {
    geom::Angle const alpha1 = this->getLongitude();
    geom::Angle const delta1 = this->getLatitude();
    geom::Angle const alpha2 = other.getLongitude();
    geom::Angle const delta2 = other.getLatitude();

    // Compute the projection of "other" on a tangent plane centered at this point
    double const sinDelta1 = std::sin(delta1);
    double const cosDelta1 = std::cos(delta1);
    double const sinDelta2 = std::sin(delta2);
    double const cosDelta2 = std::cos(delta2);
    double const cosAlphaDiff = std::cos(alpha2 - alpha1);
    double const sinAlphaDiff = std::sin(alpha2 - alpha1);

    double const div = cosDelta1 * cosAlphaDiff * cosDelta2 + sinDelta1 * sinDelta2;
    double const xi = cosDelta1 * sinAlphaDiff / div;
    double const eta = (cosDelta1 * cosAlphaDiff * sinDelta2 - sinDelta1 * cosDelta2) / div;

    return std::make_pair(xi * geom::radians, eta * geom::radians);
}

SpherePoint averageSpherePoint(std::vector<SpherePoint> const& coords) {
    if (coords.size() == 0) {
        throw LSST_EXCEPT(pex::exceptions::LengthError, "No coordinates provided to average");
    }
    sphgeom::Vector3d sum(0, 0, 0);
    sphgeom::Vector3d corr(0, 0, 0);  // Kahan summation correction
    for (auto const& sp : coords) {
        auto const point = sp.getVector();
        // Kahan summation
        auto const add = point - corr;
        auto const temp = sum + add;
        corr = (temp - sum) - add;
        sum = temp;
    }
    sum /= static_cast<double>(coords.size());
    return SpherePoint(sum);
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
}  // namespace geom
}  // namespace afw
}  // namespace lsst

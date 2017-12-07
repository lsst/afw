// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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

/*
 * Provide functions to handle coordinates
 *
 * Most (nearly all) algorithms adapted from Astronomical Algorithms, 2nd ed. (J. Meeus)
 *
 */
#include <cmath>
#include <limits>
#include <cstdio>
#include <iomanip>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include "lsst/pex/exceptions.h"
#include "boost/algorithm/string.hpp"
#include "boost/tuple/tuple.hpp"
#include "boost/format.hpp"

#include "lsst/afw/coord/Coord.h"
#include "lsst/daf/base/DateTime.h"

namespace ex = lsst::pex::exceptions;
namespace dafBase = lsst::daf::base;

namespace lsst {
namespace afw {
namespace coord {

namespace {

typedef std::map<std::string, CoordSystem> CoordSystemMap;

CoordSystemMap const getCoordSystemMap() {
    CoordSystemMap idMap;
    idMap["FK5"] = FK5;
    idMap["ICRS"] = ICRS;
    idMap["ECLIPTIC"] = ECLIPTIC;
    idMap["GALACTIC"] = GALACTIC;
    idMap["ELON"] = ECLIPTIC;
    idMap["GLON"] = GALACTIC;
    idMap["TOPOCENTRIC"] = TOPOCENTRIC;
    return idMap;
}

/** @internal Calculate angular distance between two points using the Haversine formula
 *
 * Besides the differences between the two coordinates, we also input the
 * cosine of the two declinations, so as to be thrifty with the use of trig functions.
 *
 * @param dAlpha Difference between RAs (or longitudes), alpha1-alpha2
 * @param dDelta Difference between Decs (or latitudes), delta1-delta2
 * @param cosDelta1 Cosine of the first Dec, cos(delta1)
 * @param cosDelta2 Cosine of the second Dec, cos(delta2)
 */
geom::Angle haversine(geom::Angle const dAlpha, geom::Angle const dDelta, double const cosDelta1,
                      double const cosDelta2) {
    double const havDDelta = std::sin(dDelta / 2.0) * std::sin(dDelta / 2.0);
    double const havDAlpha = std::sin(dAlpha / 2.0) * std::sin(dAlpha / 2.0);
    double const havD = havDDelta + cosDelta1 * cosDelta2 * havDAlpha;
    double const sinDHalf = std::sqrt(havD);
    geom::Angle dist = (2.0 * std::asin(sinDHalf)) * geom::radians;
    return dist;
}

/// @internal Precession to new epoch performed if two epochs differ by this.
double const epochTolerance = 1.0e-12;

/// @internal Put a pair of coordinates in a common FK5 system
std::pair<Fk5Coord, Fk5Coord> commonFk5(Coord const &c1, Coord const &c2) {
    // make sure they're fk5
    Fk5Coord fk51 = c1.toFk5();
    Fk5Coord fk5tmp = c2.toFk5();

    // make sure they have the same epoch
    Fk5Coord fk52;
    if (fabs(fk51.getEpoch() - fk5tmp.getEpoch()) > epochTolerance) {
        fk52 = fk5tmp.precess(fk51.getEpoch());
    } else {
        fk52 = fk5tmp;
    }

    return std::make_pair(fk51, fk52);
}

}  // end anonymous namespace

CoordSystem makeCoordEnum(std::string const system) {
    static CoordSystemMap idmap = getCoordSystemMap();
    if (idmap.find(system) != idmap.end()) {
        return idmap[system];
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterError, "System " + system + " not defined.");
    }
}

namespace {

double const NaN = std::numeric_limits<double>::quiet_NaN();
double const JD2000 = 2451544.50;

/*
 * A local class to handle dd:mm:ss coordinates
 *
 * This class allows a decimal or dd:mm:ss coordinate to be
 * disassembed into d, m, and s components.
 * It's in an anonymous namespace, but there are public functions
 * which perform the transformations directly:
 *
 * --> std::string dmsStr = degreesToDmsString(double deg);
 * --> double deg = dmsStringToDegrees(std::string dms);
 */
class Dms {
public:
    Dms()= default;;

    // note that isSouth is needed to specify coords between dec = 0, and dec = -1
    // otherwise, d = -0 gets carried as d = 0 ... need a way to specify it explicitly
    Dms(int const d, int const m, double const s, bool const isSouth = false) {
        sign = (d < 0 || isSouth) ? -1 : 1;
        deg = std::abs(d);
        min = m;
        sec = s;
    };
    // unit could be "degrees" or "hours"
    explicit Dms(geom::Angle const deg00, geom::AngleUnit const unit = geom::degrees) {
        double deg0 = deg00.asAngularUnits(unit);
        double const absVal = std::fabs(deg0);
        sign = (deg0 >= 0) ? 1 : -1;
        deg = static_cast<int>(std::floor(absVal));
        min = static_cast<int>(std::floor((absVal - deg) * 60.0));
        sec = ((absVal - deg) * 60.0 - min) * 60.0;
    }

    int deg;
    int min;
    double sec;
    int sign;
};

/**
 * @internal Store the Fk5 coordinates of the Galactic pole (and vice-versa) for coordinate transforms.
 *
 */
Coord const &GalacticPoleInFk5() {
    static Coord pole(192.85950 * geom::degrees, 27.12825 * geom::degrees, 2000.0);  // C&O
    return pole;
}

Coord const &Fk5PoleInGalactic() {
    static Coord pole(122.93200 * geom::degrees, 27.12825 * geom::degrees, 2000.0);  // C&O
    return pole;
}

/**
 * @internal Compute the mean Sidereal Time at Greenwich
 *
 */
geom::Angle meanSiderealTimeGreenwich(double const jd  ///< Julian Day
                                      ) {
    double const T = (jd - 2451545.0) / 36525.0;
    return (280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T * T -
            (T * T * T / 38710000.0)) *
           geom::degrees;
}

double const atPoleEpsilon = 0.0;  // std::numeric_limits<double>::epsilon();
std::pair<geom::Angle, geom::Angle> pointToLonLat(geom::Point3D const &p3d,
                                                  double const defaultLongitude = 0.0,
                                                  bool normalize = true) {
    std::pair<geom::Angle, geom::Angle> lonLat;

    if (normalize) {
        double const inorm = 1.0 / p3d.asEigen().norm();
        double const x = inorm * p3d.getX();
        double const y = inorm * p3d.getY();
        double const z = inorm * p3d.getZ();
        if (fabs(x) <= atPoleEpsilon && fabs(y) <= atPoleEpsilon) {
            lonLat.first = 0.0 * geom::radians;
            lonLat.second = ((z >= 0) ? 1.0 : -1.0) * geom::HALFPI * geom::radians;
        } else {
            lonLat.first = (atan2(y, x) * geom::radians).wrap();
            lonLat.second = asin(z) * geom::radians;
        }
    } else {
        if (fabs(p3d.getX()) <= atPoleEpsilon && fabs(p3d.getY()) <= atPoleEpsilon) {
            lonLat.first = 0.0 * geom::radians;
            lonLat.second = ((p3d.getZ() >= 0) ? 1.0 : -1.0) * geom::HALFPI * geom::radians;
        } else {
            lonLat.first = (atan2(p3d.getY(), p3d.getX()) * geom::radians).wrap();
            lonLat.second = asin(p3d.getZ()) * geom::radians;
        }
    }
    return lonLat;
}

}  // end anonymous namespace

/* ******************* Public functions ******************* */

static std::string angleToXmsString(geom::Angle const a, geom::AngleUnit const unit) {
    Dms dms(a, unit);

    // make sure rounding won't give 60.00 for sec or min
    if ((60.00 - dms.sec) < 0.005) {
        dms.sec = 0.0;
        dms.min += 1;
        if (dms.min == 60) {
            dms.min = 0;
            dms.deg += 1;
            if (dms.deg == 360) {
                dms.deg = 0;
            }
        }
    }

    std::string fmt("%02d:%02d:%05.2f");
    std::string s = (boost::format(fmt) % dms.deg % dms.min % dms.sec).str();
    if (dms.sign < 0) {
        s = "-" + s;
    }
    return s;
}

std::string angleToDmsString(geom::Angle const a) { return angleToXmsString(a, geom::degrees); }

std::string angleToHmsString(geom::Angle const a) { return angleToXmsString(a, geom::hours); }

/**
 * @internal Convert a XX:mm:ss string to Angle
 *
 * @param dms Coord as a string in dd:mm:ss format
 * @param unit the units assumed for the first part of `dms`. The second and third
 *             parts shall be defined to be 1/60 and 1/3600 of `unit`, respectively.
 */
static geom::Angle xmsStringToAngle(std::string const dms, geom::AngleUnit unit) {
    if (dms.find(":") == std::string::npos) {
        throw LSST_EXCEPT(ex::InvalidParameterError,
                          (boost::format("String is not in xx:mm:ss format: %s") % dms).str());
    }
    std::vector<std::string> elements;
    boost::split(elements, dms, boost::is_any_of(":"));
    if (elements.size() != 3) {
        throw LSST_EXCEPT(ex::InvalidParameterError,
                          (boost::format("Could not parse string as xx:mm:ss format: %s") % dms).str());
    }
    int const deg = abs(atoi(elements[0].c_str()));
    int const min = atoi(elements[1].c_str());
    double const sec = atof(elements[2].c_str());

    geom::Angle ang = (deg + min / 60.0 + sec / 3600.0) * unit;
    if ((elements[0].c_str())[0] == '-') {
        ang *= -1.0;
    }
    return ang;
}

geom::Angle hmsStringToAngle(std::string const hms) { return xmsStringToAngle(hms, geom::hours); }

geom::Angle dmsStringToAngle(std::string const dms) { return xmsStringToAngle(dms, geom::degrees); }

geom::Angle eclipticPoleInclination(double const epoch) {
    double const T = (epoch - 2000.0) / 100.0;
    return (23.0 + 26.0 / 60.0 + (21.448 - 46.82 * T - 0.0006 * T * T - 0.0018 * T * T * T) / 3600.0) *
           geom::degrees;
}

/* ============================================================
 *
 * class Coord
 *
 * ============================================================*/

Coord::Coord(geom::Point2D const &p2d, geom::AngleUnit unit, double const epoch)
        : _longitude(NaN), _latitude(NaN), _epoch(epoch) {
    _longitude = geom::Angle(p2d.getX(), unit).wrap();
    _latitude = geom::Angle(p2d.getY(), unit);
    _verifyValues();
}

Coord::Coord(geom::Point3D const &p3d, double const epoch, bool normalize, geom::Angle const defaultLongitude)
        : _longitude(0. * geom::radians), _latitude(0. * geom::radians), _epoch(epoch) {
    std::pair<geom::Angle, geom::Angle> lonLat = pointToLonLat(p3d, defaultLongitude, normalize);
    _longitude = lonLat.first;
    _latitude = lonLat.second;
    _epoch = epoch;
}

Coord::Coord(geom::Angle const ra, geom::Angle const dec, double const epoch)
        : _longitude(ra.wrap()), _latitude(dec), _epoch(epoch) {
    _verifyValues();
}

Coord::Coord(std::string const ra, std::string const dec, double const epoch)
        : _longitude(hmsStringToAngle(ra).wrap()), _latitude(dmsStringToAngle(dec)), _epoch(epoch) {
    _verifyValues();
}

Coord::Coord() : _longitude(geom::Angle(NaN)), _latitude(geom::Angle(NaN)), _epoch(NaN) {}

void Coord::_verifyValues() const {
    if (_latitude.asRadians() < -geom::HALFPI || _latitude.asRadians() > geom::HALFPI) {
        throw LSST_EXCEPT(
                ex::InvalidParameterError,
                (boost::format("Latitude coord must be: -PI/2 <= lat <= PI/2 (%f).") % _latitude).str());
    }
}

void Coord::reset(geom::Angle const longitude, geom::Angle const latitude, double const epoch) {
    _longitude = longitude.wrap();
    _latitude = latitude;
    _epoch = epoch;
    _verifyValues();
}

geom::Point2D Coord::getPosition(geom::AngleUnit unit) const {
    // treat HOURS specially, they must mean hours for RA, degrees for Dec
    if (unit == geom::hours) {
        return geom::Point2D(getLongitude().asHours(), getLatitude().asDegrees());
    } else {
        return geom::Point2D(getLongitude().asAngularUnits(unit), getLatitude().asAngularUnits(unit));
    }
}

geom::Point3D Coord::getVector() const {
    double lng = getLongitude();
    double lat = getLatitude();
    double const x = std::cos(lng) * std::cos(lat);
    double const y = std::sin(lng) * std::cos(lat);
    double const z = std::sin(lat);
    return geom::Point3D(x, y, z);
}

Coord Coord::transform(Coord const &poleTo, Coord const &poleFrom) const {
    double const alphaGP = poleFrom[0];
    double const deltaGP = poleFrom[1];
    double const lCP = poleTo[0];

    double const alpha = getLongitude();
    double const delta = getLatitude();

    geom::Angle const l = (lCP - std::atan2(std::sin(alpha - alphaGP),
                                            std::tan(delta) * std::cos(deltaGP) -
                                                    std::cos(alpha - alphaGP) * std::sin(deltaGP))) *
                          geom::radians;
    geom::Angle const b = std::asin((std::sin(deltaGP) * std::sin(delta) +
                                     std::cos(deltaGP) * std::cos(delta) * std::cos(alpha - alphaGP))) *
                          geom::radians;
    return Coord(l, b);
}

void Coord::rotate(Coord const &axis, geom::Angle const theta) {
    double const c = std::cos(theta);
    double const mc = 1.0 - c;
    double const s = std::sin(theta);

    // convert to cartesian
    geom::Point3D const x = getVector();
    geom::Point3D const u = axis.getVector();
    double const ux = u[0];
    double const uy = u[1];
    double const uz = u[2];

    // rotate
    geom::Point3D xprime;
    xprime[0] = (ux * ux + (1.0 - ux * ux) * c) * x[0] + (ux * uy * mc - uz * s) * x[1] +
                (ux * uz * mc + uy * s) * x[2];
    xprime[1] = (uy * uy + (1.0 - uy * uy) * c) * x[1] + (uy * uz * mc - ux * s) * x[2] +
                (ux * uy * mc + uz * s) * x[0];
    xprime[2] = (uz * uz + (1.0 - uz * uz) * c) * x[2] + (uz * ux * mc - uy * s) * x[0] +
                (uy * uz * mc + ux * s) * x[1];

    // in-situ
    std::pair<geom::Angle, geom::Angle> lonLat = pointToLonLat(xprime);
    _longitude = lonLat.first;
    _latitude = lonLat.second;
}

geom::Angle Coord::offset(geom::Angle const phi, geom::Angle const arcLen) {
    // let v = vector in the direction arcLen points (tangent to surface of sphere)
    // thus: |v| = arcLen
    //       angle phi = orientation of v in a tangent plane, measured wrt to a parallel of declination

    // To do the rotation, use rotate() method.
    // - must provide an axis of rotation: take the cross product r x v to get that axis (pole)

    // get the vector r
    Eigen::Vector3d r = getVector().asEigen();

    // Get the vector v:
    // let u = unit vector lying on a parallel of declination
    // let w = unit vector along line of longitude = r x u
    // the vector v must satisfy the following:
    // |v| = arcLen
    // r . v = 0
    // u . v = |v| cos(phi) = arcLen*cos(phi)
    // w . v = |v| sin(phi) = arcLen*sin(phi)

    // v is a linear combination of u and w
    // v = arcLen*cos(phi)*u + arcLen*sin(phi)*w

    // Thus, we must:
    // - create u vector
    // - solve w vector (r cross u)
    // - compute v
    Eigen::Vector3d u;
    u << -std::sin(getLongitude()), std::cos(getLongitude()), 0.0;
    Eigen::Vector3d w = r.cross(u);
    Eigen::Vector3d v = arcLen * std::cos(phi) * u + arcLen * std::sin(phi) * w;

    // take r x v to get the axis
    Eigen::Vector3d axisVector = r.cross(v);
    axisVector.normalize();
    Coord axisCoord = Coord(geom::Point3D(axisVector), getEpoch());

    rotate(axisCoord, arcLen);

    // now get the position angle at our destination
    // u2 . v2 = arcLen*cos(phi2)
    // w2 . v2 = arcLen*sin(phi2)
    // if we normalize v2:
    // phi2 = atan2(w2.v2, u2.v2)
    //
    // we need to compute u2, and then rotate v (exactly as we rotated r) to get v2
    Eigen::Vector3d r2 = getVector().asEigen();
    Eigen::Vector3d u2;
    u2 << -std::sin(getLongitude()), std::cos(getLongitude()), 0.0;
    Eigen::Vector3d w2 = r2.cross(u2);

    // make v a unit vector and rotate v exactly as we rotated r
    v.normalize();
    Coord v2Coord = Coord(geom::Point3D(v), getEpoch());
    v2Coord.rotate(axisCoord, arcLen);
    Eigen::Vector3d v2 = v2Coord.getVector().asEigen();

    geom::Angle phi2 = std::atan2(w2.dot(v2), u2.dot(v2)) * geom::radians;

    return phi2;
}

std::shared_ptr<Coord> Coord::convert(CoordSystem system, double epoch) const {
    switch (system) {
        case FK5: {
            Fk5Coord c1 = this->toFk5(epoch);
            return std::shared_ptr<Fk5Coord>(
                    new Fk5Coord(c1.getLongitude(), c1.getLatitude(), c1.getEpoch()));
        } break;
        case ICRS: {
            IcrsCoord c2 = this->toIcrs();
            return std::shared_ptr<IcrsCoord>(new IcrsCoord(c2.getLongitude(), c2.getLatitude()));
        } break;
        case GALACTIC: {
            GalacticCoord c4 = this->toGalactic();
            return std::shared_ptr<GalacticCoord>(new GalacticCoord(c4.getLongitude(), c4.getLatitude()));
        } break;
        case ECLIPTIC: {
            EclipticCoord c5 = this->toEcliptic(epoch);
            return std::shared_ptr<EclipticCoord>(
                    new EclipticCoord(c5.getLongitude(), c5.getLatitude(), c5.getEpoch()));
        } break;
        case TOPOCENTRIC:
            throw LSST_EXCEPT(ex::InvalidParameterError,
                              "Cannot make Topocentric with convert() (must also specify Observatory).\n"
                              "Instantiate TopocentricCoord() directly.");
            break;
        case UNKNOWN:
            throw LSST_EXCEPT(ex::InvalidParameterError, "Cannot convert to UNKNOWN coordinate system");
            break;
        default:
            throw LSST_EXCEPT(ex::InvalidParameterError,
                              "Undefined CoordSystem: only FK5, ICRS, GALACTIC, ECLIPTIC allowed.");
            break;
    }
}

geom::Angle Coord::angularSeparation(Coord const &c) const {
    // work in Fk5, no matter what two derived classes we're given (eg Fk5 and Galactic)
    // we'll put them in the same system.
    std::pair<Fk5Coord, Fk5Coord> const &fk5 = commonFk5(*this, c);
    geom::Angle const alpha1 = fk5.first.getRa();
    geom::Angle const delta1 = fk5.first.getDec();
    geom::Angle const alpha2 = fk5.second.getRa();
    geom::Angle const delta2 = fk5.second.getDec();

    return haversine(alpha1 - alpha2, delta1 - delta2, std::cos(delta1), std::cos(delta2));
}

std::pair<geom::Angle, geom::Angle> Coord::getOffsetFrom(Coord const &c) const {
    // work in Fk5, no matter what two derived classes we're given (eg Fk5 and Galactic)
    // we'll put them in the same system.
    std::pair<Fk5Coord, Fk5Coord> const &fk5 = commonFk5(*this, c);
    geom::Angle const alpha1 = fk5.first.getRa();
    geom::Angle const delta1 = fk5.first.getDec();
    geom::Angle const alpha2 = fk5.second.getRa();
    geom::Angle const delta2 = fk5.second.getDec();

    geom::Angle const dAlpha = alpha1 - alpha2;
    geom::Angle const dDelta = delta1 - delta2;

    double const cosDelta1 = std::cos(delta1);
    double const cosDelta2 = std::cos(delta2);

    geom::Angle separation = haversine(dAlpha, dDelta, cosDelta1, cosDelta2);

    // Formula from http://www.movable-type.co.uk/scripts/latlong.html
    double const y = std::sin(dAlpha) * cosDelta2;
    double const x = cosDelta1 * std::sin(delta2) - std::sin(delta1) * cosDelta2 * std::cos(dAlpha);
    geom::Angle bearing = std::atan2(y, x) * geom::radians - 90.0 * geom::degrees;

    return std::make_pair(bearing, separation);
}

std::pair<geom::Angle, geom::Angle> Coord::getTangentPlaneOffset(Coord const &c) const {
    // work in Fk5, no matter what two derived classes we're given (eg Fk5 and Galactic)
    // we'll put them in the same system.
    std::pair<Fk5Coord, Fk5Coord> const &fk5 = commonFk5(*this, c);
    geom::Angle const alpha1 = fk5.first.getRa();
    geom::Angle const delta1 = fk5.first.getDec();
    geom::Angle const alpha2 = fk5.second.getRa();
    geom::Angle const delta2 = fk5.second.getDec();

    // This is a projection of coord2 to the tangent plane at coord1
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

Fk5Coord Coord::toFk5(double const epoch) const {
    return Fk5Coord(getLongitude(), getLatitude(), getEpoch()).precess(epoch);
}
Fk5Coord Coord::toFk5() const { return Fk5Coord(getLongitude(), getLatitude(), getEpoch()); }

IcrsCoord Coord::toIcrs() const { return this->toFk5().toIcrs(); }

GalacticCoord Coord::toGalactic() const { return this->toFk5().toGalactic(); }

EclipticCoord Coord::toEcliptic(double const epoch) const { return this->toFk5(epoch).toEcliptic(); }
EclipticCoord Coord::toEcliptic() const { return this->toFk5().toEcliptic(); }

TopocentricCoord Coord::toTopocentric(Observatory const &obs,
                                      lsst::daf::base::DateTime const &obsDate) const {
    return this->toFk5().toTopocentric(obs, obsDate);
}

/* ============================================================
 *
 * class Fk5Coord
 *
 * ============================================================*/

Fk5Coord Fk5Coord::toFk5(double const epoch) const {
    return Fk5Coord(getLongitude(), getLatitude(), getEpoch()).precess(epoch);
}
Fk5Coord Fk5Coord::toFk5() const { return Fk5Coord(getLongitude(), getLatitude(), getEpoch()); }

IcrsCoord Fk5Coord::toIcrs() const {
    // only do the precession to 2000 if we're not already there.
    if (fabs(getEpoch() - 2000.0) > epochTolerance) {
        Fk5Coord c = precess(2000.0);
        return IcrsCoord(c.getLongitude(), c.getLatitude());
    } else {
        return IcrsCoord(getLongitude(), getLatitude());
    }
}

GalacticCoord Fk5Coord::toGalactic() const {
    // if we're epoch==2000, we can transform, otherwise we need to precess first
    Fk5Coord c;
    if (fabs(getEpoch() - 2000.0) > epochTolerance) {
        c = precess(2000.0);
    } else {
        c = *this;
    }

    Coord ct = c.transform(Fk5PoleInGalactic(), GalacticPoleInFk5());
    return GalacticCoord(ct.getLongitude(), ct.getLatitude());
}

EclipticCoord Fk5Coord::toEcliptic(double const epoch) const {
    geom::Angle const eclPoleIncl = eclipticPoleInclination(epoch);
    Coord const eclPoleInEquatorial(270.0 * geom::degrees, (90.0 * geom::degrees) - eclPoleIncl, epoch);
    Coord const equPoleInEcliptic(90.0 * geom::degrees, (90.0 * geom::degrees) - eclPoleIncl, epoch);
    Coord c = transform(equPoleInEcliptic, eclPoleInEquatorial);
    return EclipticCoord(c.getLongitude(), c.getLatitude(), epoch);
}
EclipticCoord Fk5Coord::toEcliptic() const { return this->toEcliptic(getEpoch()); }

TopocentricCoord Fk5Coord::toTopocentric(Observatory const &obs,
                                         lsst::daf::base::DateTime const &obsDate) const {
    // make sure we precess to the epoch
    Fk5Coord fk5 = precess(obsDate.get(dafBase::DateTime::EPOCH));

    // greenwich sidereal time
    geom::Angle theta0 = meanSiderealTimeGreenwich(obsDate.get(dafBase::DateTime::JD)).wrap();

    // lat/long of the observatory
    geom::Angle const phi = obs.getLatitude();
    geom::Angle const L = obs.getLongitude();

    // ra/dec of the target
    geom::Angle const alpha = fk5.getRa();
    geom::Angle const delta = fk5.getDec();

    geom::Angle const H = theta0 + L - alpha;

    // compute the altitude, h
    double const sinh = std::sin(phi) * std::sin(delta) + std::cos(phi) * std::cos(delta) * std::cos(H);
    geom::Angle const h = std::asin(sinh) * geom::radians;

    // compute the azimuth, A
    double const tanAnumerator = std::sin(H);
    double const tanAdenominator = (std::cos(H) * std::sin(phi) - std::tan(delta) * std::cos(phi));

    // Equations used here assume azimuth is with respect to South
    // but we use the North as our origin ... must add 180 deg
    geom::Angle A = ((180.0 * geom::degrees) + atan2(tanAnumerator, tanAdenominator) * geom::radians).wrap();

    return TopocentricCoord(A, h, obsDate.get(dafBase::DateTime::EPOCH), obs);
}

Fk5Coord Fk5Coord::precess(double const epochTo) const {
    // return a copy if the epochs are the same
    if (fabs(getEpoch() - epochTo) < epochTolerance) {
        return Fk5Coord(getLongitude(), getLatitude(), getEpoch());
    }

    dafBase::DateTime const dateFrom(getEpoch(), dafBase::DateTime::EPOCH, dafBase::DateTime::TAI);
    dafBase::DateTime const dateTo(epochTo, dafBase::DateTime::EPOCH, dafBase::DateTime::TAI);
    double const jd0 = dateFrom.get(dafBase::DateTime::JD);
    double const jd = dateTo.get(dafBase::DateTime::JD);

    double const T = (jd0 - JD2000) / 36525.0;
    double const t = (jd - jd0) / 36525.0;
    double const tt = t * t;
    double const ttt = tt * t;

    geom::Angle const xi = ((2306.2181 + 1.39656 * T - 0.000139 * T * T) * t + (0.30188 - 0.000344 * T) * tt +
                            0.017998 * ttt) *
                           geom::arcseconds;
    geom::Angle const z = ((2306.2181 + 1.39656 * T - 0.000139 * T * T) * t + (1.09468 + 0.000066 * T) * tt +
                           0.018203 * ttt) *
                          geom::arcseconds;
    geom::Angle const theta = ((2004.3109 - 0.85330 * T - 0.000217 * T * T) * t -
                               (0.42665 + 0.000217 * T) * tt - 0.041833 * ttt) *
                              geom::arcseconds;

    Fk5Coord fk5 = this->toFk5();
    geom::Angle const alpha0 = fk5.getRa();
    geom::Angle const delta0 = fk5.getDec();

    double const a = std::cos(delta0) * std::sin((alpha0 + xi));
    double const b =
            std::cos(theta) * std::cos(delta0) * std::cos((alpha0 + xi)) - std::sin(theta) * std::sin(delta0);
    double const c =
            std::sin(theta) * std::cos(delta0) * std::cos((alpha0 + xi)) + std::cos(theta) * std::sin(delta0);

    geom::Angle const alpha = (std::atan2(a, b) + z) * geom::radians;
    geom::Angle const delta = std::asin(c) * geom::radians;

    return Fk5Coord(alpha, delta, epochTo);
}

/* ============================================================
 *
 * class IcrsCoord
 *
 * ============================================================*/

void IcrsCoord::reset(geom::Angle const longitude, geom::Angle const latitude) {
    Coord::reset(longitude, latitude, 2000.0);
}

Fk5Coord IcrsCoord::toFk5(double const epoch) const {
    return Fk5Coord(getLongitude(), getLatitude(), 2000.0).precess(epoch);
}
Fk5Coord IcrsCoord::toFk5() const { return Fk5Coord(getLongitude(), getLatitude(), 2000.0); }

IcrsCoord IcrsCoord::toIcrs() const { return IcrsCoord(getLongitude(), getLatitude()); }

std::string IcrsCoord::toString() const {
    return (boost::format("%s(%.7f, %.7f)") % getClassName() % (*this)[0].asDegrees() %
            (*this)[1].asDegrees())
            .str();
}

/* ============================================================
 *
 * class GalacticCoord
 *
 * ============================================================*/

void GalacticCoord::reset(geom::Angle const longitudeDeg, geom::Angle const latitudeDeg) {
    Coord::reset(longitudeDeg, latitudeDeg, 2000.0);
}

Fk5Coord GalacticCoord::toFk5(double const epoch) const {
    // transform to fk5
    // galactic coords are ~constant, and the poles used are for epoch=2000, so we get J2000
    Coord c = transform(GalacticPoleInFk5(), Fk5PoleInGalactic());
    return Fk5Coord(c.getLongitude(), c.getLatitude(), 2000.0).precess(epoch);
}
Fk5Coord GalacticCoord::toFk5() const { return this->toFk5(2000.0); }

GalacticCoord GalacticCoord::toGalactic() const { return GalacticCoord(getLongitude(), getLatitude()); }

std::string GalacticCoord::toString() const {
    return (boost::format("%s(%.7f, %.7f)") % getClassName() % (*this)[0].asDegrees() %
            (*this)[1].asDegrees())
            .str();
}

/* ============================================================
 *
 * class EclipticCoord
 *
 * ============================================================*/

EclipticCoord EclipticCoord::toEcliptic(double const epoch) const {
    return EclipticCoord(getLongitude(), getLatitude(), getEpoch()).precess(epoch);
}
EclipticCoord EclipticCoord::toEcliptic() const {
    return EclipticCoord(getLongitude(), getLatitude(), getEpoch());
}

Fk5Coord EclipticCoord::toFk5(double const epoch) const {
    geom::Angle const eclPoleIncl = eclipticPoleInclination(epoch);
    geom::Angle ninety = 90. * geom::degrees;
    Coord const eclipticPoleInFk5(270.0 * geom::degrees, ninety - eclPoleIncl, epoch);
    Coord const fk5PoleInEcliptic(ninety, ninety - eclPoleIncl, epoch);
    Coord c = transform(eclipticPoleInFk5, fk5PoleInEcliptic);
    return Fk5Coord(c.getLongitude(), c.getLatitude(), epoch);
}
Fk5Coord EclipticCoord::toFk5() const { return this->toFk5(getEpoch()); }

EclipticCoord EclipticCoord::precess(double const epochTo) const {
    return this->toFk5().precess(epochTo).toEcliptic();
}

/* ============================================================
 *
 * class TopocentricCoord
 *
 * ============================================================*/

Fk5Coord TopocentricCoord::toFk5(double const epoch) const {
    geom::Angle const phi = _obs.getLatitude();
    geom::Angle const L = _obs.getLongitude();

    // Equations used here assume azimuth is with respect to South
    // but we use the North as our origin.
    geom::Angle A = (getAzimuth() + 180.0 * geom::degrees).wrap();
    geom::Angle const h = getAltitude();

    double const jd = dafBase::DateTime(epoch, dafBase::DateTime::EPOCH, dafBase::DateTime::TAI)
                              .get(dafBase::DateTime::JD);
    geom::Angle theta0 = meanSiderealTimeGreenwich(jd).wrap();

    double const tanHnum = std::sin(A);
    double const tanHdenom = std::cos(A) * std::sin(phi) + std::tan(h) * std::cos(phi);
    geom::Angle H = std::atan2(tanHnum, tanHdenom) * geom::radians;

    geom::Angle const alpha = theta0 + L - H;
    double const sinDelta = std::sin(phi) * std::sin(h) - std::cos(phi) * std::cos(h) * std::cos(A);
    geom::Angle const delta = (std::asin(sinDelta)) * geom::radians;

    return Fk5Coord(alpha, delta, epoch);
}
Fk5Coord TopocentricCoord::toFk5() const { return this->toFk5(getEpoch()); }

TopocentricCoord TopocentricCoord::toTopocentric(Observatory const &obs,
                                                 lsst::daf::base::DateTime const &date) const {
    if (obs != _obs) {
        throw LSST_EXCEPT(ex::InvalidParameterError,
                          (boost::format("Expected observatory %s, saw %s") % _obs % obs).str());
    }
    if (fabs(date.get() - getEpoch()) > std::numeric_limits<double>::epsilon()) {
        throw LSST_EXCEPT(ex::InvalidParameterError,
                          (boost::format("Expected date %g, saw %g") % getEpoch() % date.get()).str());
    }

    return TopocentricCoord(getLongitude(), getLatitude(), getEpoch(), _obs);
}

TopocentricCoord TopocentricCoord::toTopocentric() const {
    return TopocentricCoord(getLongitude(), getLatitude(), getEpoch(), _obs);
}

std::string TopocentricCoord::toString() const {
    return (boost::format("%s(%.7f, %.7f, %.12f, (%s))") % getClassName() % (*this)[0].asDegrees() %
            (*this)[1].asDegrees() % getEpoch() % getObservatory())
            .str();
}

/* ===============================================================================
 *
 * Factory function definitions:
 *
 * Each makeCoord() function has overloaded variants with and without epoch specified.
 *     It was tempting to specify default values, but some coordinate systems have
 *     no epoch, and this would be confusing to the user.  In its current form,
 *     any epochless coordinate system requested with an epoch specified will throw an exception.
 *     The epochless factories will always work, but assume default epoch = 2000.
 *
 * ===============================================================================
 */

std::shared_ptr<Coord> makeCoord(CoordSystem const system, geom::Angle const ra, geom::Angle const dec,
                                 double const epoch) {
    switch (system) {
        case FK5:
            return std::shared_ptr<Fk5Coord>(new Fk5Coord(ra, dec, epoch));
            break;
        case ICRS:
            throw LSST_EXCEPT(ex::InvalidParameterError,
                              "ICRS has no epoch, use overloaded makeCoord with args (system, ra, dec).");
            break;
        case GALACTIC:
            throw LSST_EXCEPT(ex::InvalidParameterError,
                              "Galactic has no epoch, use overloaded makeCoord with (system, ra, dec).");
            break;
        case ECLIPTIC:
            return std::shared_ptr<EclipticCoord>(new EclipticCoord(ra, dec, epoch));
            break;
        case TOPOCENTRIC:
            throw LSST_EXCEPT(ex::InvalidParameterError,
                              "Cannot make Topocentric with makeCoord() (must also specify Observatory).\n"
                              "Instantiate TopocentricCoord() directly.");
            break;
        default:
            throw LSST_EXCEPT(
                    ex::InvalidParameterError,
                    "Undefined CoordSystem: only FK5, ICRS, GALACTIC, ECLIPTIC, and TOPOCENTRIC allowed.");
            break;
    }
}

std::shared_ptr<Coord> makeCoord(CoordSystem const system, geom::Angle const ra, geom::Angle const dec) {
    switch (system) {
        case FK5:
            return std::shared_ptr<Fk5Coord>(new Fk5Coord(ra, dec, 2000.0));
            break;
        case ICRS:
            return std::shared_ptr<IcrsCoord>(new IcrsCoord(ra, dec));
            break;
        case GALACTIC:
            return std::shared_ptr<GalacticCoord>(new GalacticCoord(ra, dec));
            break;
        case ECLIPTIC:
            return std::shared_ptr<EclipticCoord>(new EclipticCoord(ra, dec, 2000.0));
            break;
        case TOPOCENTRIC:
            throw LSST_EXCEPT(ex::InvalidParameterError,
                              "Cannot make Topocentric with makeCoord() (must also specify Observatory).\n"
                              "Instantiate TopocentricCoord() directly.");
            break;
        default:
            throw LSST_EXCEPT(
                    ex::InvalidParameterError,
                    "Undefined CoordSystem: only FK5, ICRS, GALACTIC, ECLIPTIC, and TOPOCENTRIC allowed.");
            break;
    }
}

std::shared_ptr<Coord> makeCoord(CoordSystem const system, geom::Point3D const &p3d, double const epoch,
                                 bool normalize, geom::Angle const defaultLongitude) {
    Coord c(p3d, 2000.0, normalize, defaultLongitude);
    return makeCoord(system, c.getLongitude(), c.getLatitude(), epoch);
}
std::shared_ptr<Coord> makeCoord(CoordSystem const system, geom::Point3D const &p3d, bool normalize,
                                 geom::Angle const defaultLongitude) {
    Coord c(p3d, 2000.0, normalize, defaultLongitude);
    return makeCoord(system, c.getLongitude(), c.getLatitude());
}

std::shared_ptr<Coord> makeCoord(CoordSystem const system, geom::Point2D const &p2d, geom::AngleUnit unit,
                                 double const epoch) {
    if (unit == geom::hours) {
        return makeCoord(system, geom::Angle(p2d.getX(), geom::hours), geom::Angle(p2d.getY(), geom::degrees),
                         epoch);
    } else {
        return makeCoord(system, geom::Angle(p2d.getX(), unit), geom::Angle(p2d.getY(), unit), epoch);
    }
}

std::shared_ptr<Coord> makeCoord(CoordSystem const system, geom::Point2D const &p2d, geom::AngleUnit unit) {
    if (unit == geom::hours) {
        return makeCoord(system, geom::Angle(p2d.getX(), geom::hours),
                         geom::Angle(p2d.getY(), geom::degrees));
    } else {
        return makeCoord(system, geom::Angle(p2d.getX(), unit), geom::Angle(p2d.getY(), unit));
    }
}

std::shared_ptr<Coord> makeCoord(CoordSystem const system, std::string const ra, std::string const dec,
                                 double const epoch) {
    return makeCoord(system, dmsStringToAngle(ra), dmsStringToAngle(dec), epoch);
}
std::shared_ptr<Coord> makeCoord(CoordSystem const system, std::string const ra, std::string const dec) {
    return makeCoord(system, dmsStringToAngle(ra), dmsStringToAngle(dec));
}

std::shared_ptr<Coord> makeCoord(CoordSystem const system) {
    switch (system) {
        case FK5:
            return std::shared_ptr<Fk5Coord>(new Fk5Coord());
            break;
        case ICRS:
            return std::shared_ptr<IcrsCoord>(new IcrsCoord());
            break;
        case GALACTIC:
            return std::shared_ptr<GalacticCoord>(new GalacticCoord());
            break;
        case ECLIPTIC:
            return std::shared_ptr<EclipticCoord>(new EclipticCoord());
            break;
        case TOPOCENTRIC:
            throw LSST_EXCEPT(ex::InvalidParameterError,
                              "Cannot make Topocentric with makeCoord() (must also specify Observatory).\n"
                              "Instantiate TopocentricCoord() directly.");
            break;
        default:
            throw LSST_EXCEPT(ex::InvalidParameterError,
                              "Undefined CoordSystem: only FK5, ICRS, GALACTIC, ECLIPTIC, allowed.");
            break;
    }
}

std::string Coord::toString() const {
    return (boost::format("%s(%.7f, %.7f, %.2f)") % getClassName() % (*this)[0].asDegrees() %
            (*this)[1].asDegrees() % getEpoch())
            .str();
}

std::ostream &operator<<(std::ostream &os, Coord const &coord) {
    os << coord.toString();
    return os;
}

namespace {

// Heavy lifting for averageCoord
std::shared_ptr<Coord> doAverageCoord(std::vector<std::shared_ptr<Coord const>> const coords,
                                      CoordSystem system) {
    assert(system != UNKNOWN);  // Handled by caller
    assert(!coords.empty());  // Handled by caller
    geom::Point3D sum(0, 0, 0);
    geom::Point3D corr(0, 0, 0);  // Kahan summation correction
    for (auto &&cc : coords) {
        geom::Point3D const point = cc->getVector();
        // Kahan summation
        geom::Extent3D const add = point - corr;
        geom::Point3D const temp = sum + add;
        corr = (temp - geom::Extent3D(sum)) - add;
        sum = temp;
    }
    sum.scale(1.0 / coords.size());
    return makeCoord(system, sum);
}

}  // anonymous namespace

std::shared_ptr<Coord> averageCoord(std::vector<std::shared_ptr<Coord const>> const coords,
                                    CoordSystem system) {
    if (coords.empty()) {
        throw LSST_EXCEPT(ex::LengthError, "No coordinates provided to average");
    }

    if (system == UNKNOWN) {
        // Determine which system we're using, and check that every coordinate is in that system
        system = coords[0]->getCoordSystem();
        for (auto &&cc : coords) {
            if (cc->getCoordSystem() != system) {
                throw LSST_EXCEPT(ex::InvalidParameterError,
                                  (boost::format("Coordinates are not all in the same system: %d vs %d") %
                                   cc->getCoordSystem() %
                                   system).str());
            }
        }
        return doAverageCoord(coords, system);
    }

    // Convert everything to the nominated coordinate system if necessary
    std::vector<std::shared_ptr<Coord const>> converted;
    converted.reserve(coords.size());
    for (auto &&cc : coords) {
        converted.push_back(cc->getCoordSystem() == system ? cc : cc->convert(system));
    }
    return doAverageCoord(converted, system);
}
}
}
}  // end lsst::afw::coord

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
 
/**
 * @file Coord.cc
 * @brief Provide functions to handle coordinates
 * @ingroup afw
 * @author Steve Bickerton
 *
 * Most (nearly all) algorithms adapted from Astronomical Algorithms, 2nd ed. (J. Meeus)
 *
 */
#include <cmath>
#include <limits>
#include <cstdio>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include "lsst/pex/exceptions.h"
#include "boost/algorithm/string.hpp"
#include "boost/tuple/tuple.hpp"
#include "boost/format.hpp"

#include "lsst/afw/coord/Coord.h"
#include "lsst/daf/base/DateTime.h"

namespace afwCoord = lsst::afw::coord;
namespace ex       = lsst::pex::exceptions;
namespace afwGeom  = lsst::afw::geom;
namespace dafBase  = lsst::daf::base;

namespace {
    
typedef std::map<std::string, afwCoord::CoordSystem> CoordSystemMap;
    
CoordSystemMap const getCoordSystemMap() {
    CoordSystemMap idMap;
    idMap["FK5"]         = afwCoord::FK5;
    idMap["ICRS"]        = afwCoord::ICRS;
    idMap["ECLIPTIC"]    = afwCoord::ECLIPTIC;
    idMap["GALACTIC"]    = afwCoord::GALACTIC;
    idMap["ELON"]        = afwCoord::ECLIPTIC;
    idMap["GLON"]        = afwCoord::GALACTIC;
    idMap["TOPOCENTRIC"] = afwCoord::TOPOCENTRIC;
    return idMap;
}

/// Calculate angular distance between two points using the Haversine formula
///
/// Besides the differences between the two coordinates, we also input the
/// cosine of the two declinations, so as to be thrifty with the use of trig functions.
afwGeom::Angle haversine(afwGeom::Angle const dAlpha, /// Difference between RAs (or longitudes), alpha1-alpha2
                         afwGeom::Angle const dDelta, /// Difference between Decs (or latitudes), delta1-delta2
                         double const cosDelta1, /// Cosine of the first Dec, cos(delta1)
                         double const cosDelta2 /// Cosine of the second Dec, cos(delta2)
    )
{
    double const havDDelta = std::sin(dDelta/2.0) * std::sin(dDelta/2.0);
    double const havDAlpha = std::sin(dAlpha/2.0) * std::sin(dAlpha/2.0);
    double const havD = havDDelta + cosDelta1 * cosDelta2 * havDAlpha;
    double const sinDHalf = std::sqrt(havD);
    afwGeom::Angle dist = (2.0 * std::asin(sinDHalf)) * afwGeom::radians;
    return dist;
}

double const epochTolerance = 1.0e-12;  ///< Precession to new epoch performed if two epochs differ by this.

// Put a pair of coordinates in a common FK5 system
std::pair<afwCoord::Fk5Coord, afwCoord::Fk5Coord> commonFk5(afwCoord::Coord const& c1,
                                                            afwCoord::Coord const& c2
    )
{
    // make sure they're fk5
    afwCoord::Fk5Coord fk51 = c1.toFk5();
    afwCoord::Fk5Coord fk5tmp = c2.toFk5();

    // make sure they have the same epoch
    afwCoord::Fk5Coord fk52;
    if (fabs(fk51.getEpoch() - fk5tmp.getEpoch()) > epochTolerance) {
        fk52 = fk5tmp.precess(fk51.getEpoch());
    } else {
        fk52 = fk5tmp;
    }

    return std::make_pair(fk51, fk52);
}

} // end anonymous namespace


/**
 * @brief A utility function to get the enum value of a coordinate system from a string name.
 */
afwCoord::CoordSystem afwCoord::makeCoordEnum(std::string const system) {
    static CoordSystemMap idmap = getCoordSystemMap();
    if (idmap.find(system) != idmap.end()) {
        return idmap[system];
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterError, "System " + system + " not defined.");
    }
}


namespace {

double const NaN          = std::numeric_limits<double>::quiet_NaN();    
double const JD2000       = 2451544.50;

    
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
    Dms() {};
    
    // note that isSouth is needed to specify coords between dec = 0, and dec = -1
    // otherwise, d = -0 gets carried as d = 0 ... need a way to specify it explicitly
    Dms(int const d, int const m, double const s, bool const isSouth=false) {
        sign = (d < 0 || isSouth) ? -1 : 1;
        deg  = std::abs(d);
        min  = m;
        sec  = s;
    };
    // unit could be "degrees" or "hours"
    Dms(afwGeom::Angle const deg00, afwGeom::AngleUnit const unit = afwGeom::degrees) {
        double deg0 = deg00.asAngularUnits(unit);
        double const absVal = std::fabs(deg0);
        sign = (deg0 >= 0) ? 1 : -1;
        deg  = static_cast<int>(std::floor(absVal));
        min  = static_cast<int>(std::floor((absVal - deg)*60.0));
        sec  = ((absVal - deg)*60.0 - min)*60.0;
    }
    
    int deg;
    int min;
    double sec;
    int sign;
};

    
    
/**
 * Store the Fk5 coordinates of the Galactic pole (and vice-versa) for coordinate transforms.
 *
 */
afwCoord::Coord const& GalacticPoleInFk5()
{
    static afwCoord::Coord pole(192.85950*afwGeom::degrees, 27.12825*afwGeom::degrees, 2000.0); // C&O
    return pole;
}

afwCoord::Coord const& Fk5PoleInGalactic()
{
    static afwCoord::Coord pole(122.93200 * afwGeom::degrees, 27.12825 * afwGeom::degrees, 2000.0); // C&O
    return pole;
}

/**
 * @brief Compute the mean Sidereal Time at Greenwich
 *
 */
afwGeom::Angle meanSiderealTimeGreenwich(
                                 double const jd ///< Julian Day
                                ) {
    double const T = (jd - 2451545.0)/36525.0;
    return (280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T*T - (T*T*T/38710000.0)) * afwGeom::degrees;
}
    

/*
 * A pair of utility functions to go from cartesian to spherical
 */
double const atPoleEpsilon = 0.0; //std::numeric_limits<double>::epsilon();
afwGeom::Angle pointToLongitude(lsst::afw::geom::Point3D const &p3d, double const defaultLongitude=0.0) {
    afwGeom::Angle lon;
    if (fabs(p3d.getX()) <= atPoleEpsilon && fabs(p3d.getY()) <= atPoleEpsilon) {
        lon = afwGeom::Angle(0.0);
    } else {
        lon = std::atan2(p3d.getY(), p3d.getX()) * afwGeom::radians;
        lon.wrap();
    }
    return lon;
}
afwGeom::Angle pointToLatitude(lsst::afw::geom::Point3D const &p3d) {
    afwGeom::Angle lat;
    if ( fabs(p3d.getX()) <= atPoleEpsilon && fabs(p3d.getY()) <= atPoleEpsilon) {
        lat = (p3d.getZ() >= 0) ? afwGeom::Angle(afwGeom::HALFPI) : afwGeom::Angle(-afwGeom::HALFPI);
    } else {
        lat = std::asin(p3d.getZ()) * afwGeom::radians;
    }
    return lat;
}
std::pair<afwGeom::Angle, afwGeom::Angle> pointToLonLat(lsst::afw::geom::Point3D const &p3d, double const defaultLongitude=0.0, bool normalize=true) {
    std::pair<afwGeom::Angle, afwGeom::Angle> lonLat;

    if (normalize) {
        double const inorm = 1.0/p3d.asEigen().norm();
        double const x = inorm*p3d.getX();
        double const y = inorm*p3d.getY();
        double const z = inorm*p3d.getZ();
        if (fabs(x) <= atPoleEpsilon && fabs(y) <= atPoleEpsilon) {
            lonLat.first = 0.0 * afwGeom::radians;
            lonLat.second = ((z >= 0) ? 1.0 : -1.0) * afwGeom::HALFPI * afwGeom::radians;
        } else {
            lonLat.first = atan2(y, x) * afwGeom::radians;
            lonLat.first.wrap();
            lonLat.second = asin(z) * afwGeom::radians;
        }
    } else {
        if (fabs(p3d.getX()) <= atPoleEpsilon && fabs(p3d.getY()) <= atPoleEpsilon) {
            lonLat.first = 0.0 * afwGeom::radians;
            lonLat.second = ((p3d.getZ() >= 0) ? 1.0 : -1.0) * afwGeom::HALFPI * afwGeom::radians;
        } else {
            lonLat.first = atan2(p3d.getY(), p3d.getX()) * afwGeom::radians;
            lonLat.first.wrap();
            lonLat.second = asin(p3d.getZ()) * afwGeom::radians;
        }
    }
    return lonLat;
}

    
} // end anonymous namespace



/******************* Public functions ********************/

static std::string angleToXmsString(afwGeom::Angle const a, afwGeom::AngleUnit const unit) {

    Dms dms(a, unit);

    // make sure rounding won't give 60.00 for sec or min
    if ( (60.00 - dms.sec) < 0.005 ) {
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

    
/**
 * @brief a Function to convert a coordinate in decimal degrees to a string with form dd:mm:ss
 *
 * @todo allow a user specified format
 */
std::string afwCoord::angleToDmsString(afwGeom::Angle const a) {
    return angleToXmsString(a, afwGeom::degrees);
}

/**
 * @brief a function to convert decimal degrees to a string with form hh:mm:ss.s
 */
std::string afwCoord::angleToHmsString(afwGeom::Angle const a) {
    return angleToXmsString(a, afwGeom::hours);
}

/**
 * @brief Convert a XX:mm:ss string to Angle
 */
static afwGeom::Angle xmsStringToAngle(
    std::string const dms, ///< Coord as a string in dd:mm:ss format
    afwGeom::AngleUnit unit
    ) {
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
    int const deg   = abs(atoi(elements[0].c_str()));
    int const min   = atoi(elements[1].c_str());
    double const sec = atof(elements[2].c_str());

    afwGeom::Angle ang = (deg + min/60.0 + sec/3600.0) * unit;
    if ( (elements[0].c_str())[0] == '-' ) {
        ang *= -1.0;
    }
    return ang;
}

/**
 * @brief Convert a hh:mm:ss string to Angle
 */
afwGeom::Angle afwCoord::hmsStringToAngle(
    std::string const hms
    ){
    return xmsStringToAngle(hms, afwGeom::hours);
}

/**
 * @brief Convert a dd:mm:ss string to Angle
 */
afwGeom::Angle afwCoord::dmsStringToAngle(
    std::string const dms ///< Coord as a string in dd:mm:ss format
    ) {
    return xmsStringToAngle(dms, afwGeom::degrees);
}

/**
 * @brief get the inclination of the ecliptic pole (obliquity) at epoch
 *
 */
afwGeom::Angle afwCoord::eclipticPoleInclination(
                                         double const epoch ///< desired epoch for inclination
                                        ) {
    double const T = (epoch - 2000.0)/100.0;
    return (23.0 + 26.0/60.0 + (21.448 - 46.82*T - 0.0006*T*T - 0.0018*T*T*T)/3600.0) * afwGeom::degrees;
}


/* ============================================================
 *
 * class Coord
 *
 * ============================================================*/


/**
 * @brief Constructor for the Coord base class
 *
 */
afwCoord::Coord::Coord(
                       lsst::afw::geom::Point2D const &p2d,     ///< Point2D
                       afwGeom::AngleUnit unit,        ///< Rads, Degs, or Hrs
                       double const epoch               ///< epoch of coordinate
                      ) :
    _longitude(NaN), _latitude(NaN), _epoch(epoch) {
    _longitude = afwGeom::Angle(p2d.getX(), unit);
    _longitude.wrap();
    _latitude  = afwGeom::Angle(p2d.getY(), unit);
    _verifyValues();
}

/**
 * @brief Constructor for the Coord base class
 */
afwCoord::Coord::Coord(
                       afwGeom::Point3D const &p3d,   ///< Point3D
                       double const epoch,            ///< epoch of coordinate
                       bool normalize,                 ///< normalize the p3d provided
                       afwGeom::Angle const defaultLongitude  ///< longitude to use if x=y=0
                      ) :
    _longitude(0. * afwGeom::radians),
    _latitude(0. * afwGeom::radians),
    _epoch(epoch) {

    std::pair<afwGeom::Angle, afwGeom::Angle> lonLat = pointToLonLat(p3d, defaultLongitude, normalize);
    _longitude = lonLat.first;
    _latitude  = lonLat.second;
    _epoch = epoch;
}

/**
 * @brief Constructor for the Coord base class
 *
 */
afwCoord::Coord::Coord(
                       afwGeom::Angle const ra,   ///< Right ascension, decimal degrees
                       afwGeom::Angle const dec,  ///< Declination, decimal degrees
                       double const epoch ///< epoch of coordinate
                      ) :
    _longitude(ra), _latitude(dec), _epoch(epoch) {
    _longitude.wrap();
    _verifyValues();
}

/**
 * @brief Constructor for the Coord base class
 *
 */
afwCoord::Coord::Coord(
                       std::string const ra,  ///< Right ascension, hh:mm:ss.s format
                       std::string const dec, ///< Declination, dd:mm:ss.s format
                       double const epoch     ///< epoch of coordinate
                      ) :
    _longitude(hmsStringToAngle(ra)),
    _latitude(dmsStringToAngle(dec)),
    _epoch(epoch) {
    _longitude.wrap();
    _verifyValues();
}

/**
 * @brief Default constructor for the Coord base class
 *
 * Set all values to NaN
 * Don't call _veriftyValues() method ... it'll fail.
 *
 */
afwCoord::Coord::Coord() : _longitude(afwGeom::Angle(NaN)), _latitude(afwGeom::Angle(NaN)), _epoch(NaN) {}


/**
 * @brief Make sure the values we've got are in the range 0 <= x < 2PI
 */
void afwCoord::Coord::_verifyValues() const {
    if (_latitude.asRadians() < -afwGeom::HALFPI || _latitude.asRadians() > afwGeom::HALFPI) {
        throw LSST_EXCEPT(ex::InvalidParameterError,
                          (boost::format("Latitude coord must be: -PI/2 <= lat <= PI/2 (%f).") %
                           _latitude).str());
    }
}

/**
 * @brief Reset our coordinates wholesale.
 *
 * This allows the user to instantiate Coords without values, and fill them later.
 */
void afwCoord::Coord::reset(
    afwGeom::Angle const longitude, ///< Longitude coord (eg. R.A. for Fk5)
    afwGeom::Angle const latitude,  ///< Latitude coord (eg. Declination for Fk5)
    double const epoch         ///< epoch of coordinate
    ) {
    _longitude = longitude;
    _latitude  = latitude;
    _longitude.wrap();
    _epoch = epoch;
    _verifyValues();
}


/**
 * @brief Return our contents in a Point2D object
 *
 */
afwGeom::Point2D afwCoord::Coord::getPosition(afwGeom::AngleUnit unit) const {
    // treat HOURS specially, they must mean hours for RA, degrees for Dec
    if (unit == afwGeom::hours) {
        return afwGeom::Point2D(getLongitude().asHours(), getLatitude().asDegrees());
    } else {
        return afwGeom::Point2D(getLongitude().asAngularUnits(unit), getLatitude().asAngularUnits(unit));
    }
}


/**
 * @brief Return our contents in a position vector.
 *
 */
afwGeom::Point3D afwCoord::Coord::getVector() const {
    double lng = getLongitude();
    double lat = getLatitude();
    double const x = std::cos(lng) * std::cos(lat);
    double const y = std::sin(lng) * std::cos(lat);
    double const z = std::sin(lat);
    return afwGeom::Point3D(x, y, z);
}


/**
 * @brief Tranform our current coords to another spherical polar system
 *
 * Variable names assume an equaltorial/galactic tranform, but it works
 *  for any spherical polar system when the appropriate poles are supplied.
 */
afwCoord::Coord afwCoord::Coord::transform(
    Coord const &poleTo,   ///< Pole of the destination system in the current coords
    Coord const &poleFrom  ///< Pole of the current system in the destination coords
                                          ) const {
    double const alphaGP  = poleFrom[0];
    double const deltaGP  = poleFrom[1];
    double const lCP      = poleTo[0];
    
    double const alpha = getLongitude();
    double const delta = getLatitude();
    
    afwGeom::Angle const l = (lCP - std::atan2(std::sin(alpha - alphaGP), 
                                               std::tan(delta)*std::cos(deltaGP) - std::cos(alpha - alphaGP)*std::sin(deltaGP))) * afwGeom::radians;
    afwGeom::Angle const b = std::asin( (std::sin(deltaGP)*std::sin(delta) + std::cos(deltaGP)*std::cos(delta)*std::cos(alpha - alphaGP))) * afwGeom::radians;
    return Coord(l, b);
}


/**
 * @brief Rotate our current coords about a pole
 *
 */
void afwCoord::Coord::rotate(
                             Coord const &axis,   ///< axis of rotation (right handed)
                             afwGeom::Angle const theta   ///< angle to offset in radians
                            ) {

    double const c = std::cos(theta);
    double const mc = 1.0 - c;
    double const s = std::sin(theta);
    
    // convert to cartesian
    afwGeom::Point3D const x = getVector();
    afwGeom::Point3D const u = axis.getVector();
    double const ux = u[0];
    double const uy = u[1];
    double const uz = u[2];

    // rotate
    afwGeom::Point3D xprime;
    xprime[0] = (ux*ux + (1.0 - ux*ux)*c)*x[0] +  (ux*uy*mc - uz*s)*x[1] +  (ux*uz*mc + uy*s)*x[2];
    xprime[1] = (uy*uy + (1.0 - uy*uy)*c)*x[1] +  (uy*uz*mc - ux*s)*x[2] +  (ux*uy*mc + uz*s)*x[0];
    xprime[2] = (uz*uz + (1.0 - uz*uz)*c)*x[2] +  (uz*ux*mc - uy*s)*x[0] +  (uy*uz*mc + ux*s)*x[1];
    
    // in-situ
    std::pair<afwGeom::Angle, afwGeom::Angle> lonLat = pointToLonLat(xprime);
    _longitude = lonLat.first;
    _latitude  = lonLat.second;
}


/**
 * @brief offset our current coords along a great circle defined by an angle wrt a declination parallel
 *
 * @note At/near the pole, longitude becomes degenerate with angle-wrt-declination.  So
 *       at the pole the offset will trace a meridian with longitude = 90 + longitude0 + phi
 *
 * @return the angle wrt a declination parallel at new position
 */
afwGeom::Angle afwCoord::Coord::offset(
                               afwGeom::Angle const phi,    ///< angle wrt parallel to offset
                               afwGeom::Angle const arcLen  ///< angle to offset
                              ) {

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
    Eigen::Vector3d v = arcLen * std::cos(phi)*u + arcLen * std::sin(phi)*w;

    // take r x v to get the axis
    Eigen::Vector3d axisVector = r.cross(v);
    axisVector.normalize();
    Coord axisCoord = Coord(afwGeom::Point3D(axisVector), getEpoch());
    
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
    Coord v2Coord = Coord(afwGeom::Point3D(v), getEpoch());
    v2Coord.rotate(axisCoord, arcLen);
    Eigen::Vector3d v2 = v2Coord.getVector().asEigen();

    afwGeom::Angle phi2 = std::atan2(w2.dot(v2), u2.dot(v2)) * afwGeom::radians;

    return phi2;
}



/**
 * @brief Convert to a specified Coord type.
 */
afwCoord::Coord::Ptr afwCoord::Coord::convert(CoordSystem system) const {

    switch (system) {
      case FK5:
        {
            Fk5Coord c1 = this->toFk5();
            return boost::shared_ptr<Fk5Coord>(new Fk5Coord(c1.getLongitude(),
                                                            c1.getLatitude(),
                                                            c1.getEpoch()));
        }
        break;
      case ICRS:
        {
            IcrsCoord c2 = this->toIcrs();
            return boost::shared_ptr<IcrsCoord>(new IcrsCoord(c2.getLongitude(),
                                                              c2.getLatitude()));
        }
        break;
      case GALACTIC:
        {
            GalacticCoord c4 = this->toGalactic();
            return boost::shared_ptr<GalacticCoord>(new GalacticCoord(c4.getLongitude(),
                                                                      c4.getLatitude()));
        }
        break;
      case ECLIPTIC:
        {
            EclipticCoord c5 = this->toEcliptic();
            return boost::shared_ptr<EclipticCoord>(new EclipticCoord(c5.getLongitude(),
                                                                      c5.getLatitude(),
                                                                      c5.getEpoch()));
        }
        break;
      case TOPOCENTRIC:
        throw LSST_EXCEPT(ex::InvalidParameterError,
                          "Cannot make Topocentric with convert() (must also specify Observatory).\n"
                          "Instantiate TopocentricCoord() directly.");
        break;
      default:
        throw LSST_EXCEPT(ex::InvalidParameterError,
                          "Undefined CoordSystem: only FK5, ICRS, GALACTIC, ECLIPTIC allowed.");
        break;
        
    }
    
}


/**
 * @brief compute the angular separation between two Coords
 *
 */
afwGeom::Angle afwCoord::Coord::angularSeparation(
    Coord const &c ///< coordinate to compute our separation from
    ) const {

    // work in Fk5, no matter what two derived classes we're given (eg Fk5 and Galactic)
    // we'll put them in the same system.
    std::pair<afwCoord::Fk5Coord, afwCoord::Fk5Coord> const& fk5 = commonFk5(*this, c);
    afwGeom::Angle const alpha1 = fk5.first.getRa();
    afwGeom::Angle const delta1 = fk5.first.getDec();
    afwGeom::Angle const alpha2 = fk5.second.getRa();
    afwGeom::Angle const delta2 = fk5.second.getDec();

    return haversine(alpha1 - alpha2, delta1 - delta2, std::cos(delta1), std::cos(delta2));
}


/**
 * @brief Compute the offset from a coordinate
 *
 * The resulting angles are suitable for input to Coord::offset
 *
 * @return pair of Angles: bearing (angle wrt a declination parallel) and distance
 */
std::pair<afwGeom::Angle, afwGeom::Angle>
afwCoord::Coord::getOffsetFrom(afwCoord::Coord const &c ///< Coordinate from which to compute offset
    ) const
{
    // work in Fk5, no matter what two derived classes we're given (eg Fk5 and Galactic)
    // we'll put them in the same system.
    std::pair<afwCoord::Fk5Coord, afwCoord::Fk5Coord> const& fk5 = commonFk5(*this, c);
    afwGeom::Angle const alpha1 = fk5.first.getRa();
    afwGeom::Angle const delta1 = fk5.first.getDec();
    afwGeom::Angle const alpha2 = fk5.second.getRa();
    afwGeom::Angle const delta2 = fk5.second.getDec();

    afwGeom::Angle const dAlpha = alpha1-alpha2;
    afwGeom::Angle const dDelta = delta1-delta2;

    double const cosDelta1 = std::cos(delta1);
    double const cosDelta2 = std::cos(delta2);

    afwGeom::Angle separation = haversine(dAlpha, dDelta, cosDelta1, cosDelta2);

    // Formula from http://www.movable-type.co.uk/scripts/latlong.html
    double const y = std::sin(dAlpha)*cosDelta2;
    double const x = cosDelta1*std::sin(delta2) - std::sin(delta1)*cosDelta2*std::cos(dAlpha);
    afwGeom::Angle bearing = std::atan2(y, x)*afwGeom::radians - 90.0*afwGeom::degrees;

    return std::make_pair(bearing, separation);
}

/**
 * @brief Get the offset on the tangent plane
 *
 * This is suitable only for small angles.
 *
 * @return pair of Angles: Longitude and Latitude offsets
 */
std::pair<afwGeom::Angle, afwGeom::Angle>
afwCoord::Coord::getTangentPlaneOffset(afwCoord::Coord const &c ///< Coordinate from which to compute offset
    ) const
{
    // work in Fk5, no matter what two derived classes we're given (eg Fk5 and Galactic)
    // we'll put them in the same system.
    std::pair<afwCoord::Fk5Coord, afwCoord::Fk5Coord> const& fk5 = commonFk5(*this, c);
    afwGeom::Angle const alpha1 = fk5.first.getRa();
    afwGeom::Angle const delta1 = fk5.first.getDec();
    afwGeom::Angle const alpha2 = fk5.second.getRa();
    afwGeom::Angle const delta2 = fk5.second.getDec();

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

    return std::make_pair(xi*afwGeom::radians, eta*afwGeom::radians);
}


/**
 * @brief Convert ourself to Fk5: RA, Dec (precess to new epoch)
 */
afwCoord::Fk5Coord afwCoord::Coord::toFk5(double const epoch) const {
    return Fk5Coord(getLongitude(), getLatitude(), getEpoch()).precess(epoch);
}
/**
 * @brief Convert ourself to Fk5: RA, Dec (use current epoch)
 */
afwCoord::Fk5Coord afwCoord::Coord::toFk5() const {
    return Fk5Coord(getLongitude(), getLatitude(), getEpoch());
}

/**
 * @brief Convert ourself to ICRS: RA, Dec (basically J2000)
 *
 */
afwCoord::IcrsCoord afwCoord::Coord::toIcrs() const {
    return this->toFk5().toIcrs();
}

/**
 * @brief Convert ourself to Galactic: l, b
 */
afwCoord::GalacticCoord afwCoord::Coord::toGalactic() const {
    return this->toFk5().toGalactic();
}

/**
 * @brief Convert ourself to Ecliptic: lambda, beta (precess to new epoch)
 */
afwCoord::EclipticCoord afwCoord::Coord::toEcliptic(double const epoch) const {
        return this->toFk5(epoch).toEcliptic();
}
/**
 * @brief Convert ourself to Ecliptic: lambda, beta (use existing epoch)
 */
afwCoord::EclipticCoord afwCoord::Coord::toEcliptic() const {
        return this->toFk5().toEcliptic();
}

/**
 * @brief Convert ourself to Altitude/Azimuth: alt, az
 */
afwCoord::TopocentricCoord afwCoord::Coord::toTopocentric(
                                        Observatory const &obs,            ///< observatory of observation
                                        lsst::daf::base::DateTime const &obsDate   ///< date of observation
                                                         ) const {
    return this->toFk5().toTopocentric(obs, obsDate);
}




/* ============================================================
 *
 * class Fk5Coord
 *
 * ============================================================*/


/**
 * @brief Convert ourself to Fk5 (ie. a no-op): RA, Dec  (precess to new epoch)
 */
afwCoord::Fk5Coord afwCoord::Fk5Coord::toFk5(double const epoch) const {
    return Fk5Coord(getLongitude(), getLatitude(), getEpoch()).precess(epoch);
}
/**
 * @brief Convert ourself to Fk5 (ie. a no-op): RA, Dec (keep current epoch)
 */
afwCoord::Fk5Coord afwCoord::Fk5Coord::toFk5() const {
    return Fk5Coord(getLongitude(), getLatitude(), getEpoch());
}

/**
 * @brief Convert ourself to ICRS: RA, Dec (basically J2000)
 *
 */
afwCoord::IcrsCoord afwCoord::Fk5Coord::toIcrs() const {

    // only do the precession to 2000 if we're not already there.
    if ( fabs(getEpoch() - 2000.0) > epochTolerance ) {
        afwCoord::Fk5Coord c = precess(2000.0);
        return IcrsCoord(c.getLongitude(), c.getLatitude());
    } else {
        return IcrsCoord(getLongitude(), getLatitude());
    }
}


/**
 * @brief Convert ourself to Galactic: l, b
 */
afwCoord::GalacticCoord afwCoord::Fk5Coord::toGalactic() const {

    // if we're epoch==2000, we can transform, otherwise we need to precess first
    Fk5Coord c;
    if ( fabs(getEpoch() - 2000.0) > epochTolerance ) {
        c = precess(2000.0);
    } else {
        c = *this;
    }
    
    Coord ct = c.transform(Fk5PoleInGalactic(), GalacticPoleInFk5());
    return GalacticCoord(ct.getLongitude(), ct.getLatitude());
    
}

/**
 * @brief Convert ourself to Ecliptic: lambda, beta (precess to new epoch)
 */
afwCoord::EclipticCoord afwCoord::Fk5Coord::toEcliptic(double const epoch) const {
    afwGeom::Angle const eclPoleIncl = eclipticPoleInclination(epoch);
    Coord const eclPoleInEquatorial(270.0 * afwGeom::degrees, (90.0 * afwGeom::degrees) - eclPoleIncl, epoch);
    Coord const equPoleInEcliptic(90.0 * afwGeom::degrees, (90.0 * afwGeom::degrees) - eclPoleIncl, epoch);
    Coord c = transform(equPoleInEcliptic, eclPoleInEquatorial); 
    return EclipticCoord(c.getLongitude(), c.getLatitude(), epoch);
}
/**
 * @brief Convert ourself to Ecliptic: lambda, beta (use current epoch)
 */
afwCoord::EclipticCoord afwCoord::Fk5Coord::toEcliptic() const {
    return this->toEcliptic(getEpoch());
}

/**
 * @brief Convert ourself to Altitude/Azimuth: alt, az
 */
afwCoord::TopocentricCoord afwCoord::Fk5Coord::toTopocentric(
    Observatory const &obs,           ///< observatory
    lsst::daf::base::DateTime const &obsDate  ///< date of obs.
                                                            ) const {

    // make sure we precess to the epoch
    Fk5Coord fk5 = precess(obsDate.get(dafBase::DateTime::EPOCH));

    // greenwich sidereal time
    afwGeom::Angle theta0 = meanSiderealTimeGreenwich(obsDate.get(dafBase::DateTime::JD));
    theta0.wrap();

    // lat/long of the observatory
    afwGeom::Angle const phi             = obs.getLatitude();
    afwGeom::Angle const L               = obs.getLongitude();

    // ra/dec of the target
    afwGeom::Angle const alpha           = fk5.getRa();
    afwGeom::Angle const delta           = fk5.getDec();
                               
    afwGeom::Angle const H               = theta0 + L - alpha;

    // compute the altitude, h
    double const sinh            = std::sin(phi)* std::sin(delta) + std::cos(phi) * std::cos(delta) * std::cos(H);
    afwGeom::Angle const h               = std::asin(sinh) * afwGeom::radians;

    // compute the azimuth, A
    double const tanAnumerator   = std::sin(H);
    double const tanAdenominator = (std::cos(H) * std::sin(phi) - std::tan(delta) * std::cos(phi));
    
    // Equations used here assume azimuth is with respect to South
    // but we use the North as our origin ... must add 180 deg
    afwGeom::Angle A = (180.0*afwGeom::degrees) + atan2(tanAnumerator, tanAdenominator)* afwGeom::radians;
    A.wrap();
    
    return TopocentricCoord(A, h, obsDate.get(dafBase::DateTime::EPOCH), obs);
}



/**
 * @brief Precess ourselves from whence we are to a new epoch
 *
 */
afwCoord::Fk5Coord afwCoord::Fk5Coord::precess(
                                               double const epochTo ///< epoch to precess to
                                              ) const {

    // return a copy if the epochs are the same
    if ( fabs(getEpoch() - epochTo) < epochTolerance) {
        return Fk5Coord(getLongitude(), getLatitude(), getEpoch());
    }
    
    dafBase::DateTime const dateFrom(getEpoch(), dafBase::DateTime::EPOCH, dafBase::DateTime::TAI);
    dafBase::DateTime const dateTo(epochTo, dafBase::DateTime::EPOCH, dafBase::DateTime::TAI);
    double const jd0 = dateFrom.get(dafBase::DateTime::JD);
    double const jd  = dateTo.get(dafBase::DateTime::JD);

    double const T   = (jd0 - JD2000)/36525.0;
    double const t   = (jd - jd0)/36525.0;
    double const tt  = t*t;
    double const ttt = tt*t;

    afwGeom::Angle const xi    = ((2306.2181 + 1.39656*T - 0.000139*T*T)*t +
                                  (0.30188 - 0.000344*T)*tt + 0.017998*ttt) * afwGeom::arcseconds;
    afwGeom::Angle const z     = ((2306.2181 + 1.39656*T - 0.000139*T*T)*t +
                                  (1.09468 + 0.000066*T)*tt + 0.018203*ttt) * afwGeom::arcseconds;
    afwGeom::Angle const theta = ((2004.3109 - 0.85330*T - 0.000217*T*T)*t -
                                  (0.42665 + 0.000217*T)*tt - 0.041833*ttt) * afwGeom::arcseconds;

    Fk5Coord fk5 = this->toFk5();
    afwGeom::Angle const alpha0 = fk5.getRa();
    afwGeom::Angle const delta0 = fk5.getDec();
    
    double const a = std::cos(delta0) * std::sin((alpha0 + xi));
    double const b = std::cos(theta)  * std::cos(delta0) * std::cos((alpha0 + xi)) - std::sin(theta) * std::sin(delta0);
    double const c = std::sin(theta)  * std::cos(delta0) * std::cos((alpha0 + xi)) + std::cos(theta) * std::sin(delta0);

    afwGeom::Angle const alpha = (std::atan2(a,b) + z) * afwGeom::radians;
    afwGeom::Angle const delta = std::asin(c) * afwGeom::radians;
    
    return Fk5Coord(alpha, delta, epochTo);
}


/* ============================================================
 *
 * class IcrsCoord
 *
 * ============================================================*/

/**
 * @brief special reset() overload to make sure no epoch can be set
 */
void afwCoord::IcrsCoord::reset(afwGeom::Angle const longitude, afwGeom::Angle const latitude) {
    Coord::reset(longitude, latitude, 2000.0);
}

/**
 * @brief Fk5 converter for IcrsCoord. (specify epoch)
 */
afwCoord::Fk5Coord afwCoord::IcrsCoord::toFk5(double const epoch) const {
    return Fk5Coord(getLongitude(), getLatitude(), 2000.0).precess(epoch);
}
/**
 * @brief Fk5 converter for IcrsCoord. (no epoch specified)
 */
afwCoord::Fk5Coord afwCoord::IcrsCoord::toFk5() const {
    return Fk5Coord(getLongitude(), getLatitude(), 2000.0);
}

/**
 * @brief Icrs converter for IcrsCoord. (ie. a no-op)
 */
afwCoord::IcrsCoord afwCoord::IcrsCoord::toIcrs() const {
    return IcrsCoord(getLongitude(), getLatitude());
}




/* ============================================================
 *
 * class GalacticCoord
 *
 * ============================================================*/

/**
 * @brief special reset() overload to make sure no epoch can be set
 */
void afwCoord::GalacticCoord::reset(afwGeom::Angle const longitudeDeg, afwGeom::Angle const latitudeDeg) {
    Coord::reset(longitudeDeg, latitudeDeg, 2000.0);
}

/**
 * @brief Convert ourself from galactic to Fk5 (specify epoch)
 */
afwCoord::Fk5Coord afwCoord::GalacticCoord::toFk5(double const epoch) const {
    // transform to fk5
    // galactic coords are ~constant, and the poles used are for epoch=2000, so we get J2000
    Coord c = transform(GalacticPoleInFk5(), Fk5PoleInGalactic());
    return Fk5Coord(c.getLongitude(), c.getLatitude(), 2000.0).precess(epoch);
}
/**
 * @brief Convert ourself from galactic to Fk5 (no epoch specified)
 */
afwCoord::Fk5Coord afwCoord::GalacticCoord::toFk5() const {
    return this->toFk5(2000.0);
}

/**
 * @brief Convert ourself from Galactic to Galactic ... a no-op
 */
afwCoord::GalacticCoord afwCoord::GalacticCoord::toGalactic() const { 
    return GalacticCoord(getLongitude(), getLatitude());
}




/* ============================================================
 *
 * class EclipticCoord
 *
 * ============================================================*/

/**
 * @brief Convert ourself from Ecliptic to Ecliptic ... a no-op (but precess to new epoch)
 */
afwCoord::EclipticCoord afwCoord::EclipticCoord::toEcliptic(double const epoch) const {
    return EclipticCoord(getLongitude(), getLatitude(), getEpoch()).precess(epoch);
}
/**
 * @brief Convert ourself from Ecliptic to Ecliptic ... a no-op (use the current epoch)
 */
afwCoord::EclipticCoord afwCoord::EclipticCoord::toEcliptic() const {
    return EclipticCoord(getLongitude(), getLatitude(), getEpoch());
}


/**
 * @brief Convert ourself from Ecliptic to Fk5 (precess to new epoch)
 */
afwCoord::Fk5Coord afwCoord::EclipticCoord::toFk5(double const epoch) const {
    afwGeom::Angle const eclPoleIncl = eclipticPoleInclination(epoch);
    afwGeom::Angle ninety = 90. * afwGeom::degrees;
    Coord const eclipticPoleInFk5(270.0 * afwGeom::degrees, ninety - eclPoleIncl, epoch);
    Coord const fk5PoleInEcliptic(ninety, ninety - eclPoleIncl, epoch);
    Coord c = transform(eclipticPoleInFk5, fk5PoleInEcliptic);
    return Fk5Coord(c.getLongitude(), c.getLatitude(), epoch);
}
/**
 * @brief Convert ourself from Ecliptic to Fk5 (use current epoch)
 */
afwCoord::Fk5Coord afwCoord::EclipticCoord::toFk5() const {
    return this->toFk5(getEpoch());
}


/**
 * @brief precess to new epoch
 *
 * Do this by going through fk5
 */
afwCoord::EclipticCoord afwCoord::EclipticCoord::precess(
                                                         double const epochTo ///< epoch to precess to.
                                                        ) const {
    return this->toFk5().precess(epochTo).toEcliptic();
}



/* ============================================================
 *
 * class TopocentricCoord
 *
 * ============================================================*/

/**
 * @brief Convert ourself from Topocentric to Fk5
 */
afwCoord::Fk5Coord afwCoord::TopocentricCoord::toFk5(double const epoch) const {
     
    afwGeom::Angle const phi      = _obs.getLatitude();
    afwGeom::Angle const L        = _obs.getLongitude();

    // Equations used here assume azimuth is with respect to South
    // but we use the North as our origin.
    afwGeom::Angle A              = getAzimuth() + 180.0*afwGeom::degrees;
    A.wrap();
    afwGeom::Angle const h        = getAltitude();

    
    double const jd       = dafBase::DateTime(epoch,
                                              dafBase::DateTime::EPOCH,
                                              dafBase::DateTime::TAI).get(dafBase::DateTime::JD);
    afwGeom::Angle theta0   = meanSiderealTimeGreenwich(jd);
    theta0.wrap();

    double const tanHnum     = std::sin(A);
    double const tanHdenom   = std::cos(A)*std::sin(phi) + std::tan(h)*std::cos(phi);
    afwGeom::Angle H         = std::atan2(tanHnum, tanHdenom) * afwGeom::radians;

    afwGeom::Angle const alpha    = theta0 + L - H;
    double const sinDelta = std::sin(phi)*std::sin(h) - std::cos(phi)*std::cos(h)*std::cos(A);
    afwGeom::Angle const delta    = (std::asin(sinDelta)) * afwGeom::radians;

    return Fk5Coord(alpha, delta, epoch);
}
/**
 * @brief Convert outself from Topocentric to Fk5 (use current epoch)
 */
afwCoord::Fk5Coord afwCoord::TopocentricCoord::toFk5() const {
    return this->toFk5(getEpoch());
}


/**
 * @brief Convert ourself from Topocentric to Topocentric ... a no-op
 */
afwCoord::TopocentricCoord afwCoord::TopocentricCoord::toTopocentric(
    lsst::afw::coord::Observatory const &obs,             ///< observatory of observation
    lsst::daf::base::DateTime const &date        ///< date of observation
                                                                    ) const
{
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

/**
 * @brief Convert ourself from Topocentric to Topocentric with no observatory or date arguments
 *
 * As this is essentially a copy-constructor, the extra info can be obtained internally.
 */
afwCoord::TopocentricCoord afwCoord::TopocentricCoord::toTopocentric() const {
    return TopocentricCoord(getLongitude(), getLatitude(), getEpoch(), _obs);
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

/**
 * @brief Factory function to create a Coord of arbitrary type with decimal RA,Dec
 *
 * @note This factory allows the epoch to be specified but will throw if used with ICRS or Galactic
 * @note Most of the other factories (which accept epochs) just call this one indirectly.
 *
 */
afwCoord::Coord::Ptr afwCoord::makeCoord(
        CoordSystem const system,     ///< the system (equ, fk5, galactic ..)
        afwGeom::Angle const ra,              ///< right ascension
        afwGeom::Angle const dec,             ///< declination
        double const epoch            ///< epoch of coordinate
) {

    switch (system) {
      case FK5:
        return boost::shared_ptr<Fk5Coord>(new Fk5Coord(ra, dec, epoch));
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
        return boost::shared_ptr<EclipticCoord>(new EclipticCoord(ra, dec, epoch));
        break;
      case TOPOCENTRIC:
        throw LSST_EXCEPT(ex::InvalidParameterError,
                          "Cannot make Topocentric with makeCoord() (must also specify Observatory).\n"
                          "Instantiate TopocentricCoord() directly.");
        break;
      default:
        throw LSST_EXCEPT(ex::InvalidParameterError,
            "Undefined CoordSystem: only FK5, ICRS, GALACTIC, ECLIPTIC, and TOPOCENTRIC allowed.");
        break;
        
    }

}



/**
 * @brief Factory function to create a Coord of arbitrary type with decimal RA,Dec in degrees
 *
 * @note This factory assumes a default epoch
 * @note Most of the other factories (which don't accept epoch) call this one.
 */
afwCoord::Coord::Ptr afwCoord::makeCoord(
        CoordSystem const system,     ///< the system (equ, fk5, galactic ..)
        afwGeom::Angle const ra,              ///< right ascension
        afwGeom::Angle const dec             ///< declination
) {

    switch (system) {
      case FK5:
        return boost::shared_ptr<Fk5Coord>(new Fk5Coord(ra, dec, 2000.0));
        break;
      case ICRS:
        return boost::shared_ptr<IcrsCoord>(new IcrsCoord(ra, dec));
        break;
      case GALACTIC:
        return boost::shared_ptr<GalacticCoord>(new GalacticCoord(ra, dec));
        break;
      case ECLIPTIC:
        return boost::shared_ptr<EclipticCoord>(new EclipticCoord(ra, dec, 2000.0));
        break;
      case TOPOCENTRIC:
        throw LSST_EXCEPT(ex::InvalidParameterError,
                          "Cannot make Topocentric with makeCoord() (must also specify Observatory).\n"
                          "Instantiate TopocentricCoord() directly.");
        break;
      default:
        throw LSST_EXCEPT(ex::InvalidParameterError,
            "Undefined CoordSystem: only FK5, ICRS, GALACTIC, ECLIPTIC, and TOPOCENTRIC allowed.");
        break;
        
    }

}
    


/**
 * @brief Factory function to create a Coord of arbitrary type with a Point3D
 *
 * @note This factory accepts epoch.  There is an overloaded version which uses a default.
 *
 */
afwCoord::Coord::Ptr afwCoord::makeCoord(
        CoordSystem const system,     ///< the system (equ, fk5, galactic ..)
        lsst::afw::geom::Point3D const &p3d,     ///< the coord in Point3D format
        double const epoch,            ///< epoch of coordinate
        bool normalize,                 ///< normalize the p3d provided
        afwGeom::Angle const defaultLongitude ///< longitude to use if x=y=0
) {
    Coord c(p3d, 2000.0, normalize, defaultLongitude);
    return makeCoord(system, c.getLongitude(), c.getLatitude(), epoch);
}
/**
 * @brief Factory function to create a Coord of arbitrary type with a Point3D
 *
 * @note This factory uses a default epoch.  There is an overloaded version which accepts an epoch.
 *
 */
afwCoord::Coord::Ptr afwCoord::makeCoord(
        CoordSystem const system,             ///< the system (equ, fk5, galactic ..)
        lsst::afw::geom::Point3D const &p3d,  ///< the coord in Point3D format
        bool normalize,                        ///< normalize the p3d provided
        afwGeom::Angle const defaultLongitude ///< longitude to use if x=y=0
) {
    Coord c(p3d, 2000.0, normalize, defaultLongitude);
    return makeCoord(system, c.getLongitude(), c.getLatitude());
}

/**
 * @brief Factory function to create a Coord of arbitrary type with Point2D
 *
 * @note This factory accepts epoch.  There is an overloaded version which uses a default.
 */
afwCoord::Coord::Ptr afwCoord::makeCoord(
        CoordSystem const system,               ///< the system (equ, fk5, galactic ..)
        lsst::afw::geom::Point2D const &p2d,    ///< the (eg) ra,dec in a Point2D
        afwGeom::AngleUnit unit,       ///< the units (eg. degrees, radians)
        double const epoch  ///< epoch of coordinate
) {
    if (unit == afwGeom::hours) {
        return makeCoord(system, afwGeom::Angle(p2d.getX(), afwGeom::hours), afwGeom::Angle(p2d.getY(), afwGeom::degrees), epoch);
    } else {
        return makeCoord(system, afwGeom::Angle(p2d.getX(), unit), afwGeom::Angle(p2d.getY(), unit), epoch);
    }
}

/**
 * @brief Factory function to create a Coord of arbitrary type with Point2D
 *
 * @note This factory uses a default epoch.  There is an overloaded version which accepts an epoch.
 *
 */
afwCoord::Coord::Ptr afwCoord::makeCoord(
        CoordSystem const system, ///< the system (equ, fk5, galactic ..)
        lsst::afw::geom::Point2D const &p2d,    ///< the (eg) ra,dec in a Point2D
        afwGeom::AngleUnit unit        ///< the units (eg. degrees, radians)
) {
    if (unit == afwGeom::hours) {
        return makeCoord(system, afwGeom::Angle(p2d.getX(), afwGeom::hours), afwGeom::Angle(p2d.getY(), afwGeom::degrees));
    } else {
        return makeCoord(system, afwGeom::Angle(p2d.getX(), unit), afwGeom::Angle(p2d.getY(), unit));
    }
}

/**
 * @brief Factory function to create a Coord of arbitrary type with string RA [in degrees, not hours!], Dec
 *
 * @note This factory accepts epoch.  There is an overloaded version which uses a default.
 */
afwCoord::Coord::Ptr afwCoord::makeCoord(
                                   CoordSystem const system,   ///< the system (equ, fk5, galactic ..)
                                   std::string const ra,       ///< right ascension
                                   std::string const dec,      ///< declination
                                   double const epoch          ///< epoch of coordinate
                                  ) {
    return makeCoord(system, dmsStringToAngle(ra), dmsStringToAngle(dec), epoch);
}
/**
 * @brief Factory function to create a Coord of arbitrary type with string RA [in degrees, not hours!], Dec
 *
 * @note This factory uses a default epoch.  There is an overloaded version which accepts an epoch.
 */
afwCoord::Coord::Ptr afwCoord::makeCoord(
                                   CoordSystem const system,   ///< the system (equ, fk5, galactic ..)
                                   std::string const ra,       ///< right ascension
                                   std::string const dec       ///< declination
                                  ) {
    return makeCoord(system, dmsStringToAngle(ra), dmsStringToAngle(dec));
}



/**
 * @brief Lightweight factory to make an empty coord.
 */
afwCoord::Coord::Ptr afwCoord::makeCoord(
                                   CoordSystem const system ///< the system (FK5, ICRS, etc)
                                  ) {
    switch (system) {
      case FK5:
        return boost::shared_ptr<Fk5Coord>(new Fk5Coord());
        break;
      case ICRS:
        return boost::shared_ptr<IcrsCoord>(new IcrsCoord());
        break;
      case GALACTIC:
        return boost::shared_ptr<GalacticCoord>(new GalacticCoord());
        break;
      case ECLIPTIC:
        return boost::shared_ptr<EclipticCoord>(new EclipticCoord());
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

std::ostream & afwCoord::operator<<(std::ostream & os, afwCoord::Coord const & coord) {
    return os << coord.getPosition() << "@" << coord.getEpoch();
}

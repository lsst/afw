// -*- lsst-c++ -*-
/**
 * @file Coord.cc
 * @brief Provide functions to handle coordinates
 * @ingroup afw
 * @author Steve Bickerton
 *
 * Most (nearly all) algorithms adapted from Astronomical Algorithms, 2nd ed. (J. Meeus)
 *
 */
#include <sstream>
#include <iostream>
#include <cmath>
#include <limits>

#include "lsst/pex/exceptions.h"
#include "boost/algorithm/string.hpp"
#include "boost/tuple/tuple.hpp"
#include "boost/format.hpp"

#include "lsst/afw/coord/Coord.h"
//#include "lsst/afw/coord/Date.h"
#include "lsst/daf/base/DateTime.h"

namespace coord = lsst::afw::coord;
namespace ex    = lsst::pex::exceptions;
namespace geom  = lsst::afw::geom;
namespace dafBase  = lsst::daf::base;

namespace {

double const NaN          = std::numeric_limits<double>::quiet_NaN();    
double const arcsecToRad  = M_PI/(3600.0*180.0); // arcsec per radian  = 2.062648e5;
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
    Dms(double const deg0) {
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
 * @brief Adjust a large angle or negative angle to be between 0, 360 degrees
 *
 */
double reduceAngle(double theta) {

    theta = theta - (static_cast<int>(theta)/360)*360.0;
    if (theta < 0) {
        theta += 360.0;
    }
    return theta;
}

    
/**
 * Store the Fk5 coordinates of the Galactic pole (and vice-versa) for coordinate transforms.
 *
 */
coord::Coord GalacticPoleInFk5 = coord::Coord(192.85950, 27.12825, 2000.0); // C&O
coord::Coord Fk5PoleInGalactic = coord::Coord(122.93200, 27.12825, 2000.0); // C&O


/**
 * @brief Compute the mean Sidereal Time at Greenwich
 *
 */
double meanSiderealTimeGreenwich(
                                 double const jd ///< Julian Day
                                ) {
    double const T = (jd - 2451545.0)/36525.0;
    return 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T*T - (T*T*T/38710000.0);
}

    
double const epochTolerance = 1.0e-12;  ///< Precession to new epoch performed if two epochs differ by this.
    

} // end anonymous namespace



/******************* Public functions ********************/


/**
 * @brief a Function to convert a coordinate in decimal degrees to a string with form dd:mm:ss
 *
 * @todo allow a user specified format
 */
std::string coord::degreesToDmsString(
                                      double const deg ///< Coord in decimal degrees
                                     ) {
    
    Dms dms(deg);
    
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
    char s[12];
    sprintf(s, "%02d:%02d:%05.2f", dms.sign*dms.deg, dms.min, dms.sec);
    std::string dmsStr(s);
    return dmsStr;
}

/**
 * @brief a function to convert decimal degrees to a string with form hh:mm:ss.s
 */
std::string coord::degreesToHmsString(
                                      double const deg ///< coord in decimal degrees
                                     ) {
    return degreesToDmsString(deg/15.0);
}

/**
 * @brief Convert a dd:mm:ss string to decimal degrees
 */
double coord::dmsStringToDegrees(
                                 std::string const dms ///< Coord as a string in dd:mm:ss format
                                ) {
    
    std::vector<std::string> elements;
    boost::split(elements, dms, boost::is_any_of(":"));
    int const deg   = abs(atoi(elements[0].c_str()));
    int const min   = atoi(elements[1].c_str());
    double const sec = atof(elements[2].c_str());

    double degrees = deg + min/60.0 + sec/3600.0;
    if ( (elements[0].c_str())[0] == '-' ) {
        degrees *= -1.0;
    }
    return degrees;
}


/**
 * @brief a function to convert hh:mm:ss.s string to decimal degrees
 *
 */
double coord::hmsStringToDegrees(
                                 std::string const hms ///< coord as a string in hh:mm:ss.s format
                                ) {
    return 15.0*dmsStringToDegrees(hms);
}


/**
 * @brief get the inclination of the ecliptic pole (obliquity) at epoch
 *
 */
double coord::eclipticPoleInclination(
                                      double const epoch ///< desired epoch for inclination
                                     ) {
    double const T = (epoch - 2000.0)/100.0;
    return 23.0 + 26.0/60.0 + (21.448 - 46.82*T - 0.0006*T*T - 0.0018*T*T*T)/3600.0;
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
coord::Coord::Coord(
                    geom::Point2D p2d,     ///< Point2D
                    CoordUnit unit,        ///< Rads, Degs, or Hrs
                    double const epoch     ///< epoch of coordinate
                   ) :
    _longitudeRad(NaN), _latitudeRad(NaN), _epoch(epoch) {

    if (unit == DEGREES) {
        _longitudeRad = degToRad*p2d.getX();
        _latitudeRad = degToRad*p2d.getY();
    } else if (unit == RADIANS) {
        _longitudeRad = p2d.getX();
        _latitudeRad = p2d.getY();
    } else if (unit == HOURS) {
        _longitudeRad = degToRad*15.0*p2d.getX();
        _latitudeRad = degToRad*p2d.getY();
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, "CoordUnit must be DEGREES, RADIANS, or HOURS");
    }
    
    _verifyValues();
}

/**
 * @brief Constructor for the Coord base class
 */
coord::Coord::Coord(
                    geom::Point3D const p3d,   ///< Point3D
                    double const epoch   ///< epoch of coordinate
                   ) :
    _longitudeRad( atan2(p3d.getY(), p3d.getX()) ),
    _latitudeRad(asin(p3d.getZ())),
    _epoch(epoch) {}


/**
 * @brief Constructor for the Coord base class
 *
 */
coord::Coord::Coord(
                    double const ra,   ///< Right ascension, decimal degrees
                    double const dec,  ///< Declination, decimal degrees
                    double const epoch ///< epoch of coordinate
                   ) :
    _longitudeRad(degToRad*ra), _latitudeRad(degToRad*dec), _epoch(epoch) {
    _verifyValues();
}

/**
 * @brief Constructor for the Coord base class
 *
 */
coord::Coord::Coord(
                    std::string const ra,  ///< Right ascension, hh:mm:ss.s format
                    std::string const dec, ///< Declination, dd:mm:ss.s format
                    double const epoch     ///< epoch of coordinate
                   ) :
    _longitudeRad(degToRad*15.0*dmsStringToDegrees(ra)),
    _latitudeRad(degToRad*dmsStringToDegrees(dec)),
    _epoch(epoch) {
    _verifyValues();
}

/**
 * @brief Default constructor for the Coord base class
 *
 * Set all values to NaN
 * Don't call _veriftyValues() method ... it'll fail.
 *
 */
coord::Coord::Coord() : _longitudeRad(NaN), _latitudeRad(NaN), _epoch(NaN) {}



/**
 * @brief Make sure the values we've got are in the range 0 < x < 2PI
 */
void coord::Coord::_verifyValues() {
    if (_longitudeRad < 0.0 || _longitudeRad >= 2.0*M_PI) {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          (boost::format("Azimuthal coord must be: 0 < long < 2PI (%f).") %
                           _longitudeRad).str());
    }
    if (_latitudeRad < -M_PI/2.0 || _latitudeRad > M_PI/2.0) {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          (boost::format("Latitude coord must be: -PI/2 <= lat <= PI/2 (%f).") %
                           _latitudeRad).str());
    }
}

/**
 * @brief Reset our coordinates wholesale.
 *
 * This allows the user to instantiate Coords without values, and fill them later.
 */
void coord::Coord::reset(
                         double const longitudeDeg, ///< Longitude coord (eg. R.A. for Fk5)
                         double const latitudeDeg,  ///< Latitude coord (eg. Declination for Fk5)
                         double const epoch         ///< epoch of coordinate
                        ) {
    _longitudeRad = degToRad*longitudeDeg;
    _latitudeRad  = degToRad*latitudeDeg;
    _epoch = epoch;
    _verifyValues();
}



/**
 * @brief Return our contents in a Point2D object
 *
 */
geom::Point2D coord::Coord::getPosition(CoordUnit unit) {
    // treat HOURS specially, they must mean hours for RA, degrees for Dec
    if (unit == HOURS) {
        return geom::makePointD(getLongitude(unit), getLatitude(DEGREES));
    } else {
        return geom::makePointD(getLongitude(unit), getLatitude(unit));
    }
}

std::pair<std::string, std::string> coord::Coord::getCoordNames() {
    return std::pair<std::string, std::string>("RA", "Dec");
}
std::pair<std::string, std::string> coord::GalacticCoord::getCoordNames() {
    return std::pair<std::string, std::string>("L", "B");
}
std::pair<std::string, std::string> coord::EclipticCoord::getCoordNames() {
    return std::pair<std::string, std::string>("Lambda", "Beta");
}
std::pair<std::string, std::string> coord::AltAzCoord::getCoordNames() {
    return std::pair<std::string, std::string>("Az", "Alt");
}


geom::Point3D coord::Coord::getVector() {
    double const x = cos(getLongitude(RADIANS))*cos(getLatitude(RADIANS));
    double const y = sin(getLongitude(RADIANS))*cos(getLatitude(RADIANS));
    double const z = sin(getLatitude(RADIANS));
    return geom::makePointD(x, y, z);
}


double coord::Coord::operator[](int index) {

    if (index == 0) {
        return _longitudeRad;
    } else if (index == 1) {
        return _latitudeRad;
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Index must be 0 or 1.");
    }
}

/**
 * @brief The main access method for the longitudinal coordinate
 *
 * All systems store their longitudinal coordinate in _longitude,
 * be it RA, l, lambda, or azimuth.  This is how they're accessed.
 *
 */
double coord::Coord::getLongitude(CoordUnit unit) {
    switch (unit) {
      case DEGREES:
        return radToDeg*_longitudeRad;
        break;
      case RADIANS:
        return _longitudeRad;
        break;
      case HOURS:
        return radToDeg*_longitudeRad/15.0;
        break;
      default:
        throw LSST_EXCEPT(ex::InvalidParameterException, "Units must be DEGREES, RADIANS, or HOURS.");
        break;
    }
}

/**
 * @brief The main access method for the longitudinal coordinate
 *
 * All systems store their latitudinal coordinate in _latitude,
 * be it Dec, b, beta, or altitude.  This is how they're accessed.
 *
 * @note There's no reason to want a latitude in hours, so that unit will cause
 *       an exception to be thrown
 *
 */
double coord::Coord::getLatitude(CoordUnit unit) {
    switch (unit) {
      case DEGREES:
        return radToDeg*_latitudeRad;
        break;
      case RADIANS:
        return _latitudeRad;
        break;
      default:
        throw LSST_EXCEPT(ex::InvalidParameterException, "Units must be DEGREES, or RADIANS.");
        break;
    }
}

/**
 * @brief Allow quick access to the longitudinal coordinate as a string
 *
 * @note There's no reason to want a longitude in radians, so that unit will cause
 *       an exception to be thrown
 * @note There's no clear winner for a default, so the unit must always be
 *       explicitly provided.
 *
 */
std::string coord::Coord::getLongitudeStr(CoordUnit unit) {
    if (unit == HOURS || unit == DEGREES) {
        return degreesToDmsString(getLongitude(unit));
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Units must be DEGREES or HOURS");
    }
}
/**
 * @brief Allow quick access to the longitude coordinate as a string
 *
 * @note There's no reason to want a latitude in radians or hours, so
 *       the units can not be explicitly requested.
 *
 */
std::string coord::Coord::getLatitudeStr() {
    return degreesToDmsString(getLatitude(DEGREES));
}


double coord::Fk5Coord::getRa(CoordUnit unit)         { return getLongitude(unit); }
double coord::Fk5Coord::getDec(CoordUnit unit)        { return getLatitude(unit); }
std::string coord::Fk5Coord::getRaStr(CoordUnit unit) { return getLongitudeStr(unit); }
std::string coord::Fk5Coord::getDecStr()              { return getLatitudeStr(); }

double coord::IcrsCoord::getRa(CoordUnit unit)         { return getLongitude(unit); }
double coord::IcrsCoord::getDec(CoordUnit unit)        { return getLatitude(unit); }
std::string coord::IcrsCoord::getRaStr(CoordUnit unit) { return getLongitudeStr(unit); }
std::string coord::IcrsCoord::getDecStr()              { return getLatitudeStr(); }

double coord::GalacticCoord::getL(CoordUnit unit)          { return getLongitude(unit); }
double coord::GalacticCoord::getB(CoordUnit unit)          { return getLatitude(unit); }
std::string coord::GalacticCoord::getLStr(CoordUnit unit)  { return getLongitudeStr(unit); }
std::string coord::GalacticCoord::getBStr()                { return getLatitudeStr(); }

double coord::EclipticCoord::getLambda(CoordUnit unit)         { return getLongitude(unit); }
double coord::EclipticCoord::getBeta(CoordUnit unit)           { return getLatitude(unit); }
std::string coord::EclipticCoord::getLambdaStr(CoordUnit unit) { return getLongitudeStr(unit); }
std::string coord::EclipticCoord::getBetaStr()                 { return getLatitudeStr(); }

double coord::AltAzCoord::getAzimuth(CoordUnit unit)         { return getLongitude(unit); }
double coord::AltAzCoord::getAltitude(CoordUnit unit)        { return getLatitude(unit); }
std::string coord::AltAzCoord::getAzimuthStr(CoordUnit unit) { return getLongitudeStr(unit); }
std::string coord::AltAzCoord::getAltitudeStr()              { return getLatitudeStr(); }


/**
 * @brief Tranform our current coords to another spherical polar system
 *
 * Variable names assume an equaltorial/galactic tranform, but it works
 *  for any spherical polar system when the appropriate poles are supplied.
 */
coord::Coord coord::Coord::transform(
                                     Coord poleTo,   ///< Pole of the destination system in the current coords
                                     Coord poleFrom  ///< Pole of the current system in the destination coords
                                    ) {
    double const alphaGP  = poleFrom[0];
    double const deltaGP  = poleFrom[1];
    double const lCP      = poleTo[0];
    
    double const alpha = getLongitude(RADIANS);
    double const delta = getLatitude(RADIANS);
    
    double const l = radToDeg*(lCP - atan2(sin(alpha - alphaGP), 
                                           tan(delta)*cos(deltaGP) - cos(alpha - alphaGP)*sin(deltaGP)));
    double const b = radToDeg*asin( (sin(deltaGP)*sin(delta) + cos(deltaGP)*cos(delta)*cos(alpha - alphaGP)));

    return Coord(reduceAngle(l), b);
}


/**
 * @brief compute the angular separation between two Coords
 *
 */
double coord::Coord::angularSeparation(
                                       Coord &c, ///< coordinate to compute our separation from
                                       CoordUnit unit
                                      ) {

    // make sure they're fk5
    Fk5Coord fk51 = this->toFk5();
    Fk5Coord fk5tmp = c.toFk5();
    
    // make sure they have the same epoch
    Fk5Coord fk52;
    if ( fabs(fk51.getEpoch() - fk5tmp.getEpoch()) > epochTolerance ) {
        fk52 = fk5tmp.precess(fk51.getEpoch());
    } else {
        fk52 = fk5tmp;
    }

    // work in Fk5, no matter what two derived classes we're given (eg Fk5 and Galactic)
    // we'll put them in the same system.
    double const alpha1 = fk51.getRa(RADIANS);
    double const delta1 = fk51.getDec(RADIANS);
    double const alpha2 = fk52.getRa(RADIANS);
    double const delta2 = fk52.getDec(RADIANS);
    
#if 0
    // this formula breaks down near 0 and 180
    double const cosd    = sin(delta1)*sin(delta2) + cos(delta1)*cos(delta2)*cos(alpha1 - alpha2);
    double const distDeg = radToDeg*acos(cosd);
#endif

    // use haversine form.  it's stable near 0 and 180.
    double const dDelta = delta1 - delta2;
    double const dAlpha = alpha1 - alpha2;
    double const havDDelta = sin(dDelta/2.0)*sin(dDelta/2.0);
    double const havDAlpha = sin(dAlpha/2.0)*sin(dAlpha/2.0);
    double const havD = havDDelta + cos(delta1)*cos(delta2)*havDAlpha;
    double const sinDHalf = std::sqrt(havD);
    double dist = 2.0*asin(sinDHalf);

    if (unit == DEGREES) {
        dist *= radToDeg;
    }
    
    return dist;
}



/**
 * @brief Convert ourself to Fk5: RA, Dec (basically J2000)
 */
coord::Fk5Coord coord::Coord::toFk5() {
    return Fk5Coord(getLongitude(DEGREES), getLatitude(DEGREES), getEpoch());
}

/**
 * @brief Convert ourself to ICRS: RA, Dec (basically J2000)
 *
 */
coord::IcrsCoord coord::Coord::toIcrs() {
    return this->toFk5().toIcrs();
}

/**
 * @brief Convert ourself to Galactic: l, b
 */
coord::GalacticCoord coord::Coord::toGalactic() {
    return this->toFk5().toGalactic();
}

/**
 * @brief Convert ourself to Ecliptic: lambda, beta
 */
coord::EclipticCoord coord::Coord::toEcliptic() {
    return this->toFk5().toEcliptic();
}

/**
 * @brief Convert ourself to Altitude/Azimuth: alt, az
 */
coord::AltAzCoord coord::Coord::toAltAz(
                                        Observatory obs,            ///< observatory of observation
                                        dafBase::DateTime obsDate   ///< date of observation
                                       ) {
    return this->toFk5().toAltAz(obs, obsDate);
}




/* ============================================================
 *
 * class Fk5Coord
 *
 * ============================================================*/


/**
 * @brief Convert ourself to Fk5: RA, Dec (basically J2000)
 */
coord::Fk5Coord coord::Fk5Coord::toFk5() {
    return Fk5Coord(getLongitude(DEGREES), getLatitude(DEGREES), getEpoch());
}

/**
 * @brief Convert ourself to ICRS: RA, Dec (basically J2000)
 *
 */
coord::IcrsCoord coord::Fk5Coord::toIcrs() {

    // only do the precession to 2000 if we're not already there.
    if ( fabs(getEpoch() - 2000.0) > epochTolerance ) {
        coord::Fk5Coord c = precess(2000.0);
        return IcrsCoord(c.getLongitude(DEGREES), c.getLatitude(DEGREES));
    } else {
        return IcrsCoord(getLongitude(DEGREES), getLatitude(DEGREES));
    }
}

/**
 * @brief Convert ourself to Galactic: l, b
 */
coord::GalacticCoord coord::Fk5Coord::toGalactic() {

    // if we're epoch==2000, we can transform, otherwise we need to precess first
    Fk5Coord c;
    if ( fabs(getEpoch() - 2000.0) > epochTolerance ) {
        c = precess(2000.0);
    } else {
        c = *this;
    }
    
    Coord ct = c.transform(Fk5PoleInGalactic, GalacticPoleInFk5);
    return GalacticCoord(ct.getLongitude(DEGREES), ct.getLatitude(DEGREES));
    
}

/**
 * @brief Convert ourself to Ecliptic: lambda, beta
 */
coord::EclipticCoord coord::Fk5Coord::toEcliptic() {
    double const eclPoleIncl = eclipticPoleInclination(getEpoch());
    Coord const eclPoleInEquatorial(270.0, 90.0 - eclPoleIncl, getEpoch());
    Coord const equPoleInEcliptic(90.0, 90.0 - eclPoleIncl, getEpoch());
    Coord c = transform(equPoleInEcliptic, eclPoleInEquatorial); 
    return EclipticCoord(c.getLongitude(DEGREES), c.getLatitude(DEGREES), getEpoch());
}

/**
 * @brief Convert ourself to Altitude/Azimuth: alt, az
 */
coord::AltAzCoord coord::Fk5Coord::toAltAz(
                                           Observatory obs,            ///< observatory of observation
                                           dafBase::DateTime obsDate   ///< date of observation
                                          ) {

    // make sure we precess to the epoch
    Fk5Coord fk5 = precess(obsDate.get(dafBase::DateTime::EPOCH));

    // greenwich sidereal time
    double const theta0 = degToRad*reduceAngle(
                              meanSiderealTimeGreenwich(obsDate.get(dafBase::DateTime::JD)));
    double const phi    = obs.getLatitude(RADIANS);  // observatory latitude
    double const L      = obs.getLongitude(RADIANS); // observatory longitude

    double const alpha  = fk5.getRa(RADIANS);
    double const delta  = fk5.getDec(RADIANS);

    double const H         = theta0 - L - alpha;
    double const sinh      = sin(phi)*sin(delta) + cos(phi)*cos(delta)*cos(H);
    double const tanAnum   = sin(H);
    double const tanAdenom = (cos(H)*sin(phi) - tan(delta)*cos(phi));
    double const h         = radToDeg*asin(sinh);
    double const A         = reduceAngle(-90.0 - radToDeg*atan2(tanAdenom, tanAnum));
    
    return AltAzCoord(A, h, obsDate.get(dafBase::DateTime::EPOCH), obs);
}



/**
 * @brief Precess ourselves from whence we are to a new epoch
 *
 */
coord::Fk5Coord coord::Fk5Coord::precess(
                                         double const epochTo ///< epoch to precess to
                                        ) {

    // return a copy if the epochs are the same
    if ( fabs(getEpoch() - epochTo) < epochTolerance) {
        return Fk5Coord(getLongitude(DEGREES), getLatitude(DEGREES), getEpoch());
    }
    
    dafBase::DateTime dateFrom(getEpoch(), dafBase::DateTime::EPOCH, dafBase::DateTime::TAI);
    dafBase::DateTime dateTo(epochTo, dafBase::DateTime::EPOCH, dafBase::DateTime::TAI);
    double const jd0 = dateFrom.get(dafBase::DateTime::JD);
    double const jd  = dateTo.get(dafBase::DateTime::JD);

    double const T   = (jd0 - JD2000)/36525.0;
    double const t   = (jd - jd0)/36525.0;
    double const tt  = t*t;
    double const ttt = tt*t;

    double const xi    = arcsecToRad*((2306.2181 + 1.39656*T - 0.000139*T*T)*t +
                                      (0.30188 - 0.000344*T)*tt + 0.017998*ttt);
    double const z     = arcsecToRad*((2306.2181 + 1.39656*T - 0.000139*T*T)*t +
                                      (1.09468 + 0.000066*T)*tt + 0.018203*ttt);
    double const theta = arcsecToRad*((2004.3109 - 0.85330*T - 0.000217*T*T)*t -
                                      (0.42665 + 0.000217*T)*tt - 0.041833*ttt);

    Fk5Coord fk5 = this->toFk5();
    double const alpha0 = fk5.getRa(RADIANS);
    double const delta0 = fk5.getDec(RADIANS);
    
    double const A = cos(delta0)*sin(alpha0 + xi);
    double const B = cos(theta)*cos(delta0)*cos(alpha0 + xi) - sin(theta)*sin(delta0);
    double const C = sin(theta)*cos(delta0)*cos(alpha0 + xi) + cos(theta)*sin(delta0);

    double const alpha = reduceAngle(radToDeg*(atan2(A,B) + z));
    double const delta = radToDeg*asin(C);
    
    return Fk5Coord(alpha, delta, epochTo);
}


/* ============================================================
 *
 * class IcrsCoord
 *
 * ============================================================*/

/**
 * @brief 
 *
 * @note
 */
coord::Fk5Coord coord::IcrsCoord::toFk5() {
    return Fk5Coord(getLongitude(DEGREES), getLatitude(DEGREES), 2000.0);
}

coord::IcrsCoord coord::IcrsCoord::toIcrs() {
    return IcrsCoord(getLongitude(DEGREES), getLatitude(DEGREES));
}


/* ============================================================
 *
 * class GalacticCoord
 *
 * ============================================================*/

/**
 * @brief Convert ourself from galactic to Fk5
 */
coord::Fk5Coord coord::GalacticCoord::toFk5() {
    
    // transform to equatorial
    // galactic coords are ~constant, and the poles used are for epoch=2000, so we get J2000
    Coord c = transform(GalacticPoleInFk5, Fk5PoleInGalactic);
    
    //
    if ( fabs(getEpoch() - 2000.0) < epochTolerance ) {
        return Fk5Coord(c.getLongitude(DEGREES), c.getLatitude(DEGREES), 2000.0);
    } else {
        return Fk5Coord(c.getLongitude(DEGREES), c.getLatitude(DEGREES), 2000.0).precess(getEpoch());
    }
}

/**
 * @brief Convert ourself from Galactic to Galactic ... a no-op
 */
coord::GalacticCoord coord::GalacticCoord::toGalactic() { 
    return GalacticCoord(getLongitude(DEGREES), getLatitude(DEGREES), getEpoch());
}




/* ============================================================
 *
 * class EclipticCoord
 *
 * ============================================================*/

/**
 * @brief Convert ourself from Ecliptic to Ecliptic ... a no-op
 */
coord::EclipticCoord coord::EclipticCoord::toEcliptic() {
    return EclipticCoord(getLongitude(DEGREES), getLatitude(DEGREES), getEpoch());
}


/**
 * @brief Convert ourself from Ecliptic to Equatorial
 */
coord::Fk5Coord coord::EclipticCoord::toFk5() {
    double const eclPoleIncl = eclipticPoleInclination(getEpoch());
    Coord const eclipticPoleInFk5(270.0, 90.0 - eclPoleIncl, getEpoch());
    Coord const fk5PoleInEcliptic(90.0, 90.0 - eclPoleIncl, getEpoch());
    Coord c = transform(eclipticPoleInFk5, fk5PoleInEcliptic);
    return Fk5Coord(c.getLongitude(DEGREES), c.getLatitude(DEGREES), getEpoch());
}


/**
 * @brief precess to new epoch
 *
 * Do this by going through fk5
 */
coord::EclipticCoord coord::EclipticCoord::precess(
                                                   double epochTo ///< epoch to precess to.
                                                  ) {
    return this->toFk5().precess(epochTo).toEcliptic();
}



/* ============================================================
 *
 * class AltAzCoord
 *
 * ============================================================*/

/**
 * @brief Convert ourself from AltAz to Fk5
 */
coord::Fk5Coord coord::AltAzCoord::toFk5() {
    double const A        = getAzimuth(RADIANS);
    double const h        = getAltitude(RADIANS);
    double const phi      = _obs.getLatitude(RADIANS);
    double const L        = _obs.getLongitude(RADIANS);

    double const jd       = dafBase::DateTime(getEpoch(),
                                              dafBase::DateTime::EPOCH,
                                              dafBase::DateTime::TAI).get(dafBase::DateTime::JD);
    double const theta0   = degToRad*meanSiderealTimeGreenwich(jd);

    double const tanH     = sin(A) / (cos(A)*sin(phi) + tan(h)*cos(phi));
    double const alpha    = radToDeg*(theta0 - L - atan(tanH));
    double const sinDelta = sin(phi)*sin(h) - cos(phi)*cos(h)*cos(A);
    double const delta    = radToDeg*asin(sinDelta);

    return Fk5Coord(alpha, delta, getEpoch());
}

/**
 * @brief Convert ourself from AltAz to AltAz ... a no-op
 */
coord::AltAzCoord coord::AltAzCoord::toAltAz(
                                             Observatory const &obs, ///< observatory of observation
                                             dafBase::DateTime const &date        ///< date of observation
                                            ) {
    return AltAzCoord(getLongitude(DEGREES), getLatitude(DEGREES), getEpoch(), _obs);
}

/**
 * @brief Convert ourself from AltAz to AltAz with no observatory or date arguments
 *
 * As this is essentially a copy-constructor, the extra info can be obtained internally.
 */
coord::AltAzCoord coord::AltAzCoord::toAltAz() {
    return AltAzCoord(getLongitude(DEGREES), getLatitude(DEGREES), getEpoch(), _obs);
}



/**
 * @brief Factory function to create a Coord of arbitrary type with decimal RA,Dec
 *
 */
coord::Coord::Ptr coord::makeCoord(
                                   CoordSystem const system,     ///< the system (equ, fk5, galactic ..)
                                   double const ra,              ///< right ascension
                                   double const dec,             ///< declination
                                   double const epoch            ///< epoch of coordinate
                                  ) {

    switch (system) {

      case FK5:
        return boost::shared_ptr<Fk5Coord>(new Fk5Coord(ra, dec, epoch));
        break;
      case ICRS:
        // if the epoch isn't 2000.0, should we precess or throw an exception?
        if ( fabs(epoch - 2000.0) < epochTolerance ) {
            return boost::shared_ptr<IcrsCoord>(new IcrsCoord(ra, dec));
        } else {
            IcrsCoord c = Fk5Coord(ra, dec, epoch).toIcrs();
            return boost::shared_ptr<IcrsCoord>(new IcrsCoord(c.getLongitude(DEGREES),
                                                              c.getLatitude(DEGREES)));
        }
        break;
      case GALACTIC:
        return boost::shared_ptr<GalacticCoord>(new GalacticCoord(ra, dec, epoch));
        break;
      case ECLIPTIC:
        return boost::shared_ptr<EclipticCoord>(new EclipticCoord(ra, dec, epoch));
        break;
      case ALTAZ:
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Cannot make AltAz with makeCoord() (must also specify Observatory).\n"
                          "Instantiate AltAzCoord() directly.");
        break;
      default:
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Undefined CoordSystem: only FK5, ICRS, GALACTIC, ECLIPTIC, and ALTAZ allowed.");
        break;
        
    }

}


/**
 * @brief Factory function to create a Coord of arbitrary type with a Point3D
 *
 */
coord::Coord::Ptr coord::makeCoord(
                                   CoordSystem const system,     ///< the system (equ, fk5, galactic ..)
                                   geom::Point3D const p3d,      ///< the coord in Point3D format
                                   double const epoch            ///< epoch of coordinate
                                  ) {

    switch (system) {

      case FK5:
        return boost::shared_ptr<Fk5Coord>(new Fk5Coord(p3d, epoch));
        break;
      case ICRS:
        // if the epoch isn't 2000.0, should we precess or throw an exception?
        if ( fabs(epoch - 2000.0) < epochTolerance ) {
            return boost::shared_ptr<IcrsCoord>(new IcrsCoord(p3d));
        } else {
            IcrsCoord c = Fk5Coord(p3d, epoch).toIcrs();
            return boost::shared_ptr<IcrsCoord>(new IcrsCoord(c.getLongitude(DEGREES),
                                                              c.getLatitude(DEGREES)));
        }
        break;
      case GALACTIC:
        return boost::shared_ptr<GalacticCoord>(new GalacticCoord(p3d, epoch));
        break;
      case ECLIPTIC:
        return boost::shared_ptr<EclipticCoord>(new EclipticCoord(p3d, epoch));
        break;
      case ALTAZ:
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Cannot make AltAz with makeCoord() (must also specify Observatory).\n"
                          "Instantiate AltAzCoord() directly.");
        break;
      default:
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Undefined CoordSystem: only FK5, ICRS, GALACTIC, ECLIPTIC, and ALTAZ allowed.");
        break;
        
    }

}


/**
 * @brief Factory function to create a Coord of arbitrary type with Point2D
 *
 */
coord::Coord::Ptr coord::makeCoord(
                                   CoordSystem const system,     ///< the system (equ, fk5, galactic ..)
                                   geom::Point2D p2d,            ///< the (eg) ra,dec in a Point2D
                                   CoordUnit unit,               ///< the units (eg. DEGREES, RADIANS)
                                   double const epoch            ///< epoch of coordinate
                                  ) {
    switch (unit) {
      case DEGREES:
        return makeCoord(system, p2d.getX(), p2d.getY(), epoch);
        break;
      case RADIANS:
        return makeCoord(system, radToDeg*p2d.getX(), radToDeg*p2d.getY(), epoch);
        break;
      case HOURS:
        return makeCoord(system, 15.0*radToDeg*p2d.getX(), radToDeg*p2d.getY(), epoch);
        break;
      default:
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Point2D values for Coord must be DEGREES, RADIANS, or HOURS.");
        break;
    }
}


/**
 * @brief Factory function to create a Coord of arbitrary type with string RA, Dec
 *
 */
coord::Coord::Ptr coord::makeCoord(
                                   CoordSystem const system,   ///< the system (equ, fk5, galactic ..)
                                   std::string const ra,       ///< right ascension
                                   std::string const dec,      ///< declination
                                   double const epoch          ///< epoch of coordinate
                                  ) {
    return makeCoord(system, 15.0*dmsStringToDegrees(ra), dmsStringToDegrees(dec), epoch);
}

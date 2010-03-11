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
#include "lsst/afw/coord/Observatory.h"
#include "lsst/afw/coord/Date.h"

namespace coord = lsst::afw::coord;
namespace ex    = lsst::pex::exceptions;
namespace geom  = lsst::afw::geom;

namespace {

double const NaN          = std::numeric_limits<double>::quiet_NaN();    
double const arcsecToRad  = M_PI/(3600.0*180.0); // arcsec per radian  = 2.062648e5;
    
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
                    coord::CoordUnit unit, ///< Rads, Degs, or Hrs
                    double const epoch     ///< epoch of coordinate
                   ) :
    _longitudeRad(NaN), _latitudeRad(NaN), _epoch(epoch) {

    if (unit == coord::DEG) {
        _longitudeRad = degToRad*p2d.getX();
        _latitudeRad = degToRad*p2d.getY();
    } else if (unit == coord::RAD) {
        _longitudeRad = p2d.getX();
        _latitudeRad = p2d.getY();
    } else if (unit == coord::HRS) {
        _longitudeRad = degToRad*15.0*p2d.getX();
        _latitudeRad = degToRad*p2d.getY();
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, "CoordUnit must be DEG, RAD, or HRS");
    }
    
    _verifyValues();
}


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
    _longitudeRad(degToRad*15.0*coord::dmsStringToDegrees(ra)),
    _latitudeRad(degToRad*coord::dmsStringToDegrees(dec)),
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
geom::Point2D coord::Coord::getPoint2D(coord::CoordUnit unit) {
    if (unit == coord::DEG) {
        return geom::makePointD(getLongitudeDeg(), getLatitudeDeg());
    } else if (unit == coord::RAD) {
        return geom::makePointD(getLongitudeRad(), getLatitudeRad());
    } else if (unit == coord::HRS) {
        return geom::makePointD(getLongitudeHrs(), getLatitudeDeg());
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Undefined CoordUnit type.  Only DEG, RAD, HRS allowed.");
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


/**
 *
 *
 */
double coord::Coord::getLongitudeDeg()          { return radToDeg*_longitudeRad; }
double coord::Coord::getLongitudeHrs()          { return radToDeg*_longitudeRad/15.0; }
double coord::Coord::getLongitudeRad()          { return _longitudeRad; }
double coord::Coord::getLatitudeDeg()           { return radToDeg*_latitudeRad; }
double coord::Coord::getLatitudeRad()           { return _latitudeRad; }
std::string coord::Coord::getLongitudeStr()     { return degreesToDmsString(radToDeg*_longitudeRad/15.0); }
std::string coord::Coord::getLatitudeStr()      { return degreesToDmsString(radToDeg*_latitudeRad); }
                                                
double coord::Coord::getRaDeg()                 { return this->toEquatorial().getLongitudeDeg(); }
double coord::Coord::getDecDeg()                { return this->toEquatorial().getLatitudeDeg(); }
double coord::Coord::getRaHrs()                 { return this->toEquatorial().getLongitudeHrs(); }
double coord::Coord::getRaRad()                 { return this->toEquatorial().getLongitudeRad(); }
double coord::Coord::getDecRad()                { return this->toEquatorial().getLatitudeRad(); }
std::string coord::Coord::getRaStr()            { return this->toEquatorial().getLongitudeStr(); }
std::string coord::Coord::getDecStr()           { return this->toEquatorial().getLatitudeStr(); }
                                                
double coord::Coord::getLDeg()                  { return this->toGalactic().getLongitudeDeg(); }
double coord::Coord::getBDeg()                  { return this->toGalactic().getLatitudeDeg(); }
double coord::Coord::getLHrs()                  { return this->toGalactic().getLongitudeHrs(); }    
double coord::Coord::getLRad()                  { return this->toGalactic().getLongitudeRad(); }
double coord::Coord::getBRad()                  { return this->toGalactic().getLatitudeRad(); }
std::string coord::Coord::getLStr()             { return this->toGalactic().getLongitudeStr(); }
std::string coord::Coord::getBStr()             { return this->toGalactic().getLatitudeStr(); }
                                                
double coord::Coord::getLambdaDeg()             { return this->toEcliptic().getLongitudeDeg(); }
double coord::Coord::getBetaDeg()               { return this->toEcliptic().getLatitudeDeg(); }
double coord::Coord::getLambdaHrs()             { return this->toEcliptic().getLongitudeHrs(); }
double coord::Coord::getLambdaRad()             { return this->toEcliptic().getLongitudeRad(); }
double coord::Coord::getBetaRad()               { return this->toEcliptic().getLatitudeRad(); }
std::string coord::Coord::getLambdaStr()        { return this->toEcliptic().getLongitudeStr(); }
std::string coord::Coord::getBetaStr()          { return this->toEcliptic().getLatitudeStr(); }

double coord::AltAzCoord::getAzimuthDeg()       { return getLongitudeDeg(); }
double coord::AltAzCoord::getAltitudeDeg()      { return getLatitudeDeg(); }
double coord::AltAzCoord::getAzimuthHrs()       { return getLongitudeHrs(); }
double coord::AltAzCoord::getAzimuthRad()       { return getLongitudeRad(); }
double coord::AltAzCoord::getAltitudeRad()      { return getLatitudeRad(); }
std::string coord::AltAzCoord::getAzimuthStr()  { return getLongitudeStr(); }
std::string coord::AltAzCoord::getAltitudeStr() { return getLatitudeStr(); }


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
    double const alphaGP  = poleFrom.getLongitudeRad();
    double const deltaGP  = poleFrom.getLatitudeRad();
    double const lCP      = poleTo.getLongitudeRad();
    
    double const alpha = getLongitudeRad();
    double const delta = getLatitudeRad();
    
    double const l = radToDeg*(lCP - atan2(sin(alpha - alphaGP), 
                                           tan(delta)*cos(deltaGP) - cos(alpha - alphaGP)*sin(deltaGP)));
    double const b = radToDeg*asin( (sin(deltaGP)*sin(delta) + cos(deltaGP)*cos(delta)*cos(alpha - alphaGP)));

    return coord::Coord(reduceAngle(l), b);
}


/**
 * @brief compute the angular separation between two Coords
 *
 */
double coord::Coord::angularSeparation(
                                       coord::Coord &c ///< coordinate to compute our separation from
                                      ) {

    // make sure they have the same epoch
    coord::Coord cPrecess;
    if ( fabs(getEpoch() - c.getEpoch()) > epochTolerance ) {
        cPrecess = c.precess(getEpoch());
    } else {
        cPrecess = c;
    }

    // work in Fk5, no matter what two derived classes we're given (eg Fk5 and Galactic)
    // we'll put them in the same system.
    double const alpha1 = getRaRad();
    double const delta1 = getDecRad();
    double const alpha2 = cPrecess.getRaRad();
    double const delta2 = cPrecess.getDecRad();
    
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
    double const distDeg = radToDeg*2.0*asin(sinDHalf);
    
    return distDeg;
}


/**
 * @brief Precess ourselves from whence we are to a new epoch
 *
 */
coord::Coord coord::Coord::precess(
                                   double const epochTo ///< epoch to precess to
                                  ) {

    coord::Date dateFrom(getEpoch(), coord::Date::EPOCH);
    coord::Date dateTo(epochTo, coord::Date::EPOCH);
    double const jd0 = dateFrom.getJd();
    double const jd  = dateTo.getJd();

    double const T   = (jd0 - coord::JD2000)/36525.0;
    double const t   = (jd - jd0)/36525.0;
    double const tt  = t*t;
    double const ttt = tt*t;

    double const xi    = arcsecToRad*((2306.2181 + 1.39656*T - 0.000139*T*T)*t +
                                      (0.30188 - 0.000344*T)*tt + 0.017998*ttt);
    double const z     = arcsecToRad*((2306.2181 + 1.39656*T - 0.000139*T*T)*t +
                                      (1.09468 + 0.000066*T)*tt + 0.018203*ttt);
    double const theta = arcsecToRad*((2004.3109 - 0.85330*T - 0.000217*T*T)*t -
                                      (0.42665 + 0.000217*T)*tt - 0.041833*ttt);

    double const alpha0 = getRaRad();
    double const delta0 = getDecRad();
    
    double const A = cos(delta0)*sin(alpha0 + xi);
    double const B = cos(theta)*cos(delta0)*cos(alpha0 + xi) - sin(theta)*sin(delta0);
    double const C = sin(theta)*cos(delta0)*cos(alpha0 + xi) + cos(theta)*sin(delta0);

    double const alpha = reduceAngle(radToDeg*(atan2(A,B) + z));
    double const delta = radToDeg*asin(C);
    
    return coord::Coord(alpha, delta, epochTo);
}


/**
 * @brief Convert ourself to Equatorial: RA, Dec
 */
coord::EquatorialCoord coord::Coord::toEquatorial() {
    return coord::EquatorialCoord(getLongitudeDeg(), getLatitudeDeg(), getEpoch());
}


/**
 * @brief Convert ourself to Fk5: RA, Dec (basically J2000)
 */
coord::Fk5Coord coord::Coord::toFk5() {
    coord::Coord c = precess(2000.0);
    return Fk5Coord(c.getLongitudeDeg(), c.getLatitudeDeg());
}

/**
 * @brief Convert ourself to ICRS: RA, Dec (basically J2000)
 *
 * @note This currently just calls the FK5 routine.
 */
coord::IcrsCoord coord::Coord::toIcrs() {
    coord::Coord c = precess(2000.0);
    return IcrsCoord(c.getLongitudeDeg(), c.getLatitudeDeg());
}

/**
 * @brief Convert ourself to Galactic: l, b
 */
coord::GalacticCoord coord::Coord::toGalactic() {

    // if we're epoch==2000, we can transform, otherwise we need to precess first
    if ( (getEpoch() - 2000.0) < epochTolerance ) {
        Coord c = transform(Fk5PoleInGalactic, GalacticPoleInFk5);
        return GalacticCoord(c.getLongitudeDeg(), c.getLatitudeDeg(), getEpoch());
    } else {
        // precess to 2000.0 and call ourselves recursively ... then we'll land in the above 'if' block
        return precess(2000.0).toGalactic();
    }
    
}

/**
 * @brief Convert ourself to Ecliptic: lambda, beta
 */
coord::EclipticCoord coord::Coord::toEcliptic() {
    double const eclPoleIncl = eclipticPoleInclination(getEpoch());
    Coord const eclPoleInEquatorial(270.0, 90.0 - eclPoleIncl, getEpoch());
    Coord const equPoleInEcliptic(90.0, 90.0 - eclPoleIncl, getEpoch());
    Coord c = transform(equPoleInEcliptic, eclPoleInEquatorial); 
    return EclipticCoord(c.getLongitudeDeg(), c.getLatitudeDeg(), getEpoch());
}

/**
 * @brief Convert ourself to Altitude/Azimuth: alt, az
 */
coord::AltAzCoord coord::Coord::toAltAz(
                                        coord::Observatory obs, ///< observatory of observation
                                        coord::Date obsDate     ///< date of observation
                                       ) {

    // precess to the epoch
    coord::Coord coord = precess(obsDate.getEpoch());

    // greenwich sidereal time
    double const theta0 = degToRad*reduceAngle(meanSiderealTimeGreenwich(obsDate.getJd()));
    double const phi    = obs.getLatitudeRad();  // observatory latitude
    double const L      = obs.getLongitudeRad(); // observatory longitude

    double const alpha  = coord.getRaRad();
    double const delta  = coord.getDecRad();

    double const H         = theta0 - L - alpha;
    double const sinh      = sin(phi)*sin(delta) + cos(phi)*cos(delta)*cos(H);
    double const tanAnum   = sin(H);
    double const tanAdenom = (cos(H)*sin(phi) - tan(delta)*cos(phi));
    double const h         = radToDeg*asin(sinh);
    double const A         = reduceAngle(-90.0 - radToDeg*atan2(tanAdenom, tanAnum));
    
    return AltAzCoord(A, h, obsDate.getEpoch(), obs);
}




/* ============================================================
 *
 * class EquatorialCoord
 *
 * ============================================================*/


/**
 * @brief precess ourselfs to a new epoch
 */
coord::EquatorialCoord coord::EquatorialCoord::precess(
                                                       double const epochTo ///< epoch to precess to
                                                      ) {
    coord::Coord c = coord::Coord::precess(epochTo);
    return EquatorialCoord(c.getLongitudeDeg(), c.getLatitudeDeg(), c.getEpoch());
}




/* ============================================================
 *
 * class Fk5Coord
 *
 * ============================================================*/


/**
 * @brief precess ourselfs to a new epoch
 *
 * We can't precess Fk5, or it won't be Fk5 anymore.  We'll return an Equatorial
 */
coord::EquatorialCoord coord::Fk5Coord::precess(
                                                double const epochTo ///< epoch to precess to
                                               ) {
    return coord::EquatorialCoord(getLongitudeDeg(), getLatitudeDeg(), 2000.0).precess(epochTo);
}



/* ============================================================
 *
 * class IcrsCoord
 *
 * ============================================================*/

/**
 * @brief precess ourselfs to a new epoch
 *
 * Can't just let the base class Coord do this ... we need to return the correct type
 */
coord::EquatorialCoord coord::IcrsCoord::precess(
                                                 double const epochTo ///< epoch to precess to
                                                ) {
    return coord::EquatorialCoord(getLongitudeDeg(), getLatitudeDeg(), 2000.0).precess(epochTo);
}



/* ============================================================
 *
 * class GalacticCoord
 *
 * ============================================================*/

/**
 * @brief Convert ourself from galactic to Fk5
 */
coord::EquatorialCoord coord::GalacticCoord::toEquatorial() {
    
    // transform to equatorial
    // galactic coords are ~constant, and the poles used are for epoch=2000, so we get J2000
    Coord c = transform(GalacticPoleInFk5, Fk5PoleInGalactic);
    
    //
    if ( fabs(getEpoch() - 2000.0) < epochTolerance ) {
        return coord::EquatorialCoord(c.getLongitudeDeg(), c.getLatitudeDeg(), 2000.0);
    } else {
        return coord::EquatorialCoord(c.getLongitudeDeg(), c.getLatitudeDeg(), 2000.0).precess(getEpoch());
    }
}


/**
 * @brief Convert ourself from galactic to Fk5
 */
coord::Fk5Coord coord::GalacticCoord::toFk5() {
    
    // transform to equatorial
    // galactic coords are ~constant, and the poles used are for epoch=2000, so we get J2000
    Coord c = transform(GalacticPoleInFk5, Fk5PoleInGalactic);
    
    // put the new values in an Fk5Coord
    return coord::Fk5Coord(c.getLongitudeDeg(), c.getLatitudeDeg());
}

/**
 * @brief Convert ourself from galactic to Icrs
 *
 * @note This currently calls the Fk5 routines and is not strictly ICRS.
 */
coord::IcrsCoord coord::GalacticCoord::toIcrs() {
    return (this->toFk5()).toIcrs();
}


/**
 * @brief Convert ourself from Galactic to Galactic ... a no-op
 */
coord::GalacticCoord coord::GalacticCoord::toGalactic() { 
    return GalacticCoord(getLongitudeDeg(), getLatitudeDeg(), getEpoch());
}

/**
 * @brief Convert ourself from Galactic to Ecliptic
 *
 * To do this, we'll go to fk5 first, then to ecliptic
 */
coord::EclipticCoord coord::GalacticCoord::toEcliptic() {
    return (this->toFk5()).toEcliptic();
}

/**
 * @brief Convert ourself from Galactic to AltAz
 *
 * To do this, we'll go to fk5 first, then to alt/az
 */
coord::AltAzCoord coord::GalacticCoord::toAltAz(
                                                coord::Observatory const &obs, ///< observatory of observation
                                                coord::Date const &date       ///< date of observation
                                               ) {
    return (this->toFk5()).toAltAz(obs, date);
}

/**
 * @brief Precess to a new epoch
 *
 * Actually nothing to do here, just create a new GalacticCoord with the epoch
 */
coord::GalacticCoord coord::GalacticCoord::precess(double epochTo) {
    return coord::GalacticCoord(getLongitudeDeg(), getLatitudeDeg(), epochTo);
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
    return coord::EclipticCoord(getLongitudeDeg(), getLatitudeDeg(), getEpoch());
}


/**
 * @brief Convert ourself from Ecliptic to Equatorial
 */
coord::EquatorialCoord coord::EclipticCoord::toEquatorial() {
    double const eclPoleIncl = eclipticPoleInclination(getEpoch());
    Coord const eclipticPoleInFk5(270.0, 90.0 - eclPoleIncl, getEpoch());
    Coord const fk5PoleInEcliptic(90.0, 90.0 - eclPoleIncl, getEpoch());
    Coord c = transform(eclipticPoleInFk5, fk5PoleInEcliptic);
    return coord::EquatorialCoord(c.getLongitudeDeg(), c.getLatitudeDeg(), getEpoch());
}


/**
 * @brief Convert ourself from Ecliptic to Fk5
 */
coord::Fk5Coord coord::EclipticCoord::toFk5() {
    double const eclPoleIncl = eclipticPoleInclination(getEpoch());
    Coord const eclipticPoleInFk5(270.0, 90.0 - eclPoleIncl, getEpoch());
    Coord const fk5PoleInEcliptic(90.0, 90.0 - eclPoleIncl, getEpoch());
    Coord c = transform(eclipticPoleInFk5, fk5PoleInEcliptic);
    Coord cp = c.precess(2000.0);
    return Fk5Coord(cp.getLongitudeDeg(), cp.getLatitudeDeg());
}

/**
 * @brief Convert ourself from galactic to Icrs
 *
 * @note This currently calls the Fk5 routines and is not strictly ICRS.
 */
coord::IcrsCoord coord::EclipticCoord::toIcrs() {
    return (this->toFk5()).toIcrs();
}


/**
 * @brief Convert ourself from Ecliptic to Galactic
 *
 * To do this, we'll go to fk5 first, then to galactic
 */
coord::GalacticCoord coord::EclipticCoord::toGalactic() {
    return (this->toFk5()).toGalactic(); 
}


/**
 * @brief Convert ourself from Ecliptic to AltAz
 *
 * To do This, we'll go to fk5 first, then to alt/az.
 */
coord::AltAzCoord coord::EclipticCoord::toAltAz(
                                                coord::Observatory const &obs, ///< observatory of observation
                                                coord::Date const &date        ///< date of observation
                                               ) {
    return (this->toFk5()).toAltAz(obs, date);
}

/**
 * @brief precess to new epoch
 *
 * Do this by going through fk5
 */
coord::EclipticCoord coord::EclipticCoord::precess(
                                                   double epochTo ///< epoch to precess to.
                                                  ) {
    return (this->toFk5()).precess(epochTo).toEcliptic();
}


/* ============================================================
 *
 * class AltAzCoord
 *
 * ============================================================*/

/**
 * @brief Convert ourself from AltAz to Ecliptic
 *
 * Do this by going through fk5.
 */
coord::EclipticCoord coord::AltAzCoord::toEcliptic() {
    return (this->toFk5()).toEcliptic();
}

/**
 * @brief Convert ourself from AltAz to Galactic
 *
 * Do this by going through fk5
 */
coord::GalacticCoord coord::AltAzCoord::toGalactic() {
    return (this->toFk5()).toGalactic(); 
}




/**
 * @brief Convert ourself from AltAz to Fk5
 */
coord::Fk5Coord coord::AltAzCoord::toFk5() {
    double const A        = getAzimuthRad();
    double const h        = getAltitudeRad();
    double const phi      = _obs.getLatitudeRad();
    double const L        = _obs.getLongitudeRad();

    double const jd       = coord::Date(getEpoch(), coord::Date::EPOCH).getJd();
    double const theta0   = degToRad*meanSiderealTimeGreenwich(jd);

    double const tanH     = sin(A) / (cos(A)*sin(phi) + tan(h)*cos(phi));
    double const alpha    = radToDeg*(theta0 - L - atan(tanH));
    double const sinDelta = sin(phi)*sin(h) - cos(phi)*cos(h)*cos(A);
    double const delta    = radToDeg*asin(sinDelta);

    coord::Coord c(alpha, delta, getEpoch());
    coord::Coord cp = c.precess(2000.0);
    return Fk5Coord(cp.getLatitudeDeg(), cp.getLongitudeDeg());
}

/**
 * @brief Convert ourself from galactic to Icrs
 *
 * @note This currently calls the Fk5 routines and is not strictly ICRS.
 */
coord::IcrsCoord coord::AltAzCoord::toIcrs() {
    return (this->toFk5()).toIcrs();
}

/**
 * @brief Convert ourself from AltAz to Fk5
 */
coord::EquatorialCoord coord::AltAzCoord::toEquatorial() {
    return (this->toFk5()).precess(getEpoch());
}


/**
 * @brief Convert ourself from AltAz to AltAz ... a no-op
 */
coord::AltAzCoord coord::AltAzCoord::toAltAz(
                                             coord::Observatory const &obs, ///< observatory of observation
                                             coord::Date const &date        ///< date of observation
                                            ) {
    return coord::AltAzCoord(getLongitudeDeg(), getLatitudeDeg(), getEpoch(), _obs);
}

/**
 * @brief Convert ourself from AltAz to AltAz with no observatory or date arguments
 *
 * As this is essentially a copy-constructor, the extra info can be obtained internally.
 */
coord::AltAzCoord coord::AltAzCoord::toAltAz() {
    return coord::AltAzCoord(getLongitudeDeg(), getLatitudeDeg(), getEpoch(), _obs);
}

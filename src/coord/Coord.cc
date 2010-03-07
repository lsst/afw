// -*- lsst-c++ -*-
/**
 * @file Coord.cc
 * @brief Provide functions to handle coordinates
 * @ingroup afw
 * @author Steve Bickerton
 *
 * All algorithms adapted from Astronomical Algorithms, 2nd ed. (J. Meeus)
 *
 */
#include <sstream>
#include <iostream>
#include <cmath>

#include "lsst/pex/exceptions.h"
#include "boost/algorithm/string.hpp"
#include "boost/tuple/tuple.hpp"
#include "boost/format.hpp"

#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/coord/Observatory.h"
#include "lsst/afw/coord/Date.h"

namespace coord = lsst::afw::coord;
namespace ex    = lsst::pex::exceptions;

double coord::eclipticPoleInclination(double const epoch) {
    double const T = (epoch - 2000.0)/100.0;
    return 23.0 + 26.0/60.0 + (21.448 - 46.82*T - 0.0006*T*T - 0.0018*T*T*T)/3600.0;
}


namespace {
    
double reduceAngle(double theta) {

    theta = theta - (static_cast<int>(theta)/360)*360.0;
    if (theta < 0) {
        theta += 360.0;
    }
    return theta;
}

coord::Coord GalacticPoleInFk5 = coord::Coord(192.85950, 27.12825, 2000.0); // C&O
coord::Coord Fk5PoleInGalactic = coord::Coord(122.93200, 27.12825, 2000.0); // C&O

double getTheta0(double jd, double T) {
    return 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T*T - (T*T*T/38710000.0);
}

double const epochTolerance = 1.0e-8;    
    
}



/**
 * @brief A function to convert sexigesimal to decimal degrees
 *
 * @note it is left to the user to multiply by 15 for hours LONGITUDE.
 */

double coord::toDecimal(std::string const val) {
    
    std::vector<std::string> elements;
    boost::split(elements, val, boost::is_any_of(":"));
    int const deg   = abs(atoi(elements[0].c_str()));
    int const min   = atoi(elements[1].c_str());
    double const sec = atof(elements[2].c_str());

    double degrees = deg + min/60.0 + sec/3600.0;
    if ( (elements[0].c_str())[0] == '-' ) {
        degrees *= -1.0;
    }
    return degrees;
}

coord::Dms coord::toDms(double const val) {
    double absVal = std::fabs(val);
    Dms dms;
    dms.sign = (val >= 0) ? 1 : -1;
    dms.deg  = static_cast<int>(std::floor(absVal));
    dms.min  = static_cast<int>(std::floor((absVal - dms.deg)*60.0));
    dms.sec  = ((absVal - dms.deg)*60.0 - dms.min)*60.0;
    return dms;
}

std::string coord::toDmsStr(double const val) {
    coord::Dms dms = coord::toDms(val);
    
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



coord::Coord::Coord(double const ra, double const dec, double const epoch) :
    _longitudeRad(degToRad*ra), _latitudeRad(degToRad*dec), _epoch(epoch) {
    _verifyValues();
}

coord::Coord::Coord(std::string const ra, std::string const dec, double const epoch) :
    _longitudeRad(degToRad*15.0*coord::toDecimal(ra)),
    _latitudeRad(degToRad*coord::toDecimal(dec)),
    _epoch(epoch) {
    _verifyValues();
}

coord::Coord::Coord() : _longitudeRad(0.0), _latitudeRad(0.0), _epoch(2000.0) {
    _verifyValues();
}

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

void coord::Coord::reset(double const longitudeDeg, double const latitudeDeg, double const epoch) {
    _longitudeRad = degToRad*longitudeDeg;
    _latitudeRad  = degToRad*latitudeDeg;
    _epoch = epoch;
    _verifyValues();
}


double coord::Coord::getLongitudeDeg()  { return radToDeg*_longitudeRad; }
double coord::Coord::getLongitudeHrs()  { return radToDeg*_longitudeRad/15.0; }
double coord::Coord::getLongitudeRad()  { return _longitudeRad; }
double coord::Coord::getLatitudeDeg()   { return radToDeg*_latitudeRad; }
double coord::Coord::getLatitudeRad()   { return _latitudeRad; }
std::string coord::Coord::getLongitudeStr() { 
    return toDmsStr(radToDeg*_longitudeRad/15.0);
}
std::string coord::Coord::getLatitudeStr() {
    return toDmsStr(radToDeg*_latitudeRad);
}

double coord::Coord::getRaDeg()          { return this->toFk5().getLongitudeDeg(); }
double coord::Coord::getDecDeg()         { return this->toFk5().getLatitudeDeg(); }
double coord::Coord::getRaHrs()          { return this->toFk5().getLongitudeHrs(); }
double coord::Coord::getRaRad()          { return this->toFk5().getLongitudeRad(); }
double coord::Coord::getDecRad()         { return this->toFk5().getLatitudeRad(); }
std::string coord::Coord::getRaStr()     { return this->toFk5().getLongitudeStr(); }
std::string coord::Coord::getDecStr()    { return this->toFk5().getLatitudeStr(); }

double coord::Coord::getLDeg()           { return this->toGalactic().getLongitudeDeg(); }
double coord::Coord::getBDeg()           { return this->toGalactic().getLatitudeDeg(); }
double coord::Coord::getLHrs()           { return this->toGalactic().getLongitudeHrs(); }    
double coord::Coord::getLRad()           { return this->toGalactic().getLongitudeRad(); }
double coord::Coord::getBRad()           { return this->toGalactic().getLatitudeRad(); }
std::string coord::Coord::getLStr()      { return this->toGalactic().getLongitudeStr(); }
std::string coord::Coord::getBStr()      { return this->toGalactic().getLatitudeStr(); }


double coord::Coord::getLambdaDeg()      { return this->toEcliptic().getLongitudeDeg(); }
double coord::Coord::getBetaDeg()        { return this->toEcliptic().getLatitudeDeg(); }
double coord::Coord::getLambdaHrs()      { return this->toEcliptic().getLongitudeHrs(); }
double coord::Coord::getLambdaRad()      { return this->toEcliptic().getLongitudeRad(); }
double coord::Coord::getBetaRad()        { return this->toEcliptic().getLatitudeRad(); }
std::string coord::Coord::getLambdaStr() { return this->toEcliptic().getLongitudeStr(); }
std::string coord::Coord::getBetaStr()   { return this->toEcliptic().getLatitudeStr(); }


double coord::EclipticCoord::getLongitudeDeg()  { return radToDeg*_longitudeRad; }
double coord::EclipticCoord::getLongitudeHrs()  { return radToDeg*_longitudeRad/15.0; }
double coord::EclipticCoord::getLongitudeRad()  { return _longitudeRad; }
double coord::EclipticCoord::getLatitudeDeg()   { return radToDeg*_latitudeRad; }
double coord::EclipticCoord::getLatitudeRad()   { return _latitudeRad; }
std::string coord::EclipticCoord::getLongitudeStr() { 
    return toDmsStr(radToDeg*_longitudeRad/15.0);
}
std::string coord::EclipticCoord::getLatitudeStr() {
    return toDmsStr(radToDeg*_latitudeRad);
}


/*
 *
 *
 * Variable names assume an equaltorial/galactic tranform, but it works
 *  for any spherical polar system when the appropriate poles are supplied.
 */
coord::Coord coord::Coord::transform(Coord poleTo, Coord poleFrom) {
    double const alphaGP  = poleFrom.getLongitudeRad();
    double const deltaGP  = poleFrom.getLatitudeRad();
    double const lCP      = poleTo.getLongitudeRad();
    //double const bCP      = poleTo.getLatitudeRad();
    
    double alpha = getLongitudeRad();
    double delta = getLatitudeRad();
    
    double l =  (lCP - atan2( sin(alpha - alphaGP), 
                              tan(delta)*cos(deltaGP) - cos(alpha - alphaGP)*sin(deltaGP)));
    
    double const b = radToDeg*asin( (sin(deltaGP)*sin(delta) + cos(deltaGP)*cos(delta)*cos(alpha - alphaGP)));

    l *= radToDeg;

    // fix cyclicality issues
    if (l < 0) {
        l += 360.0;
    }
    if (l >= 360.0) {
        l -= 360.0;
    }

    return coord::Coord(l, b);
}



double coord::Coord::angularSeparation(coord::Coord &c) {
    double alpha1 = getLongitudeRad();
    double delta1 = getLatitudeRad();
    double alpha2 = c.getLongitudeRad();
    double delta2 = c.getLatitudeRad();
    
#if 0
    // this formula breaks down near 0 and 180
    double cosd    = sin(delta1)*sin(delta2) + cos(delta1)*cos(delta2)*cos(alpha1 - alpha2);
    double distDeg = radToDeg*acos(cosd);
#endif

    // use haversine form
    double dDelta = delta1 - delta2;
    double dAlpha = alpha1 - alpha2;
    double havDDelta = sin(dDelta/2.0)*sin(dDelta/2.0);
    double havDAlpha = sin(dAlpha/2.0)*sin(dAlpha/2.0);
    double havD = havDDelta + cos(delta1)*cos(delta2)*havDAlpha;
    double sinDHalf = std::sqrt(havD);
    double distDeg = radToDeg*2.0*asin(sinDHalf);
    
    return distDeg;
}



#if 0
double coord::GalacticCoord::angularSeparation(GalacticCoord &coo) {
    coord::Coord cb = coo.getCoord();
    return _coord.angularSeparation(cb);
}
double coord::EclipticCoord::angularSeparation(EclipticCoord &coo) {
    coord::Coord cb = coo.getCoord();
    return _coord.angularSeparation(cb);
}
double coord::AltAzCoord::angularSeparation(AltAzCoord &coo) {
    coord::Coord cb = coo.getCoord();
    return _coord.angularSeparation(cb);
}
#endif

coord::Coord coord::Coord::_precess(double epochFrom, double epochTo) {

    coord::Date dateFrom(epochFrom, coord::Date::EPOCH);
    coord::Date dateTo(epochTo, coord::Date::EPOCH);
    double jd0 = dateFrom.getJd();
    double jd = dateTo.getJd();

    double T = (jd0 - coord::JD2000) / 36525.0;
    double t =     (jd - jd0) / 36525.0;

    double asPerRad = 2.062648e5;
    double xi = (2306.2181 + 1.39656*T - 0.000139*T*T)*t + (0.30188 - 0.000344*T)*t*t + 0.017998*t*t*t;
    double z = (2306.2181 + 1.39656*T - 0.000139*T*T)*t + (1.09468 + 0.000066*T)*t*t + 0.018203*t*t*t;
    double theta = (2004.3109 - 0.85330*T - 0.000217*T*T)*t - (0.42665 + 0.000217*T)*t*t - 0.041833*t*t*t;
    xi /= asPerRad;
    z /= asPerRad;
    theta /= asPerRad;

    double alpha0 = getLongitudeRad();
    double delta0 = getLatitudeRad();
    
    double A = cos(delta0) * sin(alpha0 + xi);
    double B = cos(theta)*cos(delta0)*cos(alpha0 + xi) - sin(theta)*sin(delta0);
    double C = sin(theta)*cos(delta0)*cos(alpha0 + xi) + cos(theta)*sin(delta0);

    double alpha = reduceAngle(radToDeg * ( atan2(A,B) + z));
    double delta = radToDeg * asin(C);
    
    return coord::Coord(alpha, delta, epochTo);
}



/**
 * Fk5Coord
 */
coord::Fk5Coord coord::Coord::toFk5() { 
    return Fk5Coord(getLongitudeDeg(), getLatitudeDeg(), getEpoch()); 
}
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
coord::EclipticCoord coord::Coord::toEcliptic() {
    double eclPoleIncl = eclipticPoleInclination(getEpoch());
    Coord eclPoleInEquatorial(270.0, 90.0 - eclPoleIncl, getEpoch());
    Coord equPoleInEcliptic(90.0, 90.0 - eclPoleIncl, getEpoch());
    Coord c = transform(equPoleInEcliptic, eclPoleInEquatorial); 
    return EclipticCoord(c.getLongitudeDeg(), c.getLatitudeDeg(), getEpoch());
}

#if 0
coord::AltAzCoord coord::Fk5Coord::toAltAz(coord::Observatory obs, coord::Date obsDate) {

    // precess to the epoch
    coord::Fk5Coord coord = precess(obsDate.getEpoch());

    double jd = obsDate.getJd();
    double T = (jd - 2451545.0)/36525.0;

    // greenwich sidereal time
    double theta0deg = getTheta0(jd, T);
    int n360 = static_cast<int>(theta0deg/360);
    theta0deg -= n360*360.0;
    double theta0 = degToRad*theta0deg;

    
    double phi = obs.getLatitudeRad(); // observatory latitude
    double L   = obs.getLongitudeRad(); // observatory longitude

    double alpha = coord.getRaRad();
    double delta = coord.getDecRad();

    double H = theta0 - L - alpha;
    double sinh = sin(phi)*sin(delta) + cos(phi)*cos(delta)*cos(H);
    double tanAnum = sin(H);
    double tanAdenom = (cos(H)*sin(phi) - tan(delta)*cos(phi) );
    double h = radToDeg*asin(sinh);
    double A = -90.0 - radToDeg*atan2(tanAdenom, tanAnum);
    if (A < 0 ) {
        A += 360.0;
    }
    
    return AltAzCoord(A, h, obsDate.getEpoch(), obs);
}
#endif


/**
 * GalacticCoord
 */
coord::Fk5Coord coord::GalacticCoord::toFk5() {
    // transform to equatorial
    // galactic coords are ~constant, and the poles used are for epoch=2000, so we get J2000
    Coord c = transform(GalacticPoleInFk5, Fk5PoleInGalactic);
    // put the new values in an Fk5Coord and force epoch=2000
    Fk5Coord equ(c.getLongitudeDeg(), c.getLatitudeDeg(), 2000.0);
    // now precess the J2000 to the epoch for the original galactic
    if ( fabs(getEpoch() - 2000.0) > epochTolerance ) {
        equ.precess(getEpoch());
    }
    return equ;
}
coord::GalacticCoord coord::GalacticCoord::toGalactic() { 
    return GalacticCoord(getLongitudeDeg(), getLatitudeDeg(), getEpoch());
}
coord::EclipticCoord coord::GalacticCoord::toEcliptic() {
    return (this->toFk5()).toEcliptic();
}
#if 0
coord::AltAzCoord coord::GalacticCoord::toAltAz(coord::Observatory const &obs, coord::Date const &date) {
    return (this->toFk5()).toAltAz(obs, date);
}
#endif
/**
 * EclipticCoord
 */
coord::EclipticCoord coord::EclipticCoord::toEcliptic() {
    return coord::EclipticCoord(getLongitudeDeg(), getLatitudeDeg(), getEpoch());
}
coord::Fk5Coord coord::EclipticCoord::toFk5() {
    double eclPoleIncl = eclipticPoleInclination(getEpoch());
    Coord eclipticPoleInFk5(270.0, 90.0 - eclPoleIncl, getEpoch());
    Coord fk5PoleInEcliptic(90.0, 90.0 - eclPoleIncl, getEpoch());
    Coord c = transform(eclipticPoleInFk5, fk5PoleInEcliptic); 
    return Fk5Coord(c.getLongitudeDeg(), c.getLatitudeDeg(), getEpoch());
}
coord::GalacticCoord coord::EclipticCoord::toGalactic() {
    return (this->toFk5()).toGalactic(); 
}
#if 0
coord::AltAzCoord coord::EclipticCoord::toAltAz(coord::Observatory const &obs, coord::Date const &date) {
    return (this->toFk5()).toAltAz(obs, date);
}



/**
 * AltAzCoord
 */

coord::EclipticCoord coord::AltAzCoord::toEcliptic() {
    return (this->toFk5()).toEcliptic();
}
coord::GalacticCoord coord::AltAzCoord::toGalactic() {
    return (this->toFk5()).toGalactic(); 
}
coord::Fk5Coord coord::AltAzCoord::toFk5() {
    double A = getAzimuthRad();
    double h = getAltitudeRad();
    double phi = _obs.getLatitudeRad();
    double L = _obs.getLongitudeRad();

    double jd = coord::Date(_coord.getEpoch(), coord::Date::EPOCH).getJd();
    double T = (jd - 2451545.0)/36525.0;
    double theta0 = degToRad*getTheta0(jd, T);

    double tanH = sin(A) / (cos(A)*sin(phi) + tan(h)*cos(phi));
    double alpha = radToDeg*(theta0 - L - atan(tanH));
    double sinDelta = sin(phi)*sin(h) - cos(phi)*cos(h)*cos(A);
    double delta = radToDeg*asin(sinDelta);
    
    return Fk5Coord(alpha, delta, _coord.getEpoch());
}
coord::AltAzCoord coord::AltAzCoord::toAltAz(coord::Observatory const &obs, coord::Date const &date) {
    return coord::AltAzCoord(_coord.getLongitudeDeg(), _coord.getLatitudeDeg(), _coord.getEpoch(), _obs);
}
coord::AltAzCoord coord::AltAzCoord::toAltAz() {
    return coord::AltAzCoord(_coord.getLongitudeDeg(), _coord.getLatitudeDeg(), _coord.getEpoch(), _obs);
}
#endif

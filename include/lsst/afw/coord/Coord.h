// -*- lsst-c++ -*-
#if !defined(LSST_AFW_COORD_COORD_H)
#define LSST_AFW_COORD_COORD_H
/**
 * @file Coord.h
 * @brief Functions to handle coordinates
 * @ingroup afw
 * @author Steve Bickerton
 *
 * @todo Verify same epoch for angular separation.
 */ 

#include "lsst/afw/coord/Observatory.h"
#include "lsst/afw/coord/Date.h"

namespace lsst {
namespace afw {    
namespace coord {

double const degToRad = M_PI/180.0;
double const radToDeg = 180.0/M_PI;
    
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
    int deg;
    int min;
    double sec;
    int sign;
};

    

/**
 * @brief a Function to convert sexigesimal to decimal degrees
 *
 */
double toDecimal(std::string const dms);
Dms toDms(double const val);
std::string toDmsStr(double const val);

class Coord {
public:
    
    Coord(double const longitude, double const latitude, double const epoch = 2000.0) {
        reset(longitude, latitude, epoch);
    }
    
    Coord(std::string const longitude, std::string const latitude, double const epoch = 2000.0) {
        reset( 15.0*coord::toDecimal(longitude), coord::toDecimal(latitude), epoch);
    }

    Coord() {
        reset(0.0, 0.0, 2000.0);
    }

    void reset(double const longitude, double const latitude, double const epoch = 2000.0);
    
    double getLongitudeDeg()  { return radToDeg*_longitudeRad; }
    double getLongitudeHrs()  { return radToDeg*_longitudeRad/15.0; }
    double getLongitudeRad()  { return _longitudeRad; }
    double getLatitudeDeg()   { return radToDeg*_latitudeRad; }
    double getLatitudeRad()   { return _latitudeRad; }
    double getEpoch() { return _epoch; }
    
    Coord transform(Coord const poleFrom, Coord const poleTo);
    double angularSeparation(Coord &coord);
    
    std::string getLongitudeStr();
    std::string getLatitudeStr();

private:
    double _longitudeRad;
    double _latitudeRad;
    double _epoch;
};


class GalacticCoord;
class Fk5Coord;
class EclipticCoord;
class AltAzCoord;
 
class Fk5Coord {
public:
    
    Fk5Coord(double const ra, double const dec, double const epoch = 2000.0) : 
        _coord(Coord(ra, dec, epoch)) {}
    Fk5Coord(std::string const ra, std::string const dec, double const epoch = 2000.0) : 
        _coord(Coord(ra, dec, epoch)) {}
    Fk5Coord() : _coord(Coord()) {}

    void reset(double const longitude, double const latitude, double const epoch = 2000.0) {
        _coord.reset(longitude, latitude, epoch);
    }
    
    double getRaDeg() { return _coord.getLongitudeDeg(); }
    double getDecDeg() { return _coord.getLatitudeDeg(); }
    double getRaHrs() { return _coord.getLongitudeHrs(); }
    double getRaRad() { return _coord.getLongitudeRad(); }
    double getDecRad() { return _coord.getLatitudeRad(); }
    std::string getRaStr() { return _coord.getLongitudeStr(); }
    std::string getDecStr() { return _coord.getLatitudeStr(); }
    double getEpoch() { return _coord.getEpoch(); }
    Coord getCoord() { return _coord; }
    
    GalacticCoord toGalactic();
    Fk5Coord toFk5();
    EclipticCoord toEcliptic();
    AltAzCoord toAltAz(coord::Observatory obs, coord::Date obsDate);
    
    Fk5Coord precess(double epochTo) { return _precess(_coord.getEpoch(), epochTo); }
    double angularSeparation(Fk5Coord &coo);

private:
    Coord _coord;
    Fk5Coord _precess(double epochFrom, double epochTo);
};
 

    
class GalacticCoord {
public:
    
    GalacticCoord(double const l, double const b, double const epoch = 2000.0) : 
        _coord(Coord(l, b, epoch)) {}
    GalacticCoord(std::string const l, std::string const b, double const epoch = 2000.0) : 
        _coord(Coord(l, b, epoch)) {}
    GalacticCoord() : _coord(Coord()) {}

    void reset(double const longitude, double const latitude, double const epoch = 2000.0) {
        _coord.reset(longitude, latitude, epoch);
    }

    double getLDeg() { return _coord.getLongitudeDeg(); }
    double getBDeg() { return _coord.getLatitudeDeg(); }
    double getLHrs() { return _coord.getLongitudeHrs(); }    
    double getLRad() { return _coord.getLongitudeRad(); }
    double getBRad() { return _coord.getLatitudeRad(); }
    std::string getLStr() { return _coord.getLongitudeStr(); }
    std::string getBStr() { return _coord.getLatitudeStr(); }
    double getEpoch() { return _coord.getEpoch(); }
    Coord getCoord() { return _coord; }
    
    Fk5Coord toFk5();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic();
    AltAzCoord toAltAz(coord::Observatory const &obs, coord::Date const &date);

    // Nothing to do here, just create a new GalacticCoord with the epoch
    GalacticCoord precess(double epochTo) {
        return GalacticCoord(_coord.getLongitudeDeg(), _coord.getLatitudeDeg(), epochTo);
    }
    double angularSeparation(GalacticCoord &coo);
    
 private:
    Coord _coord;
};


class EclipticCoord {
public:
    
    EclipticCoord(double const lambda, double const beta, double const epoch = 2000.0) : 
        _coord(Coord(lambda, beta, epoch)) {}
    EclipticCoord(std::string const lambda, std::string const beta, double const epoch = 2000.0) : 
        _coord(Coord(lambda, beta, epoch)) {}
    EclipticCoord() : _coord(Coord()) {}

    void reset(double const longitude, double const latitude, double const epoch = 2000.0) {
        _coord.reset(longitude, latitude, epoch);
    }
    
    double getLambdaDeg() { return _coord.getLongitudeDeg(); }
    double getBetaDeg()   { return _coord.getLatitudeDeg(); }
    double getLambdaHrs() { return _coord.getLongitudeHrs(); }
    double getLambdaRad() { return _coord.getLongitudeRad(); }
    double getBetaRad()   { return _coord.getLatitudeRad(); }
    std::string getLambdaStr() { return _coord.getLongitudeStr(); }
    std::string getBetaStr() { return _coord.getLatitudeStr(); }
    double getEpoch() { return _coord.getEpoch(); }
    Coord getCoord() { return _coord; }
    
    Fk5Coord toFk5();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic();
    AltAzCoord toAltAz(coord::Observatory const &obs, coord::Date const &date);
    
    EclipticCoord precess(double epochTo) {
        return (this->toFk5()).precess(epochTo).toEcliptic();
    }
    double angularSeparation(EclipticCoord &coo);
    
 private:
    Coord _coord;
};



class AltAzCoord {
public:
    
    AltAzCoord(double const az, double const alt, double const epoch,
               coord::Observatory const &obs) : 
        _coord(Coord(az, alt, epoch)), _obs(obs) {}
    AltAzCoord(std::string const az, std::string const alt, double const epoch,
               coord::Observatory const &obs) : 
        _coord(Coord(az, alt, epoch)), _obs(obs) {}
    
    double getAzimuthDeg() { return _coord.getLongitudeDeg(); }
    double getAltitudeDeg() { return _coord.getLatitudeDeg(); }
    double getAzimuthHrs() { return _coord.getLongitudeHrs(); }
    double getAzimuthRad() { return _coord.getLongitudeRad(); }
    double getAltitudeRad() { return _coord.getLatitudeRad(); }
    std::string getAzimuthStr() { return _coord.getLongitudeStr(); }
    std::string getAltitudeStr() { return _coord.getLatitudeStr(); }
    double getEpoch() { return _coord.getEpoch(); }
    Coord getCoord() { return _coord; }

    Fk5Coord toFk5();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic();
    AltAzCoord toAltAz(coord::Observatory const &obs, coord::Date const &date);
    AltAzCoord toAltAz();
    double angularSeparation(AltAzCoord &coo);

    
private:
    Coord _coord;
    coord::Observatory _obs;
};
    

double eclipticPoleInclination(double const epoch);

 
 
}}}

#endif

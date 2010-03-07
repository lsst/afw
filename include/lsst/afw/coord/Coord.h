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

#if 0    
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
#endif

class GalacticCoord;
class Fk5Coord;
class EclipticCoord;
class AltAzCoord;
 
class Coord {
public:
    
    Coord(double const ra, double const dec, double const epoch = 2000.0);
    Coord(std::string const ra, std::string const dec, double const epoch = 2000.0);
    Coord();

    void reset(double const longitude, double const latitude, double const epoch = 2000.0);

    double getEpoch()         { return _epoch; }
    
    double getLongitudeDeg();
    double getLongitudeHrs();
    double getLongitudeRad();
    double getLatitudeDeg();
    double getLatitudeRad();
    std::string getLongitudeStr();
    std::string getLatitudeStr();
    
    double getRaDeg();
    double getDecDeg();
    double getRaHrs();
    double getRaRad();
    double getDecRad();
    std::string getRaStr();
    std::string getDecStr();

    double getLDeg();
    double getBDeg();
    double getLHrs();
    double getLRad();
    double getBRad();
    std::string getLStr();
    std::string getBStr();

    double getLambdaDeg();
    double getBetaDeg();
    double getLambdaHrs();
    double getLambdaRad();
    double getBetaRad();
    std::string getLambdaStr();
    std::string getBetaStr();
    
    
    //Coord getCoord() { return _coord; }
    
    Coord transform(Coord const poleFrom, Coord const poleTo);
    double angularSeparation(Coord &c);
    Coord precess(double epochTo) { return _precess(getEpoch(), epochTo); }

    Fk5Coord toFk5();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic();
#if 0    
    AltAzCoord toAltAz(coord::Observatory obs, coord::Date obsDate);
#endif
    

private:
    double _longitudeRad;
    double _latitudeRad;
    double _epoch;

    Coord _precess(double epochFrom, double epochTo);
    void _verifyValues();
};
 


class Fk5Coord : public Coord {
public:    
    Fk5Coord(double const ra, double const dec, double const epoch = 2000.0) : 
        Coord(ra, dec, epoch) {}
    Fk5Coord(std::string const ra, std::string const dec, double const epoch = 2000.0) : 
        Coord(ra, dec, epoch) {}
    Fk5Coord() : Coord() {}
private:
};
    

class GalacticCoord : public Coord {
public:
    
    GalacticCoord(double const l, double const b, double const epoch = 2000.0) : 
        Coord(l, b, epoch) {}
    GalacticCoord(std::string const l, std::string const b, double const epoch = 2000.0) : 
        Coord(l, b, epoch) {}
    GalacticCoord() : Coord() {}

    Fk5Coord toFk5();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic();
#if 0
    AltAzCoord toAltAz(coord::Observatory const &obs, coord::Date const &date);
#endif
    
    // Nothing to do here, just create a new GalacticCoord with the epoch
    GalacticCoord precess(double epochTo) {
        return GalacticCoord(getLongitudeDeg(), getLatitudeDeg(), epochTo);
    }
    //double angularSeparation(GalacticCoord &c);
    
private:
};



class EclipticCoord : public Coord {
public:
    
    EclipticCoord(double const lambda, double const beta, double const epoch = 2000.0) : 
        Coord(lambda, beta, epoch) {}
    EclipticCoord(std::string const lambda, std::string const beta, double const epoch = 2000.0) : 
        Coord(lambda, beta, epoch) {}
    EclipticCoord() : Coord() {}
    
    Fk5Coord toFk5();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic();
#if 0
    AltAzCoord toAltAz(coord::Observatory const &obs, coord::Date const &date);
    
    EclipticCoord precess(double epochTo) {
        return (this->toFk5()).precess(epochTo).toEcliptic();
    }
#endif
    //double angularSeparation(EclipticCoord &c);
    
private:

};


#if 0    
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
#endif    

double eclipticPoleInclination(double const epoch);

 
 
}}}

#endif

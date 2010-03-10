// -*- lsst-c++ -*-
#if !defined(LSST_AFW_COORD_COORD_H)
#define LSST_AFW_COORD_COORD_H
/**
 * @file Coord.h
 * @brief Functions to handle coordinates
 * @ingroup afw
 * @author Steve Bickerton
 *
 * @todo Finish python docs
 * @todo Start tex doc
 * @todo add *many* const
 * @todo for epoch=2000 for fk5 and icrs
 */ 

#include "lsst/afw/coord/Observatory.h"
#include "lsst/afw/coord/Date.h"

namespace lsst {
namespace afw {    
namespace coord {

double const degToRad = M_PI/180.0;
double const radToDeg = 180.0/M_PI;
    
class IcrsCoord;
class Fk5Coord;
class EquatorialCoord;    
class GalacticCoord;
class EclipticCoord;
class AltAzCoord;

/**
 * @class Coord
 *
 * This is the base class for spherical coordinates.
 * Derived classes include:
 *     Fk5Coord, IcrsCoord, GalacticCoord, EclipticCoord, AltAzCoord
 *
 */
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
    
    Coord transform(Coord const poleFrom, Coord const poleTo);
    double angularSeparation(Coord &c);
    Coord precess(double epochTo);

    virtual EquatorialCoord toEquatorial();
    virtual Fk5Coord toFk5();
    virtual IcrsCoord toIcrs();
    virtual GalacticCoord toGalactic();
    virtual EclipticCoord toEcliptic();
    virtual AltAzCoord toAltAz(coord::Observatory obs, coord::Date obsDate);

private:
    double _longitudeRad;
    double _latitudeRad;
    double _epoch;

    void _verifyValues();
};
 

/**
 * @class EquatorialCoord
 * @brief A class to handle Fk5 coordinates (inherits from Coord)
 */
class EquatorialCoord : public Coord {
public:    
    EquatorialCoord(double const ra, double const dec, double const epoch=2000.0) : 
        Coord(ra, dec, epoch) {}
    EquatorialCoord(std::string const ra, std::string const dec, double const epoch=2000.0) :
        Coord(ra, dec, epoch) {}
    EquatorialCoord() : Coord() {}

    EquatorialCoord precess(double epochTo);
    
private:
};


    

/**
 * @class Fk5Coord
 * @brief A class to handle Fk5 coordinates (inherits from Coord)
 */
class Fk5Coord : public Coord {
public:    
    Fk5Coord(double const ra, double const dec) : 
        Coord(ra, dec, 2000.0) {}
    Fk5Coord(std::string const ra, std::string const dec) :
        Coord(ra, dec, 2000.0) {}
    Fk5Coord() : Coord() {}

    EquatorialCoord precess(double epochTo);
    
private:
};


/**
 * @class IcrsCoord
 * @brief A class to handle Icrs coordinates (inherits from Coord)
 */
class IcrsCoord : public Coord {
public:    
    IcrsCoord(double const ra, double const dec) : 
        Coord(ra, dec, 2000.0) {}
    IcrsCoord(std::string const ra, std::string const dec) : 
        Coord(ra, dec, 2000.0) {}
    IcrsCoord() : Coord() {}

    EquatorialCoord precess(double epochTo);

private:
};
    

/**
 * @class GalacticCoord
 * @brief A class to handle Galactic coordinates (inherits from Coord)
 */
class GalacticCoord : public Coord {
public:
    
    GalacticCoord(double const l, double const b, double const epoch = 2000.0) : 
        Coord(l, b, epoch) {}
    GalacticCoord(std::string const l, std::string const b, double const epoch = 2000.0) : 
        Coord(l, b, epoch) {}
    GalacticCoord() : Coord() {}

    EquatorialCoord toEquatorial();
    Fk5Coord toFk5();
    IcrsCoord toIcrs();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic();
    AltAzCoord toAltAz(coord::Observatory const &obs, coord::Date const &date);

    GalacticCoord precess(double epochTo);
    
private:
};



/**
 * @class EclipticCoord
 * @brief A class to handle Ecliptic coordinates (inherits from Coord)
 */
class EclipticCoord : public Coord {
public:
    
    EclipticCoord(double const lambda, double const beta, double const epoch = 2000.0) : 
        Coord(lambda, beta, epoch) {}
    EclipticCoord(std::string const lambda, std::string const beta, double const epoch = 2000.0) : 
        Coord(lambda, beta, epoch) {}
    EclipticCoord() : Coord() {}

    EquatorialCoord toEquatorial();
    Fk5Coord toFk5();
    IcrsCoord toIcrs();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic();
    AltAzCoord toAltAz(coord::Observatory const &obs, coord::Date const &date);

    EclipticCoord precess(double epochTo);
    
private:

};


class AltAzCoord : public Coord {
public:
    
    AltAzCoord(double const az, double const alt, double const epoch,
               coord::Observatory const &obs) : 
        Coord(az, alt, epoch), _obs(obs) {}
    AltAzCoord(std::string const az, std::string const alt, double const epoch,
               coord::Observatory const &obs) : 
        Coord(az, alt, epoch), _obs(obs) {}

    double getAzimuthDeg();
    double getAltitudeDeg();
    double getAzimuthHrs();
    double getAzimuthRad();
    double getAltitudeRad();
    std::string getAzimuthStr();
    std::string getAltitudeStr();


    EquatorialCoord toEquatorial();
    Fk5Coord toFk5();
    IcrsCoord toIcrs();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic();
    AltAzCoord toAltAz(coord::Observatory const &obs, coord::Date const &date);
    AltAzCoord toAltAz();
    
private:
    coord::Observatory _obs;
};

double eclipticPoleInclination(double const epoch);
    
double dmsStringToDegrees(std::string const dms);
//double hmsStringToDecimalDegrees(std::string const hms);
std::string degreesToDmsString(double const deg);
//std::string degreesToHmsString(double const deg);    
    
}}}

#endif

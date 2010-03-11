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
 * @todo in factory ... if ICRS epoch != 2000 ... precess or throw?
 */ 

#include "boost/shared_ptr.hpp"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/coord/Observatory.h"
#include "lsst/afw/coord/Date.h"

namespace geom = lsst::afw::geom;

namespace lsst {
namespace afw {    
namespace coord {

double const degToRad = M_PI/180.0;
double const radToDeg = 180.0/M_PI;

enum CoordUnit { DEGREES, RADIANS, HOURS };
enum CoordSystem { FK5, ICRS, GALACTIC, ECLIPTIC, ALTAZ };  // currently unused.
    
class IcrsCoord;
class Fk5Coord;
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

    Coord(geom::Point2D const p2d, CoordUnit unit = DEGREES, double const epoch = 2000.0);
    Coord(double const ra, double const dec, double const epoch = 2000.0);
    Coord(std::string const ra, std::string const dec, double const epoch = 2000.0);
    Coord();

    typedef boost::shared_ptr<Coord> Ptr;
    
    void reset(double const longitude, double const latitude, double const epoch = 2000.0);

    double getEpoch()         { return _epoch; }

    geom::Point2D getPoint2D(CoordUnit unit = DEGREES);
    std::pair<std::string, std::string> getCoordNames();
    //std::vector<double, double, double> getPositionVector();
    
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

    virtual Fk5Coord toFk5();
    virtual IcrsCoord toIcrs();
    virtual GalacticCoord toGalactic();
    virtual EclipticCoord toEcliptic();
    virtual AltAzCoord toAltAz(Observatory obs, coord::Date obsDate);

private:
    double _longitudeRad;
    double _latitudeRad;
    double _epoch;

    void _verifyValues();
};


/**
 * @class IcrsCoord
 * @brief A class to handle Icrs coordinates (inherits from Coord)
 */
class IcrsCoord : public Coord {
public:    
    IcrsCoord(geom::Point2D const p2d, CoordUnit unit = DEGREES) :
        Coord(p2d, unit, 2000.0) {}
    IcrsCoord(double const ra, double const dec) : 
        Coord(ra, dec, 2000.0) {}
    IcrsCoord(std::string const ra, std::string const dec) : 
        Coord(ra, dec, 2000.0) {}
    IcrsCoord() : Coord() {}

    // don't need specify converters (toGalactic(), etc), base class methods are fine for Fk5
    
    
    Fk5Coord precess(double epochTo);

private:
};
    

/**
 * @class Fk5Coord
 * @brief A class to handle Fk5 coordinates (inherits from Coord)
 */
class Fk5Coord : public Coord {
public:    
    Fk5Coord(geom::Point2D const p2d, CoordUnit unit = DEGREES, double const epoch = 2000.0) :
        Coord(p2d, unit, epoch) {}
    Fk5Coord(double const ra, double const dec, double const epoch = 2000.0) : 
        Coord(ra, dec, epoch) {}
    Fk5Coord(std::string const ra, std::string const dec, double const epoch = 2000.0) :
        Coord(ra, dec, epoch) {}
    Fk5Coord() : Coord() {}

    // don't need specify converters (toGalactic(), etc), base class methods are fine for Fk5
#if 0    
    Fk5Coord toFk5();
    IcrsCoord toIcrs();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic();
    AltAzCoord toAltAz(Observatory const &obs, coord::Date const &date);
#endif
    
    Fk5Coord precess(double epochTo);
    
private:
};


/**
 * @class GalacticCoord
 * @brief A class to handle Galactic coordinates (inherits from Coord)
 */
class GalacticCoord : public Coord {
public:
    
    GalacticCoord(geom::Point2D const p2d, CoordUnit unit = DEGREES, double const epoch = 2000.0) :
        Coord(p2d, unit, epoch) {}
    GalacticCoord(double const l, double const b, double const epoch = 2000.0) : 
        Coord(l, b, epoch) {}
    GalacticCoord(std::string const l, std::string const b, double const epoch = 2000.0) : 
        Coord(l, b, epoch) {}
    GalacticCoord() : Coord() {}

    std::pair<std::string, std::string> getCoordNames();

    Fk5Coord toFk5();
    IcrsCoord toIcrs();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic();
    AltAzCoord toAltAz(Observatory const &obs, coord::Date const &date);

    GalacticCoord precess(double epochTo);
    
private:
};



/**
 * @class EclipticCoord
 * @brief A class to handle Ecliptic coordinates (inherits from Coord)
 */
class EclipticCoord : public Coord {
public:
    
    EclipticCoord(geom::Point2D const p2d, CoordUnit unit = DEGREES, double const epoch = 2000.0) :
        Coord(p2d, unit, epoch) {}
    EclipticCoord(double const lambda, double const beta, double const epoch = 2000.0) : 
        Coord(lambda, beta, epoch) {}
    EclipticCoord(std::string const lambda, std::string const beta, double const epoch = 2000.0) : 
        Coord(lambda, beta, epoch) {}
    EclipticCoord() : Coord() {}

    std::pair<std::string, std::string> getCoordNames();

    Fk5Coord toFk5();
    IcrsCoord toIcrs();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic();
    AltAzCoord toAltAz(Observatory const &obs, coord::Date const &date);

    EclipticCoord precess(double epochTo);
    
private:

};


class AltAzCoord : public Coord {
public:
    
    AltAzCoord(geom::Point2D const p2d, CoordUnit unit, double const epoch, Observatory const &obs) :
        Coord(p2d, unit, epoch), _obs(obs) {}
    AltAzCoord(double const az, double const alt, double const epoch, Observatory const &obs) : 
        Coord(az, alt, epoch), _obs(obs) {}
    AltAzCoord(std::string const az, std::string const alt, double const epoch, Observatory const &obs) : 
        Coord(az, alt, epoch), _obs(obs) {}
    
    std::pair<std::string, std::string> getCoordNames();
    
    double getAzimuthDeg();
    double getAltitudeDeg();
    double getAzimuthHrs();
    double getAzimuthRad();
    double getAltitudeRad();
    std::string getAzimuthStr();
    std::string getAltitudeStr();


    Fk5Coord toFk5();
    IcrsCoord toIcrs();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic();
    AltAzCoord toAltAz(Observatory const &obs, coord::Date const &date);
    AltAzCoord toAltAz();
    
private:
    Observatory _obs;
};

double eclipticPoleInclination(double const epoch);
    
double dmsStringToDegrees(std::string const dms);
double hmsStringToDegrees(std::string const hms);
std::string degreesToDmsString(double const deg);
std::string degreesToHmsString(double const deg);    

Coord::Ptr makeCoord(CoordSystem const system,
                     double const ra, double const dec, double const epoch=2000.0);
Coord::Ptr makeCoord(CoordSystem const system,
                     std::string const ra, std::string const dec, double const epoch=2000.0);
Coord::Ptr makeCoord(CoordSystem const system,
                     geom::Point2D p2d, CoordUnit unit, double const epoch=2000.0);
    

}}}

#endif

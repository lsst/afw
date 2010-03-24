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
 * @todo add FK4 ... as needed
 */ 
#include <limits>
#include <map>

#include "boost/shared_ptr.hpp"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/coord/Utils.h"     // this contains the enums CoordSystem CoordType and radToDec
#include "lsst/afw/coord/Observatory.h"
#include "lsst/daf/base.h"

//#include "boost/unordered_map.hpp"

namespace afwGeom    = lsst::afw::geom;
namespace dafBase = lsst::daf::base;


namespace lsst {
namespace afw {    
namespace coord {

enum CoordSystem { FK5, ICRS, GALACTIC, ECLIPTIC, TOPOCENTRIC };

namespace {    
typedef std::map<std::string, CoordSystem> CoordSystemMap;
CoordSystemMap const getCoordSystemMap() {
    CoordSystemMap idMap;
    idMap["FK5"]         = FK5;
    idMap["ICRS"]        = ICRS;
    idMap["ECLIPTIC"]    = ECLIPTIC;
    idMap["GALACTIC"]    = GALACTIC;
    idMap["ELON"]        = ECLIPTIC;
    idMap["GLON"]        = GALACTIC;
    idMap["TOPOCENTRIC"] = TOPOCENTRIC;
    return idMap;
}
}
inline CoordSystem const makeCoordEnum(std::string system) {
    static CoordSystemMap idmap = getCoordSystemMap();
    return idmap[system];
}

    
class IcrsCoord;
class Fk5Coord;
class GalacticCoord;
class EclipticCoord;
class TopocentricCoord;

    
/**
 * @class Coord
 *
 * This is the base class for spherical coordinates.
 */
class Coord {
public:

    Coord(afwGeom::Point2D const &p2d, CoordUnit unit = DEGREES, double const epoch = 2000.0);
    Coord(afwGeom::Point3D const &p3d, double const epoch = 2000.0);
    Coord(double const ra, double const dec, double const epoch = 2000.0);
    Coord(std::string const ra, std::string const dec, double const epoch = 2000.0);
    Coord();

    typedef boost::shared_ptr<Coord> Ptr;
    
    void reset(double const longitude, double const latitude, double const epoch = 2000.0);

    double getEpoch()         { return _epoch; }

    afwGeom::Point2D getPosition(CoordUnit unit = DEGREES);
    afwGeom::Point3D getVector();
    std::pair<std::string, std::string> getCoordNames();

    double getLongitude(CoordUnit unit);
    double getLatitude(CoordUnit unit);
    std::string getLongitudeStr(CoordUnit unit);
    std::string getLatitudeStr();

    double operator[](int index);
    
    Coord transform(Coord const poleFrom, Coord const poleTo);
    double angularSeparation(Coord &c, CoordUnit unit);

    Coord::Ptr convert(CoordSystem system);

    virtual Fk5Coord toFk5(double epoch);
    virtual Fk5Coord toFk5();
    virtual IcrsCoord toIcrs();
    virtual GalacticCoord toGalactic();
    virtual EclipticCoord toEcliptic(double epoch);
    virtual EclipticCoord toEcliptic();
    virtual TopocentricCoord toTopocentric(Observatory obs, dafBase::DateTime obsDate);

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
    IcrsCoord(afwGeom::Point2D const &p2d, CoordUnit unit = DEGREES) : Coord(p2d, unit, 2000.0) {}
    IcrsCoord(afwGeom::Point3D const &p3d) : Coord(p3d, 2000.0) {}
    IcrsCoord(double const ra, double const dec) : Coord(ra, dec, 2000.0) {}
    IcrsCoord(std::string const ra, std::string const dec) : Coord(ra, dec, 2000.0) {}
    IcrsCoord() : Coord() {}

    void reset(double const longitude, double const latitude);
    
    typedef boost::shared_ptr<IcrsCoord> Ptr;

    double getRa(CoordUnit unit);
    double getDec(CoordUnit unit);
    std::string getRaStr(CoordUnit unit);
    std::string getDecStr();

    Fk5Coord toFk5(double epoch);
    Fk5Coord toFk5();
    IcrsCoord toIcrs();
    
private:
};


/**
 * @class Fk5Coord
 * @brief A class to handle Fk5 coordinates (inherits from Coord)
 */
class Fk5Coord : public Coord {
public:    
    Fk5Coord(afwGeom::Point2D const &p2d, CoordUnit unit = DEGREES, double const epoch = 2000.0) :
        Coord(p2d, unit, epoch) {}
    Fk5Coord(afwGeom::Point3D const &p3d, double const epoch = 2000.0) :
        Coord(p3d, epoch) {}
    Fk5Coord(double const ra, double const dec, double const epoch = 2000.0) : 
        Coord(ra, dec, epoch) {}
    Fk5Coord(std::string const ra, std::string const dec, double const epoch = 2000.0) :
        Coord(ra, dec, epoch) {}
    Fk5Coord() : Coord() {}
    
    typedef boost::shared_ptr<Fk5Coord> Ptr;
    
    Fk5Coord precess(double epochTo);
    
    double getRa(CoordUnit unit);
    double getDec(CoordUnit unit);
    std::string getRaStr(CoordUnit unit);
    std::string getDecStr();

    Fk5Coord toFk5(double epoch);
    Fk5Coord toFk5();
    IcrsCoord toIcrs();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic(double epoch);
    EclipticCoord toEcliptic();
    TopocentricCoord toTopocentric(Observatory obs, dafBase::DateTime obsDate);

    
private:
};


/**
 * @class GalacticCoord
 * @brief A class to handle Galactic coordinates (inherits from Coord)
 */
class GalacticCoord : public Coord {
public:
    
    GalacticCoord(afwGeom::Point2D const &p2d, CoordUnit unit = DEGREES) : Coord(p2d, unit) {}
    GalacticCoord(afwGeom::Point3D const &p3d) : Coord(p3d) {}
    GalacticCoord(double const l, double const b) : Coord(l, b) {}
    GalacticCoord(std::string const l, std::string const b) : Coord(l, b) {}
    GalacticCoord() : Coord() {}

    void reset(double const longitude, double const latitude);
    
    typedef boost::shared_ptr<GalacticCoord> Ptr;
    
    std::pair<std::string, std::string> getCoordNames();

    double getL(CoordUnit unit);
    double getB(CoordUnit unit);
    std::string getLStr(CoordUnit unit);
    std::string getBStr();
    
    Fk5Coord toFk5(double epoch);
    Fk5Coord toFk5();
    GalacticCoord toGalactic();

private:
};



/**
 * @class EclipticCoord
 * @brief A class to handle Ecliptic coordinates (inherits from Coord)
 */
class EclipticCoord : public Coord {
public:
    
    EclipticCoord(afwGeom::Point2D const &p2d, CoordUnit unit = DEGREES, double const epoch = 2000.0) :
        Coord(p2d, unit, epoch) {}
    EclipticCoord(afwGeom::Point3D const &p3d, double const epoch = 2000.0) : Coord(p3d, epoch) {}
    EclipticCoord(double const lambda, double const beta, double const epoch = 2000.0) : 
        Coord(lambda, beta, epoch) {}
    EclipticCoord(std::string const lambda, std::string const beta, double const epoch = 2000.0) : 
        Coord(lambda, beta, epoch) {}
    EclipticCoord() : Coord() {}
    
    typedef boost::shared_ptr<EclipticCoord> Ptr;

    std::pair<std::string, std::string> getCoordNames();

    double getLambda(CoordUnit unit);
    double getBeta(CoordUnit unit);
    std::string getLambdaStr(CoordUnit unit);
    std::string getBetaStr();
    
    
    Fk5Coord toFk5(double epoch);
    Fk5Coord toFk5();
    EclipticCoord toEcliptic(double epoch);
    EclipticCoord toEcliptic();

    EclipticCoord precess(double epochTo);
    
private:

};


class TopocentricCoord : public Coord {
public:
    
    TopocentricCoord(afwGeom::Point2D const &p2d, CoordUnit unit, double const epoch,
                     Observatory const &obs) : Coord(p2d, unit, epoch), _obs(obs) {}
    TopocentricCoord(afwGeom::Point3D const &p3d, double const epoch,
                     Observatory const &obs) : Coord(p3d, epoch), _obs(obs) {}
    TopocentricCoord(double const az, double const alt, double const epoch,
                     Observatory const &obs) : Coord(az, alt, epoch), _obs(obs) {}
    TopocentricCoord(std::string const az, std::string const alt, double const epoch,
                     Observatory const &obs) : Coord(az, alt, epoch), _obs(obs) {}

    typedef boost::shared_ptr<TopocentricCoord> Ptr;
    
    std::pair<std::string, std::string> getCoordNames();
    
    double getAzimuth(CoordUnit unit);
    double getAltitude(CoordUnit unit);
    std::string getAzimuthStr(CoordUnit unit);
    std::string getAltitudeStr();


    Fk5Coord toFk5(double epoch);
    Fk5Coord toFk5();
    TopocentricCoord toTopocentric(Observatory const &obs, dafBase::DateTime const &date);
    TopocentricCoord toTopocentric();
    
private:
    Observatory _obs;
};

double eclipticPoleInclination(double const epoch);
    
double dmsStringToDegrees(std::string const dms);
double hmsStringToDegrees(std::string const hms);
std::string degreesToDmsString(double const deg);
std::string degreesToHmsString(double const deg);    
    
Coord::Ptr makeCoord(CoordSystem const system, double const ra, double const dec, double const epoch);
Coord::Ptr makeCoord(CoordSystem const system, std::string const ra, std::string const dec,
                     double const epoch);
Coord::Ptr makeCoord(CoordSystem const system, afwGeom::Point2D const &p2d, CoordUnit unit,
                     double const epoch);
Coord::Ptr makeCoord(CoordSystem const system, afwGeom::Point3D const &p3d, double const epoch);
Coord::Ptr makeCoord(CoordSystem const system);


Coord::Ptr makeCoord(CoordSystem const system, double const ra, double const dec);
Coord::Ptr makeCoord(CoordSystem const system, std::string const ra, std::string const dec);
Coord::Ptr makeCoord(CoordSystem const system, afwGeom::Point2D const &p2d, CoordUnit unit);
Coord::Ptr makeCoord(CoordSystem const system, afwGeom::Point3D const &p3d);
    
}}}

#endif

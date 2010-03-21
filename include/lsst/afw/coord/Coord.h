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
 */ 
#include <limits>

#include "boost/shared_ptr.hpp"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/coord/Utils.h"     // this contains the enums CoordSystem CoordType and radToDec
#include "lsst/afw/coord/Observatory.h"
#include "lsst/daf/base.h"

namespace geom    = lsst::afw::geom;
namespace dafBase = lsst::daf::base;


namespace lsst {
namespace afw {    
namespace coord {


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

    Coord(geom::Point2D const &p2d, CoordUnit unit = DEGREES, double const epoch = 2000.0);
    Coord(geom::Point3D const &p3d, double const epoch = 2000.0);
    Coord(double const ra, double const dec, double const epoch = 2000.0);
    Coord(std::string const ra, std::string const dec, double const epoch = 2000.0);
    Coord();

    typedef boost::shared_ptr<Coord> Ptr;
    
    void reset(double const longitude, double const latitude, double const epoch = 2000.0);

    double getEpoch()         { return _epoch; }

    geom::Point2D getPosition(CoordUnit unit = DEGREES);
    geom::Point3D getVector();
    std::pair<std::string, std::string> getCoordNames();

    double getLongitude(CoordUnit unit);
    double getLatitude(CoordUnit unit);
    std::string getLongitudeStr(CoordUnit unit);
    std::string getLatitudeStr();

    double operator[](int index);
    
    Coord transform(Coord const poleFrom, Coord const poleTo);
    double angularSeparation(Coord &c, CoordUnit unit);

    Coord::Ptr convert(CoordSystem system);

    virtual Fk5Coord toFk5(double epoch = std::numeric_limits<double>::quiet_NaN());
    virtual IcrsCoord toIcrs();
    virtual EquatorialCoord toEquatorial();
    virtual GalacticCoord toGalactic();
    virtual EclipticCoord toEcliptic(double epoch = std::numeric_limits<double>::quiet_NaN());
    virtual AltAzCoord toAltAz(Observatory obs, dafBase::DateTime obsDate);

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
    IcrsCoord(geom::Point2D const &p2d, CoordUnit unit = DEGREES) : Coord(p2d, unit, 2000.0) {}
    IcrsCoord(geom::Point3D const &p3d) : Coord(p3d, 2000.0) {}
    IcrsCoord(double const ra, double const dec) : Coord(ra, dec, 2000.0) {}
    IcrsCoord(std::string const ra, std::string const dec) : Coord(ra, dec, 2000.0) {}
    IcrsCoord() : Coord() {}

    typedef boost::shared_ptr<IcrsCoord> Ptr;

    double getRa(CoordUnit unit);
    double getDec(CoordUnit unit);
    std::string getRaStr(CoordUnit unit);
    std::string getDecStr();

    Fk5Coord toFk5(double epoch = std::numeric_limits<double>::quiet_NaN());
    IcrsCoord toIcrs();
    
private:
};


/**
 * @class EquatorialCoord
 * @brief A class to handle Equatorial coordinates (inherits from Coord)
 *
 * @note This is identical to Icrs
 */
class EquatorialCoord : public Coord {
public:    
    EquatorialCoord(geom::Point2D const &p2d, CoordUnit unit = DEGREES) : Coord(p2d, unit, 2000.0) {}
    EquatorialCoord(geom::Point3D const &p3d) : Coord(p3d, 2000.0) {}
    EquatorialCoord(double const ra, double const dec) : Coord(ra, dec, 2000.0) {}
    EquatorialCoord(std::string const ra, std::string const dec) : Coord(ra, dec, 2000.0) {}
    EquatorialCoord() : Coord() {}

    typedef boost::shared_ptr<EquatorialCoord> Ptr;
    
    double getRa(CoordUnit unit);
    double getDec(CoordUnit unit);
    std::string getRaStr(CoordUnit unit);
    std::string getDecStr();

    Fk5Coord toFk5(double epoch = std::numeric_limits<double>::quiet_NaN());
    EquatorialCoord toEquatorial();
    
private:
};
    
    
/**
 * @class Fk5Coord
 * @brief A class to handle Fk5 coordinates (inherits from Coord)
 */
class Fk5Coord : public Coord {
public:    
    Fk5Coord(geom::Point2D const &p2d, CoordUnit unit = DEGREES, double const epoch = 2000.0) :
        Coord(p2d, unit, epoch) {}
    Fk5Coord(geom::Point3D const &p3d, double const epoch = 2000.0) :
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

    Fk5Coord toFk5(double epoch = std::numeric_limits<double>::quiet_NaN());
    IcrsCoord toIcrs();
    EquatorialCoord toEquatorial();
    GalacticCoord toGalactic();
    EclipticCoord toEcliptic(double epoch = std::numeric_limits<double>::quiet_NaN());
    AltAzCoord toAltAz(Observatory obs, dafBase::DateTime obsDate);

    
private:
};


/**
 * @class GalacticCoord
 * @brief A class to handle Galactic coordinates (inherits from Coord)
 */
class GalacticCoord : public Coord {
public:
    
    GalacticCoord(geom::Point2D const &p2d, CoordUnit unit = DEGREES) : Coord(p2d, unit) {}
    GalacticCoord(geom::Point3D const &p3d) : Coord(p3d) {}
    GalacticCoord(double const l, double const b) : Coord(l, b) {}
    GalacticCoord(std::string const l, std::string const b) : Coord(l, b) {}
    GalacticCoord() : Coord() {}

    typedef boost::shared_ptr<GalacticCoord> Ptr;
    
    std::pair<std::string, std::string> getCoordNames();

    double getL(CoordUnit unit);
    double getB(CoordUnit unit);
    std::string getLStr(CoordUnit unit);
    std::string getBStr();
    
    Fk5Coord toFk5(double epoch = std::numeric_limits<double>::quiet_NaN());
    GalacticCoord toGalactic();

private:
};



/**
 * @class EclipticCoord
 * @brief A class to handle Ecliptic coordinates (inherits from Coord)
 */
class EclipticCoord : public Coord {
public:
    
    EclipticCoord(geom::Point2D const &p2d, CoordUnit unit = DEGREES, double const epoch = 2000.0) :
        Coord(p2d, unit, epoch) {}
    EclipticCoord(geom::Point3D const &p3d, double const epoch = 2000.0) : Coord(p3d, epoch) {}
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
    
    
    Fk5Coord toFk5(double epoch = std::numeric_limits<double>::quiet_NaN());
    EclipticCoord toEcliptic(double epoch = std::numeric_limits<double>::quiet_NaN());

    EclipticCoord precess(double epochTo);
    
private:

};


class AltAzCoord : public Coord {
public:
    
    AltAzCoord(geom::Point2D const &p2d, CoordUnit unit, double const epoch, Observatory const &obs) :
        Coord(p2d, unit, epoch), _obs(obs) {}
    AltAzCoord(geom::Point3D const &p3d, double const epoch, Observatory const &obs) :
        Coord(p3d, epoch), _obs(obs) {}
    AltAzCoord(double const az, double const alt, double const epoch, Observatory const &obs) : 
        Coord(az, alt, epoch), _obs(obs) {}
    AltAzCoord(std::string const az, std::string const alt, double const epoch, Observatory const &obs) : 
        Coord(az, alt, epoch), _obs(obs) {}

    typedef boost::shared_ptr<AltAzCoord> Ptr;
    
    std::pair<std::string, std::string> getCoordNames();
    
    double getAzimuth(CoordUnit unit);
    double getAltitude(CoordUnit unit);
    std::string getAzimuthStr(CoordUnit unit);
    std::string getAltitudeStr();


    Fk5Coord toFk5(double epoch = std::numeric_limits<double>::quiet_NaN());
    AltAzCoord toAltAz(Observatory const &obs, dafBase::DateTime const &date);
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
                     double const ra, double const dec,
                     double const epoch = std::numeric_limits<double>::quiet_NaN());
Coord::Ptr makeCoord(CoordSystem const system,
                     std::string const ra, std::string const dec,
                     double const epoch = std::numeric_limits<double>::quiet_NaN());
Coord::Ptr makeCoord(CoordSystem const system,
                     geom::Point2D const &p2d, CoordUnit unit,
                     double const epoch = std::numeric_limits<double>::quiet_NaN());
Coord::Ptr makeCoord(CoordSystem const system,
                     geom::Point3D const &p3d,
                     double const epoch = std::numeric_limits<double>::quiet_NaN());
Coord::Ptr makeCoord(CoordSystem const system);

}}}

#endif

// -*- lsst-c++ -*-
#if !defined(LSST_AFW_COORD_COORD_H)
#define LSST_AFW_COORD_COORD_H
/**
 * @file Coord.h
 * @brief Functions to handle coordinates
 * @ingroup afw
 * @author Steve Bickerton
 *
 * @todo add FK4 ... as needed
 */ 
#include <limits>
#include <map>

#include "boost/shared_ptr.hpp"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/coord/Utils.h"     // this contains the enums CoordSystem CoordType and radToDec
#include "lsst/afw/coord/Observatory.h"
#include "lsst/daf/base.h"

namespace afwGeom    = lsst::afw::geom;
namespace dafBase = lsst::daf::base;


namespace lsst {
namespace afw {    
namespace coord {

    
/*
 * Information about the coordinate system we support
 */
enum CoordSystem { FK5, ICRS, GALACTIC, ECLIPTIC, TOPOCENTRIC };
CoordSystem makeCoordEnum(std::string const system);

    
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

    typedef boost::shared_ptr<Coord> Ptr;
    typedef boost::shared_ptr<Coord const> ConstPtr;

    Coord(afwGeom::Point2D const &p2d, CoordUnit unit = DEGREES, double const epoch = 2000.0);
    Coord(afwGeom::Point3D const &p3d, double const epoch = 2000.0);
    Coord(double const ra, double const dec, double const epoch = 2000.0);
    Coord(std::string const ra, std::string const dec, double const epoch = 2000.0);
    Coord();
    virtual ~Coord() {}
    
    void reset(double const longitude, double const latitude, double const epoch = 2000.0);

    double getEpoch() const { return _epoch; }

    afwGeom::Point2D getPosition(CoordUnit unit = DEGREES) const;
    afwGeom::Point3D getVector() const;
    inline std::pair<std::string, std::string> getCoordNames() const {
        return std::pair<std::string, std::string>("RA", "Dec");
    }

    // These are inline functions and are defined at the end of this header file
    double operator[](int const index) const;
    inline double getLongitude(CoordUnit unit) const;
    inline double getLatitude(CoordUnit unit) const;
    inline std::string getLongitudeStr(CoordUnit unit) const;
    inline std::string getLatitudeStr() const;

    
    Coord transform(Coord const &poleFrom, Coord const &poleTo) const;
    double angularSeparation(Coord const &c, CoordUnit unit) const;

    Coord::Ptr convert(CoordSystem system) const;

    virtual Fk5Coord toFk5(double const epoch) const;
    virtual Fk5Coord toFk5() const;
    virtual IcrsCoord toIcrs() const;
    virtual GalacticCoord toGalactic() const;
    virtual EclipticCoord toEcliptic(double const epoch) const;
    virtual EclipticCoord toEcliptic() const;
    virtual TopocentricCoord toTopocentric(Observatory const &obs, dafBase::DateTime const &obsDate) const;

private:
    double _longitudeRad;
    double _latitudeRad;
    double _epoch;

    void _verifyValues() const;
};


/**
 * @class IcrsCoord
 * @brief A class to handle Icrs coordinates (inherits from Coord)
 */
class IcrsCoord : public Coord {
public:
    
    typedef boost::shared_ptr<IcrsCoord> Ptr;

    IcrsCoord(afwGeom::Point2D const &p2d, CoordUnit unit = DEGREES) : Coord(p2d, unit, 2000.0) {}
    IcrsCoord(afwGeom::Point3D const &p3d) : Coord(p3d, 2000.0) {}
    IcrsCoord(double const ra, double const dec) : Coord(ra, dec, 2000.0) {}
    IcrsCoord(std::string const ra, std::string const dec) : Coord(ra, dec, 2000.0) {}
    IcrsCoord() : Coord() {}

    void reset(double const longitude, double const latitude);
    
    double getRa(CoordUnit unit) const         { return getLongitude(unit); }   
    double getDec(CoordUnit unit) const        { return getLatitude(unit); }    
    std::string getRaStr(CoordUnit unit) const { return getLongitudeStr(unit); }
    std::string getDecStr() const              { return getLatitudeStr(); }     

    virtual Fk5Coord toFk5(double const epoch) const;
    virtual Fk5Coord toFk5() const;
    virtual IcrsCoord toIcrs() const;
    
private:
};


/**
 * @class Fk5Coord
 * @brief A class to handle Fk5 coordinates (inherits from Coord)
 */
class Fk5Coord : public Coord {
public:    

    typedef boost::shared_ptr<Fk5Coord> Ptr;
    
    Fk5Coord(afwGeom::Point2D const &p2d, CoordUnit unit = DEGREES, double const epoch = 2000.0) :
        Coord(p2d, unit, epoch) {}
    Fk5Coord(afwGeom::Point3D const &p3d, double const epoch = 2000.0) :
        Coord(p3d, epoch) {}
    Fk5Coord(double const ra, double const dec, double const epoch = 2000.0) : 
        Coord(ra, dec, epoch) {}
    Fk5Coord(std::string const ra, std::string const dec, double const epoch = 2000.0) :
        Coord(ra, dec, epoch) {}
    Fk5Coord() : Coord() {}
    
    Fk5Coord precess(double const epochTo) const;
    
    double getRa(CoordUnit unit) const         { return getLongitude(unit); }   
    double getDec(CoordUnit unit) const        { return getLatitude(unit); }    
    std::string getRaStr(CoordUnit unit) const { return getLongitudeStr(unit); }
    std::string getDecStr() const              { return getLatitudeStr(); }     

    virtual Fk5Coord toFk5(double const epoch) const;
    virtual Fk5Coord toFk5() const;
    virtual IcrsCoord toIcrs() const;
    virtual GalacticCoord toGalactic() const;
    virtual EclipticCoord toEcliptic(double const epoch) const;
    virtual EclipticCoord toEcliptic() const;
    virtual TopocentricCoord toTopocentric(Observatory const &obs, dafBase::DateTime const &obsDate) const;

    
private:
};


/**
 * @class GalacticCoord
 * @brief A class to handle Galactic coordinates (inherits from Coord)
 */
class GalacticCoord : public Coord {
public:
    
    typedef boost::shared_ptr<GalacticCoord> Ptr;
    
    GalacticCoord(afwGeom::Point2D const &p2d, CoordUnit unit = DEGREES) : Coord(p2d, unit) {}
    GalacticCoord(afwGeom::Point3D const &p3d) : Coord(p3d) {}
    GalacticCoord(double const l, double const b) : Coord(l, b) {}
    GalacticCoord(std::string const l, std::string const b) : Coord(l, b) {}
    GalacticCoord() : Coord() {}

    void reset(double const longitude, double const latitude);
    
    inline std::pair<std::string, std::string> getCoordNames() const {
        return std::pair<std::string, std::string>("L", "B");
    }
    
    double getL(CoordUnit unit) const         { return getLongitude(unit); }   
    double getB(CoordUnit unit) const         { return getLatitude(unit); }    
    std::string getLStr(CoordUnit unit) const { return getLongitudeStr(unit); }
    std::string getBStr() const               { return getLatitudeStr(); }     
    
    virtual Fk5Coord toFk5(double const epoch) const;
    virtual Fk5Coord toFk5() const ;
    virtual GalacticCoord toGalactic() const;

private:
};



/**
 * @class EclipticCoord
 * @brief A class to handle Ecliptic coordinates (inherits from Coord)
 */
class EclipticCoord : public Coord {
public:
    
    typedef boost::shared_ptr<EclipticCoord> Ptr;

    EclipticCoord(afwGeom::Point2D const &p2d, CoordUnit unit = DEGREES, double const epoch = 2000.0) :
        Coord(p2d, unit, epoch) {}
    EclipticCoord(afwGeom::Point3D const &p3d, double const epoch = 2000.0) : Coord(p3d, epoch) {}
    EclipticCoord(double const lambda, double const beta, double const epoch = 2000.0) : 
        Coord(lambda, beta, epoch) {}
    EclipticCoord(std::string const lambda, std::string const beta, double const epoch = 2000.0) : 
        Coord(lambda, beta, epoch) {}
    EclipticCoord() : Coord() {}
    
    std::pair<std::string, std::string> getCoordNames() const {
        return std::pair<std::string, std::string>("Lambda", "Beta");
    }
    double getLambda(CoordUnit unit) const         { return getLongitude(unit); }   
    double getBeta(CoordUnit unit) const           { return getLatitude(unit); }    
    std::string getLambdaStr(CoordUnit unit) const { return getLongitudeStr(unit); }
    std::string getBetaStr() const                 { return getLatitudeStr(); }     
    
    
    virtual Fk5Coord toFk5(double const epoch) const; 
    virtual Fk5Coord toFk5() const;
    virtual EclipticCoord toEcliptic(double const epoch) const;
    virtual EclipticCoord toEcliptic() const;

    EclipticCoord precess(double const epochTo) const;
    
private:

};


/**
 * @class TopocentricCoord
 * @brief A class to handle topocentric (AltAz) coordinates (inherits from Coord)
 */
class TopocentricCoord : public Coord {
public:
    
    typedef boost::shared_ptr<TopocentricCoord> Ptr;
    
    TopocentricCoord(afwGeom::Point2D const &p2d, CoordUnit unit, double const epoch,
                     Observatory const &obs) : Coord(p2d, unit, epoch), _obs(obs) {}
    TopocentricCoord(afwGeom::Point3D const &p3d, double const epoch,
                     Observatory const &obs) : Coord(p3d, epoch), _obs(obs) {}
    TopocentricCoord(double const az, double const alt, double const epoch,
                     Observatory const &obs) : Coord(az, alt, epoch), _obs(obs) {}
    TopocentricCoord(std::string const az, std::string const alt, double const epoch,
                     Observatory const &obs) : Coord(az, alt, epoch), _obs(obs) {}

    std::pair<std::string, std::string> getCoordNames() const {
        return std::pair<std::string, std::string>("Az", "Alt");
    }
    double getAzimuth(CoordUnit unit) const         { return getLongitude(unit); }   
    double getAltitude(CoordUnit unit) const        { return getLatitude(unit); }    
    std::string getAzimuthStr(CoordUnit unit) const { return getLongitudeStr(unit); }
    std::string getAltitudeStr() const              { return getLatitudeStr(); }     

    virtual Fk5Coord toFk5(double const epoch) const;
    virtual Fk5Coord toFk5() const;
    virtual TopocentricCoord toTopocentric(Observatory const &obs, dafBase::DateTime const &date) const;
    virtual TopocentricCoord toTopocentric() const;
    
private:
    Observatory _obs;
};


/*
 * Factory Functions
 *
 */
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


/*
 * Utility functions
 *
 */
double eclipticPoleInclination(double const epoch);
    
double dmsStringToDegrees(std::string const dms);
double hmsStringToDegrees(std::string const hms);
std::string degreesToDmsString(double const deg);
std::string degreesToHmsString(double const deg);    
    

    
}}}


/* ============================================================== 
 *
 * Definitions of inline functions
 *
 * ============================================================== */


/**
 * @brief Provide access to our contents via an index
 *
 * @note This only gets you the internal format ... RADIANS.
 */
inline double lsst::afw::coord::Coord::operator[](int const index) const {

    switch (index) {
      case 0:
        return _longitudeRad;
        break;
      case 1:
        return _latitudeRad;
        break;
      default:
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          "Index must be 0 or 1.");
        break;
    }
}

/**
 * @brief The main access method for the longitudinal coordinate
 *
 * All systems store their longitudinal coordinate in _longitude,
 * be it RA, l, lambda, or azimuth.  This is how they're accessed.
 *
 */
inline double lsst::afw::coord::Coord::getLongitude(CoordUnit unit) const {
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
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          "Units must be DEGREES, RADIANS, or HOURS.");
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
inline double lsst::afw::coord::Coord::getLatitude(CoordUnit unit) const {
    switch (unit) {
      case DEGREES:
        return radToDeg*_latitudeRad;
        break;
      case RADIANS:
        return _latitudeRad;
        break;
      default:
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          "Units must be DEGREES, or RADIANS.");
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
inline std::string lsst::afw::coord::Coord::getLongitudeStr(CoordUnit unit) const {
    if (unit == HOURS || unit == DEGREES) {
        return degreesToDmsString(getLongitude(unit));
    } else {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          "Units must be DEGREES or HOURS");
    }
}
/**
 * @brief Allow quick access to the longitude coordinate as a string
 *
 * @note There's no reason to want a latitude in radians or hours, so
 *       the units can not be explicitly requested.
 *
 */
inline std::string lsst::afw::coord::Coord::getLatitudeStr() const {
    return degreesToDmsString(getLatitude(DEGREES));
}



#endif

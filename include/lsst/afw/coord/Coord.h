// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
#if !defined(LSST_AFW_COORD_COORD_H)
#define LSST_AFW_COORD_COORD_H
/**
 * @file
 * @brief Functions to handle coordinates
 * @ingroup afw
 * @author Steve Bickerton
 *
 * @todo add FK4 ... as needed
 */ 
#include <iostream>
#include <limits>
#include <map>

#include "boost/shared_ptr.hpp"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/coord/Observatory.h"
#include "lsst/daf/base.h"

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

    Coord(lsst::afw::geom::Point2D const &p2d, lsst::afw::geom::AngleUnit unit = lsst::afw::geom::degrees, double const epoch = 2000.0);
    Coord(lsst::afw::geom::Point3D const &p3d, double const epoch = 2000.0,
          bool normalize=true,
          lsst::afw::geom::Angle const defaultLongitude = lsst::afw::geom::Angle(0.));
    Coord(lsst::afw::geom::Angle const ra, lsst::afw::geom::Angle const dec, double const epoch = 2000.0);
    Coord(std::string const ra, std::string const dec, double const epoch = 2000.0);
    Coord();
    virtual ~Coord() {}

    virtual Coord::Ptr clone() const { return Coord::Ptr(new Coord(*this)); }
    
    virtual void reset(lsst::afw::geom::Angle const longitude, lsst::afw::geom::Angle const latitude) {
        double const epoch = 2000.0;
        reset(longitude, latitude, epoch);
    }
    virtual void reset(lsst::afw::geom::Angle const longitude, lsst::afw::geom::Angle const latitude,
                       double const epoch);

    double getEpoch() const { return _epoch; }

    lsst::afw::geom::Point2D getPosition(lsst::afw::geom::AngleUnit unit = lsst::afw::geom::degrees) const;
    lsst::afw::geom::Point3D getVector() const;
    virtual std::pair<std::string, std::string> getCoordNames() const {
        return std::pair<std::string, std::string>("RA", "Dec");
    }

    virtual std::string getClassName() const { return "Coord"; }

    // These are inline functions and are defined at the end of this header file
    lsst::afw::geom::Angle operator[](int const index) const;
    bool operator==(Coord const &rhs) const;
    /**
     * @brief The main access method for the longitudinal coordinate
     *
     * All systems store their longitudinal coordinate in _longitude,
     * be it RA, l, lambda, or azimuth.  This is how they're accessed.
     *
     */
    inline lsst::afw::geom::Angle getLongitude() const { return _longitude; };
    /**
     * @brief The main access method for the latitudinal coordinate
     *
     * All systems store their latitudinal coordinate in _latitude,
     * be it Dec, b, beta, or altitude.  This is how they're accessed.
     */
    inline lsst::afw::geom::Angle getLatitude() const { return _latitude; };
    inline std::string getLongitudeStr(lsst::afw::geom::AngleUnit unit) const;
    inline std::string getLatitudeStr() const;

    
    Coord transform(Coord const &poleFrom, Coord const &poleTo) const;
    lsst::afw::geom::Angle angularSeparation(Coord const &c) const;

    std::pair<lsst::afw::geom::Angle, lsst::afw::geom::Angle> getOffsetFrom(Coord const &c) const;
    std::pair<lsst::afw::geom::Angle, lsst::afw::geom::Angle> getTangentPlaneOffset(Coord const &c) const;

    void rotate(Coord const &axis, lsst::afw::geom::Angle const theta);
    lsst::afw::geom::Angle offset(lsst::afw::geom::Angle const phi, lsst::afw::geom::Angle const arcLen);
    
    Coord::Ptr convert(CoordSystem system, double epoch=2000) const;

    virtual Fk5Coord toFk5(double const epoch) const;
    virtual Fk5Coord toFk5() const;
    virtual IcrsCoord toIcrs() const;
    virtual GalacticCoord toGalactic() const;
    virtual EclipticCoord toEcliptic(double const epoch) const;
    virtual EclipticCoord toEcliptic() const;
    virtual TopocentricCoord toTopocentric(Observatory const &obs,
        lsst::daf::base::DateTime const &obsDate) const;

private:
    lsst::afw::geom::Angle _longitude;
    lsst::afw::geom::Angle _latitude;
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

    IcrsCoord(lsst::afw::geom::Point2D const &p2d, lsst::afw::geom::AngleUnit unit = lsst::afw::geom::degrees) : Coord(p2d, unit, 2000.0) {}
    IcrsCoord(lsst::afw::geom::Point3D const &p3d, bool normalize=true, lsst::afw::geom::Angle const defaultLongitude = lsst::afw::geom::Angle(0.)) :
        Coord(p3d, 2000.0, normalize, defaultLongitude) {}
    IcrsCoord(lsst::afw::geom::Angle const ra, lsst::afw::geom::Angle const dec) : Coord(ra, dec, 2000.0) {}
    IcrsCoord(std::string const ra, std::string const dec) : Coord(ra, dec, 2000.0) {}
    IcrsCoord() : Coord() {}

    virtual Coord::Ptr clone() const { return IcrsCoord::Ptr(new IcrsCoord(*this)); }

    virtual std::string getClassName() const { return "IcrsCoord"; }
    
    virtual void reset(lsst::afw::geom::Angle const longitude, lsst::afw::geom::Angle const latitude);
    
    lsst::afw::geom::Angle getRa() const         { return getLongitude(); }   
    lsst::afw::geom::Angle getDec() const        { return getLatitude(); }    
    std::string getRaStr(lsst::afw::geom::AngleUnit unit) const { return getLongitudeStr(unit); }
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
    
    Fk5Coord(lsst::afw::geom::Point2D const &p2d, lsst::afw::geom::AngleUnit unit = lsst::afw::geom::degrees, double const epoch = 2000.0) :
        Coord(p2d, unit, epoch) {}
    Fk5Coord(lsst::afw::geom::Point3D const &p3d, double const epoch = 2000.0,
             bool normalize=true,
             lsst::afw::geom::Angle const defaultLongitude= lsst::afw::geom::Angle(0.)) :
        Coord(p3d, epoch, normalize, defaultLongitude) {}
    Fk5Coord(lsst::afw::geom::Angle const ra, lsst::afw::geom::Angle const dec, double const epoch = 2000.0) : 
        Coord(ra, dec, epoch) {}
    Fk5Coord(std::string const ra, std::string const dec, double const epoch = 2000.0) :
        Coord(ra, dec, epoch) {}
    Fk5Coord() : Coord() {}
    
    virtual Coord::Ptr clone() const { return Fk5Coord::Ptr(new Fk5Coord(*this)); }

    virtual std::string getClassName() const { return "Fk5Coord"; }

    Fk5Coord precess(double const epochTo) const;
    
    lsst::afw::geom::Angle getRa() const         { return getLongitude(); }   
    lsst::afw::geom::Angle getDec() const        { return getLatitude(); }    
    std::string getRaStr(lsst::afw::geom::AngleUnit unit) const { return getLongitudeStr(unit); }
    std::string getDecStr() const              { return getLatitudeStr(); }     

    virtual Fk5Coord toFk5(double const epoch) const;
    virtual Fk5Coord toFk5() const;
    virtual IcrsCoord toIcrs() const;
    virtual GalacticCoord toGalactic() const;
    virtual EclipticCoord toEcliptic(double const epoch) const;
    virtual EclipticCoord toEcliptic() const;
    virtual TopocentricCoord toTopocentric(Observatory const &obs,
        lsst::daf::base::DateTime const &obsDate) const;

    
private:
};


/**
 * @class GalacticCoord
 * @brief A class to handle Galactic coordinates (inherits from Coord)
 */
class GalacticCoord : public Coord {
public:
    
    typedef boost::shared_ptr<GalacticCoord> Ptr;
    
    GalacticCoord(lsst::afw::geom::Point2D const &p2d, lsst::afw::geom::AngleUnit unit = lsst::afw::geom::degrees) : Coord(p2d, unit) {}
    GalacticCoord(lsst::afw::geom::Point3D const &p3d,
                  bool normalize=true, lsst::afw::geom::Angle const defaultLongitude= lsst::afw::geom::Angle(0.)) :
        Coord(p3d, normalize, defaultLongitude) {}
    GalacticCoord(lsst::afw::geom::Angle const l, lsst::afw::geom::Angle const b) : Coord(l, b) {}
    GalacticCoord(std::string const l, std::string const b) : Coord(l, b) {}
    GalacticCoord() : Coord() {}

    virtual Coord::Ptr clone() const { return GalacticCoord::Ptr(new GalacticCoord(*this)); }

    virtual std::string getClassName() const { return "GalacticCoord"; }

    virtual void reset(lsst::afw::geom::Angle const longitude, lsst::afw::geom::Angle const latitude);
    
    virtual std::pair<std::string, std::string> getCoordNames() const {
        return std::pair<std::string, std::string>("L", "B");
    }
    
    lsst::afw::geom::Angle getL() const         { return getLongitude(); }   
    lsst::afw::geom::Angle getB() const         { return getLatitude(); }    
    std::string getLStr(lsst::afw::geom::AngleUnit unit) const { return getLongitudeStr(unit); }
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

    EclipticCoord(lsst::afw::geom::Point2D const &p2d,
                  lsst::afw::geom::AngleUnit unit = lsst::afw::geom::degrees,
                  double const epoch = 2000.0) :
        Coord(p2d, unit, epoch) {}
    EclipticCoord(lsst::afw::geom::Point3D const &p3d, double const epoch = 2000.0,
                  bool normalize=true,
                  lsst::afw::geom::Angle const defaultLongitude= lsst::afw::geom::Angle(0.)) :
        Coord(p3d, epoch, normalize, defaultLongitude) {}
    
    // note the abbreviation of lambda -> lamd to avoid swig warnings for python keyword 'lambda'
    EclipticCoord(lsst::afw::geom::Angle const lamb, lsst::afw::geom::Angle const beta,
                  double const epoch = 2000.0) : 
        Coord(lamb, beta, epoch) {}
    EclipticCoord(std::string const lamb, std::string const beta, double const epoch = 2000.0) : 
        Coord(lamb, beta, epoch) {}
    
    EclipticCoord() : Coord() {}
    
    virtual Coord::Ptr clone() const { return EclipticCoord::Ptr(new EclipticCoord(*this)); }

    virtual std::string getClassName() const { return "EclipticCoord"; }

    virtual std::pair<std::string, std::string> getCoordNames() const {
        return std::pair<std::string, std::string>("Lambda", "Beta");
    }
    lsst::afw::geom::Angle getLambda() const         { return getLongitude(); }   
    lsst::afw::geom::Angle getBeta() const           { return getLatitude(); }    
    std::string getLambdaStr(lsst::afw::geom::AngleUnit unit) const { return getLongitudeStr(unit); }
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
    
    TopocentricCoord(lsst::afw::geom::Point2D const &p2d, lsst::afw::geom::AngleUnit unit, double const epoch,
                     Observatory const &obs) : Coord(p2d, unit, epoch), _obs(obs) {}
    TopocentricCoord(lsst::afw::geom::Point3D const &p3d, double const epoch,
                     Observatory const &obs, bool normalize=true,
                     lsst::afw::geom::Angle const defaultLongitude= lsst::afw::geom::Angle(0.)) :
        Coord(p3d, epoch, normalize, defaultLongitude), _obs(obs) {}
    TopocentricCoord(lsst::afw::geom::Angle const az, lsst::afw::geom::Angle const alt, double const epoch,
                     Observatory const &obs) : Coord(az, alt, epoch), _obs(obs) {}
    TopocentricCoord(std::string const az, std::string const alt, double const epoch,
                     Observatory const &obs) : Coord(az, alt, epoch), _obs(obs) {}

    virtual Coord::Ptr clone() const { return TopocentricCoord::Ptr(new TopocentricCoord(*this)); }

    virtual std::string getClassName() const { return "TopocentricCoord"; }

    Observatory getObservatory() const { return _obs; }

    virtual std::pair<std::string, std::string> getCoordNames() const {
        return std::pair<std::string, std::string>("Az", "Alt");
    }
    lsst::afw::geom::Angle getAzimuth() const         { return getLongitude(); }   
    lsst::afw::geom::Angle getAltitude() const        { return getLatitude(); }    
    std::string getAzimuthStr(lsst::afw::geom::AngleUnit unit) const { return getLongitudeStr(unit); }
    std::string getAltitudeStr() const              { return getLatitudeStr(); }     

    virtual Fk5Coord toFk5(double const epoch) const;
    virtual Fk5Coord toFk5() const;
    virtual TopocentricCoord toTopocentric(Observatory const &obs,
        lsst::daf::base::DateTime const &date) const;
    virtual TopocentricCoord toTopocentric() const;
    
private:
    Observatory _obs;
};


/*
 * Factory Functions
 *
 */
Coord::Ptr makeCoord(CoordSystem const system, lsst::afw::geom::Angle const ra, lsst::afw::geom::Angle const dec, double const epoch);
Coord::Ptr makeCoord(CoordSystem const system, std::string const ra, std::string const dec,
                     double const epoch);
Coord::Ptr makeCoord(CoordSystem const system, lsst::afw::geom::Point2D const &p2d, lsst::afw::geom::AngleUnit unit,
                     double const epoch);
Coord::Ptr makeCoord(CoordSystem const system, lsst::afw::geom::Point3D const &p3d, double const epoch,
                     bool normalize=true,
                     lsst::afw::geom::Angle const defaultLongitude=lsst::afw::geom::Angle(0.));
Coord::Ptr makeCoord(CoordSystem const system);

Coord::Ptr makeCoord(CoordSystem const system, lsst::afw::geom::Angle const ra, lsst::afw::geom::Angle const dec);
Coord::Ptr makeCoord(CoordSystem const system, std::string const ra, std::string const dec);
Coord::Ptr makeCoord(CoordSystem const system, lsst::afw::geom::Point2D const &p2d, lsst::afw::geom::AngleUnit unit);
Coord::Ptr makeCoord(CoordSystem const system, lsst::afw::geom::Point3D const &p3d,
                     bool normalize=true,
                     lsst::afw::geom::Angle const defaultLongitude=lsst::afw::geom::Angle(0.));

/*
 * Utility functions
 *
 */
lsst::afw::geom::Angle eclipticPoleInclination(double const epoch);
    
lsst::afw::geom::Angle dmsStringToAngle(std::string const dms);
lsst::afw::geom::Angle hmsStringToAngle(std::string const hms);
std::string angleToDmsString(lsst::afw::geom::Angle const deg);
std::string angleToHmsString(lsst::afw::geom::Angle const deg);    
    
std::ostream & operator<<(std::ostream & os, Coord const & coord);

}}}


/* ============================================================== 
 *
 * Definitions of inline functions
 *
 * ============================================================== */


/**
 * @brief Provide access to our contents via an index
 *
 */
inline lsst::afw::geom::Angle lsst::afw::coord::Coord::operator[](int const index) const {

    switch (index) {
      case 0:
        return _longitude;
        break;
      case 1:
        return _latitude;
        break;
      default:
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "Index must be 0 or 1.");
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
inline std::string lsst::afw::coord::Coord::getLongitudeStr(lsst::afw::geom::AngleUnit unit) const {
    if (unit == lsst::afw::geom::hours) {
        return angleToHmsString(getLongitude());
    } else if (unit == lsst::afw::geom::degrees) {
        return angleToDmsString(getLongitude());
    } else {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "Units must be 'degrees' or 'hours'");
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
    return angleToDmsString(getLatitude());
}

/**
 * @brief Equality operator, compares each element directly
 */
inline bool lsst::afw::coord::Coord::operator==(lsst::afw::coord::Coord const &rhs) const {
    return (_longitude == rhs._longitude) &&
        (_latitude == rhs._latitude) &&
        (_epoch == rhs._epoch);
}

/**
 * @brief Inequality; the complement of equality
 */
inline bool operator!=(lsst::afw::coord::Coord const &lhs, lsst::afw::coord::Coord const &rhs) {
    return !(lhs == rhs);
}

#endif

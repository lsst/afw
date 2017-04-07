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
/*
 * Functions to handle coordinates
 */
#include <iostream>
#include <limits>
#include <map>
#include <memory>

#include "lsst/base.h"
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
enum CoordSystem { UNKNOWN=-1, FK5, ICRS, GALACTIC, ECLIPTIC, TOPOCENTRIC };
/**
 * A utility function to get the enum value of a coordinate system from a string name.
 */
CoordSystem makeCoordEnum(std::string const system);

class IcrsCoord;
class Fk5Coord;
class GalacticCoord;
class EclipticCoord;
class TopocentricCoord;


/**
 * This is the base class for spherical coordinates.
 */
class Coord {
public:

    /**
     * Constructor for the Coord base class
     *
     * @param p2d Point2D
     * @param unit Rads, Degs, or Hrs
     * @param epoch epoch of coordinate
     */
    Coord(lsst::afw::geom::Point2D const &p2d, lsst::afw::geom::AngleUnit unit = lsst::afw::geom::degrees, double const epoch = 2000.0);
    /**
     * Constructor for the Coord base class
     *
     * @param p3d Point3D
     * @param epoch epoch of coordinate
     * @param normalize normalize the p3d provided
     * @param defaultLongitude longitude to use if x=y=0
     */
    Coord(lsst::afw::geom::Point3D const &p3d, double const epoch = 2000.0,
          bool normalize=true,
          lsst::afw::geom::Angle const defaultLongitude = lsst::afw::geom::Angle(0.));
    /**
     * Constructor for the Coord base class
     *
     * @param ra Right ascension, decimal degrees
     * @param dec Declination, decimal degrees
     * @param epoch epoch of coordinate
     */
    Coord(lsst::afw::geom::Angle const ra, lsst::afw::geom::Angle const dec, double const epoch = 2000.0);
    /**
     * Constructor for the Coord base class
     *
     * @param ra Right ascension, hh:mm:ss.s format
     * @param dec Declination, dd:mm:ss.s format
     * @param epoch epoch of coordinate
     */
    Coord(std::string const ra, std::string const dec, double const epoch = 2000.0);
    /**
     * Default constructor for the Coord base class
     *
     * Set all values to NaN
     * Don't call _veriftyValues() method ... it'll fail.
     *
     */
    Coord();
    virtual ~Coord() {}

    virtual PTR(Coord) clone() const { return PTR(Coord)(new Coord(*this)); }

    virtual void reset(lsst::afw::geom::Angle const longitude, lsst::afw::geom::Angle const latitude) {
        double const epoch = 2000.0;
        reset(longitude, latitude, epoch);
    }
    /**
     * Reset our coordinates wholesale.
     *
     * This allows the user to instantiate Coords without values, and fill them later.
     *
     * @param longitude Longitude coord (eg. R.A. for Fk5)
     * @param latitude Latitude coord (eg. Declination for Fk5)
     * @param epoch epoch of coordinate
     */
    virtual void reset(lsst::afw::geom::Angle const longitude, lsst::afw::geom::Angle const latitude,
                       double const epoch);

    double getEpoch() const { return _epoch; }

    /**
     * Return our contents in a Point2D object
     *
     */
    lsst::afw::geom::Point2D getPosition(lsst::afw::geom::AngleUnit unit = lsst::afw::geom::degrees) const;
    /**
     * Return our contents in a position vector.
     *
     */
    lsst::afw::geom::Point3D getVector() const;
    virtual std::pair<std::string, std::string> getCoordNames() const {
        return std::pair<std::string, std::string>("RA", "Dec");
    }

    virtual std::string getClassName() const { return "Coord"; }

    virtual CoordSystem getCoordSystem() const { return UNKNOWN; }

    /**
     * Get string representation
     */
    virtual std::string toString() const;

    // These are inline functions and are defined at the end of this header file
    lsst::afw::geom::Angle operator[](int const index) const;
    bool operator==(Coord const &rhs) const;
    /**
     * The main access method for the longitudinal coordinate
     *
     * All systems store their longitudinal coordinate in _longitude,
     * be it RA, l, lambda, or azimuth.  This is how they're accessed.
     *
     */
    inline lsst::afw::geom::Angle getLongitude() const { return _longitude; };
    /**
     * The main access method for the latitudinal coordinate
     *
     * All systems store their latitudinal coordinate in _latitude,
     * be it Dec, b, beta, or altitude.  This is how they're accessed.
     */
    inline lsst::afw::geom::Angle getLatitude() const { return _latitude; };
    inline std::string getLongitudeStr(lsst::afw::geom::AngleUnit unit) const;
    inline std::string getLatitudeStr() const;


    /**
     * Transform our current coords to another spherical polar system
     *
     * Variable names assume an equatorial/galactic transform, but it works
     *  for any spherical polar system when the appropriate poles are supplied.
     *
     * @param poleTo Pole of the destination system in the current coords
     * @param poleFrom Pole of the current system in the destination coords
     */
    Coord transform(
    Coord const &poleTo,
    Coord const &poleFrom
                                          ) const;
    /**
     * compute the angular separation between two Coords
     *
     * @param c coordinate to compute our separation from
     */
    lsst::afw::geom::Angle angularSeparation(Coord const &c) const;

    /**
     * Compute the offset from a coordinate
     *
     * The resulting angles are suitable for input to Coord::offset
     *
     * @param c Coordinate from which to compute offset
     * @returns pair of Angles: bearing (angle wrt a declination parallel) and distance
     */
    std::pair<lsst::afw::geom::Angle, lsst::afw::geom::Angle> getOffsetFrom(Coord const &c) const;
    /**
     * Get the offset on the tangent plane
     *
     * This is suitable only for small angles.
     *
     * @param c Coordinate from which to compute offset
     * @returns pair of Angles: Longitude and Latitude offsets
     */
    std::pair<lsst::afw::geom::Angle, lsst::afw::geom::Angle> getTangentPlaneOffset(Coord const &c) const;

    /**
     * Rotate our current coords about a pole
     *
     * @param axis axis of rotation (right handed)
     * @param theta angle to offset in radians
     */
    void rotate(Coord const &axis, lsst::afw::geom::Angle const theta);
    /**
     * offset our current coords along a great circle defined by an angle wrt a declination parallel
     *
     * @param phi angle wrt parallel to offset
     * @param arcLen angle to offset
     * @returns the angle wrt a declination parallel at new position
     *
     * @note At/near the pole, longitude becomes degenerate with angle-wrt-declination.  So
     *       at the pole the offset will trace a meridian with longitude = 90 + longitude0 + `phi`
     */
    lsst::afw::geom::Angle offset(lsst::afw::geom::Angle const phi, lsst::afw::geom::Angle const arcLen);

    /**
     * Convert to a specified Coord type at a specified epoch.
     *
     * @param[in] system  coordinate system to which to convert
     * @param[in] epoch  epoch of coordinate system; only relevant for FK5 and Ecliptic coordinates
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if `system` = `TOPOCENTRIC`
     *         (because observatory data is required) or if `system` not recognized
     */
    PTR(Coord) convert(CoordSystem system, double epoch=2000) const;

    /**
     * Convert ourself to Fk5: RA, Dec (precess to new epoch)
     */
    virtual Fk5Coord toFk5(double const epoch) const;
    /**
     * Convert ourself to Fk5: RA, Dec (use current epoch)
     */
    virtual Fk5Coord toFk5() const;
    /**
     * Convert ourself to ICRS: RA, Dec (basically J2000)
     *
     */
    virtual IcrsCoord toIcrs() const;
    /**
     * Convert ourself to Galactic: l, b
     */
    virtual GalacticCoord toGalactic() const;
    /**
     * Convert ourself to Ecliptic: lambda, beta (precess to new epoch)
     */
    virtual EclipticCoord toEcliptic(double const epoch) const;
    /**
     * Convert ourself to Ecliptic: lambda, beta (use existing epoch)
     */
    virtual EclipticCoord toEcliptic() const;
    /**
     * Convert ourself to Altitude/Azimuth: alt, az
     *
     * @param obs observatory of observation
     * @param obsDate date of observation
     */
    virtual TopocentricCoord toTopocentric(Observatory const &obs,
        lsst::daf::base::DateTime const &obsDate) const;

private:
    lsst::afw::geom::Angle _longitude;
    lsst::afw::geom::Angle _latitude;
    double _epoch;

    /**
     * Make sure the values we've got are in the range 0 <= x < 2PI
     */
    void _verifyValues() const;
};

/**
 * A class to handle Icrs coordinates (inherits from Coord)
 */
class IcrsCoord : public Coord {
public:

    IcrsCoord(lsst::afw::geom::Point2D const &p2d, lsst::afw::geom::AngleUnit unit = lsst::afw::geom::degrees) : Coord(p2d, unit, 2000.0) {}
    IcrsCoord(lsst::afw::geom::Point3D const &p3d, bool normalize=true, lsst::afw::geom::Angle const defaultLongitude = lsst::afw::geom::Angle(0.)) :
        Coord(p3d, 2000.0, normalize, defaultLongitude) {}
    IcrsCoord(lsst::afw::geom::Angle const ra, lsst::afw::geom::Angle const dec) : Coord(ra, dec, 2000.0) {}
    IcrsCoord(std::string const ra, std::string const dec) : Coord(ra, dec, 2000.0) {}
    IcrsCoord() : Coord() {}

    virtual PTR(Coord) clone() const { return PTR(IcrsCoord)(new IcrsCoord(*this)); }

    virtual std::string getClassName() const { return "IcrsCoord"; }

    virtual CoordSystem getCoordSystem() const { return ICRS; }

    /**
     * Get string representation
     */
    virtual std::string toString() const;

    /**
     * special reset() overload to make sure no epoch can be set
     */
    virtual void reset(lsst::afw::geom::Angle const longitude, lsst::afw::geom::Angle const latitude);

    lsst::afw::geom::Angle getRa() const         { return getLongitude(); }
    lsst::afw::geom::Angle getDec() const        { return getLatitude(); }
    std::string getRaStr(lsst::afw::geom::AngleUnit unit) const { return getLongitudeStr(unit); }
    std::string getDecStr() const              { return getLatitudeStr(); }

    /**
     * Fk5 converter for IcrsCoord. (specify epoch)
     */
    virtual Fk5Coord toFk5(double const epoch) const;
    /**
     * Fk5 converter for IcrsCoord. (no epoch specified)
     */
    virtual Fk5Coord toFk5() const;
    /**
     * Icrs converter for IcrsCoord. (ie. a no-op)
     */
    virtual IcrsCoord toIcrs() const;

private:
};


/**
 * A class to handle Fk5 coordinates (inherits from Coord)
 */
class Fk5Coord : public Coord {
public:

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

    virtual PTR(Coord) clone() const { return PTR(Fk5Coord)(new Fk5Coord(*this)); }

    virtual std::string getClassName() const { return "Fk5Coord"; }

    virtual CoordSystem getCoordSystem() const { return FK5; }

    /**
     * Precess ourselves from whence we are to a new epoch
     *
     * @param epochTo epoch to precess to
     */
    Fk5Coord precess(double const epochTo) const;

    lsst::afw::geom::Angle getRa() const         { return getLongitude(); }
    lsst::afw::geom::Angle getDec() const        { return getLatitude(); }
    std::string getRaStr(lsst::afw::geom::AngleUnit unit) const { return getLongitudeStr(unit); }
    std::string getDecStr() const              { return getLatitudeStr(); }

    /**
     * Convert ourself to Fk5 (ie. a no-op): RA, Dec  (precess to new epoch)
     */
    virtual Fk5Coord toFk5(double const epoch) const;
    /**
     * Convert ourself to Fk5 (ie. a no-op): RA, Dec (keep current epoch)
     */
    virtual Fk5Coord toFk5() const;
    /**
     * Convert ourself to ICRS: RA, Dec (basically J2000)
     *
     */
    virtual IcrsCoord toIcrs() const;
    /**
     * Convert ourself to Galactic: l, b
     */
    virtual GalacticCoord toGalactic() const;
    /**
     * Convert ourself to Ecliptic: lambda, beta (precess to new epoch)
     */
    virtual EclipticCoord toEcliptic(double const epoch) const;
    /**
     * Convert ourself to Ecliptic: lambda, beta (use current epoch)
     */
    virtual EclipticCoord toEcliptic() const;
    /**
     * Convert ourself to Altitude/Azimuth: alt, az
     *
     * @param obs observatory
     * @param obsDate date of obs.
     */
    virtual TopocentricCoord toTopocentric(Observatory const &obs,
        lsst::daf::base::DateTime const &obsDate) const;


private:
};


/**
 * A class to handle Galactic coordinates (inherits from Coord)
 */
class GalacticCoord : public Coord {
public:

    GalacticCoord(lsst::afw::geom::Point2D const &p2d, lsst::afw::geom::AngleUnit unit = lsst::afw::geom::degrees) : Coord(p2d, unit) {}
    GalacticCoord(lsst::afw::geom::Point3D const &p3d,
                  bool normalize=true, lsst::afw::geom::Angle const defaultLongitude= lsst::afw::geom::Angle(0.)) :
        Coord(p3d, normalize, defaultLongitude) {}
    GalacticCoord(lsst::afw::geom::Angle const l, lsst::afw::geom::Angle const b) : Coord(l, b) {}
    GalacticCoord(std::string const l, std::string const b) : Coord(l, b) {}
    GalacticCoord() : Coord() {}

    virtual PTR(Coord) clone() const { return PTR(GalacticCoord)(new GalacticCoord(*this)); }

    virtual std::string getClassName() const { return "GalacticCoord"; }

    virtual CoordSystem getCoordSystem() const { return GALACTIC; }

    /**
     * Get string representation
     */
    virtual std::string toString() const;

    /**
     * special reset() overload to make sure no epoch can be set
     */
    virtual void reset(lsst::afw::geom::Angle const longitude, lsst::afw::geom::Angle const latitude);

    virtual std::pair<std::string, std::string> getCoordNames() const {
        return std::pair<std::string, std::string>("L", "B");
    }

    lsst::afw::geom::Angle getL() const         { return getLongitude(); }
    lsst::afw::geom::Angle getB() const         { return getLatitude(); }
    std::string getLStr(lsst::afw::geom::AngleUnit unit) const { return getLongitudeStr(unit); }
    std::string getBStr() const               { return getLatitudeStr(); }

    /**
     * Convert ourself from galactic to Fk5 (specify epoch)
     */
    virtual Fk5Coord toFk5(double const epoch) const;
    /**
     * Convert ourself from galactic to Fk5 (no epoch specified)
     */
    virtual Fk5Coord toFk5() const ;
    /**
     * Convert ourself from Galactic to Galactic ... a no-op
     */
    virtual GalacticCoord toGalactic() const;

private:
};



/**
 * A class to handle Ecliptic coordinates (inherits from Coord)
 */
class EclipticCoord : public Coord {
public:

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

    virtual PTR(Coord) clone() const { return PTR(EclipticCoord)(new EclipticCoord(*this)); }

    virtual std::string getClassName() const { return "EclipticCoord"; }

    virtual CoordSystem getCoordSystem() const { return ECLIPTIC; }

    virtual std::pair<std::string, std::string> getCoordNames() const {
        return std::pair<std::string, std::string>("Lambda", "Beta");
    }
    lsst::afw::geom::Angle getLambda() const         { return getLongitude(); }
    lsst::afw::geom::Angle getBeta() const           { return getLatitude(); }
    std::string getLambdaStr(lsst::afw::geom::AngleUnit unit) const { return getLongitudeStr(unit); }
    std::string getBetaStr() const                 { return getLatitudeStr(); }


    /**
     * Convert ourself from Ecliptic to Fk5 (precess to new epoch)
     */
    virtual Fk5Coord toFk5(double const epoch) const;
    /**
     * Convert ourself from Ecliptic to Fk5 (use current epoch)
     */
    virtual Fk5Coord toFk5() const;
    /**
     * Convert ourself from Ecliptic to Ecliptic ... a no-op (but precess to new epoch)
     */
    virtual EclipticCoord toEcliptic(double const epoch) const;
    /**
     * Convert ourself from Ecliptic to Ecliptic ... a no-op (use the current epoch)
     */
    virtual EclipticCoord toEcliptic() const;

    /**
     * precess to new epoch
     *
     * @param epochTo epoch to precess to.
     *
     * @note Do this by going through fk5
     */
    EclipticCoord precess(double const epochTo) const;

private:
};


/**
 * A class to handle topocentric (AltAz) coordinates (inherits from Coord)
 */
class TopocentricCoord : public Coord {
public:

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

    virtual PTR(Coord) clone() const { return PTR(TopocentricCoord)(new TopocentricCoord(*this)); }

    virtual std::string getClassName() const { return "TopocentricCoord"; }

    virtual CoordSystem getCoordSystem() const { return TOPOCENTRIC; }

    /**
     * Get string representation
     */
    virtual std::string toString() const;

    Observatory getObservatory() const { return _obs; }

    virtual std::pair<std::string, std::string> getCoordNames() const {
        return std::pair<std::string, std::string>("Az", "Alt");
    }
    lsst::afw::geom::Angle getAzimuth() const         { return getLongitude(); }
    lsst::afw::geom::Angle getAltitude() const        { return getLatitude(); }
    std::string getAzimuthStr(lsst::afw::geom::AngleUnit unit) const { return getLongitudeStr(unit); }
    std::string getAltitudeStr() const              { return getLatitudeStr(); }

    /**
     * Convert ourself from Topocentric to Fk5
     */
    virtual Fk5Coord toFk5(double const epoch) const;
    /**
     * Convert outself from Topocentric to Fk5 (use current epoch)
     */
    virtual Fk5Coord toFk5() const;
    /**
     * Convert ourself from Topocentric to Topocentric ... a no-op
     *
     * @param obs observatory of observation
     * @param date date of observation
     */
    virtual TopocentricCoord toTopocentric(Observatory const &obs,
        lsst::daf::base::DateTime const &date) const;
    /**
     * Convert ourself from Topocentric to Topocentric with no observatory or date arguments
     *
     * @note As this is essentially a copy-constructor, the extra info can be obtained internally.
     */
    virtual TopocentricCoord toTopocentric() const;

private:
    Observatory _obs;
};


/*
 * Factory Functions
 *
 */
/**
 * Factory function to create a Coord of arbitrary type with decimal RA,Dec
 *
 * @param system the system (equ, fk5, galactic ..)
 * @param ra right ascension
 * @param dec declination
 * @param epoch epoch of coordinate
 *
 * @note This factory allows the epoch to be specified but will throw if used with ICRS or Galactic
 * @note Most of the other factories (which accept epochs) just call this one indirectly.
 *
 */
PTR(Coord) makeCoord(CoordSystem const system, lsst::afw::geom::Angle const ra, lsst::afw::geom::Angle const dec, double const epoch);
/**
 * Factory function to create a Coord of arbitrary type with string RA [in degrees, not hours!], Dec
 *
 * @param system the system (equ, fk5, galactic ..)
 * @param ra right ascension
 * @param dec declination
 * @param epoch epoch of coordinate
 *
 * @note This factory accepts epoch.  There is an overloaded version which uses a default.
 */
PTR(Coord) makeCoord(CoordSystem const system, std::string const ra, std::string const dec,
                     double const epoch);
/**
 * Factory function to create a Coord of arbitrary type with Point2D
 *
 * @param system the system (equ, fk5, galactic ..)
 * @param p2d the (eg) ra,dec in a Point2D
 * @param unit the units (eg. degrees, radians)
 * @param epoch epoch of coordinate
 *
 * @note This factory accepts epoch.  There is an overloaded version which uses a default.
 */
PTR(Coord) makeCoord(CoordSystem const system, lsst::afw::geom::Point2D const &p2d, lsst::afw::geom::AngleUnit unit,
                     double const epoch);
/**
 * Factory function to create a Coord of arbitrary type with a Point3D
 *
 * @param system the system (equ, fk5, galactic ..)
 * @param p3d the coord in Point3D format
 * @param epoch epoch of coordinate
 * @param normalize normalize the p3d provided
 * @param defaultLongitude longitude to use if x=y=0
 *
 * @note This factory accepts epoch.  There is an overloaded version which uses a default.
 *
 */
PTR(Coord) makeCoord(CoordSystem const system, lsst::afw::geom::Point3D const &p3d, double const epoch,
                     bool normalize=true,
                     lsst::afw::geom::Angle const defaultLongitude=lsst::afw::geom::Angle(0.));
/**
 * Lightweight factory to make an empty coord.
 *
 * @param system the system (FK5, ICRS, etc)
 */
PTR(Coord) makeCoord(CoordSystem const system);

/**
 * Factory function to create a Coord of arbitrary type with decimal RA,Dec in degrees
 *
 * @param system the system (equ, fk5, galactic ..)
 * @param ra right ascension
 * @param dec declination
 *
 * @note This factory assumes a default epoch
 * @note Most of the other factories (which don't accept epoch) call this one.
 */
PTR(Coord) makeCoord(CoordSystem const system, lsst::afw::geom::Angle const ra, lsst::afw::geom::Angle const dec);
/**
 * Factory function to create a Coord of arbitrary type with string RA [in degrees, not hours!], Dec
 *
 * @param system the system (equ, fk5, galactic ..)
 * @param ra right ascension
 * @param dec declination
 *
 * @note This factory uses a default epoch.  There is an overloaded version which accepts an epoch.
 */
PTR(Coord) makeCoord(CoordSystem const system, std::string const ra, std::string const dec);
/**
 * Factory function to create a Coord of arbitrary type with Point2D
 *
 * @param system the system (equ, fk5, galactic ..)
 * @param p2d the (eg) ra,dec in a Point2D
 * @param unit the units (eg. degrees, radians)
 *
 * @note This factory uses a default epoch.  There is an overloaded version which accepts an epoch.
 *
 */
PTR(Coord) makeCoord(CoordSystem const system, lsst::afw::geom::Point2D const &p2d, lsst::afw::geom::AngleUnit unit);
/**
 * Factory function to create a Coord of arbitrary type with a Point3D
 *
 * @param system the system (equ, fk5, galactic ..)
 * @param p3d the coord in Point3D format
 * @param normalize normalize the p3d provided
 * @param defaultLongitude longitude to use if x=y=0
 *
 * @note This factory uses a default epoch.  There is an overloaded version which accepts an epoch.
 *
 */
PTR(Coord) makeCoord(CoordSystem const system, lsst::afw::geom::Point3D const &p3d,
                     bool normalize=true,
                     lsst::afw::geom::Angle const defaultLongitude=lsst::afw::geom::Angle(0.));

/**
 * Return average of a list of coordinates
 *
 * @param[in] coords  list of coords to average
 * @param[in] system  coordinate system of returned result;
 *                    if UNKNOWN then all input coordinates must have the same coordinate system,
 *                    which is used for the result
 *
 * @throws  lsst::pex::exceptions::InvalidParameterError if system is UNKNOWN
 *          and the coords do not all have the same coordinate system
 */
PTR(Coord) averageCoord(
    std::vector<PTR(Coord const)> const coords,
    CoordSystem system=UNKNOWN
    );

/*
 * Utility functions
 *
 */
/**
 * get the inclination of the ecliptic pole (obliquity) at epoch
 *
 * @param epoch desired epoch for inclination
 */
lsst::afw::geom::Angle eclipticPoleInclination(double const epoch);

/**
 * Convert a dd:mm:ss string to Angle
 *
 * @param dms Coord as a string in dd:mm:ss format
 */
lsst::afw::geom::Angle dmsStringToAngle(std::string const dms);
/// Convert a hh:mm:ss string to Angle
lsst::afw::geom::Angle hmsStringToAngle(std::string const hms);
/**
 * a Function to convert a coordinate in decimal degrees to a string with form dd:mm:ss
 *
 * @todo allow a user specified format
 */
std::string angleToDmsString(lsst::afw::geom::Angle const deg);
/// a function to convert decimal degrees to a string with form hh:mm:ss.s
std::string angleToHmsString(lsst::afw::geom::Angle const deg);

std::ostream & operator<<(std::ostream & os, Coord const & coord);

}}}


/* ==============================================================
 *
 * Definitions of inline functions
 *
 * ============================================================== */


/**
 * Provide access to our contents via an index
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
 * Allow quick access to the longitudinal coordinate as a string
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
 * Allow quick access to the longitude coordinate as a string
 *
 * @note There's no reason to want a latitude in radians or hours, so
 *       the units can not be explicitly requested.
 *
 */
inline std::string lsst::afw::coord::Coord::getLatitudeStr() const {
    return angleToDmsString(getLatitude());
}

/**
 * Equality operator, compares each element directly
 */
inline bool lsst::afw::coord::Coord::operator==(lsst::afw::coord::Coord const &rhs) const {
    return (_longitude == rhs._longitude) &&
        (_latitude == rhs._latitude) &&
        (_epoch == rhs._epoch);
}

/**
 * Inequality; the complement of equality
 */
inline bool operator!=(lsst::afw::coord::Coord const &lhs, lsst::afw::coord::Coord const &rhs) {
    return !(lhs == rhs);
}

#endif

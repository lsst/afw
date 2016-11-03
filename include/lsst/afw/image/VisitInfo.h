// -*- LSST-C++ -*- // fixed format comment for emacs
/*
 * LSST Data Management System
 * Copyright 2016 LSST Corporation.
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

#ifndef LSST_AFW_IMAGE_VISITINFO_H_INCLUDED
#define LSST_AFW_IMAGE_VISITINFO_H_INCLUDED

#include <cmath>
#include <limits>

#include "lsst/base.h"
#include "lsst/daf/base.h"
#include "lsst/afw/coord/Observatory.h"
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/coord/Weather.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/table/misc.h"  // for RecordId
#include "lsst/afw/table/io/Persistable.h"

namespace lsst { namespace afw { namespace image {

/// Type of rotation
enum class RotType {
    UNKNOWN,    ///< Rotation angle is unknown. Note: if there is no instrument rotator then it is better
                ///< to compute SKY or HORIZON and use that rotation type rather than specify UNKNOWN.
    SKY,        ///< Orientation of E,N with respected to detector X,Y;
                ///< X is flipped, if necessary, to match the handedness of E,N.
    HORIZON,    ///< orientation of Az/Alt with respect to detector X,Y;
                ///< X is flipped, if necessary, to match the handedness of Az,Alt.
    MOUNT       ///< The position sent to the instrument rotator; the details depend on the rotator.
};

/**
 * Information about a single exposure of an imaging camera.
 *
 * Includes exposure duration and date, and telescope pointing and orientation.
 *
 * All information is for the middle of the exposure
 * and at the boresight (center of the focal plane).
 * Thus for a mosaic camera VisitInfo is the same for all detectors in the mosaic.
 *
 * VisitInfo is immutable.
 */
class VisitInfo : public table::io::PersistableFacade<VisitInfo>, public table::io::Persistable {
public:

    /**
     * Construct a VisitInfo
     *
     * @param[in] exposureId  exposure ID
     * @param[in] exposureTime  exposure duration (shutter open time); (sec)
     * @param[in] darkTime  time from CCD flush to readout, including shutter open time (despite the name);
                    (sec)
     * @param[in] date  TAI (international atomic time) MJD date at middle of exposure
     * @param[in] ut1  UT1 (universal time) MJD date at middle of exposure
     * @param[in] era  earth rotation angle at middle of exposure
     * @param[in] boresightRaDec  ICRS RA/Dec of boresight at middle of exposure
     * @param[in] boresightAzAlt  refracted apparent topocentric Az/Alt of boresight at middle of exposure;
                    for now this is represented as a plain coord::Coord,
                    but in the long run it will be a SphPoint or a specialized Coord if one is added
                    (we have TopocentricCoord, but not with refraction)
     * @param[in] boresightAirmass  airmass at the boresight, relative to zenith at sea level
     * @param[in] boresightRotAngle  rotation angle at boresight at middle of exposure;
                        see getBoresightRotAngle for details
     * @param[in] rotType  rotation type
     * @param[in] observatory  observatory longitude, latitude and altitude
     * @param[in] weather  basic weather information for computing air mass
     */
    explicit VisitInfo(
        table::RecordId exposureId,
        double exposureTime,
        double darkTime,
        daf::base::DateTime const & date,
        double ut1,
        geom::Angle const & era,
        coord::IcrsCoord const & boresightRaDec,
        coord::Coord const & boresightAzAlt,
        double boresightAirmass,
        geom::Angle const & boresightRotAngle,
        RotType const & rotType,
        coord::Observatory const & observatory,
        coord::Weather const & weather
    ) :
        _exposureId(exposureId),
        _exposureTime(exposureTime),
        _darkTime(darkTime),
        _date(date),
        _ut1(ut1),
        _era(era),
        _boresightRaDec(boresightRaDec),
        _boresightAzAlt(boresightAzAlt),
        _boresightAirmass(boresightAirmass),
        _boresightRotAngle(boresightRotAngle),
        _rotType(rotType),
        _observatory(observatory),
        _weather(weather)
    {};

    explicit VisitInfo(daf::base::PropertySet const & metadata);

    ~VisitInfo() {};

    VisitInfo(VisitInfo const &) = default;
    VisitInfo(VisitInfo &&) = default;
    VisitInfo & operator=(VisitInfo const &) = default;
    VisitInfo & operator=(VisitInfo &&) = default;

    bool operator==(VisitInfo const & other) const;
    bool operator!=(VisitInfo const & other) const { return !(*this == other); };

    /// get exposure ID
    table::RecordId getExposureId() const { return _exposureId; }

    /// get exposure duration (shutter open time); (sec)
    double getExposureTime() const { return _exposureTime; }

    /// get time from CCD flush to exposure readout, including shutter open time (despite the name); (sec)
    double getDarkTime() const { return _darkTime; }

    /// get uniform date and time at middle of exposure
    daf::base::DateTime getDate() const { return _date; }

    /// get UT1 (universal time) MJD date at middle of exposure
    double getUt1() const { return _ut1; }

    /// get earth rotation angle at middle of exposure
    geom::Angle getEra() const { return _era; }

    /// get ICRS RA/Dec position at the boresight
    /// (and at the middle of the exposure, if it varies with time)
    coord::IcrsCoord getBoresightRaDec() const { return _boresightRaDec; }

    /// get refracted apparent topocentric Az/Alt position at the boresight
    /// (and at the middle of the exposure, if it varies with time)
    coord::Coord getBoresightAzAlt() const { return _boresightAzAlt; }

    /// get airmass at the boresight, relative to zenith at sea level
    /// (and at the middle of the exposure, if it varies with time)
    double getBoresightAirmass() const { return _boresightAirmass; }

    /**
    * Get rotation angle at boresight at middle of exposure
    *
    * Rotation angle is angle of coordinate system specified by boresightRotType with respect to detector.
    * For example if boresightRotType is SKY then at a rotation angle of 0 North is along detector Y axis
    * (never flipped) and East is along the detector X axis (flipped if necessary)
    */
    geom::Angle getBoresightRotAngle() const { return _boresightRotAngle; }

    /// get rotation type of boresightRotAngle
    RotType getRotType() const { return _rotType; }

    /// get observatory longitude, latitude and elevation
    coord::Observatory getObservatory() const { return _observatory; }

    /// get basic weather information
    coord::Weather getWeather() const { return _weather; }

    bool isPersistable() const { return true; }

    // get the local sidereal time on the meridian (equivalent, but not equal, to Local Mean Sidereal Time)
    geom::Angle getLocalEra() const;

    // get hour angle at the boresight
    geom::Angle getBoresightHourAngle() const;

protected:

    virtual std::string getPersistenceName() const;

    virtual void write(OutputArchiveHandle & handle) const;

private:
    table::RecordId _exposureId;
    double _exposureTime;
    double _darkTime;
    daf::base::DateTime _date;
    double _ut1;
    geom::Angle _era;
    coord::IcrsCoord _boresightRaDec;
    coord::Coord _boresightAzAlt;
    double _boresightAirmass;
    geom::Angle _boresightRotAngle;
    RotType _rotType;
    coord::Observatory _observatory;
    coord::Weather _weather;
};

namespace detail {

/**
 * Set FITS metadata from a VisitInfo
 *
 * @param[in,out] metadata  FITS keyword metadata to set
 * @param[in] visitInfo  instance of VisitInfo from which to set metadata
 */
void setVisitInfoMetadata(daf::base::PropertyList &metadata, VisitInfo const &visitInfo);

/**
 * Remove VisitInfo-related keywords from the metadata
 *
 * @param[in,out] metadata  FITS keyword metadata
 *
 * @return Number of keywords stripped
 */
int stripVisitInfoKeywords(daf::base::PropertySet & metadata);

} // lsst::afw::image::detail

}}} // lsst::afw::image

#endif // !LSST_AFW_IMAGE_VISITINFO_H_INCLUDED

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
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>

#include "boost/algorithm/string/trim.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/aggregates.h"  // for CoordKey
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/misc.h"  // for RecordId
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"  // needed, but why?
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/image/VisitInfo.h"

using lsst::daf::base::DateTime;

namespace lsst { namespace afw { namespace image {

/// the following persistence-related code emulates that in Calib.cc

namespace {

auto const nan = std::numeric_limits<double>::quiet_NaN();
auto const nanAngle = nan*geom::radians;

/**
 * Get a specified double from a PropertySet, or nan if not present
 *
 * @param[in] metadata  metadata to get
 * @param[in] key  key name; the associated value must be of type double if the key exists
 * @return value of metadata for the specified key, as a double,
 *   with a value of nan if the key is not present
 */
double getDouble(daf::base::PropertySet const & metadata, std::string const & key) {
    return metadata.exists(key) ? metadata.getAsDouble(key) : nan;
}

/**
 * Get a specified angle, as a float in degrees, from a PropertySet, or nan if not present
 *
 * @param[in] metadata  metadata to get
 * @param[in] key  key name; the associated value is treated as an angle in degrees, if it exists
 * @return value of metadata for the specified key, as an lsst::afw::geom::Angle,
 *   with a value of nan if the key is not present
 */
geom::Angle getAngle(daf::base::PropertySet const & metadata, std::string const & key) {
    return getDouble(metadata, key)*geom::degrees;
}

/**
 * Set a specified double in a PropertySet, if value is finite
 *
 * @param[in,out] metadata  metadata to set
 * @param[in] key  name of key to set
 * @param[in] value  value of key
 * @return true if item set, false otherwise
 */
bool setDouble(
    daf::base::PropertySet & metadata,
    std::string const & key,
    double value,
    std::string const & comment
) {
    if (std::isfinite(value)) {
        metadata.set(key, value);
        return true;
    }
    return false;
}

/**
 * Set a specified angle in a PropertySet, in degrees, if angle is finite
 *
 * @param[in,out] metadata  metadata to set
 * @param[in] key  name of key to set
 * @param[in] angle  value of key
 * @return true if item set, false otherwise
 */
bool setAngle(
    daf::base::PropertySet & metadata,
    std::string const & key,
    geom::Angle const & angle,
    std::string const & comment
) {
    return setDouble(metadata, key, angle.asDegrees(), comment);
}

/**
 * Get rotation type as a string to use for a for FITS keyword value, given an enum
 *
 * @throw lsst::pex::exceptions::RuntimeError if the enum is not recognized
 */
std::string rotTypeStrFromEnum(RotType rotType) {
    switch(rotType) {
        case RotType::UNKNOWN:
            return "UNKNOWN";
        case RotType::SKY:
            return "SKY";
        case RotType::HORIZON:
            return "HORIZON";
        case RotType::MOUNT:
            return "MOUNT";
    }
    std::ostringstream os;
    os << "Unknown RotType enum: " << static_cast<int>(rotType);
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, os.str());
}

/**
 * Get rotation type as an enum, given the FITS keyword value string equivalent
 *
 * @throw lsst::pex::exceptions::RuntimeError if the string is not recognized
 */
RotType rotTypeEnumFromStr(std::string const & rotTypeName) {
    if (rotTypeName == "UNKNOWN") {
        return RotType::UNKNOWN;
    } else if (rotTypeName == "SKY") {
        return RotType::SKY;
    } else if (rotTypeName == "HORIZON") {
        return RotType::HORIZON;
    } else if (rotTypeName == "MOUNT") {
        return RotType::MOUNT;
    }
    std::ostringstream os;
    os << "Unknown RotType name: \"" << rotTypeName << "\"";
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, os.str());
}

class VisitInfoSchema {
public:
    table::Schema schema;
    table::Key<table::RecordId> exposureId;
    table::Key<double> exposureTime;
    table::Key<double> darkTime;
    table::Key<std::int64_t> tai;
    table::Key<double> ut1;
    table::Key<geom::Angle> era;
    table::CoordKey  boresightRaDec;
    table::Key<geom::Angle> boresightAzAlt_az;
    table::Key<geom::Angle> boresightAzAlt_alt;
    table::Key<double> boresightAirmass;
    table::Key<geom::Angle> boresightRotAngle;
    table::Key<int> rotType;
    // observatory data
    table::Key<geom::Angle> latitude;
    table::Key<geom::Angle> longitude;
    table::Key<double> elevation;
    // weather data
    table::Key<double> airTemperature;
    table::Key<double> airPressure;
    table::Key<double> humidity;

    static VisitInfoSchema const & get() {
        static VisitInfoSchema instance;
        return instance;
    }

    // No copying
    VisitInfoSchema (const VisitInfoSchema&) = delete;
    VisitInfoSchema& operator=(const VisitInfoSchema&) = delete;

    // No moving
    VisitInfoSchema (VisitInfoSchema&&) = delete;
    VisitInfoSchema& operator=(VisitInfoSchema&&) = delete;

private:
    VisitInfoSchema() :
        schema(),
        exposureId(schema.addField<table::RecordId>("exposureid", "exposure ID", "")),
        exposureTime(schema.addField<double>("exposuretime", "exposure duration", "s")),
        darkTime(schema.addField<double>("darktime", "time from CCD flush to readout", "s")),
        tai(schema.addField<std::int64_t>("tai",
            "TAI date and time at middle of exposure as nsec from unix epoch", "nsec")),
        ut1(schema.addField<double>("ut1", "UT1 date and time at middle of exposure", "MJD")),
        era(schema.addField<geom::Angle>("era", "earth rotation angle at middle of exposure", "")),
        boresightRaDec(table::CoordKey::addFields(schema, "boresightradec",
            "sky position of boresight at middle of exposure")),
        // CoordKey is only for IcrsCoord; we have no equivalent for other kinds of coordinates
        // so use a pair of Angle fields to save boresightAzAlt
        boresightAzAlt_az(schema.addField<geom::Angle>("boresightazalt_az",
            "refracted apparent topocentric position of boresight at middle of exposure", "")),
        boresightAzAlt_alt(schema.addField<geom::Angle>("boresightazalt_alt",
            "refracted apparent topocentric position of boresight at middle of exposure", "")),
        boresightAirmass(schema.addField<double>("boresightairmass",
            "airmass at boresight, relative to zenith at sea level", "")),
        boresightRotAngle(schema.addField<geom::Angle>("boresightrotangle",
            "rotation angle at boresight at middle of exposure", "")),
        rotType(schema.addField<int>("rottype",
            "rotation type; see VisitInfo.getRotType for details", "MJD")),
        // observatory data
        latitude(schema.addField<geom::Angle>("latitude",
            "latitude of telescope (+ is east of Greenwich)", "")),
        longitude(schema.addField<geom::Angle>("longitude", "longitude of telescope", "")),
        elevation(schema.addField<double>("elevation", "elevation of telescope", "")),
        // weather data
        airTemperature(schema.addField<double>("airtemperature", "air temperature", "C")),
        airPressure(schema.addField<double>("airpressure", "air pressure", "Pascal")),
        humidity(schema.addField<double>("humidity", "humidity (%)", ""))
    {
        schema.getCitizen().markPersistent();
    }
};

class VisitInfoFactory : public table::io::PersistableFactory {
public:

    virtual PTR(table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        VisitInfoSchema const & keys = VisitInfoSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        table::BaseRecord const & record = catalogs.front().front();
        PTR(VisitInfo) result(new VisitInfo(
            record.get(keys.exposureId),
            record.get(keys.exposureTime),
            record.get(keys.darkTime),
            ::DateTime(record.get(keys.tai), ::DateTime::TAI),
            record.get(keys.ut1),
            record.get(keys.era),
            record.get(keys.boresightRaDec),
            coord::Coord(record.get(keys.boresightAzAlt_az), record.get(keys.boresightAzAlt_alt)),
            record.get(keys.boresightAirmass),
            record.get(keys.boresightRotAngle),
            static_cast<RotType>(record.get(keys.rotType)),
            coord::Observatory(
                record.get(keys.longitude),
                record.get(keys.latitude),
                record.get(keys.elevation)
            ),
            coord::Weather(
                record.get(keys.airTemperature),
                record.get(keys.airPressure),
                record.get(keys.humidity)
            )
        ));
        return result;
    }

    explicit VisitInfoFactory(std::string const & name) : table::io::PersistableFactory(name) {}

};

std::string getVisitInfoPersistenceName() { return "VisitInfo"; }

VisitInfoFactory registration(getVisitInfoPersistenceName());

} // anonymous

namespace detail {

int stripVisitInfoKeywords(daf::base::PropertySet & metadata) {
    int nstripped = 0;

    std::vector<std::string> keyList = {
        "EXPID", "EXPTIME", "DARKTIME", "DATE-AVG", "TIMESYS", "TIME-MID", "MJD-AVG-UT1", "AVG-ERA",
        "BORE-RA", "BORE-DEC","BORE-AZ", "BORE-ALT", "BORE-AIRMASS", "BORE-ROTANG", "ROTTYPE",
        "OBS-LONG", "OBS-LAT", "OBS-ELEV",
        "AIRTEMP", "AIRPRESS", "HUMIDITY"
    };
    for (auto&& key : keyList) {
        if (metadata.exists(key)) {
            metadata.remove(key);
            nstripped++;
        }
    }
    return nstripped;
}

void setVisitInfoMetadata(daf::base::PropertyList &metadata, VisitInfo const &visitInfo) {
    if (visitInfo.getExposureId() != 0) {
        metadata.set("EXPID", visitInfo.getExposureId());
    }
    setDouble(metadata, "EXPTIME", visitInfo.getExposureTime(), "Exposure time (sec)");
    setDouble(metadata, "DARKTIME", visitInfo.getDarkTime(), "Time from CCD flush to readout (sec)");
    if (visitInfo.getDate().isValid()) {
        metadata.set("DATE-AVG", visitInfo.getDate().toString(::DateTime::TAI),
            "TAI date at middle of observation");
        metadata.set("TIMESYS", "TAI");
    }
    setDouble(metadata, "MJD-AVG-UT1", visitInfo.getUt1(), "UT1 MJD date at ctr of obs");
    setAngle(metadata, "AVG-ERA", visitInfo.getEra(), "Earth rot ang at ctr of obs (deg)");
    auto boresightRaDec = visitInfo.getBoresightRaDec();
    setAngle(metadata, "BORE-RA", boresightRaDec[0], "ICRS RA (deg) at boresight");
    setAngle(metadata, "BORE-DEC", boresightRaDec[1], "ICRS Dec (deg) at boresight");
    auto boresightAzAlt = visitInfo.getBoresightAzAlt();
    setAngle(metadata, "BORE-AZ", boresightAzAlt[0],  "Refr app topo az (deg) at bore");
    setAngle(metadata, "BORE-ALT", boresightAzAlt[1], "Refr app topo alt (deg) at bore");
    setDouble(metadata, "BORE-AIRMASS", visitInfo.getBoresightAirmass(), "Airmass at boresight");
    setAngle(metadata, "BORE-ROTANG", visitInfo.getBoresightRotAngle(), "Rotation angle (deg) at boresight");
    metadata.set("ROTTYPE", rotTypeStrFromEnum(visitInfo.getRotType()), "Type of rotation angle");
    auto observatory = visitInfo.getObservatory();
    setAngle(metadata, "OBS-LONG", observatory.getLongitude(), "Telescope longitude (+E, deg)");
    setAngle(metadata, "OBS-LAT", observatory.getLatitude(), "Telescope latitude (deg)");
    setDouble(metadata, "OBS-ELEV", observatory.getElevation(), "Telescope elevation (m)");
    auto weather = visitInfo.getWeather();
    setDouble(metadata, "AIRTEMP", weather.getAirTemperature(), "Outside air temperature (C)");
    setDouble(metadata, "AIRPRESS", weather.getAirPressure() , "Outdoor air pressure (P)");
    setDouble(metadata, "HUMIDITY", weather.getHumidity(), "Relative humidity (%)");
}

}  // lsst::afw::image::detail


VisitInfo::VisitInfo(daf::base::PropertySet const & metadata) :
    _exposureId(0),
    _exposureTime(nan),  // don't use getDouble because str values are also accepted
    _darkTime(getDouble(metadata, "DARKTIME")),
    _date(),
    _ut1(getDouble(metadata, "MJD-AVG-UT1")),
    _era(getAngle(metadata, "AVG-ERA")),
    _boresightRaDec(coord::IcrsCoord(getAngle(metadata, "BORE-RA"), getAngle(metadata, "BORE-DEC"))),
    _boresightAzAlt(coord::Coord(getAngle(metadata, "BORE-AZ"), getAngle(metadata, "BORE-ALT"))),
    _boresightAirmass(getDouble(metadata, "BORE-AIRMASS")),
    _boresightRotAngle(getAngle(metadata, "BORE-ROTANG")),
    _rotType(RotType::UNKNOWN),
    _observatory(
        getAngle(metadata, "OBS-LONG"),
        getAngle(metadata, "OBS-LAT"),
        getDouble(metadata, "OBS-ELEV")),
    _weather(
        getDouble(metadata, "AIRTEMP"),
        getDouble(metadata, "AIRPRESS"),
        getDouble(metadata, "HUMIDITY"))
{
    auto key = "EXPID";
    if (metadata.exists(key)) {
        _exposureId = metadata.getAsInt64(key);
    }

    key = "EXPTIME";
    if (metadata.exists(key)) {
        try {
            _exposureTime = metadata.getAsDouble(key);
        } catch (lsst::pex::exceptions::TypeError & err) {
            // some old exposures have EXPTIME stored as a string
            std::string exptimeStr = metadata.getAsString(key);
            _exposureTime = std::stod(exptimeStr);
        }
    }

    key = "DATE-AVG";
    if (metadata.exists(key)) {
        if (metadata.exists("TIMESYS")) {
            auto timesysName = boost::algorithm::trim_right_copy(metadata.getAsString("TIMESYS"));
            if (timesysName != "TAI") {
                // rather than try to deal with all the possible choices, which requires
                // appending or deleting a "Z", depending on the time system, just give up.
                // VisitInfo should be used on FITS headers that have been sanitized!
                std::ostringstream os;
                os << "TIMESYS = \"" << timesysName <<
                    "\"; VisitInfo requires TIMESYS to exist and to equal \"TAI\"";
                throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, os.str());
            }
        } else {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                "TIMESYS not found; VistitInfo requires TIMESYS to exist and to equal \"TAI\"");
        }
        _date = ::DateTime(boost::algorithm::trim_right_copy(metadata.getAsString(key)), ::DateTime::TAI);
    } else {
        // DATE-AVG not found. For backwards compatibility look for TIME-MID, an outdated LSST keyword
        // whose time system was UTC, despite a FITS comment claiming it was TAI. Ignore TIMESYS.
        key = "TIME-MID";
        if (metadata.exists(key)) {
            _date = ::DateTime(boost::algorithm::trim_right_copy(metadata.getAsString(key)), ::DateTime::UTC);
        }
    }

    key = "ROTTYPE";
    if (metadata.exists(key)) {
        _rotType = rotTypeEnumFromStr(metadata.getAsString(key));
    }
}

bool VisitInfo::operator==(VisitInfo const & other) const {
    return _exposureId == other.getExposureId()
        && _exposureTime == other.getExposureTime()
        && _darkTime == other.getDarkTime()
        && _date == other.getDate()
        && _ut1 == other.getUt1()
        && _era == other.getEra()
        && _boresightRaDec == other.getBoresightRaDec()
        && _boresightAzAlt == other.getBoresightAzAlt()
        && _boresightAirmass == other.getBoresightAirmass()
        && _boresightRotAngle == other.getBoresightRotAngle()
        && _rotType == other.getRotType()
        && _observatory == other.getObservatory()
        && _weather == other.getWeather();
}

std::string VisitInfo::getPersistenceName() const {
    return getVisitInfoPersistenceName();
}

void VisitInfo::write(OutputArchiveHandle & handle) const {
    VisitInfoSchema const & keys = VisitInfoSchema::get();
    table::BaseCatalog cat = handle.makeCatalog(keys.schema);
    PTR(table::BaseRecord) record = cat.addNew();
    record->set(keys.exposureId, getExposureId());
    record->set(keys.exposureTime, getExposureTime());
    record->set(keys.darkTime, getDarkTime());
    record->set(keys.tai, getDate().nsecs(::DateTime::TAI));
    record->set(keys.ut1, getUt1());
    record->set(keys.era, getEra());
    record->set(keys.boresightRaDec, getBoresightRaDec());
    auto boresightAzAlt = getBoresightAzAlt();
    record->set(keys.boresightAzAlt_az, boresightAzAlt[0]);
    record->set(keys.boresightAzAlt_alt, boresightAzAlt[1]);
    record->set(keys.boresightAirmass, getBoresightAirmass());
    record->set(keys.boresightRotAngle, getBoresightRotAngle());
    record->set(keys.rotType, static_cast<int>(getRotType()));
    auto observatory = getObservatory();
    record->set(keys.latitude, observatory.getLatitude());
    record->set(keys.longitude, observatory.getLongitude());
    record->set(keys.elevation, observatory.getElevation());
    auto weather = getWeather();
    record->set(keys.airTemperature, weather.getAirTemperature());
    record->set(keys.airPressure, weather.getAirPressure());
    record->set(keys.humidity, weather.getHumidity());
    handle.saveCatalog(cat);
}

geom::Angle VisitInfo::getLocalEra() const {
    return getEra() + getObservatory().getLongitude();
}

geom::Angle VisitInfo::getBoresightHourAngle() const {
    return getLocalEra() - getBoresightRaDec()[0];
}

}}}  // namespace

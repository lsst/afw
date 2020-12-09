// -*- LSST-C++ -*- // fixed format comment for emacs
/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/afw/image/ExposureInfo.h"
#include "lsst/afw/image/PhotoCalib.h"
#include "lsst/afw/geom/polygon/Polygon.h"
#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/afw/image/ApCorrMap.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/image/TransmissionCurve.h"
#include "lsst/afw/fits.h"

using namespace std::string_literals;

namespace {
LOG_LOGGER _log = LOG_GET("afw.image.ExposureInfo");

using MapClass = lsst::afw::image::detail::StorableMap;
}  // namespace

namespace lsst {
namespace afw {
namespace image {

// Important: do *not* change the keys' strings; doing so will break compatibility with old FITS files

typehandling::Key<std::string, std::shared_ptr<geom::SkyWcs const>> const ExposureInfo::KEY_WCS =
        typehandling::makeKey<std::shared_ptr<geom::SkyWcs const>>("SKYWCS"s);

bool ExposureInfo::hasWcs() const { return hasComponent(KEY_WCS); }
std::shared_ptr<geom::SkyWcs const> ExposureInfo::getWcs() const { return getComponent(KEY_WCS); }
void ExposureInfo::setWcs(std::shared_ptr<geom::SkyWcs const> wcs) { setComponent(KEY_WCS, wcs); }

typehandling::Key<std::string, std::shared_ptr<detection::Psf const>> const ExposureInfo::KEY_PSF =
        typehandling::makeKey<std::shared_ptr<detection::Psf const>>("PSF"s);

bool ExposureInfo::hasPsf() const { return hasComponent(KEY_PSF); }
std::shared_ptr<detection::Psf const> ExposureInfo::getPsf() const { return getComponent(KEY_PSF); }
void ExposureInfo::setPsf(std::shared_ptr<detection::Psf const> psf) { setComponent(KEY_PSF, psf); }

typehandling::Key<std::string, std::shared_ptr<PhotoCalib const>> const ExposureInfo::KEY_PHOTO_CALIB =
        typehandling::makeKey<std::shared_ptr<PhotoCalib const>>("PHOTOCALIB"s);

bool ExposureInfo::hasPhotoCalib() const { return hasComponent(KEY_PHOTO_CALIB); }
std::shared_ptr<PhotoCalib const> ExposureInfo::getPhotoCalib() const {
    return getComponent(KEY_PHOTO_CALIB);
}
void ExposureInfo::setPhotoCalib(std::shared_ptr<PhotoCalib const> photoCalib) {
    setComponent(KEY_PHOTO_CALIB, photoCalib);
}

typehandling::Key<std::string, std::shared_ptr<cameraGeom::Detector const>> const ExposureInfo::KEY_DETECTOR =
        typehandling::makeKey<std::shared_ptr<cameraGeom::Detector const>>("DETECTOR"s);

bool ExposureInfo::hasDetector() const { return hasComponent(KEY_DETECTOR); }
std::shared_ptr<cameraGeom::Detector const> ExposureInfo::getDetector() const {
    return getComponent(KEY_DETECTOR);
}
void ExposureInfo::setDetector(std::shared_ptr<cameraGeom::Detector const> detector) {
    setComponent(KEY_DETECTOR, detector);
}

typehandling::Key<std::string, std::shared_ptr<geom::polygon::Polygon const>> const
        ExposureInfo::KEY_VALID_POLYGON =
                typehandling::makeKey<std::shared_ptr<geom::polygon::Polygon const>>("VALID_POLYGON"s);

bool ExposureInfo::hasValidPolygon() const { return hasComponent(KEY_VALID_POLYGON); }
std::shared_ptr<geom::polygon::Polygon const> ExposureInfo::getValidPolygon() const {
    return getComponent(KEY_VALID_POLYGON);
}
void ExposureInfo::setValidPolygon(std::shared_ptr<geom::polygon::Polygon const> polygon) {
    setComponent(KEY_VALID_POLYGON, polygon);
}

typehandling::Key<std::string, std::shared_ptr<CoaddInputs const>> const ExposureInfo::KEY_COADD_INPUTS =
        typehandling::makeKey<std::shared_ptr<CoaddInputs const>>("COADD_INPUTS"s);

bool ExposureInfo::hasCoaddInputs() const { return hasComponent(KEY_COADD_INPUTS); }
void ExposureInfo::setCoaddInputs(std::shared_ptr<CoaddInputs const> coaddInputs) {
    setComponent(KEY_COADD_INPUTS, coaddInputs);
}
std::shared_ptr<CoaddInputs const> ExposureInfo::getCoaddInputs() const {
    return getComponent(KEY_COADD_INPUTS);
}

typehandling::Key<std::string, std::shared_ptr<ApCorrMap const>> const ExposureInfo::KEY_AP_CORR_MAP =
        typehandling::makeKey<std::shared_ptr<ApCorrMap const>>("AP_CORR_MAP"s);

bool ExposureInfo::hasApCorrMap() const { return hasComponent(KEY_AP_CORR_MAP); }
std::shared_ptr<ApCorrMap const> ExposureInfo::getApCorrMap() const { return getComponent(KEY_AP_CORR_MAP); }
void ExposureInfo::setApCorrMap(std::shared_ptr<ApCorrMap const> apCorrMap) {
    setComponent(KEY_AP_CORR_MAP, apCorrMap);
}

typehandling::Key<std::string, std::shared_ptr<TransmissionCurve const>> const
        ExposureInfo::KEY_TRANSMISSION_CURVE =
                typehandling::makeKey<std::shared_ptr<TransmissionCurve const>>("TRANSMISSION_CURVE"s);

bool ExposureInfo::hasTransmissionCurve() const { return hasComponent(KEY_TRANSMISSION_CURVE); }
std::shared_ptr<TransmissionCurve const> ExposureInfo::getTransmissionCurve() const {
    return getComponent(KEY_TRANSMISSION_CURVE);
}
void ExposureInfo::setTransmissionCurve(std::shared_ptr<TransmissionCurve const> tc) {
    setComponent(KEY_TRANSMISSION_CURVE, tc);
}

// Compatibility code defined inside ExposureFitsReader.cc, to be removed in DM-27177
std::shared_ptr<FilterLabel> makeFilterLabel(Filter const& filter);
Filter makeFilter(FilterLabel const& label);

Filter ExposureInfo::getFilter() const {
    std::shared_ptr<FilterLabel const> label = getFilterLabel();
    if (label) {
        return makeFilter(*label);
    } else {
        // Old exposures always had a Filter, even if only the default
        return Filter();
    }
}

void ExposureInfo::setFilter(Filter const& filter) { setFilterLabel(makeFilterLabel(filter)); }

typehandling::Key<std::string, std::shared_ptr<FilterLabel const>> const ExposureInfo::KEY_FILTER =
        typehandling::makeKey<std::shared_ptr<FilterLabel const>>("FILTER"s);
bool ExposureInfo::hasFilterLabel() const { return hasComponent(KEY_FILTER); }
std::shared_ptr<FilterLabel const> ExposureInfo::getFilterLabel() const { return getComponent(KEY_FILTER); }
void ExposureInfo::setFilterLabel(std::shared_ptr<FilterLabel const> label) {
    setComponent(KEY_FILTER, label);
}

int ExposureInfo::getFitsSerializationVersion() {
    // Version history:
    // unversioned and 0: photometric calibration via Calib, WCS via SkyWcs using AST.
    // 1: photometric calibration via PhotoCalib, generic components
    // 2: remove Filter, replaced with (generic) FilterLabel
    static int const version = 2;
    return version;
}

std::string const& ExposureInfo::getFitsSerializationVersionName() {
    static std::string const versionName("EXPINFO_V");
    return versionName;
}

// Clone various components; defined here so that we don't have to expose their insides in Exposure.h

std::shared_ptr<ApCorrMap> ExposureInfo::_cloneApCorrMap(std::shared_ptr<ApCorrMap const> apCorrMap) {
    if (apCorrMap) {
        return std::make_shared<ApCorrMap>(*apCorrMap);
    }
    return std::shared_ptr<ApCorrMap>();
}

ExposureInfo::ExposureInfo(std::shared_ptr<geom::SkyWcs const> const& wcs,
                           std::shared_ptr<detection::Psf const> const& psf,
                           std::shared_ptr<PhotoCalib const> const& photoCalib,
                           std::shared_ptr<cameraGeom::Detector const> const& detector,
                           std::shared_ptr<geom::polygon::Polygon const> const& polygon, Filter const& filter,
                           std::shared_ptr<daf::base::PropertySet> const& metadata,
                           std::shared_ptr<CoaddInputs> const& coaddInputs,
                           std::shared_ptr<ApCorrMap> const& apCorrMap,
                           std::shared_ptr<image::VisitInfo const> const& visitInfo,
                           std::shared_ptr<TransmissionCurve const> const& transmissionCurve)
        : _metadata(metadata ? metadata
                             : std::shared_ptr<daf::base::PropertySet>(new daf::base::PropertyList())),
          _visitInfo(visitInfo),
          _components(std::make_unique<MapClass>()) {
    static Filter const DEFAULT;
    // Avoid putting dummy filters into the FilterLabel store. getFilter()
    // will preserve old default behavior.
    // Default filter has id=UNKNOWN, but others do too.
    if (filter.getId() != DEFAULT.getId() || filter.getName() != DEFAULT.getName()) {
        setFilter(filter);
    }
    setWcs(wcs);
    setPsf(psf);
    setPhotoCalib(photoCalib);
    setDetector(detector);
    setValidPolygon(polygon);
    setCoaddInputs(coaddInputs);
    setApCorrMap(_cloneApCorrMap(apCorrMap));
    setTransmissionCurve(transmissionCurve);
}

ExposureInfo::ExposureInfo(ExposureInfo const& other) : ExposureInfo(other, false) {}

// Delegate to copy-constructor for backwards compatibility
ExposureInfo::ExposureInfo(ExposureInfo&& other) : ExposureInfo(other) {}

ExposureInfo::ExposureInfo(ExposureInfo const& other, bool copyMetadata)
        : _metadata(other._metadata),
          _visitInfo(other._visitInfo),
          // ExposureInfos can (historically) share objects, but should each have their own pointers to them
          _components(std::make_unique<MapClass>(*(other._components))) {
    if (copyMetadata) _metadata = _metadata->deepCopy();
}

ExposureInfo& ExposureInfo::operator=(ExposureInfo const& other) {
    if (&other != this) {
        _metadata = other._metadata;
        _visitInfo = other._visitInfo;
        // ExposureInfos can (historically) share objects, but should each have their own pointers to them
        _components = std::make_unique<MapClass>(*(other._components));
    }
    return *this;
}
// Delegate to copy-assignment for backwards compatibility
ExposureInfo& ExposureInfo::operator=(ExposureInfo&& other) { return *this = other; }

void ExposureInfo::initApCorrMap() { setApCorrMap(std::make_shared<ApCorrMap>()); }

ExposureInfo::~ExposureInfo() = default;

int ExposureInfo::_addToArchive(FitsWriteData& data, table::io::Persistable const& object, std::string key,
                                std::string comment) {
    int componentId = data.archive.put(object);
    data.metadata->set(key, componentId, comment);
    return componentId;
}

int ExposureInfo::_addToArchive(FitsWriteData& data,
                                std::shared_ptr<table::io::Persistable const> const& object, std::string key,
                                std::string comment) {
    // Don't delegate to Persistable const& version because OutputArchive::put
    // has special handling of shared_ptr
    int componentId = data.archive.put(object);
    data.metadata->set(key, componentId, comment);
    return componentId;
}

// Standardized strings for _startWriteFits
namespace {
std::string _getOldHeaderKey(std::string mapKey) { return mapKey + "_ID"; }
std::string _getNewHeaderKey(std::string mapKey) { return "ARCHIVE_ID_" + mapKey; }

std::string _getHeaderComment(std::string mapKey) {
    return "archive ID for generic component '" + mapKey + "'";
}
}  // namespace

ExposureInfo::FitsWriteData ExposureInfo::_startWriteFits(lsst::geom::Point2I const& xy0) const {
    FitsWriteData data;

    data.metadata.reset(new daf::base::PropertyList());
    data.imageMetadata.reset(new daf::base::PropertyList());
    data.maskMetadata = data.imageMetadata;
    data.varianceMetadata = data.imageMetadata;

    data.metadata->combine(getMetadata());

    data.metadata->set(getFitsSerializationVersionName(), getFitsSerializationVersion());

    // In the future, we might not have exactly three image HDUs, but we always do right now,
    // so 0=primary, 1=image, 2=mask, 3=variance, 4+=archive
    //
    // Historically the AR_HDU keyword was 1-indexed (see RFC-304), and to maintain file compatibility
    // this is still the case so we're setting AR_HDU to 5 == 4 + 1
    //
    data.metadata->set("AR_HDU", 5, "HDU (1-indexed) containing the archive used to store ancillary objects");
    for (auto const& keyValue : *_components) {
        std::string const& key = keyValue.first.getId();
        std::shared_ptr<typehandling::Storable const> const& object = keyValue.second;

        if (object && object->isPersistable()) {
            std::string comment = _getHeaderComment(key);
            // Store archive ID in two header keys:
            //     - old-style key for backwards compatibility,
            //     - and new-style key because it's much safer to parse
            int id = _addToArchive(data, object, _getOldHeaderKey(key), comment);
            data.metadata->set(_getNewHeaderKey(key), id, comment);
        }
    }

    // LSST convention is that Wcs is in pixel coordinates (i.e relative to bottom left
    // corner of parent image, if any). The Wcs/Fits convention is that the Wcs is in
    // image coordinates. When saving an image we convert from pixel to index coordinates.
    // In the case where this image is a parent image, the reference pixels are unchanged
    // by this transformation
    if (hasWcs() && getWcs()->isFits()) {
        // Try to save the WCS as FITS-WCS metadata; if an exact representation
        // is not possible then skip it
        auto shift = lsst::geom::Extent2D(lsst::geom::Point2I(0, 0) - xy0);
        auto newWcs = getWcs()->copyAtShiftedPixelOrigin(shift);
        std::shared_ptr<daf::base::PropertyList> wcsMetadata;
        try {
            wcsMetadata = newWcs->getFitsMetadata(true);
        } catch (pex::exceptions::RuntimeError const&) {
            // cannot represent this WCS as FITS-WCS; don't write its metadata
        }
        if (wcsMetadata) {
            data.imageMetadata->combine(newWcs->getFitsMetadata(true));
        }
    }

    // For the sake of ds9, store _x0 and _y0 as -LTV1, -LTV2.
    // This is in addition to saving _x0 and _y0 as WCS A, which is done elsewhere
    // and is what LSST uses to read _x0 and _y0.
    // LTV is a convention used by STScI (see \S2.6.2 of HST Data Handbook for STIS, version 5.0
    // http://www.stsci.edu/hst/stis/documents/handbooks/currentDHB/ch2_stis_data7.html#429287)
    // and recognized by ds9.
    data.imageMetadata->set("LTV1", static_cast<double>(-xy0.getX()));
    data.imageMetadata->set("LTV2", static_cast<double>(-xy0.getY()));

    if (hasDetector()) {
        data.metadata->set("DETNAME", getDetector()->getName());
        data.metadata->set("DETSER", getDetector()->getSerial());
    }

    auto visitInfoPtr = getVisitInfo();
    if (visitInfoPtr) {
        detail::setVisitInfoMetadata(*(data.metadata), *visitInfoPtr);
    }

    return data;
}

void ExposureInfo::_finishWriteFits(fits::Fits& fitsfile, FitsWriteData const& data) const {
    data.archive.writeFits(fitsfile);
}

}  // namespace image
}  // namespace afw
}  // namespace lsst

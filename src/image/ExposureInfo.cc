// -*- LSST-C++ -*- // fixed format comment for emacs
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

#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/afw/image/ExposureInfo.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/geom/polygon/Polygon.h"
#include "lsst/afw/image/ApCorrMap.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/image/TransmissionCurve.h"
#include "lsst/afw/fits.h"

namespace {
LOG_LOGGER _log = LOG_GET("afw.image.ExposureInfo");
}

namespace lsst {
namespace afw {
namespace image {

namespace {

// Return an int value from a PropertySet if it exists and remove it, or return 0.
int popInt(daf::base::PropertySet& metadata, std::string const& name) {
    int r = 0;
    if (metadata.exists(name)) {
        r = metadata.get<int>(name);
        metadata.remove(name);
    }
    return r;
}

}  // anonymous

// Clone various components; defined here so that we don't have to expose their insides in Exposure.h

std::shared_ptr<Calib> ExposureInfo::_cloneCalib(std::shared_ptr<Calib const> calib) {
    if (calib) return std::shared_ptr<Calib>(new Calib(*calib));
    return std::shared_ptr<Calib>();
}

std::shared_ptr<Wcs> ExposureInfo::_cloneWcs(std::shared_ptr<Wcs const> wcs) {
    if (wcs) return wcs->clone();
    return std::shared_ptr<Wcs>();
}

std::shared_ptr<ApCorrMap> ExposureInfo::_cloneApCorrMap(std::shared_ptr<ApCorrMap const> apCorrMap) {
    if (apCorrMap) {
        return std::make_shared<ApCorrMap>(*apCorrMap);
    }
    return std::shared_ptr<ApCorrMap>();
}

ExposureInfo::ExposureInfo(std::shared_ptr<Wcs const> const& wcs,
                           std::shared_ptr<detection::Psf const> const& psf,
                           std::shared_ptr<Calib const> const& calib,
                           std::shared_ptr<cameraGeom::Detector const> const& detector,
                           std::shared_ptr<geom::polygon::Polygon const> const& polygon, Filter const& filter,
                           std::shared_ptr<daf::base::PropertySet> const& metadata,
                           std::shared_ptr<CoaddInputs> const& coaddInputs,
                           std::shared_ptr<ApCorrMap> const& apCorrMap,
                           std::shared_ptr<image::VisitInfo const> const& visitInfo,
                           std::shared_ptr<TransmissionCurve> const & transmissionCurve)
        : _wcs(_cloneWcs(wcs)),
          _psf(std::const_pointer_cast<detection::Psf>(psf)),
          _calib(calib ? _cloneCalib(calib) : std::shared_ptr<Calib>(new Calib())),
          _detector(detector),
          _validPolygon(polygon),
          _filter(filter),
          _metadata(metadata ? metadata
                             : std::shared_ptr<daf::base::PropertySet>(new daf::base::PropertyList())),
          _coaddInputs(coaddInputs),
          _apCorrMap(_cloneApCorrMap(apCorrMap)),
          _visitInfo(visitInfo),
          _transmissionCurve(transmissionCurve)
{}

ExposureInfo::ExposureInfo(ExposureInfo const& other)
        : _wcs(_cloneWcs(other._wcs)),
          _psf(other._psf),
          _calib(_cloneCalib(other._calib)),
          _detector(other._detector),
          _validPolygon(other._validPolygon),
          _filter(other._filter),
          _metadata(other._metadata),
          _coaddInputs(other._coaddInputs),
          _apCorrMap(_cloneApCorrMap(other._apCorrMap)),
          _visitInfo(other._visitInfo),
          _transmissionCurve(other._transmissionCurve) {}

ExposureInfo::ExposureInfo(ExposureInfo const& other, bool copyMetadata)
        : _wcs(_cloneWcs(other._wcs)),
          _psf(other._psf),
          _calib(_cloneCalib(other._calib)),
          _detector(other._detector),
          _validPolygon(other._validPolygon),
          _filter(other._filter),
          _metadata(other._metadata),
          _coaddInputs(other._coaddInputs),
          _apCorrMap(_cloneApCorrMap(other._apCorrMap)),
          _visitInfo(other._visitInfo),
          _transmissionCurve(other._transmissionCurve) {
    if (copyMetadata) _metadata = _metadata->deepCopy();
}

ExposureInfo& ExposureInfo::operator=(ExposureInfo const& other) {
    if (&other != this) {
        _wcs = _cloneWcs(other._wcs);
        _psf = other._psf;
        _calib = _cloneCalib(other._calib);
        _detector = other._detector;
        _validPolygon = other._validPolygon;
        _filter = other._filter;
        _metadata = other._metadata;
        _coaddInputs = other._coaddInputs;
        _apCorrMap = _cloneApCorrMap(other._apCorrMap);
        _visitInfo = other._visitInfo;
        _transmissionCurve = other._transmissionCurve;
    }
    return *this;
}

void ExposureInfo::initApCorrMap() { _apCorrMap = std::make_shared<ApCorrMap>(); }

ExposureInfo::~ExposureInfo() {}

ExposureInfo::FitsWriteData ExposureInfo::_startWriteFits(afw::geom::Point2I const& xy0) const {
    FitsWriteData data;

    data.metadata.reset(new daf::base::PropertyList());
    data.imageMetadata.reset(new daf::base::PropertyList());
    data.maskMetadata = data.imageMetadata;
    data.varianceMetadata = data.imageMetadata;

    data.metadata->combine(getMetadata());

    // In the future, we might not have exactly three image HDUs, but we always do right now,
    // so 0=primary, 1=image, 2=mask, 3=variance, 4+=archive
    //
    // Historically the AR_HDU keyword was 1-indexed (see RFC-304), and to maintain file compatibility
    // this is still the case so we're setting AR_HDU to 5 == 4 + 1
    //
    data.metadata->set("AR_HDU", 5, "HDU (1-indexed) containing the archive used to store ancillary objects");
    if (hasCoaddInputs()) {
        int coaddInputsId = data.archive.put(getCoaddInputs());
        data.metadata->set("COADD_INPUTS_ID", coaddInputsId, "archive ID for coadd inputs catalogs");
    }
    if (hasApCorrMap()) {
        int apCorrMapId = data.archive.put(getApCorrMap());
        data.metadata->set("AP_CORR_MAP_ID", apCorrMapId, "archive ID for aperture correction map");
    }
    if (hasPsf() && getPsf()->isPersistable()) {
        int psfId = data.archive.put(getPsf());
        data.metadata->set("PSF_ID", psfId, "archive ID for the Exposure's main Psf");
    }
    if (hasWcs() && getWcs()->isPersistable()) {
        int wcsId = data.archive.put(getWcs());
        data.metadata->set("WCS_ID", wcsId, "archive ID for the Exposure's main Wcs");
    }
    if (hasValidPolygon() && getValidPolygon()->isPersistable()) {
        int polygonId = data.archive.put(getValidPolygon());
        data.metadata->set("VALID_POLYGON_ID", polygonId, "archive ID for the Exposure's valid polygon");
    }
    if (hasTransmissionCurve() && getTransmissionCurve()->isPersistable()) {
        int transmissionCurveId = data.archive.put(getTransmissionCurve());
        data.metadata->set("TRANSMISSION_CURVE_ID", transmissionCurveId,
                           "archive ID for the Exposure's transmission curve");
    }

    // LSST convention is that Wcs is in pixel coordinates (i.e relative to bottom left
    // corner of parent image, if any). The Wcs/Fits convention is that the Wcs is in
    // image coordinates. When saving an image we convert from pixel to index coordinates.
    // In the case where this image is a parent image, the reference pixels are unchanged
    // by this transformation
    if (hasWcs()) {
        std::shared_ptr<Wcs> newWcs = getWcs()->clone();  // Create a copy
        newWcs->shiftReferencePixel(-xy0.getX(), -xy0.getY());

        // We want the WCS to appear in all HDUs
        data.imageMetadata->combine(newWcs->getFitsMetadata());
    }

    // Store _x0 and _y0. If this exposure is a portion of a larger image, _x0 and _y0
    // indicate the origin (the position of the bottom left corner) of the sub-image with
    // respect to the origin of the parent image.
    // This is stored in the fits header using the LTV convention used by STScI
    //(see \S2.6.2 of HST Data Handbook for STIS, version 5.0
    // http://www.stsci.edu/hst/stis/documents/handbooks/currentDHB/ch2_stis_data7.html#429287).
    // This is not a fits standard keyword, but is recognised by ds9
    // LTV keywords use the opposite convention to the LSST, in that they represent
    // the position of the origin of the parent image relative to the origin of the sub-image.
    // _x0, _y0 >= 0, while LTV1 and LTV2 <= 0

    data.imageMetadata->set("LTV1", static_cast<double>(-xy0.getX()));
    data.imageMetadata->set("LTV2", static_cast<double>(-xy0.getY()));

    data.metadata->set("FILTER", getFilter().getName());
    if (hasDetector()) {
        data.metadata->set("DETNAME", getDetector()->getName());
        data.metadata->set("DETSER", getDetector()->getSerial());
    }

    auto visitInfoPtr = getVisitInfo();
    if (visitInfoPtr) {
        detail::setVisitInfoMetadata(*(data.metadata), *visitInfoPtr);
    }

    /*
     * We need to define these keywords properly! XXX
     */
    data.metadata->set("FLUXMAG0", getCalib()->getFluxMag0().first);
    data.metadata->set("FLUXMAG0ERR", getCalib()->getFluxMag0().second);

    return data;
}

void ExposureInfo::_finishWriteFits(fits::Fits& fitsfile, FitsWriteData const& data) const {
    data.archive.writeFits(fitsfile);
}

void ExposureInfo::_readFits(fits::Fits& fitsfile, std::shared_ptr<daf::base::PropertySet> metadata,
                             std::shared_ptr<daf::base::PropertySet> imageMetadata) {
    // true: strip keywords that are related to the created WCS from the input metadata
    _wcs = makeWcs(imageMetadata, true);

    if (!imageMetadata->exists("INHERIT")) {
        // New-style exposures put everything but the Wcs in the primary HDU, use
        // INHERIT keyword in the others.  For backwards compatibility, if we don't
        // find the INHERIT keyword, we ignore the primary HDU metadata and expect
        // everything to be in the image HDU metadata.  Note that we can't merge them,
        // because they're probably duplicates.
        metadata = imageMetadata;
    }

    _filter = Filter(metadata, true);
    detail::stripFilterKeywords(metadata);

    _visitInfo = std::shared_ptr<VisitInfo const>(new VisitInfo(*metadata));
    detail::stripVisitInfoKeywords(*metadata);

    std::shared_ptr<Calib> newCalib(new Calib(metadata));
    setCalib(newCalib);
    detail::stripCalibKeywords(metadata);

    int archiveHdu = popInt(*metadata, "AR_HDU");

    if (archiveHdu) {
        --archiveHdu;  // see note above in _startWriteFits;  AR_HDU is *one* indexed

        fitsfile.setHdu(archiveHdu);
        table::io::InputArchive archive = table::io::InputArchive::readFits(fitsfile);
        // Load the Psf and Wcs from the archive; id=0 results in a null pointer.
        // Note that the binary table Wcs, if present, clobbers the FITS header one,
        // because the former might be an approximation to something we can't represent
        // using the FITS WCS standard but can represent with binary tables.
        int psfId = popInt(*metadata, "PSF_ID");
        try {
            _psf = archive.get<detection::Psf>(psfId);
        } catch (pex::exceptions::NotFoundError& err) {
            LOGLS_WARN(_log, "Could not read PSF; setting to null: " << err.what());
        }
        int wcsId = popInt(*metadata, "WCS_ID");
        try {
            auto archiveWcs = archive.get<Wcs>(wcsId);
            if (archiveWcs) {
                _wcs = archiveWcs;
            } else {
                LOGLS_DEBUG(_log, "Null WCS seen in binary table");
            }
        } catch (pex::exceptions::NotFoundError& err) {
            auto msg = str(boost::format("Could not read WCS extension; setting to null: %s") % err.what());
            if (_wcs) {
                msg += " ; using WCS from FITS header";
            }
            LOGLS_WARN(_log, msg);
        }
        int coaddInputsId = popInt(*metadata, "COADD_INPUTS_ID");
        try {
            _coaddInputs = archive.get<CoaddInputs>(coaddInputsId);
        } catch (pex::exceptions::NotFoundError& err) {
            LOGLS_WARN(_log, "Could not read CoaddInputs; setting to null: " << err.what());
        }
        int apCorrMapId = popInt(*metadata, "AP_CORR_MAP_ID");
        try {
            _apCorrMap = archive.get<ApCorrMap>(apCorrMapId);
        } catch (pex::exceptions::NotFoundError& err) {
            LOGLS_WARN(_log, "Could not read ApCorrMap; setting to null: " << err.what());
        }
        int validPolygonId = popInt(*metadata, "VALID_POLYGON_ID");
        try {
            _validPolygon = archive.get<geom::polygon::Polygon>(validPolygonId);
        } catch (pex::exceptions::NotFoundError& err) {
            LOGLS_WARN(_log, "Could not read ValidPolygon; setting to null: " << err.what());
        }
        int transmissionCurveId = popInt(*metadata, "TRANSMISSION_CURVE_ID");
        try {
            _transmissionCurve = archive.get<TransmissionCurve>(transmissionCurveId);
        } catch (pex::exceptions::NotFoundError & err) {
            LOGLS_WARN(_log, "Could not read TransmissionCurve; setting to null: " << err.what());
        }
    }

    _metadata = metadata;
}
}
}
}  // namespace lsst::afw::image

/*
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

#include "lsst/log/Log.h"

#include "lsst/afw/image/Calib.h"
#include "lsst/afw/geom/polygon/Polygon.h"
#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/afw/image/ApCorrMap.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/image/TransmissionCurve.h"
#include "lsst/afw/image/ExposureFitsReader.h"

namespace lsst { namespace afw { namespace image {

namespace {

LOG_LOGGER _log = LOG_GET("afw.image.fits.ExposureFitsReader");

}  // anonymous

class ExposureFitsReader::MetadataReader {
public:

    MetadataReader(
        std::shared_ptr<daf::base::PropertyList> primaryMetadata,
        std::shared_ptr<daf::base::PropertyList> imageMetadata,
        lsst::geom::Point2I const & xy0
    ) {
        // Try to read WCS from image metadata, and if found, strip the keywords used
        try {
            wcs = afw::geom::makeSkyWcs(*imageMetadata, true);
        } catch (lsst::pex::exceptions::TypeError) {
            LOGLS_DEBUG(_log, "No WCS found in FITS metadata");
        }
        if (wcs && any(xy0.ne(lsst::geom::Point2I(0, 0)))) {
            wcs = wcs->copyAtShiftedPixelOrigin(lsst::geom::Extent2D(xy0));
        }

        // Strip LTV1, LTV2 from imageMetadata, because we don't use it internally
        imageMetadata->remove("LTV1");
        imageMetadata->remove("LTV2");

        if (!imageMetadata->exists("INHERIT")) {
            // New-style exposures put everything but the Wcs in the primary HDU, use
            // INHERIT keyword in the others.  For backwards compatibility, if we don't
            // find the INHERIT keyword, we ignore the primary HDU metadata and expect
            // everything to be in the image HDU metadata.  Note that we can't merge them,
            // because they're probably duplicates.
            metadata = imageMetadata;
        } else {
            metadata = primaryMetadata;
        }

        filter = Filter(metadata, true);
        detail::stripFilterKeywords(metadata);

        visitInfo = std::make_shared<VisitInfo>(*metadata);
        detail::stripVisitInfoKeywords(*metadata);

        calib = std::make_shared<Calib>(metadata);
        detail::stripCalibKeywords(metadata);

        // Strip MJD-OBS and DATE-OBS from metadata; those may be read by
        // either SkyWcs or VisitInfo or both, so neither can strip them.
        metadata->remove("MJD-OBS");
        metadata->remove("DATE-OBS");

        // Strip DETSER, DETNAME; these are added when writing an Exposure
        // with a Detector
        metadata->remove("DETNAME");
        metadata->remove("DETSER");
    }

    std::shared_ptr<daf::base::PropertyList> metadata;
    Filter filter;
    std::shared_ptr<afw::geom::SkyWcs> wcs;
    std::shared_ptr<Calib> calib;
    std::shared_ptr<VisitInfo> visitInfo;
};


class ExposureFitsReader::ArchiveReader{
public:

    enum Component {
        PSF=0,
        WCS,
        COADD_INPUTS,
        AP_CORR_MAP,
        VALID_POLYGON,
        TRANSMISSION_CURVE,
        N_ARCHIVE_COMPONENTS
    };

    explicit ArchiveReader(daf::base::PropertyList & metadata) {
        auto popInt = [&metadata](std::string const& name) {
            // The default of zero will cause archive.get to return a
            // null/empty pointer, just as if a null/empty pointer was
            // originally written to the archive.
            int r = 0;
            if (metadata.exists(name)) {
                r = metadata.get<int>(name);
                // We remove metadata entries to maintaing our practice
                // of stripped metadata entries that have been used to
                // construct more structured components.
                metadata.remove(name);
            }
            return r;
        };
        _hdu = popInt("AR_HDU");
        if (_hdu == 0) {
            _state = ArchiveState::MISSING;
        } else {
            --_hdu;  // Switch from FITS 1-indexed convention to LSST 0-indexed convention.
            _state = ArchiveState::PRESENT;
        }
        _ids[PSF] = popInt("PSF_ID");
        _ids[WCS] = popInt("SKYWCS_ID");
        _ids[COADD_INPUTS] = popInt("COADD_INPUTS_ID");
        _ids[AP_CORR_MAP] = popInt("AP_CORR_MAP_ID");
        _ids[VALID_POLYGON] = popInt("VALID_POLYGON_ID");
        _ids[TRANSMISSION_CURVE] = popInt("TRANSMISSION_CURVE_ID");
    }

    template <typename T>
    std::shared_ptr<T> readComponent(afw::fits::Fits * fitsFile, Component c) {
        if (!_ensureLoaded(fitsFile)) {
            return nullptr;
        }
        return _archive.get<T>(_ids[c]);
    }

private:

    bool _ensureLoaded(afw::fits::Fits * fitsFile) {
        if (_state == ArchiveState::MISSING) {
            return false;
        }
        if (_state == ArchiveState::PRESENT) {
            afw::fits::HduMoveGuard guard(*fitsFile, _hdu);
            _archive = table::io::InputArchive::readFits(*fitsFile);
            _state = ArchiveState::LOADED;
        }
        assert(_state == ArchiveState::LOADED);  // constructor body should guarantee it's not UNKNOWN
        return true;
    }

    enum class ArchiveState { UNKNOWN, MISSING, PRESENT, LOADED };

    int _hdu = 0;
    ArchiveState _state = ArchiveState::UNKNOWN;
    table::io::InputArchive _archive;
    std::array<int, N_ARCHIVE_COMPONENTS> _ids = {0};
};


ExposureFitsReader::ExposureFitsReader(std::string const& fileName) :
    _maskedImageReader(fileName)
{}

ExposureFitsReader::ExposureFitsReader(fits::MemFileManager& manager) :
    _maskedImageReader(manager)
{}

ExposureFitsReader::ExposureFitsReader(fits::Fits * fitsFile) :
    _maskedImageReader(fitsFile)
{}

ExposureFitsReader::~ExposureFitsReader() noexcept = default;

lsst::geom::Box2I ExposureFitsReader::readBBox(ImageOrigin origin) {
    return _maskedImageReader.readBBox(origin);
}

lsst::geom::Point2I ExposureFitsReader::readXY0(lsst::geom::Box2I const & bbox, ImageOrigin origin) {
    return _maskedImageReader.readXY0(bbox, origin);
}

std::string ExposureFitsReader::readImageDType() const { return _maskedImageReader.readImageDType(); }

std::string ExposureFitsReader::readMaskDType() const { return _maskedImageReader.readMaskDType(); }

std::string ExposureFitsReader::readVarianceDType() const { return _maskedImageReader.readVarianceDType(); }

std::shared_ptr<daf::base::PropertyList> ExposureFitsReader::readMetadata() {
    _ensureReaders();
    return _metadataReader->metadata;
}

std::shared_ptr<afw::geom::SkyWcs> ExposureFitsReader::readWcs() {
    _ensureReaders();
    auto r = _archiveReader->readComponent<afw::geom::SkyWcs>(_getFitsFile(), ArchiveReader::WCS);
    if (!r) {
        r = _metadataReader->wcs;
    }
    return r;
}

Filter ExposureFitsReader::readFilter() {
    _ensureReaders();
    return _metadataReader->filter;
}

std::shared_ptr<Calib> ExposureFitsReader::readCalib() {
    _ensureReaders();
    return _metadataReader->calib;
}

std::shared_ptr<detection::Psf> ExposureFitsReader::readPsf() {
    _ensureReaders();
    return _archiveReader->readComponent<detection::Psf>(_getFitsFile(), ArchiveReader::PSF);
}

std::shared_ptr<afw::geom::polygon::Polygon> ExposureFitsReader::readValidPolygon() {
    _ensureReaders();
    return _archiveReader->readComponent<afw::geom::polygon::Polygon>(_getFitsFile(),
                                                                      ArchiveReader::VALID_POLYGON);
}

std::shared_ptr<ApCorrMap> ExposureFitsReader::readApCorrMap() {
    _ensureReaders();
    return _archiveReader->readComponent<ApCorrMap>(_getFitsFile(), ArchiveReader::AP_CORR_MAP);
}

std::shared_ptr<CoaddInputs> ExposureFitsReader::readCoaddInputs() {
    _ensureReaders();
    return _archiveReader->readComponent<CoaddInputs>(_getFitsFile(), ArchiveReader::COADD_INPUTS);
}

std::shared_ptr<VisitInfo> ExposureFitsReader::readVisitInfo() {
    _ensureReaders();
    return _metadataReader->visitInfo;
}

std::shared_ptr<TransmissionCurve> ExposureFitsReader::readTransmissionCurve() {
    _ensureReaders();
    return _archiveReader->readComponent<TransmissionCurve>(_getFitsFile(), ArchiveReader::TRANSMISSION_CURVE);
}

std::shared_ptr<ExposureInfo> ExposureFitsReader::readExposureInfo() {
    auto result = std::make_shared<ExposureInfo>();
    result->setMetadata(readMetadata());
    result->setFilter(readFilter());
    result->setCalib(readCalib());
    result->setVisitInfo(readVisitInfo());
    // When reading an ExposureInfo (as opposed to reading individual
    // components), we warn and try to proceed when a component is present
    // but can't be read due its serialization factory not being set up
    // (that's what throws the NotFoundErrors caught below).
    try {
        result->setPsf(readPsf());
    } catch (pex::exceptions::NotFoundError& err) {
        LOGLS_WARN(_log, "Could not read PSF; setting to null: " << err.what());
    }
    try {
        result->setCoaddInputs(readCoaddInputs());
    } catch (pex::exceptions::NotFoundError& err) {
        LOGLS_WARN(_log, "Could not read CoaddInputs; setting to null: " << err.what());
    }
    try {
        result->setApCorrMap(readApCorrMap());
    } catch (pex::exceptions::NotFoundError& err) {
        LOGLS_WARN(_log, "Could not read ApCorrMap; setting to null: " << err.what());
    }
    try {
        result->setValidPolygon(readValidPolygon());
    } catch (pex::exceptions::NotFoundError& err) {
        LOGLS_WARN(_log, "Could not read ValidPolygon; setting to null: " << err.what());
    }
    try {
        result->setTransmissionCurve(readTransmissionCurve());
    } catch (pex::exceptions::NotFoundError& err) {
        LOGLS_WARN(_log, "Could not read TransmissionCurve; setting to null: " << err.what());
    }
    // In the case of WCS, we fall back to the metadata WCS if the one from
    // the archive can't be read.
    _ensureReaders();
    result->setWcs(_metadataReader->wcs);
    try {
        auto wcs = _archiveReader->readComponent<afw::geom::SkyWcs>(_getFitsFile(), ArchiveReader::WCS);
        if (!wcs) {
            LOGLS_DEBUG(_log, "No WCS found in binary table");
        } else {
            result->setWcs(wcs);
        }
    } catch (pex::exceptions::NotFoundError & err) {
        auto msg = str(boost::format("Could not read WCS extension; setting to null: %s") % err.what());
        if (result->hasWcs()) {
            msg += " ; using WCS from FITS header";
        }
        LOGLS_WARN(_log, msg);
    }
    return result;
}

template <typename ImagePixelT>
Image<ImagePixelT> ExposureFitsReader::readImage(lsst::geom::Box2I const & bbox, ImageOrigin origin,
                                                 bool allowUnsafe) {
    return _maskedImageReader.readImage<ImagePixelT>(bbox, origin, allowUnsafe);
}

template <typename ImagePixelT>
ndarray::Array<ImagePixelT, 2, 2> ExposureFitsReader::readImageArray(lsst::geom::Box2I const & bbox,
                                                                     ImageOrigin origin,
                                                                     bool allowUnsafe) {
    return _maskedImageReader.readImageArray<ImagePixelT>(bbox, origin, allowUnsafe);
}

template <typename MaskPixelT>
Mask<MaskPixelT> ExposureFitsReader::readMask(lsst::geom::Box2I const & bbox, ImageOrigin origin,
                                              bool conformMasks, bool allowUnsafe) {
    return _maskedImageReader.readMask<MaskPixelT>(bbox, origin, conformMasks, allowUnsafe);
}

template <typename MaskPixelT>
ndarray::Array<MaskPixelT, 2, 2> ExposureFitsReader::readMaskArray(lsst::geom::Box2I const & bbox,
                                                                   ImageOrigin origin,
                                                                   bool allowUnsafe) {
    return _maskedImageReader.readMaskArray<MaskPixelT>(bbox, origin, allowUnsafe);
}

template <typename VariancePixelT>
Image<VariancePixelT> ExposureFitsReader::readVariance(lsst::geom::Box2I const & bbox, ImageOrigin origin,
                                                       bool allowUnsafe) {
    return _maskedImageReader.readVariance<VariancePixelT>(bbox, origin, allowUnsafe);
}

template <typename VariancePixelT>
ndarray::Array<VariancePixelT, 2, 2> ExposureFitsReader::readVarianceArray(lsst::geom::Box2I const & bbox,
                                                                           ImageOrigin origin,
                                                                           bool allowUnsafe) {
    return _maskedImageReader.readVarianceArray<VariancePixelT>(bbox, origin, allowUnsafe);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> ExposureFitsReader::readMaskedImage(
    lsst::geom::Box2I const & bbox,
    ImageOrigin origin,
    bool conformMasks,
    bool allowUnsafe
) {
    return _maskedImageReader.read<ImagePixelT, MaskPixelT, VariancePixelT>(bbox, origin, conformMasks,
                                                                            /* needAllHdus= */false,
                                                                            allowUnsafe);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
Exposure<ImagePixelT, MaskPixelT, VariancePixelT> ExposureFitsReader::read(
    lsst::geom::Box2I const & bbox,
    ImageOrigin origin,
    bool conformMasks,
    bool allowUnsafe
) {
    auto mi = readMaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>(bbox, origin, conformMasks,
                                                                       allowUnsafe);
    return Exposure<ImagePixelT, MaskPixelT, VariancePixelT>(mi, readExposureInfo());
}

void ExposureFitsReader::_ensureReaders() {
    if (!_metadataReader) {
        auto metadataReader = std::make_unique<MetadataReader>(
            _maskedImageReader.readPrimaryMetadata(),
            _maskedImageReader.readImageMetadata(),
            _maskedImageReader.readXY0()
        );
        _archiveReader = std::make_unique<ArchiveReader>(*metadataReader->metadata);
        _metadataReader = std::move(metadataReader);  // deferred for exception safety
    }
    assert(_archiveReader);  // should always be initialized with _metadataReader.
}

#define INSTANTIATE(ImagePixelT) \
    template Exposure<ImagePixelT, MaskPixel, VariancePixel> ExposureFitsReader::read( \
        lsst::geom::Box2I const &, \
        ImageOrigin, \
        bool, bool \
    ); \
    template Image<ImagePixelT> ExposureFitsReader::readImage( \
        lsst::geom::Box2I const &, \
        ImageOrigin, bool \
    ); \
    template ndarray::Array<ImagePixelT, 2, 2> ExposureFitsReader::readImageArray(\
        lsst::geom::Box2I const &, \
        ImageOrigin, \
        bool \
    ); \
    template MaskedImage<ImagePixelT, MaskPixel, VariancePixel> ExposureFitsReader::readMaskedImage( \
        lsst::geom::Box2I const &, \
        ImageOrigin, \
        bool, bool \
    )

INSTANTIATE(std::uint16_t);
INSTANTIATE(int);
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(std::uint64_t);

template Mask<MaskPixel> ExposureFitsReader::readMask(
    lsst::geom::Box2I const &,
    ImageOrigin,
    bool,
    bool
);
template ndarray::Array<MaskPixel, 2, 2> ExposureFitsReader::readMaskArray(
    lsst::geom::Box2I const &,
    ImageOrigin,
    bool
);

template Image<VariancePixel> ExposureFitsReader::readVariance(
    lsst::geom::Box2I const &,
    ImageOrigin,
    bool
);
template ndarray::Array<VariancePixel, 2, 2> ExposureFitsReader::readVarianceArray(
    lsst::geom::Box2I const &,
    ImageOrigin,
    bool
);



}}} // lsst::afw::image

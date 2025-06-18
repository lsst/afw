/*
 * LSST Data Management System
 * Copyright 2017 AURA/LSST.
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
#include <exception>
#include <memory>
#include <vector>

#include "astshim.h"

#include "lsst/geom/Angle.h"
#include "lsst/geom/Point.h"
#include "lsst/geom/SpherePoint.h"
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/geom/detail/frameSetUtils.h"
#include "lsst/afw/geom/wcsUtils.h"
#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/daf/base/PropertyList.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/io/Persistable.cc"

namespace lsst {
namespace afw {

template std::shared_ptr<geom::SkyWcs> table::io::PersistableFacade<geom::SkyWcs>::dynamicCast(
        std::shared_ptr<table::io::Persistable> const&);

namespace geom {
namespace {

int const SERIALIZATION_VERSION = 1;

// TIGHT_FITS_TOL is used by getFitsMetadata to determine if a WCS can accurately be represented as a FITS
// WCS. It specifies the maximum departure from linearity (in pixels) allowed on either axis of the mapping
// from pixel coordinates to Intermediate World Coordinates over a range of 100 x 100 pixels
// (or the region specified by NAXIS[12], if provided, but we do not pass this to AST as of 2018-01-17).
// For more information,
// see FitsTol in the AST manual http://starlink.eao.hawaii.edu/devdocs/sun211.htx/sun211.html
double const TIGHT_FITS_TOL = 0.0001;

class SkyWcsPersistenceHelper {
public:
    table::Schema schema;
    table::Key<table::Array<std::uint8_t>> wcs;
    table::Key<table::Array<std::uint8_t>> approx;

    // No copying
    SkyWcsPersistenceHelper(const SkyWcsPersistenceHelper&) = delete;
    SkyWcsPersistenceHelper& operator=(const SkyWcsPersistenceHelper&) = delete;

    // No moving
    SkyWcsPersistenceHelper(SkyWcsPersistenceHelper&&) = delete;
    SkyWcsPersistenceHelper& operator=(SkyWcsPersistenceHelper&&) = delete;

    explicit SkyWcsPersistenceHelper(bool hasFitsApproximation)
            : schema(),
              wcs(schema.addField<table::Array<std::uint8_t>>("wcs", "wcs string representation", "")) {
        if (hasFitsApproximation) {
            approx = schema.addField<table::Array<std::uint8_t>>(
                    "approx", "wcs string representation of FITS approximation", "");
        }
    }

    explicit SkyWcsPersistenceHelper(table::Schema const& schema)
            : schema(schema), wcs(schema["wcs"]), approx() {
        try {
            approx = schema["approx"];
        } catch (pex::exceptions::NotFoundError&) {
        }
    }
};

class SkyWcsFactory : public table::io::PersistableFactory {
public:
    explicit SkyWcsFactory(std::string const& name) : table::io::PersistableFactory(name) {}

    std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                 CatalogVector const& catalogs) const override {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        SkyWcsPersistenceHelper keys(catalogs.front().getSchema());
        table::BaseRecord const& record = catalogs.front().front();
        std::string stringRep = formatters::bytesToString(record.get(keys.wcs));
        auto result = SkyWcs::readString(stringRep);
        if (keys.approx.isValid()) {
            auto bytes = record.get(keys.approx);
            if (!bytes.isEmpty()) {
                auto approxStringRep = formatters::bytesToString(bytes);
                result = result->copyWithFitsApproximation(SkyWcs::readString(approxStringRep));
            }
        }
        return result;
    }
};

std::string getSkyWcsPersistenceName() { return "SkyWcs"; }

SkyWcsFactory registration(getSkyWcsPersistenceName());

ast::FrameDict makeSkyWcsFrameDict(TransformPoint2ToPoint2 const& pixelsToFieldAngle,
                                   lsst::geom::Angle const& orientation, bool flipX,
                                   lsst::geom::SpherePoint const& crval,
                                   std::string const& projection = "TAN") {
    auto const orientationAndFlipXMatrix = makeCdMatrix(1 * lsst::geom::degrees, orientation, flipX);
    auto const initialWcs =
            makeSkyWcs(lsst::geom::Point2D(0, 0), crval, orientationAndFlipXMatrix, projection);
    auto const initialFrameDict = initialWcs->getFrameDict();
    auto const iwcToSkyMap = initialFrameDict->getMapping("IWC", "SKY");
    auto const pixelFrame = initialFrameDict->getFrame("PIXELS");
    auto const iwcFrame = initialFrameDict->getFrame("IWC");
    auto const skyFrame = initialFrameDict->getFrame("SKY");
    // Field angle is in radians and is aligned to focal plane x and y;
    // IWC is basically the same thing, but in degrees and with rotation and flipX applied
    ndarray::Array<double, 2, 2> fieldAngleToIwcNdArray = ndarray::allocate(2, 2);
    asEigenMatrix(fieldAngleToIwcNdArray) = orientationAndFlipXMatrix * 180.0 / lsst::geom::PI;
    auto const pixelsToFieldAngleMap = pixelsToFieldAngle.getMapping();
    auto const fieldAngleToIwcMap = ast::MatrixMap(fieldAngleToIwcNdArray);
    auto const pixelsToIwcMap = pixelsToFieldAngleMap->then(fieldAngleToIwcMap);
    auto finalFrameDict = ast::FrameDict(*pixelFrame, pixelsToIwcMap, *iwcFrame);
    finalFrameDict.addFrame("IWC", *iwcToSkyMap, *skyFrame);
    return finalFrameDict;
}

}  // namespace

Eigen::Matrix2d makeCdMatrix(lsst::geom::Angle const& scale, lsst::geom::Angle const& orientation,
                             bool flipX) {
    Eigen::Matrix2d cdMatrix;
    double orientRad = orientation.asRadians();
    double scaleDeg = scale.asDegrees();
    double xmult = flipX ? 1.0 : -1.0;
    cdMatrix(0, 0) = std::cos(orientRad) * scaleDeg * xmult;
    cdMatrix(0, 1) = std::sin(orientRad) * scaleDeg;
    cdMatrix(1, 0) = -std::sin(orientRad) * scaleDeg * xmult;
    cdMatrix(1, 1) = std::cos(orientRad) * scaleDeg;
    return cdMatrix;
}

std::shared_ptr<TransformPoint2ToPoint2> makeWcsPairTransform(SkyWcs const& src, SkyWcs const& dst) {
    auto const dstInverse = dst.getTransform()->inverted();
    return src.getTransform()->then(*dstInverse);
}

SkyWcs::SkyWcs(daf::base::PropertySet& metadata, bool strip)
        : SkyWcs(detail::readLsstSkyWcs(metadata, strip)) {}

SkyWcs::SkyWcs(ast::FrameDict const& frameDict) : SkyWcs(_checkFrameDict(frameDict)) {}

bool SkyWcs::operator==(SkyWcs const& other) const { return writeString() == other.writeString(); }

lsst::geom::Angle SkyWcs::getPixelScale(lsst::geom::Point2D const& pixel) const {
    // Compute pixVec containing the pixel position and two nearby points
    // (use a vector so all three points can be converted to sky in a single call)
    double const side = 1.0;
    std::vector<lsst::geom::Point2D> pixVec = {
            pixel,
            pixel + lsst::geom::Extent2D(side, 0),
            pixel + lsst::geom::Extent2D(0, side),
    };

    auto skyVec = pixelToSky(pixVec);

    // Work in 3-space to avoid RA wrapping and pole issues
    auto skyLL = skyVec[0].getVector();
    auto skyDx = skyVec[1].getVector() - skyLL;
    auto skyDy = skyVec[2].getVector() - skyLL;

    // Compute pixel scale in radians = sqrt(pixel area in radians^2)
    // pixel area in radians^2 = area of parallelogram with sides skyDx, skyDy = |skyDx cross skyDy|
    // Use squared norm to avoid two square roots
    double skyAreaSq = skyDx.cross(skyDy).getSquaredNorm();
    return (std::pow(skyAreaSq, 0.25) / side) * lsst::geom::radians;
}

lsst::geom::SpherePoint SkyWcs::getSkyOrigin() const {
    // CRVAL is stored as the SkyRef property of the sky frame (the current frame of the SkyWcs)
    auto skyFrame = std::dynamic_pointer_cast<ast::SkyFrame>(
            getFrameDict()->getFrame(ast::FrameDict::CURRENT, false));  // false: do not copy
    if (!skyFrame) {
        throw LSST_EXCEPT(pex::exceptions::LogicError, "Current frame is not a SkyFrame");
    }
    auto const crvalRad = skyFrame->getSkyRef();
    return lsst::geom::SpherePoint(crvalRad[0] * lsst::geom::radians, crvalRad[1] * lsst::geom::radians);
}

Eigen::Matrix2d SkyWcs::getCdMatrix(lsst::geom::Point2D const& pixel) const {
    auto const pixelToIwc = getFrameDict()->getMapping(ast::FrameSet::BASE, "IWC");
    auto const pixelToIwcTransform = TransformPoint2ToPoint2(*pixelToIwc);
    return pixelToIwcTransform.getJacobian(pixel);
}

Eigen::Matrix2d SkyWcs::getCdMatrix() const { return getCdMatrix(getPixelOrigin()); }

std::shared_ptr<SkyWcs> SkyWcs::getTanWcs(lsst::geom::Point2D const& pixel) const {
    auto const crval = pixelToSky(pixel);
    auto const cdMatrix = getCdMatrix(pixel);
    auto metadata = makeSimpleWcsMetadata(pixel, crval, cdMatrix);
    return std::make_shared<SkyWcs>(*metadata);
}

std::shared_ptr<SkyWcs> SkyWcs::copyAtShiftedPixelOrigin(lsst::geom::Extent2D const& shift) const {
    auto newToOldPixel = TransformPoint2ToPoint2(ast::ShiftMap({-shift[0], -shift[1]}));
    return makeModifiedWcs(newToOldPixel, *this, true);
}

std::shared_ptr<daf::base::PropertyList> SkyWcs::_getDirectFitsMetadata() const {
    // Make a FrameSet that maps from GRID to SKY; GRID = the base frame (PIXELS or ACTUAL_PIXELS) + 1
    auto const gridToPixel = ast::ShiftMap({-1.0, -1.0});
    auto thisDict = getFrameDict();
    auto const pixelToIwc = thisDict->getMapping(ast::FrameSet::BASE, "IWC");
    auto const iwcToSky = thisDict->getMapping("IWC", "SKY");
    auto const gridToSky = gridToPixel.then(*pixelToIwc).then(*iwcToSky);
    ast::FrameSet frameSet(ast::Frame(2, "Domain=GRID"), gridToSky, *thisDict->getFrame("SKY", false));

    // Write frameSet to a FitsChan and extract the metadata
    std::ostringstream os;
    os << "Encoding=FITS-WCS, CDMatrix=1, FitsAxisOrder=<copy>, FitsTol=" << TIGHT_FITS_TOL;
    ast::StringStream strStream;
    ast::FitsChan fitsChan(strStream, os.str());
    int const nObjectsWritten = fitsChan.write(frameSet);
    if (nObjectsWritten == 0) {
        return nullptr;
    }
    std::shared_ptr<daf::base::PropertyList> header = detail::getPropertyListFromFitsChan(fitsChan);

    // Remove DATE-OBS, MJD-OBS: AST writes these if the EQUINOX is set, but we set them via other mechanisms.
    header->remove("DATE-OBS");
    header->remove("MJD-OBS");

    // If CD matrix is present, explicitly set any missing entries to zero, as a convenience to the user
    bool const hasCd11 = header->exists("CD1_1");
    bool const hasCd12 = header->exists("CD1_2");
    bool const hasCd21 = header->exists("CD2_1");
    bool const hasCd22 = header->exists("CD2_2");
    if (hasCd11 || hasCd12 || hasCd21 || hasCd22) {
        if (!hasCd11) header->set("CD1_1", 0.0, "Transformation matrix element");
        if (!hasCd12) header->set("CD1_2", 0.0, "Transformation matrix element");
        if (!hasCd21) header->set("CD2_1", 0.0, "Transformation matrix element");
        if (!hasCd22) header->set("CD2_2", 0.0, "Transformation matrix element");
    }

    return header;
}

std::shared_ptr<daf::base::PropertyList> SkyWcs::getFitsMetadata(bool precise) const {
    if (!precise && hasFitsApproximation()) {
        return getFitsApproximation()->_getDirectFitsMetadata();
    }
    auto result = _getDirectFitsMetadata();
    if (!result) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError,
                          precise ? "WCS is not directly FITS-compatible."
                                  : "WCS does not have an attached FITS approximation.");
    }
    return result;
}

std::shared_ptr<const ast::FrameDict> SkyWcs::getFrameDict() const { return _frameDict; }

std::shared_ptr<SkyWcs> SkyWcs::copyWithFitsApproximation(std::shared_ptr<SkyWcs> fitsApproximation) const {
    if (fitsApproximation->hasFitsApproximation()) {
        throw LSST_EXCEPT(pex::exceptions::LogicError,
                          "Cannot add a FITS approximation that itself already has a FITS approximation.");
    }
    auto result = std::make_shared<SkyWcs>(*this);
    result->_fitsApproximation = fitsApproximation;
    return result;
}

bool SkyWcs::isFits() const { return bool(_getDirectFitsMetadata()); }

lsst::geom::AffineTransform SkyWcs::linearizePixelToSky(lsst::geom::SpherePoint const& coord,
                                                        lsst::geom::AngleUnit const& skyUnit) const {
    return _linearizePixelToSky(skyToPixel(coord), coord, skyUnit);
}
lsst::geom::AffineTransform SkyWcs::linearizePixelToSky(lsst::geom::Point2D const& pix,
                                                        lsst::geom::AngleUnit const& skyUnit) const {
    return _linearizePixelToSky(pix, pixelToSky(pix), skyUnit);
}

lsst::geom::AffineTransform SkyWcs::linearizeSkyToPixel(lsst::geom::SpherePoint const& coord,
                                                        lsst::geom::AngleUnit const& skyUnit) const {
    return _linearizeSkyToPixel(skyToPixel(coord), coord, skyUnit);
}

lsst::geom::AffineTransform SkyWcs::linearizeSkyToPixel(lsst::geom::Point2D const& pix,
                                                        lsst::geom::AngleUnit const& skyUnit) const {
    return _linearizeSkyToPixel(pix, pixelToSky(pix), skyUnit);
}

std::string SkyWcs::getShortClassName() { return "SkyWcs"; };

bool SkyWcs::isFlipped() const {
    double det = getCdMatrix().determinant();
    if (det == 0) {
        throw(LSST_EXCEPT(pex::exceptions::RuntimeError, "CD matrix is singular"));
    }
    return (det > 0);
}

std::shared_ptr<SkyWcs> SkyWcs::readStream(std::istream& is) {
    int version;
    is >> version;
    if (version != 1) {
        throw LSST_EXCEPT(pex::exceptions::TypeError, "Unsupported version " + std::to_string(version));
    }
    std::string shortClassName;
    is >> shortClassName;
    if (shortClassName != SkyWcs::getShortClassName()) {
        std::ostringstream os;
        os << "Class name in stream " << shortClassName << " != " << SkyWcs::getShortClassName();
        throw LSST_EXCEPT(pex::exceptions::TypeError, os.str());
    }
    bool hasFitsApproximation;
    is >> hasFitsApproximation;
    auto astStream = ast::Stream(&is, nullptr);
    auto astObjectPtr = ast::Channel(astStream).read();
    auto frameSet = std::dynamic_pointer_cast<ast::FrameSet>(astObjectPtr);
    if (!frameSet) {
        std::ostringstream os;
        os << "The AST serialization was read as a " << astObjectPtr->getClassName()
           << " instead of a FrameSet";
        throw LSST_EXCEPT(pex::exceptions::TypeError, os.str());
    }
    ast::FrameDict frameDict(*frameSet);
    return std::make_shared<SkyWcs>(frameDict);
}

std::shared_ptr<SkyWcs> SkyWcs::readString(std::string& str) {
    std::istringstream is(str);
    return SkyWcs::readStream(is);
}

void SkyWcs::writeStream(std::ostream& os) const {
    os << SERIALIZATION_VERSION << " " << SkyWcs::getShortClassName() << " " << hasFitsApproximation() << " ";
    getFrameDict()->show(os, false);  // false = do not write comments
}

std::string SkyWcs::writeString() const {
    std::ostringstream os;
    writeStream(os);
    return os.str();
}

std::shared_ptr<typehandling::Storable> SkyWcs::cloneStorable() const {
    return std::make_unique<SkyWcs>(*this);
}

std::string SkyWcs::toString() const {
    std::ostringstream os;
    if (isFits()) {
        os << "FITS standard SkyWcs:";
    } else {
        os << "Non-standard SkyWcs (Frames: ";
        // Print the frames in index order (frames are numbered from 1).
        std::string delimiter = "";
        for (size_t i = 1; i <= getFrameDict()->getAllDomains().size(); ++i) {
            os << delimiter << getFrameDict()->getFrame(i)->getDomain();
            delimiter = ", ";
        }
        os << "): ";
    }
    std::string delimiter = "\n";
    os << delimiter << "Sky Origin: " << getSkyOrigin();
    os << delimiter << "Pixel Origin: " << getPixelOrigin();
    os << delimiter << "Pixel Scale: " << getPixelScale().asArcseconds() << " arcsec/pixel";
    return os.str();
}

bool SkyWcs::equals(typehandling::Storable const& other) const noexcept {
    return singleClassEquals(*this, other);
}

std::string SkyWcs::getPersistenceName() const { return getSkyWcsPersistenceName(); }

std::string SkyWcs::getPythonModule() const { return "lsst.afw.geom"; }

void SkyWcs::write(OutputArchiveHandle& handle) const {
    SkyWcsPersistenceHelper const keys(hasFitsApproximation());
    table::BaseCatalog cat = handle.makeCatalog(keys.schema);
    std::shared_ptr<table::BaseRecord> record = cat.addNew();
    record->set(keys.wcs, formatters::stringToBytes(writeString()));
    if (hasFitsApproximation()) {
        record->set(keys.approx, formatters::stringToBytes(getFitsApproximation()->writeString()));
    }
    handle.saveCatalog(cat);
}

SkyWcs::SkyWcs(std::shared_ptr<ast::FrameDict> frameDict)
        : _frameDict(frameDict), _transform(), _pixelOrigin(), _pixelScaleAtOrigin(0 * lsst::geom::radians) {
    _computeCache();
};

std::shared_ptr<ast::FrameDict> SkyWcs::_checkFrameDict(ast::FrameDict const& frameDict) const {
    // Check that each frame is present and has the right type and number of axes
    std::vector<std::string> const domainNames = {"ACTUAL_PIXELS", "PIXELS", "IWC", "SKY"};
    for (auto const& domainName : domainNames) {
        if (frameDict.hasDomain(domainName)) {
            auto const frame = frameDict.getFrame(domainName, false);
            if (frame->getNAxes() != 2) {
                throw LSST_EXCEPT(lsst::pex::exceptions::TypeError,
                                  "Frame " + domainName + " has " + std::to_string(frame->getNAxes()) +
                                          " axes instead of 2");
            }
            auto desiredClassName = domainName == "SKY" ? "SkyFrame" : "Frame";
            if (frame->getClassName() != desiredClassName) {
                throw LSST_EXCEPT(lsst::pex::exceptions::TypeError,
                                  "Frame " + domainName + " is of type " + frame->getClassName() +
                                          " instead of " + desiredClassName);
            }
        } else if (domainName != "ACTUAL_PIXELS") {
            // This is a required frame
            throw LSST_EXCEPT(lsst::pex::exceptions::TypeError,
                              "No frame with domain " + domainName + " found");
        }
    }

    // The base frame must have domain "PIXELS" or "ACTUAL_PIXELS"
    auto baseDomain = frameDict.getFrame(ast::FrameSet::BASE, false)->getDomain();
    if (baseDomain != "ACTUAL_PIXELS" && baseDomain != "PIXELS") {
        throw LSST_EXCEPT(lsst::pex::exceptions::TypeError,
                          "Base frame has domain " + baseDomain + " instead of PIXELS or ACTUAL_PIXELS");
    }

    // The current frame must have domain "SKY"
    auto currentDomain = frameDict.getFrame(ast::FrameSet::CURRENT, false)->getDomain();
    if (currentDomain != "SKY") {
        throw LSST_EXCEPT(lsst::pex::exceptions::TypeError,
                          "Current frame has domain " + currentDomain + " instead of SKY");
    }

    return frameDict.copy();
}

lsst::geom::AffineTransform SkyWcs::_linearizePixelToSky(lsst::geom::Point2D const& pix00,
                                                         lsst::geom::SpherePoint const& coord,
                                                         lsst::geom::AngleUnit const& skyUnit) const {
    // Figure out the (0, 0), (0, 1), and (1, 0) ra/dec coordinates of the corners
    // of a square drawn in pixel. It'd be better to center the square at sky00,
    // but that would involve another conversion between sky and pixel coordinates
    const double side = 1.0;  // length of the square's sides in pixels
    auto const sky00 = coord.getPosition(skyUnit);
    auto const dsky10 = coord.getTangentPlaneOffset(pixelToSky(pix00 + lsst::geom::Extent2D(side, 0)));
    auto const dsky01 = coord.getTangentPlaneOffset(pixelToSky(pix00 + lsst::geom::Extent2D(0, side)));

    Eigen::Matrix2d m;
    m(0, 0) = dsky10.first.asAngularUnits(skyUnit) / side;
    m(0, 1) = dsky01.first.asAngularUnits(skyUnit) / side;
    m(1, 0) = dsky10.second.asAngularUnits(skyUnit) / side;
    m(1, 1) = dsky01.second.asAngularUnits(skyUnit) / side;

    Eigen::Vector2d sky00v;
    sky00v << sky00.getX(), sky00.getY();
    Eigen::Vector2d pix00v;
    pix00v << pix00.getX(), pix00.getY();
    // return lsst::geom::AffineTransform(m, lsst::geom::Extent2D(sky00v - m * pix00v));
    return lsst::geom::AffineTransform(m, (sky00v - m * pix00v));
}

lsst::geom::AffineTransform SkyWcs::_linearizeSkyToPixel(lsst::geom::Point2D const& pix00,
                                                         lsst::geom::SpherePoint const& coord,
                                                         lsst::geom::AngleUnit const& skyUnit) const {
    lsst::geom::AffineTransform inverse = _linearizePixelToSky(pix00, coord, skyUnit);
    return inverse.inverted();
}

std::shared_ptr<SkyWcs> makeFlippedWcs(SkyWcs const& wcs, bool flipLR, bool flipTB,
                                       lsst::geom::Point2D const& center) {
    double const dx = 1000;  // any "reasonable" number of pixels will do
    std::vector<double> inLL = {center[0] - dx, center[1] - dx};
    std::vector<double> inUR = {center[0] + dx, center[1] + dx};
    std::vector<double> outLL(inLL);
    std::vector<double> outUR(inUR);
    if (flipLR) {
        outLL[0] = inUR[0];
        outUR[0] = inLL[0];
    }
    if (flipTB) {
        outLL[1] = inUR[1];
        outUR[1] = inLL[1];
    }
    auto const flipPix = TransformPoint2ToPoint2(ast::WinMap(inLL, inUR, outLL, outUR));
    return makeModifiedWcs(flipPix, wcs, true);
}

std::shared_ptr<SkyWcs> makeModifiedWcs(TransformPoint2ToPoint2 const& pixelTransform, SkyWcs const& wcs,
                                        bool modifyActualPixels) {
    auto const pixelMapping = pixelTransform.getMapping();
    auto oldFrameDict = wcs.getFrameDict();
    bool const hasActualPixels = oldFrameDict->hasDomain("ACTUAL_PIXELS");
    auto const pixelFrame = oldFrameDict->getFrame("PIXELS", false);
    auto const iwcFrame = oldFrameDict->getFrame("IWC", false);
    auto const skyFrame = oldFrameDict->getFrame("SKY", false);
    auto const oldPixelToIwc = oldFrameDict->getMapping("PIXELS", "IWC");
    auto const iwcToSky = oldFrameDict->getMapping("IWC", "SKY");

    std::shared_ptr<ast::FrameDict> newFrameDict;
    std::shared_ptr<ast::Mapping> newPixelToIwc;
    if (hasActualPixels) {
        auto const actualPixelFrame = oldFrameDict->getFrame("ACTUAL_PIXELS", false);
        auto const oldActualPixelToPixels = oldFrameDict->getMapping("ACTUAL_PIXELS", "PIXELS");
        std::shared_ptr<ast::Mapping> newActualPixelsToPixels;
        if (modifyActualPixels) {
            newActualPixelsToPixels = pixelMapping->then(*oldActualPixelToPixels).simplified();
            newPixelToIwc = oldPixelToIwc;
        } else {
            newActualPixelsToPixels = oldActualPixelToPixels;
            newPixelToIwc = pixelMapping->then(*oldPixelToIwc).simplified();
        }
        newFrameDict =
                std::make_shared<ast::FrameDict>(*actualPixelFrame, *newActualPixelsToPixels, *pixelFrame);
        newFrameDict->addFrame("PIXELS", *newPixelToIwc, *iwcFrame);
    } else {
        newPixelToIwc = pixelMapping->then(*oldPixelToIwc).simplified();
        newFrameDict = std::make_shared<ast::FrameDict>(*pixelFrame, *newPixelToIwc, *iwcFrame);
    }
    newFrameDict->addFrame("IWC", *iwcToSky, *skyFrame);
    return std::make_shared<SkyWcs>(*newFrameDict);
}

std::shared_ptr<SkyWcs> makeSkyWcs(daf::base::PropertySet& metadata, bool strip) {
    return std::make_shared<SkyWcs>(metadata, strip);
}

std::shared_ptr<SkyWcs> makeSkyWcs(lsst::geom::Point2D const& crpix, lsst::geom::SpherePoint const& crval,
                                   Eigen::Matrix2d const& cdMatrix, std::string const& projection) {
    auto metadata = makeSimpleWcsMetadata(crpix, crval, cdMatrix, projection);
    return std::make_shared<SkyWcs>(*metadata);
}

std::shared_ptr<SkyWcs> makeSkyWcs(TransformPoint2ToPoint2 const& pixelsToFieldAngle,
                                   lsst::geom::Angle const& orientation, bool flipX,
                                   lsst::geom::SpherePoint const& boresight, std::string const& projection) {
    auto frameDict = makeSkyWcsFrameDict(pixelsToFieldAngle, orientation, flipX, boresight, projection);
    return std::make_shared<SkyWcs>(frameDict);
}

std::shared_ptr<SkyWcs> makeTanSipWcs(lsst::geom::Point2D const& crpix, lsst::geom::SpherePoint const& crval,
                                      Eigen::Matrix2d const& cdMatrix, Eigen::MatrixXd const& sipA,
                                      Eigen::MatrixXd const& sipB) {
    auto metadata = makeTanSipMetadata(crpix, crval, cdMatrix, sipA, sipB);
    return std::make_shared<SkyWcs>(*metadata);
}

std::shared_ptr<SkyWcs> makeTanSipWcs(lsst::geom::Point2D const& crpix, lsst::geom::SpherePoint const& crval,
                                      Eigen::Matrix2d const& cdMatrix, Eigen::MatrixXd const& sipA,
                                      Eigen::MatrixXd const& sipB, Eigen::MatrixXd const& sipAp,
                                      Eigen::MatrixXd const& sipBp) {
    auto metadata = makeTanSipMetadata(crpix, crval, cdMatrix, sipA, sipB, sipAp, sipBp);
    return std::make_shared<SkyWcs>(*metadata);
}

std::shared_ptr<TransformPoint2ToSpherePoint> getIntermediateWorldCoordsToSky(SkyWcs const& wcs,
                                                                              bool simplify) {
    auto iwcToSky = wcs.getFrameDict()->getMapping("IWC", "SKY");
    return std::make_shared<TransformPoint2ToSpherePoint>(*iwcToSky, simplify);
}

std::shared_ptr<TransformPoint2ToPoint2> getPixelToIntermediateWorldCoords(SkyWcs const& wcs, bool simplify) {
    auto pixelToIwc = wcs.getFrameDict()->getMapping(ast::FrameSet::BASE, "IWC");
    return std::make_shared<TransformPoint2ToPoint2>(*pixelToIwc, simplify);
}

std::ostream& operator<<(std::ostream& os, SkyWcs const& wcs) {
    os << wcs.toString();
    return os;
};

}  // namespace geom
}  // namespace afw
}  // namespace lsst

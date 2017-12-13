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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <ostream>
#include <sstream>
#include <vector>

#include "astshim.h"

#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/table.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/geom/detail/frameSetUtils.h"
#include "lsst/afw/geom/detail/transformUtils.h"
#include "lsst/afw/geom/wcsUtils.h"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/daf/base/PropertyList.h"
#include "lsst/pex/exceptions.h"

namespace lsst {
namespace afw {
namespace geom {
namespace {

int const SERIALIZATION_VERSION = 1;

double const TIGHT_FITS_TOL = 0.0001;

inline double square(double x) { return x * x; }

class SkyWcsPersistenceHelper {
public:
    table::Schema schema;
    table::Key<table::Array<std::uint8_t>> wcs;

    static SkyWcsPersistenceHelper const& get() {
        static SkyWcsPersistenceHelper instance;
        return instance;
    }

    // No copying
    SkyWcsPersistenceHelper(const SkyWcsPersistenceHelper&) = delete;
    SkyWcsPersistenceHelper& operator=(const SkyWcsPersistenceHelper&) = delete;

    // No moving
    SkyWcsPersistenceHelper(SkyWcsPersistenceHelper&&) = delete;
    SkyWcsPersistenceHelper& operator=(SkyWcsPersistenceHelper&&) = delete;

private:
    SkyWcsPersistenceHelper()
            : schema(),
              wcs(schema.addField<table::Array<std::uint8_t>>("wcs", "wcs string representation", "")) {
        schema.getCitizen().markPersistent();
    }
};

class SkyWcsFactory : public table::io::PersistableFactory {
public:
    explicit SkyWcsFactory(std::string const& name) : table::io::PersistableFactory(name) {}

    virtual std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                         CatalogVector const& catalogs) const {
        SkyWcsPersistenceHelper const& keys = SkyWcsPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        table::BaseRecord const& record = catalogs.front().front();
        std::string stringRep = formatters::bytesToString(record.get(keys.wcs));
        return SkyWcs::readString(stringRep);
    }
};

std::string getSkyWcsPersistenceName() { return "SkyWcs"; }

SkyWcsFactory registration(getSkyWcsPersistenceName());

// from https://stackoverflow.com/a/2072890
inline bool endsWith(std::string const& value, std::string const& ending) {
    if (ending.size() > value.size()) {
        return false;
    }
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

// Get a double precision value from metadata, or 0 if absent
// Useful for arrays, since omitted coefficients are 0
double getDoubleOrZero(daf::base::PropertySet& metadata, std::string const& name) {
    if (metadata.exists(name)) {
        return metadata.getAsDouble(name);
    }
    return 0.0;
}

}  // namespace

Eigen::Matrix2d makeCdMatrix(Angle const& scale, Angle const& orientation, bool flipX) {
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
    auto const dstInverse = dst.getTransform()->getInverse();
    return src.getTransform()->then(*dstInverse);
}

SkyWcs::SkyWcs(daf::base::PropertySet& metadata, bool strip)
        : SkyWcs(detail::readLsstSkyWcs(metadata, strip)) {}

SkyWcs::SkyWcs(ast::FrameDict const& frameDict) : SkyWcs(_checkFrameDict(frameDict)) {}

bool SkyWcs::operator==(SkyWcs const& other) const { return writeString() == other.writeString(); }

Angle SkyWcs::getPixelScale(Point2D const& pixel) const {
    // Compute pixVec containing the pixel position and two nearby points
    // (use a vector so all three points can be converted to sky in a single call)
    double const side = 1.0;
    std::vector<Point2D> pixVec = {
            pixel, pixel + Extent2D(side, 0), pixel + Extent2D(0, side),
    };

    auto skyVec = pixelToSky(pixVec);

    // Work in 3-space to avoid RA wrapping and pole issues
    // (warning: getVector().asEigen() here produces incorrect results; I'm not sure why)
    auto skyLL = skyVec[0].getVector();
    auto skyDx = skyVec[1].getVector() - skyLL;
    auto skyDy = skyVec[2].getVector() - skyLL;

    // Compute pixel scale in radians = sqrt(pixel area in radians^2)
    // pixel area in radians^2 = area of parallelogram with sides skyDx, skyDy = |skyDx cross skyDy|
    // Use squared norm to avoid two square roots
    double skyAreaSq = skyDx.asEigen().cross(skyDy.asEigen()).squaredNorm();
    return (std::pow(skyAreaSq, 0.25) / side) * radians;
}

coord::IcrsCoord SkyWcs::getSkyOrigin() const {
    // CRVAL is stored as the SkyRef property of the sky frame (the current frame of the SkyWcs)
    auto skyFrame = std::dynamic_pointer_cast<ast::SkyFrame>(
            getFrameDict()->getFrame(ast::FrameDict::CURRENT, false));  // false: do not copy
    if (!skyFrame) {
        throw LSST_EXCEPT(pex::exceptions::LogicError, "Current frame is not a SkyFrame");
    }
    auto const crvalRad = skyFrame->getSkyRef();
    return coord::IcrsCoord(crvalRad[0] * radians, crvalRad[1] * radians);
}

Eigen::Matrix2d SkyWcs::getCdMatrix(Point2D const& pixel) const {
    auto const pixelToIwc = getFrameDict()->getMapping(ast::FrameSet::BASE, "IWC");
    auto const pixelToIwcTransform = TransformPoint2ToPoint2(*pixelToIwc);
    return pixelToIwcTransform.getJacobian(pixel);
}

Eigen::Matrix2d SkyWcs::getCdMatrix() const { return getCdMatrix(getPixelOrigin()); }

std::shared_ptr<SkyWcs> SkyWcs::getTanWcs(Point2D const& pixel) const {
    auto const crval = pixelToSky(pixel);
    auto const cdMatrix = getCdMatrix(pixel);
    auto metadata = makeSimpleWcsMetadata(pixel, crval, cdMatrix);
    return std::make_shared<SkyWcs>(*metadata);
}

std::shared_ptr<SkyWcs> SkyWcs::copyAtShiftedPixelOrigin(Extent2D const& shift) const {
    auto newToOldPixel = TransformPoint2ToPoint2(ast::ShiftMap({-shift[0], -shift[1]}));
    return makeModifiedWcs(newToOldPixel, *this, true);
}

std::shared_ptr<daf::base::PropertyList> SkyWcs::getFitsMetadata(bool precise) const {
    // Make a FrameSet that maps from GRID to SKY; GRID = PIXELS + 1
    // (ignore ACTUAL_PIXELS, if present, as it cannot be represented with FITS WCS)
    auto const gridToPixel = ast::ShiftMap({-1.0, -1.0});
    auto thisDict = getFrameDict();
    auto const pixelToIwc = thisDict->getMapping("PIXELS", "IWC");
    auto const iwcToSky = thisDict->getMapping("IWC", "SKY");
    auto const gridToSky = gridToPixel.then(*pixelToIwc).then(*iwcToSky);
    ast::FrameSet frameSet(ast::Frame(2, "Domain=GRID"), gridToSky, *thisDict->getFrame("SKY", false));

    // Write frameSet to a FitsChan and extract the metadata
    std::ostringstream os;
    os << "Encoding=FITS-WCS, CDMatrix=1, FitsTol=" << TIGHT_FITS_TOL;
    ast::StringStream strStream;
    ast::FitsChan fitsChan(strStream, os.str());
    int const nObjectsWritten = fitsChan.write(frameSet);
    if (nObjectsWritten == 0) {
        if (precise) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                              "Could not represent this SkyWcs using FITS-WCS metadata");
        } else {
            // A large FitsTol was not sufficient; write a local TAN WCS approximation
            auto tanWcs = getTanWcs(getPixelOrigin());
            return tanWcs->getFitsMetadata(true);
        }
    }
    return detail::getPropertyListFromFitsChan(fitsChan);
}

std::shared_ptr<const ast::FrameDict> SkyWcs::getFrameDict() const {
    auto frameDict = std::static_pointer_cast<const ast::FrameDict>(_transform->getFrameSet());
    if (!frameDict) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicError, "Could not cast FrameSet to FrameDict");
    }
    return frameDict;
}

bool SkyWcs::isFits() const {
    try {
        getFitsMetadata(true);
    } catch (const lsst::pex::exceptions::RuntimeError&) {
        return false;
    } catch (const std::runtime_error&) {
        return false;
    }
    return true;
}

AffineTransform SkyWcs::linearizePixelToSky(coord::IcrsCoord const& coord, AngleUnit const& skyUnit) const {
    return _linearizePixelToSky(skyToPixel(coord), coord, skyUnit);
}
AffineTransform SkyWcs::linearizePixelToSky(Point2D const& pix, AngleUnit const& skyUnit) const {
    return _linearizePixelToSky(pix, pixelToSky(pix), skyUnit);
}

AffineTransform SkyWcs::linearizeSkyToPixel(coord::IcrsCoord const& coord, AngleUnit const& skyUnit) const {
    return _linearizeSkyToPixel(skyToPixel(coord), coord, skyUnit);
}

AffineTransform SkyWcs::linearizeSkyToPixel(Point2D const& pix, AngleUnit const& skyUnit) const {
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
        throw LSST_EXCEPT(pex::exceptions::TypeError,
                          "Unsupported version " + std::to_string(version));
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

std::string SkyWcs::getPersistenceName() const { return getSkyWcsPersistenceName(); }

std::string SkyWcs::getPythonModule() const { return "lsst.afw.geom.skyWcs"; }

void SkyWcs::write(OutputArchiveHandle& handle) const {
    SkyWcsPersistenceHelper const& keys = SkyWcsPersistenceHelper::get();
    table::BaseCatalog cat = handle.makeCatalog(keys.schema);
    std::shared_ptr<table::BaseRecord> record = cat.addNew();
    record->set(keys.wcs, formatters::stringToBytes(writeString()));
    handle.saveCatalog(cat);
}

SkyWcs::SkyWcs(std::shared_ptr<ast::FrameDict> frameDict)
        : _transform(new TransformPoint2ToIcrsCoord(std::move(frameDict))),
          _pixelOrigin(),
          _pixelScaleAtOrigin(0 * radians) {
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

AffineTransform SkyWcs::_linearizePixelToSky(Point2D const& pix00, coord::IcrsCoord const& coord,
                                             AngleUnit const& skyUnit) const {
    // Figure out the (0, 0), (0, 1), and (1, 0) ra/dec coordinates of the corners
    // of a square drawn in pixel. It'd be better to center the square at sky00,
    // but that would involve another conversion between sky and pixel coordinates
    const double side = 1.0;  // length of the square's sides in pixels
    auto const sky00 = coord.getPosition(skyUnit);
    auto const dsky10 = coord.getTangentPlaneOffset(pixelToSky(pix00 + Extent2D(side, 0)));
    auto const dsky01 = coord.getTangentPlaneOffset(pixelToSky(pix00 + Extent2D(0, side)));

    Eigen::Matrix2d m;
    m(0, 0) = dsky10.first.asAngularUnits(skyUnit) / side;
    m(0, 1) = dsky01.first.asAngularUnits(skyUnit) / side;
    m(1, 0) = dsky10.second.asAngularUnits(skyUnit) / side;
    m(1, 1) = dsky01.second.asAngularUnits(skyUnit) / side;

    Eigen::Vector2d sky00v;
    sky00v << sky00.getX(), sky00.getY();
    Eigen::Vector2d pix00v;
    pix00v << pix00.getX(), pix00.getY();
    // return AffineTransform(m, Extent2D(sky00v - m * pix00v));
    return AffineTransform(m, (sky00v - m * pix00v));
}

AffineTransform SkyWcs::_linearizeSkyToPixel(Point2D const& pix00, coord::IcrsCoord const& coord,
                                             AngleUnit const& skyUnit) const {
    AffineTransform inverse = _linearizePixelToSky(pix00, coord, skyUnit);
    return inverse.invert();
}

std::shared_ptr<SkyWcs> makeFlippedWcs(SkyWcs const& wcs, bool flipLR, bool flipTB,
                                       geom::Point2D const& center) {
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
    auto const pixelMapping = pixelTransform.getFrameSet()->getMapping();
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
            newActualPixelsToPixels = pixelMapping->then(*oldActualPixelToPixels).simplify();
            newPixelToIwc = oldPixelToIwc;
        } else {
            newActualPixelsToPixels = oldActualPixelToPixels;
            newPixelToIwc = pixelMapping->then(*oldPixelToIwc).simplify();
        }
        newFrameDict =
                std::make_shared<ast::FrameDict>(*actualPixelFrame, *newActualPixelsToPixels, *pixelFrame);
        newFrameDict->addFrame("PIXELS", *newPixelToIwc, *iwcFrame);
    } else {
        newPixelToIwc = pixelMapping->then(*oldPixelToIwc).simplify();
        newFrameDict = std::make_shared<ast::FrameDict>(*pixelFrame, *newPixelToIwc, *iwcFrame);
    }
    newFrameDict->addFrame("IWC", *iwcToSky, *skyFrame);
    return std::make_shared<SkyWcs>(*newFrameDict);
}

std::shared_ptr<SkyWcs> makeSkyWcs(daf::base::PropertySet& metadata, bool strip) {
    return std::make_shared<SkyWcs>(metadata, strip);
}

std::shared_ptr<SkyWcs> makeSkyWcs(Point2D const& crpix, coord::IcrsCoord const& crval,
                                   Eigen::Matrix2d const& cdMatrix, std::string const& projection) {
    auto metadata = makeSimpleWcsMetadata(crpix, crval, cdMatrix, projection);
    return std::make_shared<SkyWcs>(*metadata);
}

std::shared_ptr<SkyWcs> makeTanSipWcs(Point2D const& crpix, coord::IcrsCoord const& crval,
                                      Eigen::Matrix2d const& cdMatrix, Eigen::MatrixXd const& sipA,
                                      Eigen::MatrixXd const& sipB) {
    auto metadata = makeTanSipMetadata(crpix, crval, cdMatrix, sipA, sipB);
    return std::make_shared<SkyWcs>(*metadata);
}

std::shared_ptr<SkyWcs> makeTanSipWcs(Point2D const& crpix, coord::IcrsCoord const& crval,
                                      Eigen::Matrix2d const& cdMatrix, Eigen::MatrixXd const& sipA,
                                      Eigen::MatrixXd const& sipB, Eigen::MatrixXd const& sipAp,
                                      Eigen::MatrixXd const& sipBp) {
    auto metadata = makeTanSipMetadata(crpix, crval, cdMatrix, sipA, sipB, sipAp, sipBp);
    return std::make_shared<SkyWcs>(*metadata);
}

std::shared_ptr<TransformPoint2ToIcrsCoord> getIntermediateWorldCoordsToSky(SkyWcs const& wcs,
                                                                            bool simplify) {
    auto iwcToSky = wcs.getFrameDict()->getMapping("IWC", "SKY");
    return std::make_shared<TransformPoint2ToIcrsCoord>(*iwcToSky, simplify);
}

std::shared_ptr<TransformPoint2ToPoint2> getPixelToIntermediateWorldCoords(SkyWcs const& wcs, bool simplify) {
    auto pixelToIwc = wcs.getFrameDict()->getMapping(ast::FrameSet::BASE, "IWC");
    return std::make_shared<TransformPoint2ToPoint2>(*pixelToIwc, simplify);
}

}  // namespace geom
}  // namespace afw
}  // namespace lsst

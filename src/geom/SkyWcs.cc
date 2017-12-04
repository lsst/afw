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
#include <exception>
#include <memory>
#include <ostream>
#include <sstream>
#include <vector>

#include "astshim.h"

#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/geom/detail/frameSetUtils.h"
#include "lsst/afw/geom/detail/transformUtils.h"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/daf/base/PropertyList.h"
#include "lsst/pex/exceptions.h"

namespace lsst {
namespace afw {
namespace geom {
namespace {

inline double square(double x) { return x * x; }

}  // namespace

Eigen::Matrix2d makeCdMatrix(Angle const& scale, Angle const& orientation, bool flipX) {
    Eigen::Matrix2d cdMatrix;
    double orientRad = orientation.asRadians();
    double scaleDeg = scale.asDegrees();
    double xmult = flipX ? -1.0 : 1.0;
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

SkyWcs::SkyWcs(Point2D const& crpix, coord::IcrsCoord const& crval, Eigen::Matrix2d const& cdMatrix)
        : SkyWcs(detail::readLsstSkyWcs(*detail::makeTanWcsMetadata(crpix, crval, cdMatrix))) {}

SkyWcs::SkyWcs(daf::base::PropertyList& metadata, bool strip)
        : SkyWcs(detail::readLsstSkyWcs(metadata, strip)) {}

SkyWcs::SkyWcs(ast::FrameSet const& frameSet) : TransformPoint2ToIcrsCoord(_checkFrameSet(frameSet)) {}

Angle SkyWcs::getPixelScale(Point2D const& pixel) const {
    // Compute pixVec containing the pixel position of three corners of the pixel
    // (It would be slightly more accurate to measure across the pixel center,
    // but that would require one additional pixel-to-sky transformation)
    std::vector<Point2D> pixVec;
    pixVec.push_back(pixel - Extent2D(0.5, 0.5));   // lower left corner
    pixVec.push_back(pixel + Extent2D(0.5, -0.5));  // lower right corner
    pixVec.push_back(pixel + Extent2D(-0.5, 0.5));  // upper left corner

    auto skyVec = applyForward(pixVec);

    // Work in 3-space to avoid RA wrapping and pole issues.
    auto skyLL = skyVec[0].getVector();
    auto skyDx = skyVec[1].getVector() - skyLL;
    auto skyDy = skyVec[2].getVector() - skyLL;

    // Compute pixel scale in radians = sqrt(pixel area in radians^2)
    // pixel area in radians^2 = area of parallelogram with sides dx,dy, in radians^2
    //                         = |skyDx cross skyDy|
    // This cross product computes the distance *through* the unit sphere, rather than over its surface,
    // but the difference should be negligible for pixels of sane size

    double pixelAreaSq = square(skyDx[1] * skyDy[2] - skyDx[2] * skyDy[1]) +
                         square(skyDx[2] * skyDy[0] - skyDx[0] * skyDy[2]) +
                         square(skyDx[0] * skyDy[1] - skyDx[1] * skyDy[0]);
    return std::pow(pixelAreaSq, 0.25) * radians;
}

Angle SkyWcs::getPixelScale() const { return getPixelScale(getPixelOrigin()); }

Point2D SkyWcs::getPixelOrigin() const { return this->applyInverse(getSkyOrigin()); }

coord::IcrsCoord SkyWcs::getSkyOrigin() const {
    // CRVAL is stored as the SkyRef property of the sky frame (the current frame of the SkyWcs)
    auto const skyFrame = detail::getSkyFrame(*getFrameSet(), ast::FrameSet::CURRENT, false);
    auto const crvalRad = skyFrame->getSkyRef();
    return coord::IcrsCoord(crvalRad[0] * radians, crvalRad[1] * radians);
}

Eigen::Matrix2d SkyWcs::getCdMatrix(Point2D const& pixel) const {
    // Use a copy because looking up the IWC frame alters the FrameSet
    auto frameSet = getFrameSet()->copy();
    auto prototype = ast::Frame(2, "Domain=IWC");
    auto iwcFrameSet = frameSet->findFrame(prototype);
    if (!iwcFrameSet) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicError, "IWC frame not found");
    }
    auto iwcTransform = TransformPoint2ToPoint2(*iwcFrameSet);
    return iwcTransform.getJacobian(pixel);
}

Eigen::Matrix2d SkyWcs::getCdMatrix() const { return getCdMatrix(getPixelOrigin()); }

SkyWcs SkyWcs::getTanWcs(Point2D const& pixel) const {
    auto const crval = applyForward(pixel);
    auto const cdMatrix = getCdMatrix(pixel);
    return SkyWcs(pixel, crval, cdMatrix);
}

SkyWcs SkyWcs::copyAtShiftedPixelOrigin(Extent2D const& shift) const {
    auto frameSet = getFrameSet()->copy();

    // Save the SkyFrame so we can search for it at the end, to make it current again
    auto skyFrameIndex = frameSet->getCurrent();

    // Find the GRID frame
    if (!frameSet->findFrame(ast::Frame(2, "Domain=GRID"))) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicError, "GRID frame not found");
    }
    auto gridFrameIndex = frameSet->getCurrent();

    // Find the PIXEL0 frame and the old mapping from GRID to PIXEL0
    frameSet->setBase(gridFrameIndex);
    auto oldGridToPixel0Map = frameSet->findFrame(ast::Frame(2, "Domain=PIXEL0"));
    if (!oldGridToPixel0Map) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicError, "PIXEL0 frame not found");
    }
    auto oldPixel0Index = frameSet->getCurrent();
    auto pixel0Frame = frameSet->getFrame(oldPixel0Index);

    // Look for the optional ACTUAL_PIXEL0 frame, and the mapping from PIXEL0 to ACTUAL_PIXEL0
    frameSet->setBase(oldPixel0Index);
    auto oldActualPixel0Index = ast::FrameSet::NOFRAME;
    std::shared_ptr<ast::Frame> actualPixel0Frame;
    auto pixelToActualPixelMap = frameSet->findFrame(ast::Frame(2, "Domain=ACTUAL_PIXEL0"));
    if (pixelToActualPixelMap) {
        actualPixel0Frame = frameSet->getFrame(ast::FrameSet::CURRENT);
        oldActualPixel0Index = frameSet->getCurrent();
    }

    // Compute new mapping from GRID to PIXEL0
    std::vector<double> shiftVec{shift[0], shift[1]};
    auto deltaShiftMap = ast::ShiftMap(shiftVec);
    auto newGridToPixel0Map = oldGridToPixel0Map->then(deltaShiftMap).simplify();

    // Replace the mapping from GRID to PIXEL0.
    // We actually remove the old PIXEL0 and ACTUAL_PIXEL0 frames (if present)
    // and then add them in again, since there no direct way to replace a mapping.
    frameSet->removeFrame(oldPixel0Index);
    if (gridFrameIndex > oldPixel0Index) {
        --gridFrameIndex;
    }
    if (skyFrameIndex > oldPixel0Index) {
        --skyFrameIndex;
    }
    if (pixelToActualPixelMap) {
        if (oldActualPixel0Index > oldPixel0Index) {
            --oldActualPixel0Index;
        }
        frameSet->removeFrame(oldActualPixel0Index);
        if (gridFrameIndex > oldActualPixel0Index) {
            --gridFrameIndex;
        }
        if (skyFrameIndex > oldActualPixel0Index) {
            --skyFrameIndex;
        }
    }
    frameSet->addFrame(gridFrameIndex, *newGridToPixel0Map, *pixel0Frame);

    // If ACTUAL_PIXEL0 exists, re-add it
    if (pixelToActualPixelMap) {
        frameSet->addFrame(ast::FrameSet::CURRENT, *pixelToActualPixelMap, *actualPixel0Frame);
    }

    // Set the last frame added (PIXEL0 or ACTUAL_PIXEL0) to the base frame
    // and the SkyFrame to the current frame
    frameSet->setBase(frameSet->getCurrent());
    frameSet->setCurrent(skyFrameIndex);
    return SkyWcs(std::move(frameSet));
}

std::pair<Angle, Angle> SkyWcs::pixelToSky(double x, double y) const {
    auto sky = applyForward(Point2D(x, y));
    return std::pair<Angle, Angle>(sky[0], sky[1]);
}

/**
Compute the pixel position from the sky position
*/
std::pair<double, double> SkyWcs::skyToPixel(Angle const& ra, Angle const& dec) const {
    auto pixel = applyInverse(coord::IcrsCoord(ra, dec));
    return std::pair<double, double>(pixel[0], pixel[1]);
};

std::string SkyWcs::getShortClassName() { return "SkyWcs"; };

SkyWcs SkyWcs::readStream(std::istream& is) { return detail::readStream<SkyWcs>(is); }

SkyWcs SkyWcs::readString(std::string& str) {
    std::istringstream is(str);
    return SkyWcs::readStream(is);
}

void SkyWcs::writeStream(std::ostream& os) const { detail::writeStream<SkyWcs>(*this, os); }

std::string SkyWcs::writeString() const {
    std::ostringstream os;
    writeStream(os);
    return os.str();
}

SkyWcs::SkyWcs(std::shared_ptr<ast::FrameSet>&& frameSet) : TransformPoint2ToIcrsCoord(std::move(frameSet)){};

std::shared_ptr<ast::FrameSet> SkyWcs::_checkFrameSet(ast::FrameSet const& frameSet) const {
    // checking alters the frameSet current pointer, so use a copy
    auto frameSetCopy = frameSet.copy();
    int baseIndex = frameSetCopy->getBase();
    int currentIndex = frameSetCopy->getCurrent();

    auto baseFrame = frameSetCopy->getFrame(baseIndex, false);
    auto baseDomain = baseFrame->getDomain();
    if (baseDomain != "ACTUAL_PIXEL0" && baseDomain != "PIXEL0") {
        std::ostringstream os;
        os << "Base frame has domain " << baseDomain << " instead of PIXEL0 or ACTUAL_PIXEL0";
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, os.str());
    }
    if (baseFrame->getNAxes() != 2) {
        std::ostringstream os;
        os << "Base frame has " << baseFrame->getNAxes() << " instead of 2";
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, os.str());
    }
    if (!frameSetCopy->findFrame(ast::Frame(2, "Domain=GRID"))) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "No frame with domain GRID found");
    }
    if (!frameSetCopy->findFrame(ast::Frame(2, "Domain=IWC"))) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "No frame with domain IWC found");
    }
    auto currentFrame = frameSetCopy->getFrame(currentIndex, false);
    auto currentClass = currentFrame->getClassName();
    if (currentClass != "SkyFrame") {
        std::ostringstream os;
        os << "Current frame has type " << currentClass << " instead of SkyFrame";
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, os.str());
    }
    frameSetCopy->setBase(baseIndex);
    frameSetCopy->setCurrent(currentIndex);
    return frameSetCopy;
}

}  // namespace geom
}  // namespace afw
}  // namespace lsst

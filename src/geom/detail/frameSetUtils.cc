/*
 * LSST Data Management System
 * Copyright 2017 LSST Corporation.
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
#include <exception>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <vector>

#include "astshim.h"

#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/detail/frameSetUtils.h"
#include "lsst/daf/base/PropertyList.h"
#include "lsst/pex/exceptions.h"

namespace lsst {
namespace afw {
namespace geom {
namespace detail {
namespace {

// destructively make a set of strings from a vector of strings
std::set<std::string> setFromVector(std::vector<std::string>&& vec) {
    return std::set<std::string>(std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end()));
}

} // namespace

std::shared_ptr<ast::SkyFrame> getSkyFrame(ast::FrameSet const& frameSet, int index, bool copy) {
    auto frame = frameSet.getFrame(index, false);
    auto skyFrame = std::dynamic_pointer_cast<ast::SkyFrame>(frame);
    if (!skyFrame) {
        std::ostringstream os;
        os << "Bug! Frame at index=" << index << " is a " << frame->getClass() << ", not a SkyFrame";
        throw LSST_EXCEPT(pex::exceptions::RuntimeError, os.str());
    }
    return skyFrame;
}

std::shared_ptr<daf::base::PropertyList> makeTanWcsMetadata(Point2D const& crpix, SpherePoint const& crval,
                                                            Eigen::Matrix2d const& cdMatrix) {
    auto pl = std::make_shared<daf::base::PropertyList>();
    pl->add("RADESYS", "ICRS");
    pl->add("EQUINOX", 2000);  // not needed, but may help some older code
    pl->add("CTYPE1", "RA---TAN");
    pl->add("CTYPE2", "DEC--TAN");
    pl->add("CRVAL1", crval[0].asDegrees());
    pl->add("CRVAL2", crval[1].asDegrees());
    pl->add("CRPIX1", crpix[0] + 1);
    pl->add("CRPIX2", crpix[1] + 1);
    pl->add("CD1_1", cdMatrix(0, 0));
    pl->add("CD1_2", cdMatrix(0, 1));
    pl->add("CD2_1", cdMatrix(1, 0));
    pl->add("CD2_2", cdMatrix(1, 1));
    pl->add("CUNIT1", "deg");
    pl->add("CUNIT2", "deg");
    return pl;
}

std::shared_ptr<ast::FrameSet> readFitsWcs(daf::base::PropertyList& metadata, bool strip) {
    // exclude LTV1/2 from the FitsChan because afw handles those specially
    // exclude comments to reduce clutter
    std::set<std::string> excludeNames = {"LTV1", "LTV2", "COMMENT"};
    std::string hdr = formatters::formatFitsProperties(metadata, excludeNames);
    ast::StringStream stream(hdr);
    ast::FitsChan channel(stream, "Encoding=FITS-WCS, IWC=1");
    auto const initialNames = strip ? setFromVector(channel.getAllCardNames()) : std::set<std::string>();
    auto obj = channel.read();
    auto frameSet = std::dynamic_pointer_cast<ast::FrameSet>(obj);
    if (!frameSet) {
        std::ostringstream os;
        os << "metadata describes a " << obj->getClass() << ", not a FrameSet";
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, os.str());
    }
    if (strip) {
        auto const finalNames = setFromVector(channel.getAllCardNames());

        // FITS keywords that FitsChan stripped
        std::set<std::string> namesChannelStripped;
        std::set_difference(initialNames.begin(), initialNames.end(), finalNames.begin(), finalNames.end(),
                            std::inserter(namesChannelStripped, namesChannelStripped.begin()));

        // FITS keywords that FitsChan may strip that we want to keep in `metadata`
        // TODO DM-10411: remove TIMESYS from this list after starlink_ast is updated (no rush)
        std::set<std::string> const namesToKeep = {"DATE-OBS", "TIMESYS"};

        std::set<std::string> namesToStrip;  // names to strip from metadata
        std::set_difference(namesChannelStripped.begin(), namesChannelStripped.end(), namesToKeep.begin(),
                            namesToKeep.end(), std::inserter(namesToStrip, namesToStrip.begin()));
        for (auto const& name : namesToStrip) {
            metadata.remove(name);
        }
    }
    return frameSet;
}

std::shared_ptr<ast::FrameSet> readLsstSkyWcs(daf::base::PropertyList& metadata, bool strip) {
    std::vector<double> crvalRad = {degToRad(metadata.getAsDouble("CRVAL1")),
                                    degToRad(metadata.getAsDouble("CRVAL2"))};
    auto frameSet = readFitsWcs(metadata, strip);

    // Standardize the Sky frame to ICRS with axes order RA, Dec,
    // but inherit what we can from the original sky frame (e.g. epoch)

    // Create the desired sky frame, to use as a template
    auto stdSkyFrame = ast::SkyFrame("System=ICRS");

    // Locate a Frame in the target FrameSet that looks like the template
    // and hence can be used as the original sky frame.
    // We ignore the frame set returned by findFrame because that goes from pixels to sky,
    // and using it would add an unwanted extra branch to our WCS; instead, later on,
    // we compute a mapping from the old sky frame to the new sky frame and add that.
    if (!frameSet->findFrame(stdSkyFrame)) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Could not find a SkyFrame");
    }
    auto initialSkyIndex = frameSet->getCurrent();

    // Compute a frame set that maps from the original sky frame to our desired sky frame;
    // this is what we will insert into the frame set.
    // (Temporarily set the base frame to the sky frame, because findFrame
    // produces a mapping from base to the found frame).
    auto initialBaseIndex = frameSet->getBase();
    frameSet->setBase(initialSkyIndex);
    auto stdSkyFrameSet = frameSet->findFrame(stdSkyFrame);
    if (!stdSkyFrameSet) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError,
                          "Bug: found a SkyFrame the first time, but not the second time");
    }
    frameSet->setBase(initialBaseIndex);

    // Add the new mapping into the frameSet, connecting it to the original SkyFrame.
    // Note: we use stdSkyFrameSet as the new frame (meaning stdSkyFrameSet's current frame),
    // because, unlike stdSkyFrame, stdSkyFrameSet's current frame has inherited some
    // potentially useful attributes from the old sky frame, such as epoch.
    frameSet->addFrame(initialSkyIndex, *stdSkyFrameSet, *stdSkyFrameSet);

    // Delete the original SkyFrame, to avoid bloat.
    frameSet->removeFrame(initialSkyIndex);

    // Record the index of the standard sky frame (this must be done after removing
    // the initial sky frame, because removing a frame renumbers the other frames).
    auto stdSkyIndex = frameSet->getCurrent();

    // Record CRVAL as SkyRef in the SkyFrame so it can easily be obtained later;
    // set SkyRefIs = "Ignored" (the default) so SkyRef value is ignored instead of used as an offset
    auto skyFrame = getSkyFrame(*frameSet, stdSkyIndex, true);
    skyFrame->setSkyRefIs("Ignored");
    skyFrame->setSkyRef(crvalRad);

    // Add a PIXEL0 frame, where 0,0 is the lower left corner of the parent image;
    // this is the desired base frame for the WCS.

    // The PIXEL0 frame is a simple shift from the standard FITS pixel frame. However, in general,
    // there is no way of finding the standard FITS pixel frame, though usually it will be the base frame.
    // What we do is look for a frame named "GRID" of type Frame with 2 axes.
    // If that is found then we use it (since that follows the standard AST convention).
    // If that is not found then we check if the base frame has a blank Domain,
    // is of type Frame, and has 2 axes, and if so, use that as the GRID frame.

    auto gridIndex = ast::FrameSet::NOFRAME;
    if (frameSet->findFrame(ast::Frame(2, "Domain=GRID"))) {
        // Found a Frame with domain "GRID", type Frame and # axes = 2
        gridIndex = frameSet->getCurrent();
    }

    if (gridIndex == ast::FrameSet::NOFRAME) {
        // No appropriate GRID frame found; if the original base frame is of type Frame
        // with 2 axes and a blank domain then use that, else give up

        // set current frame to initial base frame so we can operate on it via the frame set
        frameSet->setCurrent(initialBaseIndex);
        auto const baseClassName = frameSet->getClass();
        if (baseClassName != "Frame") {
            std::ostringstream os;
            os << "Could not find a GRID frame, and the base frame is of type " << baseClassName
               << "instead of Frame";
            throw LSST_EXCEPT(pex::exceptions::RuntimeError, os.str());
        }
        if (frameSet->getNaxes() != 2) {
            std::ostringstream os;
            os << "Could not find a GRID frame, and the base frame has " << frameSet->getNaxes()
               << " axes instead of 2";
            throw LSST_EXCEPT(pex::exceptions::RuntimeError, os.str());
        }
        auto const baseDomain = frameSet->getDomain();
        if (baseDomain != "") {
            std::ostringstream os;
            os << "Could not find a GRID frame, and the base frame has domain \""
               << baseDomain << "\" instead of blank";
            throw LSST_EXCEPT(pex::exceptions::RuntimeError, os.str());
        }
        // Original base frame has 
        // Set its domain to GRID, and set some other potentially useful attributes.
        frameSet->setDomain("GRID");
        frameSet->setTitle("FITS pixel coordinates - first pixel at (1,1)");
        frameSet->setLabel(1, "Grid x");
        frameSet->setLabel(2, "Grid y");
        frameSet->setUnit(1, "Pixel");
        frameSet->setUnit(2, "Pixel");
        frameSet->setSymbol(1, "gx");
        frameSet->setSymbol(2, "gy");
        gridIndex = initialBaseIndex;
    }

    // Create the PIXEL0 frame, connecting it to the GRID Frame with a ShiftMap
    // that compensates for LSST's zero-based indexing vs. FITS 1-based indexing
    // and moves the origin to the position indicated by xy0.
    auto xy0 = Extent2I(metadata.get("LTV1", 0), metadata.get("LTV2", 0));
    std::vector<double> offsetArray = {xy0[0] - 1.0, xy0[1] - 1.0};
    auto offsetMapping = ast::ShiftMap(offsetArray);
    auto pixelFrame = ast::Frame(2);
    pixelFrame.setDomain("PIXEL0");
    pixelFrame.setTitle("Title=Pixel coordinates within parent image -- first pixel at (0, 0)");
    pixelFrame.setLabel(1, "Pixel x");
    pixelFrame.setLabel(2, "Pixel y");
    pixelFrame.setUnit(1, "Pixel");
    pixelFrame.setUnit(2, "Pixel");
    pixelFrame.setSymbol(1, "px");
    pixelFrame.setSymbol(2, "py");
    frameSet->addFrame(gridIndex, offsetMapping, pixelFrame);
    auto pixel0Index = frameSet->getCurrent();

    // Set PIXEL0 as the base frame and our ICRS sky frame as the current frame
    frameSet->setBase(pixel0Index);
    frameSet->setCurrent(stdSkyIndex);

    return frameSet;
}

}  // namespace detail
}  // namespace geom
}  // namespace afw
}  // namespace lsst

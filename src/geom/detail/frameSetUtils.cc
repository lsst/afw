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
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/wcsUtils.h"
#include "lsst/afw/geom/detail/frameSetUtils.h"
#include "lsst/afw/image/ImageBase.h"  // for wcsNameForXY0
#include "lsst/daf/base/PropertyList.h"
#include "lsst/daf/base/PropertySet.h"
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

/*
 * Copy a FITS header card from a FitsChan to a PropertyList.
 *
 * Internal function for use by getPropertyListFromFitsChan.
 *
 * @param[in,out] metadata  PropertyList to which to copy the value
 * @param[in] name  FITS header card name; used as the name for the new entry in `metadata`
 * @param[in] foundValue  Value and found flag returned by ast::FitsChan.getFits{X};
 *      foundValue.found must be true.
 * @param[in] comment  Card comment; if blank then no comment is written
 *
 * @throw lsst::pex::exceptions::LogicError if foundValue.found false.
 */
template <typename T>
void setMetadataFromFoundValue(daf::base::PropertyList& metadata, std::string const& name,
                               ast::FoundValue<T> const& foundValue, std::string const& comment = "") {
    if (!foundValue.found) {
        throw LSST_EXCEPT(pex::exceptions::LogicError, "Bug! FitsChan card \"" + name + "\" not found");
    }
    if (comment.empty()) {
        metadata.set(name, foundValue.value);
    } else {
        metadata.set(name, foundValue.value, comment);
    }
}

}  // namespace

std::shared_ptr<ast::FrameSet> readFitsWcs(daf::base::PropertySet& metadata, bool strip) {
    // Exclude WCS A keywords because LSST uses them to store XY0
    auto wcsANames = createTrivialWcsMetadata("A", Point2I(0, 0))->names();
    std::set<std::string> excludeNames(wcsANames.begin(), wcsANames.end());
    // Ignore NAXIS1, NAXIS2 because if they are zero then AST will fail to read a WCS
    // Ignore LTV1/2 because LSST adds it and this code should ignore it and not strip it
    // Exclude comments and history to reduce clutter
    std::set<std::string> moreNames{"NAXIS1", "NAXIS2", "LTV1", "LTV2", "COMMENT", "HISTORY"};
    excludeNames.insert(moreNames.begin(), moreNames.end());

    // Replace RADECSYS with RADESYS if only the former is present
    if (metadata.exists("RADECSYS") && !metadata.exists("RADESYS")) {
        metadata.set("RADESYS", metadata.getAsString("RADECSYS"));
        metadata.remove("RADECSYS");
    }

    std::string hdr = formatters::formatFitsProperties(metadata, excludeNames);
    ast::StringStream stream(hdr);
    ast::FitsChan channel(stream, "Encoding=FITS-WCS, IWC=1, SipReplace=0");
    auto const initialNames = strip ? setFromVector(channel.getAllCardNames()) : std::set<std::string>();
    std::shared_ptr<ast::Object> obj;
    try {
        obj = channel.read();
    } catch (std::runtime_error) {
        throw LSST_EXCEPT(pex::exceptions::TypeError,
                          "The metadata does not describe an AST object");
    }
    auto frameSet = std::dynamic_pointer_cast<ast::FrameSet>(obj);
    if (!frameSet) {
        throw LSST_EXCEPT(pex::exceptions::TypeError,
                          "metadata describes a " + obj->getClassName() + ", not a FrameSet");
    }
    if (strip) {
        auto const finalNames = setFromVector(channel.getAllCardNames());

        // FITS keywords that FitsChan stripped
        std::set<std::string> namesChannelStripped;
        std::set_difference(initialNames.begin(), initialNames.end(), finalNames.begin(), finalNames.end(),
                            std::inserter(namesChannelStripped, namesChannelStripped.begin()));

        // FITS keywords that FitsChan may strip that we want to keep in `metadata`
        // TODO DM-10411: remove TIMESYS from this list after starlink_ast is updated (no rush)
        std::set<std::string> const namesToKeep = {"DATE-OBS", "MJD-OBS", "TIMESYS"};

        std::set<std::string> namesToStrip;  // names to strip from metadata
        std::set_difference(namesChannelStripped.begin(), namesChannelStripped.end(), namesToKeep.begin(),
                            namesToKeep.end(), std::inserter(namesToStrip, namesToStrip.begin()));
        for (auto const& name : namesToStrip) {
            metadata.remove(name);
        }
    }
    return frameSet;
}

std::shared_ptr<ast::FrameDict> readLsstSkyWcs(daf::base::PropertySet& metadata, bool strip) {
    // Record CRPIX in GRID coordinates
    // so we can compute CRVAL after standardizing the SkyFrame to ICRS
    // (that standardization is why we don't simply save CRVAL now)
    std::vector<double> crpixGrid(2);
    try {
        crpixGrid[0] = metadata.getAsDouble("CRPIX1");
        crpixGrid[1] = metadata.getAsDouble("CRPIX2");
    } catch (lsst::pex::exceptions::NotFoundError& e) {
        // std::string used because e.what() returns a C string and two C strings cannot be added
        throw LSST_EXCEPT(lsst::pex::exceptions::TypeError,
                          e.what() + std::string("; cannot read metadata as a SkyWcs"));
    }

    auto rawFrameSet = readFitsWcs(metadata, strip);
    auto const initialBaseIndex = rawFrameSet->getBase();

    // Find the GRID frame
    auto gridIndex = ast::FrameSet::NOFRAME;
    if (rawFrameSet->findFrame(ast::Frame(2, "Domain=GRID"))) {
        gridIndex = rawFrameSet->getCurrent();
    } else {
        // No appropriate GRID frame found; if the original base frame is of type Frame
        // with 2 axes and a blank domain then use that, else give up
        auto const baseFrame = rawFrameSet->getFrame(initialBaseIndex, false);
        auto const baseClassName = rawFrameSet->getClassName();
        if (baseFrame->getClassName() != "Frame") {
            throw LSST_EXCEPT(pex::exceptions::TypeError,
                              "The base frame is of type " + baseFrame->getClassName() +
                                      "instead of Frame; cannot read metadata as a SkyWcs");
        }
        if (baseFrame->getNAxes() != 2) {
            throw LSST_EXCEPT(pex::exceptions::TypeError,
                              "The base frame has " + std::to_string(baseFrame->getNAxes()) +
                                      " axes instead of 2; cannot read metadata as a SkyWcs");
        }
        if (baseFrame->getDomain() != "") {
            throw LSST_EXCEPT(pex::exceptions::TypeError,
                              "The base frame has domain \"" + baseFrame->getDomain() +
                                      "\" instead of blank or GRID; cannot read metadata as a SkyWcs");
        }
        // Original base frame has a blank Domain, is of type Frame, and has 2 axes, so
        // Set its domain to GRID, and set some other potentially useful attributes.
        gridIndex = initialBaseIndex;
    }

    // Find the IWC frame
    if (!rawFrameSet->findFrame(ast::Frame(2, "Domain=IWC"))) {
        throw LSST_EXCEPT(pex::exceptions::TypeError,
                          "No IWC frame found; cannot read metadata as a SkyWcs");
    }
    auto const iwcIndex = rawFrameSet->getCurrent();
    auto const iwcFrame = rawFrameSet->getFrame(iwcIndex);

    // Create a standard sky frame: ICRS with axis order RA, Dec

    // Create the a template for the standard sky frame
    auto const stdSkyFrameTemplate = ast::SkyFrame("System=ICRS");

    // Locate a Frame in the target FrameSet that looks like the template
    // and hence can be used as the original sky frame.
    // We ignore the frame set returned by findFrame because that goes from pixels to sky,
    // and using it would add an unwanted extra branch to our WCS; instead, later on,
    // we compute a mapping from the old sky frame to the new sky frame and add that.
    if (!rawFrameSet->findFrame(stdSkyFrameTemplate)) {
        throw LSST_EXCEPT(pex::exceptions::TypeError,
                          "Could not find a SkyFrame; cannot read metadata as a SkyWcs");
    }
    auto initialSkyIndex = rawFrameSet->getCurrent();

    // Compute a frame set that maps from the original sky frame to our desired sky frame template;
    // this contains the mapping and sky frame we will insert into the frame set.
    // (Temporarily set the base frame to the sky frame, because findFrame
    // produces a mapping from base to the found frame).
    rawFrameSet->setBase(initialSkyIndex);
    auto stdSkyFrameSet = rawFrameSet->findFrame(stdSkyFrameTemplate);
    if (!stdSkyFrameSet) {
        throw LSST_EXCEPT(pex::exceptions::LogicError,
                          "Bug: found a SkyFrame the first time, but not the second time");
    }

    // Add the new mapping into rawFrameSet, connecting it to the original SkyFrame.
    // Note: we use stdSkyFrameSet as the new frame (meaning stdSkyFrameSet's current frame),
    // because, unlike stdSkyFrameTemplate, stdSkyFrameSet's current frame has inherited some
    // potentially useful attributes from the old sky frame, such as epoch.
    rawFrameSet->addFrame(initialSkyIndex, *stdSkyFrameSet->getMapping()->simplify(),
                          *stdSkyFrameSet->getFrame(ast::FrameSet::CURRENT));
    auto const stdSkyIndex = rawFrameSet->getCurrent();
    auto const stdSkyFrame = rawFrameSet->getFrame(stdSkyIndex, false);

    // Compute a mapping from PIXELS (0-based in parent coordinates)
    // to GRID (1-based in local coordinates)
    auto xy0 = getImageXY0FromMetadata(metadata, image::detail::wcsNameForXY0, strip);
    std::vector<double> pixelToGridArray = {1.0 - xy0[0], 1.0 - xy0[1]};  // 1.0 for FITS vs LSST convention
    auto pixelToGrid = ast::ShiftMap(pixelToGridArray);

    // Now construct the returned FrameDict
    auto const gridToIwc = rawFrameSet->getMapping(gridIndex, iwcIndex)->simplify();
    auto const pixelToIwc = pixelToGrid.then(*gridToIwc).simplify();
    auto const iwcToStdSky = rawFrameSet->getMapping(iwcIndex, stdSkyIndex);

    auto frameDict = std::make_shared<ast::FrameDict>(ast::Frame(2, "Domain=PIXELS"), *pixelToIwc, *iwcFrame);
    frameDict->addFrame("IWC", *iwcToStdSky, *stdSkyFrame);

    // Record CRVAL as SkyRef in the SkyFrame so it can easily be obtained later;
    // set SkyRefIs = "Ignored" (the default) so SkyRef value is ignored instead of used as an offset
    auto crpixPixels = pixelToGrid.applyInverse(crpixGrid);
    auto crvalRad = frameDict->applyForward(crpixPixels);
    auto skyFrame = std::dynamic_pointer_cast<ast::SkyFrame>(frameDict->getFrame("SKY", false));
    if (!skyFrame) {
        throw LSST_EXCEPT(pex::exceptions::LogicError, "SKY frame is not a SkyFrame");
    }
    skyFrame->setSkyRefIs("Ignored");
    skyFrame->setSkyRef(crvalRad);

    return frameDict;
}

std::shared_ptr<daf::base::PropertyList> getPropertyListFromFitsChan(ast::FitsChan& fitsChan) {
    int const numCards = fitsChan.getNCard();
    auto metadata = std::make_shared<daf::base::PropertyList>();
    for (int cardNum = 1; cardNum <= numCards; ++cardNum) {
        fitsChan.setCard(cardNum);
        auto const cardType = fitsChan.getCardType();
        auto const cardName = fitsChan.getCardName();
        auto const cardComment = fitsChan.getCardComm();
        switch (cardType) {
            case ast::CardType::FLOAT: {
                auto foundValue = fitsChan.getFitsF();
                setMetadataFromFoundValue(*metadata, cardName, foundValue, cardComment);
                break;
            }
            case ast::CardType::INT: {
                auto foundValue = fitsChan.getFitsI();
                setMetadataFromFoundValue(*metadata, cardName, foundValue, cardComment);
                break;
            }
            case ast::CardType::STRING: {
                auto foundValue = fitsChan.getFitsS();
                setMetadataFromFoundValue(*metadata, cardName, foundValue, cardComment);
                break;
            }
            case ast::CardType::LOGICAL: {
                auto foundValue = fitsChan.getFitsL();
                setMetadataFromFoundValue(*metadata, cardName, foundValue, cardComment);
                break;
            }
            case ast::CardType::CONTINUE: {
                auto foundValue = fitsChan.getFitsCN();
                setMetadataFromFoundValue(*metadata, cardName, foundValue, cardComment);
                break;
            }
            case ast::CardType::COMMENT:
                // Drop HISTORY and COMMENT cards
                break;
            case ast::CardType::COMPLEXF:
            case ast::CardType::COMPLEXI:
            case ast::CardType::UNDEF: {
                // PropertyList supports neither complex numbers nor cards with no value
                std::ostringstream os;
                os << "Card " << cardNum << " with name \"" << cardName << "\" has type "
                   << static_cast<int>(cardType) << ", which is not supported by PropertyList";
                throw LSST_EXCEPT(lsst::pex::exceptions::TypeError, os.str());
            }
            case ast::CardType::NOTYPE: {
                // This should only occur if cardNum is invalid, and that should be impossible
                std::ostringstream os;
                os << "Bug! Card " << cardNum << " with name \"" << cardName
                   << "\" has type NOTYPE, which should not be possible";
                throw LSST_EXCEPT(lsst::pex::exceptions::TypeError, os.str());
            }
        }
    }
    return metadata;
}

}  // namespace detail
}  // namespace geom
}  // namespace afw
}  // namespace lsst

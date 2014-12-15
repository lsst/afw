/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
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
#include "boost/bind.hpp"

#include "lsst/afw/detection/FootprintMerge.h"
#include "lsst/afw/detection/FootprintSet.h"
#include "lsst/afw/table/IdFactory.h"
#include "lsst/pex/logging/Debug.h"

namespace lsst { namespace afw { namespace detection {


namespace {

FootprintSet mergeFootprintPair(Footprint const &foot1, Footprint const &foot2) {

    geom::Box2I bbox(foot1.getBBox());
    bbox.include(foot2.getBBox());

    boost::uint16_t bits = 0x1;
    image::Mask<boost::uint16_t> mask(bbox);
    setMaskFromFootprint(&mask, foot1, bits);
    setMaskFromFootprint(&mask, foot2, bits);
    FootprintSet fpSet(mask, Threshold(bits, Threshold::BITMASK));
    return fpSet;
}
} // anonymous namespace

FootprintMerge::FootprintMerge():
    _merge(PTR(Footprint)())
{
    _merge->getPeaks() = PeakCatalog(PeakTable::makeMinimalSchema());
}

FootprintMerge::FootprintMerge(PTR(Footprint) foot):
    _footprints(1,foot),
    _merge(boost::make_shared<Footprint>(*foot))
{
}

bool FootprintMerge::overlaps(Footprint const &foot) const
{

    return mergeFootprintPair(*_merge, foot).getFootprints()->size() == 1u;
}

void FootprintMerge::add(PTR(Footprint) foot, float minNewPeakDist)
{
    FootprintSet fpSet = mergeFootprintPair(*_merge, *foot);
    if (fpSet.getFootprints()->size() != 1u) return;

    _merge->_bbox.include(foot->getBBox());
    _merge->getSpans().swap(fpSet.getFootprints()->front()->getSpans());
    _footprints.push_back(foot);

    if (minNewPeakDist < 0) return;

    PeakCatalog &currentPeaks = _merge->getPeaks();
    PeakCatalog &otherPeaks = foot->getPeaks();

    // Create new list of peaks
    PeakCatalog newPeaks(currentPeaks.getTable());
    float minNewPeakDist2 = minNewPeakDist*minNewPeakDist;
    for (PeakCatalog::const_iterator otherIter = otherPeaks.begin();
         otherIter != otherPeaks.end(); ++otherIter) {

        float minDist2 = std::numeric_limits<float>::infinity();
        for (PeakCatalog::const_iterator currentIter = currentPeaks.begin();
             currentIter != currentPeaks.end(); ++currentIter) {
            float dist2 = otherIter->getI().distanceSquared(currentIter->getI());
            if (dist2 < minDist2) {
                minDist2 = dist2;
            }
        }

        if (minDist2 > minNewPeakDist2) {
            newPeaks.push_back(otherIter);
        }
    }

    _merge->getPeaks().insert(_merge->getPeaks().end(), newPeaks.begin(), newPeaks.end(), true);
}



FootprintMergeList::FootprintMergeList(afw::table::Schema & schema,
                                       std::vector<std::string> const &filterList)
{

    // Add Flags for the filters
    for (std::vector<std::string>::const_iterator iter=filterList.begin();
         iter!=filterList.end();  ++iter) {
        _filterMap[*iter] = schema.addField<afw::table::Flag>("merge."+*iter,
                                                              "Object detected in filter "+*iter);
    }
}

namespace {
bool ContainsId(std::vector<afw::table::RecordId> const &idList,
                FootprintMergeList::SourceMerge const &merge) {
    return std::find(idList.begin(), idList.end(), merge.src->getId()) != idList.end();
}
} // anonymous namespace
 
void FootprintMergeList::addCatalog(PTR(afw::table::SourceTable) &table,
                                    afw::table::SourceCatalog const &inputCat, std::string filter,
                                    float minNewPeakDist, bool doMerge)
{
    pex::logging::Debug log("afw.detection.FootprintMerge");
    if (_filterMap.find(filter) == _filterMap.end()) {
        pex::logging::Log::getDefaultLog().warn(
            boost::format("Filter %s is not in inital filter List: %s") % filter
            );
        return;
    }

    // If list is empty don't check for any matches, just add all the objects
    bool checkForMatches = (_mergeList.size() > 0);

    for (afw::table::SourceCatalog::const_iterator srcIter = inputCat.begin(); srcIter != inputCat.end();
         ++srcIter) {

        // Only consider unblended objects
        if (srcIter->getParent() != 0) continue;

        PTR(Footprint) foot = srcIter->getFootprint();

        // Empty pointer to account for the first match in the catalog.  If there is more than one
        // match, subsequent matches will be merged with this one
        PTR(FootprintMerge) first = PTR(FootprintMerge)();

        if (checkForMatches) {

            // This is a list of objects that have been merged to others and need to be deleted
            std::vector<afw::table::RecordId> removeList;
            for (FootprintMergeVec::iterator iter = _mergeList.begin(); iter != _mergeList.end(); ++iter)  {

                // skip this entry if we are going to remove it
                if (std::find(removeList.begin(), removeList.end(), iter->src->getId()) != removeList.end()) {
                    continue;
                }

                // Grow by one pixel to allow for touching
                geom::Box2I box(iter->merge->getBBox());
                box.grow(geom::Extent2I(1,1));

                if (!box.overlaps(foot->getBBox())) continue;

                if (iter->merge->overlaps(*foot)) {
                    if (!first) {
                        first = iter->merge;
                        // Add Footprint to existing merge and set flag for this band
                        if(doMerge) {
                            first->add(foot, minNewPeakDist);
                            iter->src->set(_filterMap[filter], true);
                        }
                    }
                    else {
                        // Add merged Footprint to first
                        if(doMerge) {
                            first->add(iter->merge->getMergedFootprint(), minNewPeakDist);
                            removeList.push_back(iter->src->getId());
                        }
                    }
                }
            }

            // Remove entries that were merged to other objects
            _mergeList.erase(
                std::remove_if(_mergeList.begin(), _mergeList.end(),
                               boost::bind(ContainsId,removeList, _1)),
                _mergeList.end()
                );
        }

        if (!first) {
            SourceMerge newSource;
            newSource.src = table->makeRecord();
            newSource.src->set(_filterMap[filter], true);

            newSource.merge = boost::make_shared<FootprintMerge>(foot);
            _mergeList.push_back(newSource);
        }
    }
}

void FootprintMergeList::getFinalSources(afw::table::SourceCatalog &outputCat, bool doNorm)
{
    // Now set the merged footprint as the footprint of the SourceRecord
    for (FootprintMergeVec::iterator iter = _mergeList.begin(); iter != _mergeList.end(); ++iter)  {
        if (doNorm) iter->merge->getMergedFootprint()->normalize();
        iter->src->setFootprint(iter->merge->getMergedFootprint());
        outputCat.push_back(iter->src);
    }
}

}}} // namespace lsst::afw::detection

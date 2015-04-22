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

class FootprintMerge {
public:

    typedef FootprintMergeList::FlagKey FlagKey;
    typedef FootprintMergeList::FilterMap FilterMap;

    explicit FootprintMerge(
        PTR(Footprint) footprint,
        PTR(afw::table::SourceTable) sourceTable,
        FlagKey const & flagKey
    ) :
        _footprints(1, footprint),
        _source(sourceTable->makeRecord())
    {
        _source->setFootprint(boost::make_shared<Footprint>(*footprint));
        _source->set(flagKey, true);
    }

    /*
     *  Does this Footprint overlap the merged Footprint.
     *
     *  The current implementation just builds an image from the two Footprints and
     *  detects the number of peaks.  This is not very efficient and will be changed
     *  within the Footprint class in the future.
     */
    bool overlaps(Footprint const &rhs) const {
        return mergeFootprintPair(*getMergedFootprint(), rhs).getFootprints()->size() == 1u;
    }

    /*
     *  Add this Footprint to the merge.
     *
     *  If minNewPeakDist >= 0, it will add all peaks from foot to the merged Footprint
     *  that are greater than minNewPeakDist away from the closest existing peak.
     *  If minNewPeakDist < 0, no peaks will be added from foot.
     *
     *  If foot does not overlap it will do nothing.
     */
    void add(PTR(Footprint) footprint, FlagKey const & flagKey, float minNewPeakDist=-1.) {
        if (_addImpl(footprint, minNewPeakDist)) {
            _footprints.push_back(footprint);
            _source->set(flagKey, true);
        }
    }

    /*
     *  Merge an already-merged clump of Footprints into this
     *
     *  If minNewPeakDist >= 0, it will add all peaks from foot to the merged Footprint
     *  that are greater than minNewPeakDist away from the closest existing peak.
     *  If minNewPeakDist < 0, no peaks will be added from foot.
     *
     *  If foot does not overlap it will do nothing.
     */
    void add(FootprintMerge const & other, FilterMap const & keys, float minNewPeakDist=-1.) {
        if (_addImpl(other.getMergedFootprint(), minNewPeakDist)) {
            _footprints.insert(_footprints.end(), other._footprints.begin(), other._footprints.end());
            // Set source flags to the OR of the flags of the two inputs
            for (FilterMap::const_iterator i = keys.begin(); i != keys.end(); ++i) {
                _source->set(i->second, _source->get(i->second) || other._source->get(i->second));
            }
        }
    }

    // Get the bounding box of the merge
    afw::geom::Box2I getBBox() const { return getMergedFootprint()->getBBox(); }

    PTR(Footprint) getMergedFootprint() const { return _source->getFootprint(); }

    PTR(afw::table::SourceRecord) getSource() const { return _source; }

private:

    // Implementation helper for add() methods; returns true if the Footprint actually overlapped
    // and was merged, and false otherwise.
    bool _addImpl(PTR(Footprint) footprint, float minNewPeakDist);

    std::vector<PTR(Footprint)> _footprints;
    PTR(afw::table::SourceRecord) _source;
};

bool FootprintMerge::_addImpl(PTR(Footprint) footprint, float minNewPeakDist) {
    FootprintSet fpSet = mergeFootprintPair(*getMergedFootprint(), *footprint);
    if (fpSet.getFootprints()->size() != 1u) return false;

    getMergedFootprint()->_bbox.include(footprint->getBBox());
    getMergedFootprint()->getSpans().swap(fpSet.getFootprints()->front()->getSpans());

    if (minNewPeakDist < 0) return true;

    PeakCatalog &currentPeaks = getMergedFootprint()->getPeaks();
    PeakCatalog &otherPeaks = footprint->getPeaks();

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

    getMergedFootprint()->getPeaks().insert(
        getMergedFootprint()->getPeaks().end(),
        newPeaks.begin(), newPeaks.end(),
        true // deep-copy
    );

    return true;
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
                PTR(FootprintMerge) const & merge) {
    return std::find(idList.begin(), idList.end(), merge->getSource()->getId()) != idList.end();
}

} // anonymous namespace

void FootprintMergeList::addCatalog(
    PTR(afw::table::SourceTable) table,
    afw::table::SourceCatalog const &inputCat,
    std::string const & filter,
    float minNewPeakDist, bool doMerge
) {
    FilterMap::const_iterator keyIter = _filterMap.find(filter);
    if (keyIter == _filterMap.end()) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicError,
            (boost::format("Filter %s not in original list") % filter).str()
        );
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
                if (
                    std::find(removeList.begin(), removeList.end(), (**iter).getSource()->getId())
                    != removeList.end()
                ) {
                    continue;
                }

                // Grow by one pixel to allow for touching
                geom::Box2I box((**iter).getBBox());
                box.grow(geom::Extent2I(1,1));

                if (!box.overlaps(foot->getBBox())) continue;

                if ((**iter).overlaps(*foot)) {
                    if (!first) {
                        first = *iter;
                        // Add Footprint to existing merge and set flag for this band
                        if (doMerge) {
                            first->add(foot, keyIter->second, minNewPeakDist);
                        }
                    } else {
                        // Add merged Footprint to first
                        if (doMerge) {
                            first->add(**iter, _filterMap, minNewPeakDist);
                            removeList.push_back((**iter).getSource()->getId());
                        }
                    }
                }
            }

            // Remove entries that were merged to other objects
            _mergeList.erase(
                std::remove_if(_mergeList.begin(), _mergeList.end(),
                               boost::bind(ContainsId, removeList, _1)),
                _mergeList.end()
            );
        }

        if (!first) {
            _mergeList.push_back(boost::make_shared<FootprintMerge>(foot, table, keyIter->second));
        }
    }
}

void FootprintMergeList::getFinalSources(afw::table::SourceCatalog &outputCat, bool doNorm)
{
    // Now set the merged footprint as the footprint of the SourceRecord
    for (FootprintMergeVec::iterator iter = _mergeList.begin(); iter != _mergeList.end(); ++iter)  {
        if (doNorm) (**iter).getMergedFootprint()->normalize();
        outputCat.push_back((**iter).getSource());
    }
}

}}} // namespace lsst::afw::detection

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

    typedef FootprintMergeList::KeyTuple KeyTuple;
    typedef FootprintMergeList::FilterMap FilterMap;

    explicit FootprintMerge(
        PTR(Footprint) footprint,
        PTR(afw::table::SourceTable) sourceTable,
        PTR(PeakTable) peakTable,
        afw::table::SchemaMapper const & peakSchemaMapper,
        KeyTuple const & keys
    ) :
        _footprints(1, footprint),
        _source(sourceTable->makeRecord())
    {
        PTR(Footprint) newFootprint = boost::make_shared<Footprint>(*footprint);

        _source->set(keys.footprint, true);
        // Replace all the Peaks in the merged Footprint with new ones that include the origin flags
        newFootprint->getPeaks() = PeakCatalog(peakTable);
        for (
            PeakCatalog::iterator iter = footprint->getPeaks().begin();
            iter != footprint->getPeaks().end();
            ++iter
        ) {
            PTR(PeakRecord) newPeak = peakTable->copyRecord(*iter, peakSchemaMapper);
            newPeak->set(keys.peak, true);
            newFootprint->getPeaks().push_back(newPeak);
        }
        _source->setFootprint(newFootprint);
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
    void add(
        PTR(Footprint) footprint,
        afw::table::SchemaMapper const & peakSchemaMapper,
        KeyTuple const & keys,
        float minNewPeakDist=-1.,
        float maxSamePeakDist=-1.
    ) {
        if (_addSpans(footprint)) {
            _footprints.push_back(footprint);
            _source->set(keys.footprint, true);
            _addPeaks(footprint->getPeaks(), &peakSchemaMapper, &keys, minNewPeakDist, maxSamePeakDist);
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
    void add(
        FootprintMerge const & other,
        FilterMap const & keys,
        float minNewPeakDist=-1.,
        float maxSamePeakDist=-1.
        ) {
        if (_addSpans(other.getMergedFootprint())) {
            _footprints.insert(_footprints.end(), other._footprints.begin(), other._footprints.end());
            // Set source flags to the OR of the flags of the two inputs
            for (FilterMap::const_iterator i = keys.begin(); i != keys.end(); ++i) {
                afw::table::Key<afw::table::Flag> const & flagKey = i->second.footprint;
                _source->set(flagKey, _source->get(flagKey) || other._source->get(flagKey));
            }
            _addPeaks(other.getMergedFootprint()->getPeaks(), NULL, NULL, minNewPeakDist, maxSamePeakDist);
        }
    }

    // Get the bounding box of the merge
    afw::geom::Box2I getBBox() const { return getMergedFootprint()->getBBox(); }

    PTR(Footprint) getMergedFootprint() const { return _source->getFootprint(); }

    PTR(afw::table::SourceRecord) getSource() const { return _source; }

private:

    // Implementation helper for add() methods; returns true if the Footprint actually overlapped
    // and was merged, and false otherwise.
    bool _addSpans(PTR(Footprint) footprint) {
        FootprintSet fpSet = mergeFootprintPair(*getMergedFootprint(), *footprint);
        if (fpSet.getFootprints()->size() != 1u) return false;
        getMergedFootprint()->_bbox.include(footprint->getBBox());
        getMergedFootprint()->getSpans().swap(fpSet.getFootprints()->front()->getSpans());
        return true;
    }

    void _addPeaks(
        PeakCatalog const & otherPeaks,
        afw::table::SchemaMapper const * peakSchemaMapper,
        KeyTuple const * keys,
        float minNewPeakDist,
        float maxSamePeakDist
    ) {
        if (minNewPeakDist < 0 && maxSamePeakDist < 0) return;

        PeakCatalog & currentPeaks = getMergedFootprint()->getPeaks();
        PTR(PeakRecord) nearestPeak;
        // Create new list of peaks
        PeakCatalog newPeaks(currentPeaks.getTable());
        float minNewPeakDist2 = minNewPeakDist*minNewPeakDist;
        float maxSamePeakDist2 = maxSamePeakDist*maxSamePeakDist;
        for (PeakCatalog::const_iterator otherIter = otherPeaks.begin();
             otherIter != otherPeaks.end(); ++otherIter) {

            float minDist2 = std::numeric_limits<float>::infinity();

            for (PeakCatalog::const_iterator currentIter = currentPeaks.begin();
                 currentIter != currentPeaks.end(); ++currentIter) {
                float dist2 = otherIter->getI().distanceSquared(currentIter->getI());

                if (dist2 < minDist2) {
                    minDist2 = dist2;
                    nearestPeak = currentIter;
                }
            }

            if (minDist2 < maxSamePeakDist2 && nearestPeak && keys && maxSamePeakDist > 0) {
                nearestPeak->set(keys->peak, true);
            } else if (minDist2 > minNewPeakDist2 && !(minNewPeakDist < 0)) {
                if (peakSchemaMapper) {
                    PTR(PeakRecord) newPeak = newPeaks.addNew();
                    newPeak->assign(*otherIter, *peakSchemaMapper);
                    newPeak->set(keys->peak, true);
                } else {
                    newPeaks.push_back(otherIter);
                }
            }

        }

        getMergedFootprint()->getPeaks().insert(
            getMergedFootprint()->getPeaks().end(),
            newPeaks.begin(), newPeaks.end(),
            true // deep-copy
        );
    }

    std::vector<PTR(Footprint)> _footprints;
    PTR(afw::table::SourceRecord) _source;
};


FootprintMergeList::FootprintMergeList(
    afw::table::Schema & sourceSchema,
    std::vector<std::string> const & filterList,
    afw::table::Schema const & initialPeakSchema
) : _peakSchemaMapper(initialPeakSchema) {
    _initialize(sourceSchema, filterList);
}

FootprintMergeList::FootprintMergeList(
    afw::table::Schema & sourceSchema,
    std::vector<std::string> const & filterList
) : _peakSchemaMapper(PeakTable::makeMinimalSchema()) {
    _initialize(sourceSchema, filterList);
}

void FootprintMergeList::_initialize(
    afw::table::Schema & sourceSchema,
    std::vector<std::string> const & filterList
) {
    _peakSchemaMapper.addMinimalSchema(_peakSchemaMapper.getInputSchema(), true);
    // Add Flags for the filters
    for (
        std::vector<std::string>::const_iterator iter=filterList.begin();
        iter != filterList.end();
        ++iter
    ) {
        KeyTuple & keys = _filterMap[*iter];
        keys.footprint = sourceSchema.addField<afw::table::Flag>(
            "merge_footprint_" + *iter,
            "Detection footprint overlapped with a detection from filter " + *iter
        );
        keys.peak = _peakSchemaMapper.editOutputSchema().addField<afw::table::Flag>(
            "merge_peak_" + *iter,
            "Peak detected in filter " + *iter
        );
    }
    _peakTable = PeakTable::make(_peakSchemaMapper.getOutputSchema());
}

void FootprintMergeList::addCatalog(
    PTR(afw::table::SourceTable) sourceTable,
    afw::table::SourceCatalog const &inputCat,
    std::string const & filter,
    float minNewPeakDist, bool doMerge, float maxSamePeakDist
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
            FootprintMergeVec::iterator iter = _mergeList.begin();
            while (iter != _mergeList.end()) {
                // Grow by one pixel to allow for touching
                geom::Box2I box((**iter).getBBox());
                box.grow(geom::Extent2I(1,1));
                if (box.overlaps(foot->getBBox()) && (**iter).overlaps(*foot)) {
                    if (!first) {
                        first = *iter;
                        // Add Footprint to existing merge and set flag for this band
                        if (doMerge) {
                            first->add(foot, _peakSchemaMapper, keyIter->second, minNewPeakDist,
                                       maxSamePeakDist);
                        }
                    } else {
                        // Add merged Footprint to first
                        if (doMerge) {
                            first->add(**iter, _filterMap, minNewPeakDist, maxSamePeakDist);
                            iter = _mergeList.erase(iter);
                            continue;
                        }
                    }
                }
                ++iter;
            }
        }

        if (!first) {
            _mergeList.push_back(
                boost::make_shared<FootprintMerge>(
                    foot, sourceTable, _peakTable, _peakSchemaMapper, keyIter->second
                )
            );
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

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

#ifndef LSST_AFW_DETECTION_FOOTPRINTMERGE_H
#define LSST_AFW_DETECTION_FOOTPRINTMERGE_H

#include <vector>
#include <map>

#include "lsst/afw/table/Source.h"

namespace lsst { namespace afw { namespace detection {
/**
 *  @brief Object that represents a merge of Footprints.
 *
 *  Contains a vector of individual Footprints and a composite Footprint that is the union of them.
 *  The merging and overlap functions are currently not very efficient for large Footprints as
 *  they draw Footprints into a mask and then detect the results as a FootprintSet.  Improvements to
 *  these Functions will be addressed in future changes of the Footprint code.
 *
 *  Given a set of overlapping Footprints, the final merged Footprint will depend on the order
 *  that they are added.
 */
class FootprintMerge {
public:

    FootprintMerge();
    FootprintMerge(PTR(Footprint) foot);

    /**
     *  @brief Does this Footprint overlap the merged Footprint.
     *
     *  The current implementation just builds an image from the two Footprints and
     *  detects the number of peaks.  This is not very efficient and will be changed
     *  within the Footprint class in the future.
     */
    bool overlaps(Footprint const &rhs);

    /**
     *  @brief Add this Footprint to the merge.
     *
     *  If minNewPeakDist >= 0, it will add all peaks from foot to the merged Footprint
     *  that are greater than minNewPeakDist away from the closest existing peak.
     *  If minNewPeakDist < 0, no peaks will be added from foot.
     *
     *  If foot does not overlap it will do nothing.
     */
    void add(PTR(Footprint) foot, float minNewPeakDist=-1.);

    // Get the bounding box of the merge
    afw::geom::Box2I getBBox() const { return merge->getBBox(); }
    afw::geom::Box2I & getBBox() { return merge->getBBox(); }

    std::vector<PTR(Footprint)> & getFootprints() { return _footprints; }

    PTR(Footprint) getMergedFootprint() const { return merge; }

private:

    std::vector<PTR(Footprint)> _footprints;
    PTR(Footprint) merge;
};


/**
 *  @brief List of Merged Footprints.
 *
 *  Stores a vector of FootprintMerges and SourceRecords that contain the union of different footprints and
 *  which filters it was detected in.  Individual Footprints from a SourceCatalog can be added to
 *  the vector (note that any SourceRecords with parent!=0 will be skipped).  If a Footprint overlaps an
 *  existing FootprintMerge, the Footprint will be added to it.  If not, then a new FootprintMerge will be
 *  created and added to the vector.
 *
 *  The search algorithm uses a brute force approach over the current list.  This should be fine if we 
 *  are operating on smallish number of objects, such as at the tract level.
 *
 */
 class FootprintMergeList
 {
 public:


     FootprintMergeList(afw::table::Schema &schema,
                        std::vector<std::string> const &filterList);

     /**
     *  @brief Add objects from a SourceCatalog in the specified filter
     *
     *  Iterate over all objects that have not been deblendend and search for an overlapping
     *  FootprintMerge in _mergeList.  If it overlaps, then it will be added to it,
     *  otherwise it will create a new one.  If minNewPeakDist < 0, then new peaks will
     *  not be added to existing footprints.  If minNewPeakDist >= 0, then new peaks will be added
     *  that are farther away than minNewPeakDist to the nearest existing peak.
     *
     *  The SourceTable is used to create new SourceRecords that store the filter information.
     */
     void addCatalog(PTR(afw::table::SourceTable) &table, afw::table::SourceCatalog const &inputCat,
                     std::string filter, float minNewPeakDist=-1.);

     /**
     *  @brief Clear entries in the current vector
     */
     void clearCatalog() { _mergeList.clear(); }

     /**
     *  @brief Get SourceCatalog with entries that contain the final Footprint and SourceRecord for each entry
     *
     *  The resulting Footprints will be normalized, meaning that there peaks are sorted, and
     *  areas are calculated.
     */
     void getFinalSources(afw::table::SourceCatalog &outputCat);

#ifndef SWIG
     // Class to store SourceRecord and FootprintMerge.
     struct SourceMerge{
         PTR(afw::table::SourceRecord) src;
         PTR(FootprintMerge) merge;
     };
#endif

 private:

     typedef std::vector<SourceMerge> FootprintMergeVec;

     FootprintMergeVec _mergeList;
     afw::table::Schema _schema;
     std::map<std::string, afw::table::Key<afw::table::Flag> > _filterMap;
 };

}}} // namespace lsst::afw::detection

#endif // !LSST_AFW_DETECTION_FOOTPRINTMERGE_H

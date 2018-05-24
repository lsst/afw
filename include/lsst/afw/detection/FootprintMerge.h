// -*- lsst-c++ -*-
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

namespace lsst {
namespace afw {
namespace detection {

/**
 *  FootprintMerge is a private helper class for FootprintMergeList; it's only declared here (it's defined
 *  in the .cc file) so FootprintMergeList can hold a vector without an extra PImpl layer, and so Footprint
 *  can friend it.
 */
class FootprintMerge;

/**
 *  List of Merged Footprints.
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
class FootprintMergeList {
public:
    /**
     *  Initialize the merge with a custom initial peak schema
     *
     *  @param[in,out]  sourceSchema    Input schema for SourceRecords to be merged, modified on return
     *                                  to include 'merge_footprint_<filter>' Flag fields that will
     *                                  indicate the origin of the source.
     *  @param[in]      filterList      Sequence of filter names to be used in Flag fields.
     *  @param[in]      initialPeakSchema    Input schema of PeakRecords in Footprints to be merged.
     *
     *  The output schema for PeakRecords will include additional 'merge_peak_<filter>' Flag fields that
     *  indicate the origin of peaks.  This can be accessed by getPeakSchema().
     */
    FootprintMergeList(afw::table::Schema &sourceSchema, std::vector<std::string> const &filterList,
                       afw::table::Schema const &initialPeakSchema);

    /**
     *  Initialize the merge with the default peak schema
     *
     *  @param[in,out]  sourceSchema    Input schema for SourceRecords to be merged, modified on return
     *                                  to include 'merge_footprint_<filter>' Flag fields that will
     *                                  indicate the origin of the source.
     *  @param[in]      filterList      Sequence of filter names to be used in Flag fields.
     *
     *  The output schema for PeakRecords will include additional 'merge_peak_<filter>' Flag fields that
     *  indicate the origin of peaks.  This can be accessed by getPeakSchema().
     */
    FootprintMergeList(afw::table::Schema &sourceSchema, std::vector<std::string> const &filterList);

    ~FootprintMergeList();
    FootprintMergeList(FootprintMergeList const &);
    FootprintMergeList(FootprintMergeList &&);
    FootprintMergeList &operator=(FootprintMergeList const &);
    FootprintMergeList &operator=(FootprintMergeList &&);

    /// Return the schema for PeakRecords in the merged footprints.
    afw::table::Schema getPeakSchema() const { return _peakTable->getSchema(); }

    /**
     *  Add objects from a SourceCatalog in the specified filter
     *
     *  Iterate over all objects that have not been deblendend and search for an overlapping
     *  FootprintMerge in _mergeList.  If it overlaps, then it will be added to it,
     *  otherwise it will create a new one.  If minNewPeakDist < 0, then new peaks will
     *  not be added to existing footprints.  If minNewPeakDist >= 0, then new peaks will be added
     *  that are farther away than minNewPeakDist to the nearest existing peak.
     *
     *  The SourceTable is used to create new SourceRecords that store the filter information.
     */
    void addCatalog(std::shared_ptr<afw::table::SourceTable> sourceTable,
                    afw::table::SourceCatalog const &inputCat, std::string const &filter,
                    float minNewPeakDist = -1., bool doMerge = true, float maxSamePeakDist = -1.);

    /**
     *  Clear entries in the current vector
     */
    void clearCatalog() { _mergeList.clear(); }

    /**
     *  Get SourceCatalog with entries that contain the final Footprint and SourceRecord for each entry
     *
     *  The resulting Footprints will be normalized, meaning that there peaks are sorted, and
     *  areas are calculated.
     */
    void getFinalSources(afw::table::SourceCatalog &outputCat);

private:
    typedef afw::table::Key<afw::table::Flag> FlagKey;

    struct KeyTuple {
        FlagKey footprint;
        FlagKey peak;
    };

    typedef std::vector<std::shared_ptr<FootprintMerge>> FootprintMergeVec;
    typedef std::map<std::string, KeyTuple> FilterMap;

    friend class FootprintMerge;

    void _initialize(afw::table::Schema &sourceSchema, std::vector<std::string> const &filterList);

    FootprintMergeVec _mergeList;
    FilterMap _filterMap;
    afw::table::SchemaMapper _peakSchemaMapper;
    std::shared_ptr<PeakTable> _peakTable;
};
}  // namespace detection
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_DETECTION_FOOTPRINTMERGE_H

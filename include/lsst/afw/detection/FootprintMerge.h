// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#ifndef LSST_AFW_DETECTION_FOOTPRINTMERGE_H
#define LSST_AFW_DETECTION_FOOTPRINTMERGE_H

#include <vector>
#include <map>

#include "lsst/afw/table/Source.h"

namespace lsst { namespace afw { namespace detection {

/**
 *  FootprintMerge is a private helper class for FootprintMergeList; it's only declared here (it's defined
 *  in the .cc file) so FootprintMergeList can hold a vector without an extra PImpl layer, and so Footprint
 *  can friend it.
 */
class FootprintMerge;

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
    FootprintMergeList(
        afw::table::Schema & sourceSchema,
        std::vector<std::string> const & filterList,
        afw::table::Schema const & initialPeakSchema
    );

    /**
     *  Initialize the merge with the default peak schema
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
    FootprintMergeList(
        afw::table::Schema & sourceSchema,
        std::vector<std::string> const & filterList
    );

    /// Return the schema for PeakRecords in the merged footprints.
    afw::table::Schema getPeakSchema() const { return _peakTable->getSchema(); }

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
    void addCatalog(
        PTR(afw::table::SourceTable) sourceTable,
        afw::table::SourceCatalog const &inputCat,
        std::string const & filter,
        float minNewPeakDist=-1.,
        bool doMerge=true,
        float maxSamePeakDist=-1.
    );

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
    void getFinalSources(afw::table::SourceCatalog &outputCat, bool doNorm=true);

private:

    typedef afw::table::Key<afw::table::Flag> FlagKey;

    struct KeyTuple {
        FlagKey footprint;
        FlagKey peak;
    };

    typedef std::vector<PTR(FootprintMerge)> FootprintMergeVec;
    typedef std::map<std::string,KeyTuple> FilterMap;

    friend class FootprintMerge;

    void _initialize(
        afw::table::Schema & sourceSchema,
        std::vector<std::string> const & filterList
    );

    FootprintMergeVec _mergeList;
    FilterMap _filterMap;
    afw::table::SchemaMapper _peakSchemaMapper;
    PTR(PeakTable) _peakTable;
};

}}} // namespace lsst::afw::detection

#endif // !LSST_AFW_DETECTION_FOOTPRINTMERGE_H

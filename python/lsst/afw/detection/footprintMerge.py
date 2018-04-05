from __future__ import absolute_import, division, print_function

from ._footprintMerge import FootprintMergeList

__all__ = []  # only imported for side effects


def getMergedSourceCatalog(self, catalogs, filters,
                           peakDist, schema, idFactory, samePeakDist):
    """Add multiple catalogs and get the SourceCatalog with merged Footprints"""
    import lsst.afw.table as afwTable

    table = afwTable.SourceTable.make(schema, idFactory)
    mergedList = afwTable.SourceCatalog(table)

    # if peak is not an array, create an array the size of catalogs
    try:
        len(samePeakDist)
    except TypeError:
        samePeakDist = [samePeakDist] * len(catalogs)

    try:
        len(peakDist)
    except TypeError:
        peakDist = [peakDist] * len(catalogs)

    if len(peakDist) != len(catalogs):
        raise ValueError("Number of catalogs (%d) does not match length of peakDist (%d)"
                         % (len(catalogs), len(peakDist)))

    if len(samePeakDist) != len(catalogs):
        raise ValueError("Number of catalogs (%d) does not match length of samePeakDist (%d)"
                         % (len(catalogs), len(samePeakDist)))

    if len(filters) != len(catalogs):
        raise ValueError("Number of catalogs (%d) does not match number of filters (%d)"
                         % (len(catalogs), len(filters)))

    self.clearCatalog()
    for cat, filter, dist, sameDist in zip(catalogs, filters, peakDist, samePeakDist):
        self.addCatalog(table, cat, filter, dist, True, sameDist)

    self.getFinalSources(mergedList)
    return mergedList


FootprintMergeList.getMergedSourceCatalog = getMergedSourceCatalog

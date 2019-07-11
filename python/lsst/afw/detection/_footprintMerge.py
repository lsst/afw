# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__all__ = []  # only imported for side effects

from lsst.utils import continueClass
from ._detection import FootprintMergeList


@continueClass  # noqa: F811
class FootprintMergeList:
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

# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
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

from lsst.utils import continueClass
from lsst.utils.deprecated import deprecate_pybind11

from ._imageLib import ExposureInfo

__all__ = []  # import this module only for its side effects


@continueClass
class ExposureInfo:  # noqa: F811
    KEY_SUMMARY_STATS = 'SUMMARY_STATS'

    def getSummaryStats(self):
        """Get exposure summary statistics component.

        Returns
        -------
        summaryStats : `lsst.afw.image.ExposureSummaryStats`
        """
        return self.getComponent(self.KEY_SUMMARY_STATS)

    def setSummaryStats(self, summaryStats):
        """Set exposure summary statistics component.

        Parameters
        ----------
        summaryStats : `lsst.afw.image.ExposureSummaryStats`
        """
        self.setComponent(self.KEY_SUMMARY_STATS, summaryStats)

    def hasSummaryStats(self):
        """Check if exposureInfo has a summary statistics component.

        Returns
        -------
        hasSummaryStats : `bool`
            True if exposureInfo has a summary statistics component.
        """
        return self.hasComponent(self.KEY_SUMMARY_STATS)


ExposureInfo.hasFilterLabel = deprecate_pybind11(
    ExposureInfo.hasFilterLabel,
    reason="Replaced by hasFilter. Will be removed after v24.",
    version="v24.0")
ExposureInfo.getFilterLabel = deprecate_pybind11(
    ExposureInfo.getFilterLabel,
    reason="Replaced by getFilter. Will be removed after v24.",
    version="v24.0")
ExposureInfo.setFilterLabel = deprecate_pybind11(
    ExposureInfo.setFilterLabel,
    reason="Replaced by setFilter. Will be removed after v24.",
    version="v24.0")

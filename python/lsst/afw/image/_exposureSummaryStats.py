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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List
import yaml
import warnings

from ..typehandling import Storable, StorableHelperFactory


__all__ = ("ExposureSummaryStats", )


def _default_corners():
    return [float('nan')]*4


@dataclass
class ExposureSummaryStats(Storable):
    _persistence_name = 'ExposureSummaryStats'

    _factory = StorableHelperFactory(__name__, _persistence_name)

    version: int = 0

    psfSigma: float = float('nan')
    """PSF determinant radius (pixels)."""

    psfArea: float = float('nan')
    """PSF effective area (pixels**2)."""

    psfIxx: float = float('nan')
    """PSF shape Ixx (pixels**2)."""

    psfIyy: float = float('nan')
    """PSF shape Iyy (pixels**2)."""

    psfIxy: float = float('nan')
    """PSF shape Ixy (pixels**2)."""

    ra: float = float('nan')
    """Bounding box center Right Ascension (degrees)."""

    decl: float = float('nan')
    """Bounding box center Declination (degrees)."""

    zenithDistance: float = float('nan')
    """Bounding box center zenith distance (degrees)."""

    zeroPoint: float = float('nan')
    """Mean zeropoint in detector (mag)."""

    skyBg: float = float('nan')
    """Average sky background (ADU)."""

    skyNoise: float = float('nan')
    """Average sky noise (ADU)."""

    meanVar: float = float('nan')
    """Mean variance of the weight plane (ADU**2)."""

    raCorners: List[float] = field(default_factory=_default_corners)
    """Right Ascension of bounding box corners (degrees)."""

    decCorners: List[float] = field(default_factory=_default_corners)
    """Declination of bounding box corners (degrees)."""

    astromOffsetMean: float = float('nan')
    """Astrometry match offset mean."""

    astromOffsetStd: float = float('nan')
    """Astrometry match offset stddev."""

    nPsfStar: int = 0
    """Number of stars used for psf model."""

    psfStarDeltaE1Median: float = float('nan')
    """Psf stars median E1 residual (starE1 - psfE1)."""

    psfStarDeltaE2Median: float = float('nan')
    """Psf stars median E2 residual (starE2 - psfE2)."""

    psfStarDeltaE1Scatter: float = float('nan')
    """Psf stars MAD E1 scatter (starE1 - psfE1)."""

    psfStarDeltaE2Scatter: float = float('nan')
    """Psf stars MAD E2 scatter (starE2 - psfE2)."""

    psfStarDeltaSizeMedian: float = float('nan')
    """Psf stars median size residual (starSize - psfSize)."""

    psfStarDeltaSizeScatter: float = float('nan')
    """Psf stars MAD size scatter (starSize - psfSize)."""

    psfStarScaledDeltaSizeScatter: float = float('nan')
    """Psf stars MAD size scatter scaled by psfSize**2."""

    def __post_init__(self):
        Storable.__init__(self)

    def isPersistable(self):
        return True

    def _getPersistenceName(self):
        return self._persistence_name

    def _getPythonModule(self):
        return __name__

    def _write(self):
        return yaml.dump(asdict(self), encoding='utf-8')

    @staticmethod
    def _read(bytes):
        yamlDict = yaml.load(bytes, Loader=yaml.SafeLoader)
        # For forwards compatibility, filter out any fields that are
        # not defined in the dataclass.
        droppedFields = []
        for _field in list(yamlDict.keys()):
            if _field not in ExposureSummaryStats.__dataclass_fields__:
                droppedFields.append(_field)
                yamlDict.pop(_field)
        if len(droppedFields) > 0:
            droppedFieldString = ', '.join([str(f) for f in droppedFields])
            warnings.warn((f"Could not read summary fields [{droppedFieldString}]. "
                           "Please use a newer stack."), FutureWarning)
        return ExposureSummaryStats(**yamlDict)

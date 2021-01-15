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
import yaml

from ..typehandling import Storable, StorableHelperFactory


__all__ = ("ExposureSummary", )


def _default_corners():
    return [float('nan')]*4


@dataclass
class ExposureSummary(Storable):
    _factory = StorableHelperFactory(__name__, "ExposureSummary")

    version: int = 0
    # PSF determinant radius (pixels)
    psfSigma: float = float('nan')
    # PSF effective area (pixels**2)
    psfArea: float = float('nan')
    # PSF shape Ixx (pixels**2)
    psfIxx: float = float('nan')
    # PSF shape Iyy (pixels**2)
    psfIyy: float = float('nan')
    # PSF shape Ixy (pixels**2)
    psfIxy: float = float('nan')
    # Bounding box center Right Ascension (degrees)
    ra: float = float('nan')
    # Bounding box center Declination (degrees)
    decl: float = float('nan')
    # Bounding box center zenith distance (degrees)
    zenithDistance: float = float('nan')
    # Mean zeropoint in detector (mag)
    zeroPoint: float = float('nan')
    # Average sky background (ADU)
    skyBg: float = float('nan')
    # Average sky noise (ADU)
    skyNoise: float = float('nan')
    # Mean variance of the weight plane (ADU**2)
    meanVar: float = float('nan')
    # Right Ascension of bounding box corners (degrees)
    raCorners: list[float] = field(default_factory=_default_corners)
    # Declination of bounding box corners (degrees)
    decCorners: list[float] = field(default_factory=_default_corners)

    def __post_init__(self):
        Storable.__init__(self)  # required for trampoline

    def isPersistable(self):
        return True

    def _getPersistenceName(self):
        return "ExposureSummary"

    def _getPythonModule(self):
        return __name__

    def _write(self):
        return yaml.dump(asdict(self), encoding='utf-8')

    @staticmethod
    def _read(bytes):
        return ExposureSummary(**yaml.load(bytes, Loader=yaml.SafeLoader))

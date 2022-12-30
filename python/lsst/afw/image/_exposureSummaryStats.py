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

import dataclasses
from typing import TYPE_CHECKING
import yaml
import warnings

from ..typehandling import Storable, StorableHelperFactory

if TYPE_CHECKING:
    from ..table import BaseRecord, Schema

__all__ = ("ExposureSummaryStats", )


def _default_corners():
    return [float("nan")] * 4


@dataclasses.dataclass
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

    raCorners: list[float] = dataclasses.field(default_factory=_default_corners)
    """Right Ascension of bounding box corners (degrees)."""

    decCorners: list[float] = dataclasses.field(default_factory=_default_corners)
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

    psfTraceRadiusDelta: float = float('nan')
    """Delta (max - min) of the model psf trace radius values evaluated on a
    grid of unmasked pixels (pixels).
    """

    maxDistToNearestPsf: float = float('nan')
    """Maximum distance of an unmasked pixel to its nearest model psf star
    (pixels).
    """

    def __post_init__(self):
        Storable.__init__(self)

    def isPersistable(self):
        return True

    def _getPersistenceName(self):
        return self._persistence_name

    def _getPythonModule(self):
        return __name__

    def _write(self):
        return yaml.dump(dataclasses.asdict(self), encoding='utf-8')

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
            droppedFieldString = ", ".join([str(f) for f in droppedFields])
            warnings.warn(
                (
                    f"Could not read summary fields [{droppedFieldString}]. "
                    "Please use a newer stack."
                ),
                FutureWarning,
            )
        return ExposureSummaryStats(**yamlDict)

    @classmethod
    def update_schema(cls, schema: Schema) -> None:
        """Update an schema to includes for all summary statistic fields.

        Parameters
        -------
        schema : `lsst.afw.table.Schema`
            Schema to add which fields will be added.
        """
        schema.addField(
            "psfSigma",
            type="F",
            doc="PSF model second-moments determinant radius (center of chip) (pixel)",
        )
        schema.addField(
            "psfArea",
            type="F",
            doc="PSF model effective area (center of chip) (pixel**2)",
        )
        schema.addField(
            "psfIxx", type="F", doc="PSF model Ixx (center of chip) (pixel**2)"
        )
        schema.addField(
            "psfIyy", type="F", doc="PSF model Iyy (center of chip) (pixel**2)"
        )
        schema.addField(
            "psfIxy", type="F", doc="PSF model Ixy (center of chip) (pixel**2)"
        )
        schema.addField(
            "raCorners",
            type="ArrayD",
            size=4,
            doc="Right Ascension of bounding box corners (degrees)",
        )
        schema.addField(
            "decCorners",
            type="ArrayD",
            size=4,
            doc="Declination of bounding box corners (degrees)",
        )
        schema.addField(
            "ra", type="D", doc="Right Ascension of bounding box center (degrees)"
        )
        schema.addField(
            "decl", type="D", doc="Declination of bounding box center (degrees)"
        )
        schema.addField(
            "zenithDistance",
            type="F",
            doc="Zenith distance of bounding box center (degrees)",
        )
        schema.addField("zeroPoint", type="F", doc="Mean zeropoint in detector (mag)")
        schema.addField("skyBg", type="F", doc="Average sky background (ADU)")
        schema.addField("skyNoise", type="F", doc="Average sky noise (ADU)")
        schema.addField(
            "meanVar", type="F", doc="Mean variance of the weight plane (ADU**2)"
        )
        schema.addField(
            "astromOffsetMean",
            type="F",
            doc="Mean offset of astrometric calibration matches (arcsec)",
        )
        schema.addField(
            "astromOffsetStd",
            type="F",
            doc="Standard deviation of offsets of astrometric calibration matches (arcsec)",
        )
        schema.addField("nPsfStar", type="I", doc="Number of stars used for PSF model")
        schema.addField(
            "psfStarDeltaE1Median",
            type="F",
            doc="Median E1 residual (starE1 - psfE1) for psf stars",
        )
        schema.addField(
            "psfStarDeltaE2Median",
            type="F",
            doc="Median E2 residual (starE2 - psfE2) for psf stars",
        )
        schema.addField(
            "psfStarDeltaE1Scatter",
            type="F",
            doc="Scatter (via MAD) of E1 residual (starE1 - psfE1) for psf stars",
        )
        schema.addField(
            "psfStarDeltaE2Scatter",
            type="F",
            doc="Scatter (via MAD) of E2 residual (starE2 - psfE2) for psf stars",
        )
        schema.addField(
            "psfStarDeltaSizeMedian",
            type="F",
            doc="Median size residual (starSize - psfSize) for psf stars (pixel)",
        )
        schema.addField(
            "psfStarDeltaSizeScatter",
            type="F",
            doc="Scatter (via MAD) of size residual (starSize - psfSize) for psf stars (pixel)",
        )
        schema.addField(
            "psfStarScaledDeltaSizeScatter",
            type="F",
            doc="Scatter (via MAD) of size residual scaled by median size squared",
        )
        schema.addField(
            "psfTraceRadiusDelta",
            type="F",
            doc="Delta (max - min) of the model psf trace radius values evaluated on a grid of "
            "unmasked pixels (pixel).",
        )
        schema.addField(
            "maxDistToNearestPsf",
            type="F",
            doc="Maximum distance of an unmasked pixel to its nearest model psf star (pixel).",
        )

    def update_record(self, record: BaseRecord) -> None:
        """Write summary-statistic columns into a record.

        Parameters
        ----------
        record : `lsst.afw.table.BaseRecord`
            Record to update.  This is expected to frequently be an
            `ExposureRecord` instance (with higher-level code adding other
            columns and objects), but this method can work with any record
            type.
        """
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name == "version":
                continue
            elif field.type.startswith("list"):
                record[field.name][:] = value
            else:
                record[field.name] = value

    @classmethod
    def from_record(cls, record: BaseRecord) -> ExposureSummaryStats:
        """Read summary-statistic columns from a record into ``self``.

        Parameters
        ----------
        record : `lsst.afw.table.BaseRecord`
            Record to read from.  This is expected to frequently be an
            `ExposureRecord` instance (with higher-level code adding other
            columns and objects), but this method can work with any record
            type, ignoring any attributes or columns it doesn't recognize.

        Returns
        -------
        summary : `ExposureSummaryStats`
            Summary statistics object created from the given record.
        """
        return cls(
            **{
                field.name: (
                    record[field.name] if not field.type.startswith("list")
                    else [float(v) for v in record[field.name]]
                )
                for field in dataclasses.fields(cls)
                if field.name != "version"
            }
        )

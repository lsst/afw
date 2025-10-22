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

__all__ = ["Fits"]

from collections.abc import Mapping

from lsst.utils import continueClass
from ._fits import (
    CompressionAlgorithm,
    CompressionOptions,
    DitherAlgorithm,
    Fits,
    QuantizationOptions,
    ScalingAlgorithm,
)


@continueClass
class Fits:  # noqa: F811
    def __enter__(self):
        return self

    def __exit__(self, cls, exc, traceback):
        self.closeFile()


@continueClass
class QuantizationOptions:  # noqa: F811
    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> QuantizationOptions:
        """Construct from a dictionary with keys matching this struct's fields.

        Parameters
        ----------
        mapping : `~collections.abc.Mapping`
            Mapping from string to value.  Enumeration values may be passed as
            string value names.

        Returns
        -------
        options : `QuantizationOptions`
            An instance of this options class.

        Notes
        -----
        Allowed keys are:

        - ``dither``: `str`, one of ``NO_DITHER``, ``SUBTRACTIVE_DITHER_1``, or
          ``SUBTRACTIVE_DITHER_2``.  Defaults to ``NO_DITHER``.
        - ``scaling``: `str`, one of ``STDEV_CFITSIO``, ``STDEV_MASKED``,
          ``RANGE``, or ``MANUAL`` (see C++ docs for definitions).
        - ``mask_planes`` : `list` of `str`, names of mask planes to reject in
          ``STDEV_MASKED`` and ``RANGE`` (but only when writing objects with
          mask planes)
        - ``level``: `float`, target compression level (see C++ docs).
        - ``seed``: random number seed for dithering.  Default is zero, which
          uses a timestamp-based seed when used directly, but a data ID-based
          seed when used via `lsst.obs.base.FitsExposureFormatter`.
        """
        copy = dict(mapping)
        result = QuantizationOptions()
        if "dither" in copy:
            result.dither = DitherAlgorithm[copy.pop("dither")]
        if "scaling" in copy:
            result.scaling = ScalingAlgorithm[copy.pop("scaling")]
        if "mask_planes" in copy:
            result.mask_planes = copy.pop("mask_planes")
        if "level" in copy:
            result.level = copy.pop("level")
        if "seed" in copy:
            result.seed = copy.pop("seed")
        if copy:
            raise ValueError(f"Unrecognized quantization options: {list(copy.keys())}.")
        return result

    def to_dict(self) -> dict[str, object]:
        """"Return the mapping representation of these options.

        Returns
        -------
        mapping : `dict`
            See `from_mapping`.
        """
        return {
            "dither": self.dither.name,
            "scaling": self.scaling.name,
            "mask_planes": list(self.mask_planes),
            "level": self.level,
            "seed": self.seed,
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(dither={self.dither!r}, scaling={self.scaling!r}, "
            f"mask_planes={self.mask_planes!r}, level={self.level!r}, seed={self.seed!r})"
        )


@continueClass
class CompressionOptions:  # noqa: F811
    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> CompressionOptions:
        """Construct from a dictionary with keys matching this struct's fields.

        Parameters
        ----------
        mapping : `~collections.abc.Mapping`
            Mapping from string to value.  Enumeration values may be passed as
            string value names.  Missing keys are mapped to default values.

        Returns
        -------
        options : `CompressionOptions`
            An instance of this options class.

        Notes
        -----
        Allowed keys are:

        - ``algorithm``: `str`, one of ``GZIP_1``, ``GZIP_2``, or ``RICE_1``.
        - ``tile_width``: `int`, zero to use entire rows.
        - ``tile_height``: `int`, zero to use entire columns.
        - ``quantization``: `dict` or `None` (see
           `QuantizationOptions.from_mapping`).

        Missing keys are replaced by defaults that reflect lossless compression
        (``GZIP_2``) with single rows as tiles.
        """
        copy = dict(mapping)
        result = CompressionOptions()
        if "algorithm" in copy:
            result.algorithm = CompressionAlgorithm[copy.pop("algorithm")]
        if "tile_width" in copy:
            result.tile_width = copy.pop("tile_width")
        if "tile_height" in copy:
            result.tile_height = copy.pop("tile_height")
        if (quantization := copy.pop("quantization", None)) is not None:
            result.quantization = QuantizationOptions.from_mapping(quantization)
        if copy:
            raise ValueError(f"Unrecognized compression options: {list(copy.keys())}.")
        return result

    def to_dict(self) -> dict[str, object]:
        """"Return the mapping representation of these options.

        Returns
        -------
        mapping : `dict`
            See `from_mapping`.
        """
        return {
            "algorithm": self.algorithm.name,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "quantization": self.quantization.to_dict() if self.quantization is not None else None,
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(algorithm={self.algorithm!r}, "
            f"tile_width={self.tile_width!r}, tile_height={self.tile_height!r}, "
            f"quantization={self.quantization!r})"
        )

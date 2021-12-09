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

__all__ = ['Fits']

from lsst.utils import continueClass
from ._fits import (Fits, ImageWriteOptions, ImageCompressionOptions, ImageScalingOptions,
                    compressionAlgorithmToString, scalingAlgorithmToString)


@continueClass
class Fits:  # noqa: F811
    def __enter__(self):
        return self

    def __exit__(self, cls, exc, traceback):
        self.closeFile()


@continueClass
class ImageWriteOptions:  # noqa: F811
    def __repr__(self):
        return f"{self.__class__.__name__}(compression={self.compression!r}, scaling={self.scaling!r})"


@continueClass
class ImageCompressionOptions:  # noqa: F811
    def __repr__(self):
        return (f"{self.__class__.__name__}(algorithm={compressionAlgorithmToString(self.algorithm)!r}, "
                f"tiles={self.tiles.tolist()!r}, quantizeLevel={self.quantizeLevel:f})")


@continueClass
class ImageScalingOptions:  # noqa: F811
    def __repr__(self):
        return (f"{self.__class__.__name__}(algorithm={scalingAlgorithmToString(self.algorithm)!r}, "
                f"bitpix={self.bitpix}, maskPlanes={self.maskPlanes}, seed={self.seed} "
                f"quantizeLevel={self.quantizeLevel}, quantizePad={self.quantizePad}, "
                f"fuzz={self.fuzz}, bscale={self.bscale}, bzero={self.bzero})")

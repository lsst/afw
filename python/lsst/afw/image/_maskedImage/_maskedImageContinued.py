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
import numpy as np

from lsst.utils import TemplateMeta
from ._maskedImage import MaskedImageI, MaskedImageF, MaskedImageD, MaskedImageU, MaskedImageL

from .._image._fitsIoWithOptions import imageReadFitsWithOptions, exposureWriteFitsWithOptions
from .._image._slicing import supportSlicing
from .._image._disableArithmetic import disableImageArithmetic

__all__ = ["MaskedImage", "VariancePixel"]


VariancePixel = np.float32


class MaskedImage(metaclass=TemplateMeta):

    def set(self, value, mask=None, variance=None):
        """Assign a tuple of scalars to the entirety of all three planes.
        """
        if mask is None and variance is None:
            try:
                value, mask, variance = value
            except TypeError:
                pass
        if mask is None:
            mask = 0
        if variance is None:
            variance = 0.0
        self.image.set(value)
        self.mask.set(mask)
        self.variance.set(variance)

    def _set(self, index, value, origin):
        """Set the pixel at the given index to a triple (value, mask, variance).

        Parameters
        ----------
        index : `geom.Point2I`
            Position of the pixel to assign to.
        value : `tuple`
            A tuple of (value, mask, variance) scalars.
        origin : `ImageOrigin`
            Coordinate system of ``index`` (`PARENT` or `LOCAL`).
        """
        self.image[index, origin] = value[0]
        self.mask[index, origin] = value[1]
        self.variance[index, origin] = value[2]

    def _get(self, index, origin):
        """Return a triple (value, mask, variance) at the given index.

        Parameters
        ----------
        index : `geom.Point2I`
            Position of the pixel to assign to.
        origin : `ImageOrigin`
            Coordinate system of ``index`` (`PARENT` or `LOCAL`).
        """
        return (self.image[index, origin],
                self.mask[index, origin],
                self.variance[index, origin])

    def getArrays(self):
        """Return a tuple (value, mask, variance) numpy arrays."""
        return (self.image.array if self.image else None,
                self.mask.array if self.mask else None,
                self.variance.array if self.variance else None)

    def convertF(self):
        return MaskedImageF(self, True)

    def convertD(self):
        return MaskedImageD(self, True)

    def __reduce__(self):
        from lsst.afw.fits import reduceToFits
        return reduceToFits(self)

    def __str__(self):
        string = "image={},\nmask={}, maskPlaneDict={}\nvariance={}, bbox={}"
        return string.format(self.image.array,
                             self.mask.array,
                             self.mask.getMaskPlaneDict(),
                             self.variance.array,
                             self.getBBox())

    def __repr__(self):
        return "{}.{}=({})".format(self.__module__, self.__class__.__name__, str(self))

    readFitsWithOptions = classmethod(imageReadFitsWithOptions)

    writeFitsWithOptions = exposureWriteFitsWithOptions


MaskedImage.register(np.int32, MaskedImageI)
MaskedImage.register(np.float32, MaskedImageF)
MaskedImage.register(np.float64, MaskedImageD)
MaskedImage.register(np.uint16, MaskedImageU)
MaskedImage.register(np.uint64, MaskedImageL)
MaskedImage.alias("I", MaskedImageI)
MaskedImage.alias("F", MaskedImageF)
MaskedImage.alias("D", MaskedImageD)
MaskedImage.alias("U", MaskedImageU)
MaskedImage.alias("L", MaskedImageL)

for cls in set(MaskedImage.values()):
    supportSlicing(cls)
    disableImageArithmetic(cls)

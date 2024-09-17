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

__all__ = ["Image", "DecoratedImage"]

import numpy as np
from lsst.utils import TemplateMeta

from ._slicing import supportSlicing
from ._disableArithmetic import disableImageArithmetic
from ._fitsIoWithOptions import imageReadFitsWithOptions, imageWriteFitsWithOptions
from ._imageLib import ImageI, ImageF, ImageD, ImageU, ImageL
from ._imageLib import DecoratedImageI, DecoratedImageF, DecoratedImageD, DecoratedImageU, DecoratedImageL


class Image(metaclass=TemplateMeta):

    def __reduce__(self):
        from lsst.afw.fits import reduceToFits
        return reduceToFits(self)

    def __deepcopy__(self, memo=None):
        return self.clone()

    def __str__(self):
        return "{}, bbox={}".format(self.array, self.getBBox())

    def __repr__(self):
        return "{}.{}={}".format(self.__module__, self.__class__.__name__, str(self))

    def __array__(self, dtype=None, copy=None):
        if dtype is None:
            if copy:
                return self.array.copy()
            else:
                return self.array
        else:
            if dtype != self.array.dtype and copy is False:
                raise ValueError("copy=False and dtype change requires copy.")
            if copy is None:
                copy = False
            return self.array.astype(dtype, copy=copy)

    readFitsWithOptions = classmethod(imageReadFitsWithOptions)

    writeFitsWithOptions = imageWriteFitsWithOptions


Image.register(np.int32, ImageI)
Image.register(np.float32, ImageF)
Image.register(np.float64, ImageD)
Image.register(np.uint16, ImageU)
Image.register(np.uint64, ImageL)
Image.alias("I", ImageI)
Image.alias("F", ImageF)
Image.alias("D", ImageD)
Image.alias("U", ImageU)
Image.alias("L", ImageL)


class DecoratedImage(metaclass=TemplateMeta):

    def convertF(self):
        return ImageF(self, deep=True)

    def convertD(self):
        return ImageD(self, deep=True)

    readFitsWithOptions = classmethod(imageReadFitsWithOptions)

    writeFitsWithOptions = imageWriteFitsWithOptions


DecoratedImage.register(np.int32, DecoratedImageI)
DecoratedImage.register(np.float32, DecoratedImageF)
DecoratedImage.register(np.float64, DecoratedImageD)
DecoratedImage.register(np.uint16, DecoratedImageU)
DecoratedImage.register(np.uint64, DecoratedImageL)
DecoratedImage.alias("I", DecoratedImageI)
DecoratedImage.alias("F", DecoratedImageF)
DecoratedImage.alias("D", DecoratedImageD)
DecoratedImage.alias("U", DecoratedImageU)
DecoratedImage.alias("L", DecoratedImageL)


for cls in set(Image.values()):
    supportSlicing(cls)
    disableImageArithmetic(cls)

for cls in set(DecoratedImage.values()):
    supportSlicing(cls)
    disableImageArithmetic(cls)

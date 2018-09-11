#
# LSST Data Management System
# Copyright 2008-2017 LSST/AURA.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

__all__ = ["Image", "DecoratedImage"]

import numpy as np
from lsst.utils import TemplateMeta

from ..slicing import supportSlicing
from .fitsIoWithOptions import imageReadFitsWithOptions, imageWriteFitsWithOptions
from .image import ImageI, ImageF, ImageD, ImageU, ImageL
from .image import DecoratedImageI, DecoratedImageF, DecoratedImageD, DecoratedImageU, DecoratedImageL


class Image(metaclass=TemplateMeta):

    def __reduce__(self):
        from lsst.afw.fits import reduceToFits
        return reduceToFits(self)

    readFitsWithOptions = classmethod(imageReadFitsWithOptions)

    writeFitsWithOptions = imageWriteFitsWithOptions


Image.register(np.int32, ImageI)
Image.register(np.float32, ImageF)
Image.register(np.float64, ImageD)
Image.register(np.uint16, ImageU)
Image.register(np.int64, ImageL)
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
DecoratedImage.register(np.int64, DecoratedImageL)
DecoratedImage.alias("I", DecoratedImageI)
DecoratedImage.alias("F", DecoratedImageF)
DecoratedImage.alias("D", DecoratedImageD)
DecoratedImage.alias("U", DecoratedImageU)
DecoratedImage.alias("L", DecoratedImageL)


for cls in set(Image.values()):
    supportSlicing(cls)

for cls in set(DecoratedImage.values()):
    supportSlicing(cls)

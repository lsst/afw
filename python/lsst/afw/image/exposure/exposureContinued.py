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

__all__ = ["Exposure"]

import numpy as np

from lsst.utils import TemplateMeta

from ..slicing import supportSlicing
from .exposure import ExposureI, ExposureF, ExposureD, ExposureU, ExposureL


class Exposure(metaclass=TemplateMeta):

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
        self.maskedImage._set(index, value=value, origin=origin)

    def _get(self, index, origin):
        """Return a triple (value, mask, variance) at the given index.

        Parameters
        ----------
        index : `geom.Point2I`
            Position of the pixel to assign to.
        origin : `ImageOrigin`
            Coordinate system of ``index`` (`PARENT` or `LOCAL`).
        """
        return self.maskedImage._get(index, origin=origin)

    def __reduce__(self):
        from lsst.afw.fits import reduceToFits
        return reduceToFits(self)

    def convertF(self):
        return ExposureF(self, deep=True)

    def convertD(self):
        return ExposureD(self, deep=True)

    def getImage(self):
        return self.maskedImage.image

    def setImage(self, image):
        self.maskedImage.image = image

    image = property(getImage, setImage)

    def getMask(self):
        return self.maskedImage.mask

    def setMask(self, mask):
        self.maskedImage.mask = mask

    mask = property(getMask, setMask)

    def getVariance(self):
        return self.maskedImage.variance

    def setVariance(self, variance):
        self.maskedImage.variance = variance

    variance = property(getVariance, setVariance)


Exposure.register(np.int32, ExposureI)
Exposure.register(np.float32, ExposureF)
Exposure.register(np.float64, ExposureD)
Exposure.register(np.uint16, ExposureU)
Exposure.register(np.int64, ExposureL)
Exposure.alias("I", ExposureI)
Exposure.alias("F", ExposureF)
Exposure.alias("D", ExposureD)
Exposure.alias("U", ExposureU)
Exposure.alias("L", ExposureL)

for cls in set(Exposure.values()):
    supportSlicing(cls)

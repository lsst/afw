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

from __future__ import absolute_import, division, print_function

__all__ = ["Exposure"]

from future.utils import with_metaclass
import numpy as np

from lsst.utils import TemplateMeta
import lsst.afw.geom as afwGeom

from .exposure import ExposureI, ExposureF, ExposureD, ExposureU, ExposureL


class Exposure(with_metaclass(TemplateMeta, object)):

    def __reduce__(self):
        from lsst.afw.fits import reduceToFits
        return reduceToFits(self)

    def convertF(self):
        return ExposureF(self, deep=True)

    def convertD(self):
        return ExposureD(self, deep=True)

    @classmethod
    def Factory(cls, *args, **kwargs):
        """Construct a new Exposure of the same dtype.
        """
        return cls(*args, **kwargs)

    def clone(self):
        """Return a deep copy of self"""
        return self.Factory(self, deep=True)

    def __getitem__(self, imageSlice):
        return self.Factory(self.getMaskedImage()[imageSlice])

    def __setitem__(self, imageSlice, rhs):
        self.getMaskedImage[imageSlice] = rhs.getMaskedImage()

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
    
    def getCutout(self, coord, height, width=None):
        """Return a new Exposure that is a small cutout of the original.
    
        Parameters
        ----------
        coord : `lsst.afw.geom.SpherePoint`
            desired center of cutout (e.g., in RA and Dec)
        height : `int`
            height of cutout in pixels
        width : `int`, in pixels
            width of cutout in pixels, default = height
    
        Returns
        -------
        `lsst.afw.image.Exposure`
            The cutout exposure is centered on coord, or centered near coord
            (if coord is within height/2 or width/2 of the exposure edge)
        """
        width = height if width is None else width  # default to square cutout
        wcsPixel = self.getWcs().skyToPixel(coord)
        wcsPixel.shift(afwGeom.Extent2D(-width/2, -height/2))  # adjust origin so coord is centered
        bbox = afwGeom.Box2I(afwGeom.Point2I(wcsPixel), afwGeom.Extent2I(width, height))
        bbox.clip(self.getBBox())  # ensure new bbox is within original image (may un-center coord somewhat)
        return self.Factory(self, bbox)


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

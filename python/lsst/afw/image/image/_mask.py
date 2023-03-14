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

__all__ = ["Mask", "MaskPixel"]

import warnings

import numpy as np

from lsst.utils import TemplateMeta
from ._fitsIoWithOptions import imageReadFitsWithOptions, imageWriteFitsWithOptions
from ._imageLib import MaskX
from ._slicing import supportSlicing
from ._disableArithmetic import disableMaskArithmetic
from ..maskPlaneDict import MaskPlaneDict

MaskPixel = np.int32


class Mask(metaclass=TemplateMeta):
    TEMPLATE_PARAMS = ("dtype",)
    TEMPLATE_DEFAULTS = (MaskPixel,)

    def __reduce__(self):
        from lsst.afw.fits import reduceToFits
        return reduceToFits(self)

    def __str__(self):
        return "{}, bbox={}, maskPlaneDict={}".format(self.array, self.getBBox(), self.getMaskPlaneDict())

    def __repr__(self):
        return "{}.{}={}".format(self.__module__, self.__class__.__name__, str(self))

    readFitsWithOptions = classmethod(imageReadFitsWithOptions)

    def writeFitsWithOptions(self, dest, options, item=None):
        """Write an Mask to FITS, with options

        Parameters
        ----------
        dest : `str`
            Fits file path to which to write the mask.
        options : `lsst.daf.base.PropertySet`
            Write options. The item ``item`` is read.
            It must contain an `lsst.daf.base.PropertySet` with data for
            ``lsst.afw.fits.ImageWriteOptions``.
        item : `str`, optional
            Item to read from the ``options`` parameter.
            If not specified it will default to "mask" if present, else
            will fallback to the generic "image" options.
        """
        if item is None:
            # Fallback to "image" if "mask" is missing. This allows older
            # code that assumed "image" to still function.
            item = "mask" if "mask" in options else "image"
        return imageWriteFitsWithOptions(self, dest, options, item=item)

    @staticmethod
    def addMaskPlane(name, doc=None):
        # Temporary overload helper for deprecation message.
        if doc is None:
            warnings.warn("Doc field will become non-optional. Will be removed after v26.", FutureWarning)
            doc = ""
        return Mask.addMaskPlaneWithDoc(name, doc)

    def getMaskPlaneDict(self):
        planesDict = self._getMaskPlaneDict()
        return MaskPlaneDict(planesDict)

    def conformMaskPlanes(self, maskPlaneDict):
        return self._conformMaskPlanes(maskPlaneDict._maskDict)


Mask.register(MaskPixel, MaskX)
Mask.alias("X", MaskX)

for cls in (MaskX, ):
    supportSlicing(cls)
    disableMaskArithmetic(cls)

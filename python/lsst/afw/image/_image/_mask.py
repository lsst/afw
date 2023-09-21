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

import collections.abc
import warnings
import types

import numpy as np

from lsst.utils import continueClass, TemplateMeta
from ._fitsIoWithOptions import imageReadFitsWithOptions, imageWriteFitsWithOptions
from ._imageLib import MaskX, MaskDict
from ._slicing import supportSlicing
from ._disableArithmetic import disableMaskArithmetic

MaskPixel = np.int32


class Mask(metaclass=TemplateMeta):
    TEMPLATE_PARAMS = ("dtype",)
    TEMPLATE_DEFAULTS = (MaskPixel,)

    def __reduce__(self):
        from lsst.afw.fits import reduceToFits
        return reduceToFits(self)

    def __str__(self):
        return f"{self.array}, bbox={self.getBBox()}, maskPlaneDict={self.getMaskDict()}"

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

    @property
    def maskDict(self):
        return self.getMaskDict()

    @staticmethod
    def addMaskPlane(name, doc=None):
        # TODO DM-XXXXX: Temporary overload helper for deprecation message.
        if doc is None:
            warnings.warn("Doc field will become non-optional. Will be removed after v28.", FutureWarning)
            doc = ""
        return Mask.addMaskPlaneWithDoc(name, doc)

    def getMaskPlaneDict(self):
        """For backwards compatibility.
        """
        # TODO DM-XXXXX: remove
        return self.getMaskDict()

    # TODO DM-XXXXX: remove
    @staticmethod
    def getPlaneBitMask(name):
        warnings.warn("Replaced by non-static getBitMask. Will be removed after v28.", FutureWarning)
        return Mask._getPlaneBitMask(name)

    @staticmethod
    def removeMaskPlane(name):
        warnings.warn("Replaced by non-static removeAndClearMaskPlane. Will be removed after v28.",
                      FutureWarning)
        return Mask._removeMaskPlane(name)

    @staticmethod
    def getMaskPlane(name):
        warnings.warn("Replaced by non-static getPlaneId. Will be removed after v28.", FutureWarning)
        return Mask._getMaskPlane(name)


@continueClass
class MaskDict(collections.abc.Mapping):  # noqa: F811
    """Dict-like view to the underlying C++ MaskDict object.

    TODO: More docs of what it is here?
    """

    @property
    def doc(self):
        """A view of the docstring mapping for these mask planes.
        """
        return types.MappingProxyType(self._getMaskPlaneDocDict())

    def __iter__(self):
        return iter(self._getPlaneNames())

    def __len__(self):
        return self._size()

    def __getitem__(self, name):
        if (id := self._getPlaneId(name)) < 0:
            raise KeyError(name)
        return id

    def __repr__(self):
        return self._print()


Mask.register(MaskPixel, MaskX)
Mask.alias("X", MaskX)

for cls in (MaskX, ):
    supportSlicing(cls)
    disableMaskArithmetic(cls)

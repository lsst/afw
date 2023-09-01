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

    @staticmethod
    def addMaskPlane(name, doc=None):
        # Temporary overload helper for deprecation message.
        if doc is None:
            warnings.warn("Doc field will become non-optional. Will be removed after v26.", FutureWarning)
            doc = ""
        return Mask.addMaskPlaneWithDoc(name, doc)

    def getMaskPlaneDict(self):
        """For backwards compatibility."""
        return self.getMaskDict()

    @property
    def maskDict(self):
        return self.getMaskDict()


# @collections.abc.Mapping.register
@continueClass
class MaskDict(collections.abc.Mapping):  # noqa: F811

    @property
    def doc(self):
        """A view of the docstrings for these mask planes.
        """
        return types.MappingProxyType(self._getMaskPlaneDocDict())

    # def __contains__(self, name):
    #     # TODO: try this??
    #     return collections.abc.Mapping.__contains__(self, name)

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

    # def keys(self):
    #     None

    # def items(self):
    #     None

    # def values(self):
    #     None

    # def get(self, name):
    #     None

    # def __eq__(self, other):
    #     None

    # TODO: check that we get this for free
    # def __ne__(self, other):
    #     None


# @collections.abc.Mapping.register
# class MaskDict(C_MaskDict):
#     """
#     Mapping view of the underlying C++ MaskDict state.

#     Modifying the mask planes this is a view of may invalidate this view
#     instance.

#     Notes
#     -----
#     This class directly implements all of the abc.Mapping interface, instead
#     of inheriting, to avoid metaclass conflicts with pybind11. We need the
#     C_MaskDict parent to allow passing this into pybind11.
#     """

#     def __init__(self, c_maskDict):
#         self._c_maskDict = c_maskDict
#         # self._doc = _MaskDocMappingView(c_maskDict)

#     @property
#     def doc(self):
#         return self._doc

#     def __contains__(self, name):
#         # TODO: try this??
#         return collections.abc.Mapping.__contains__(self, name)

#     def __iter__(self):
#         return iter(self._c_maskDict.getPlaneNames())

#     def __len__(self):
#         return self._c_maskDict.size()

#     def __getitem__(self, name):
#         if (id := self._c_maskDict.getPlaneId(name)) < 0:
#             raise KeyError(name)
#         return id

#     def __repr__(self):
#         return self._c_maskDict.print()

#     def keys(self):
#         None

#     def items(self):
#         None

#     def values(self):
#         None

#     def get(self, name):
#         None

#     def __eq__(self, other):
#         None

    # TODO: check that we get this for free
    # def __ne__(self, other):
    #     None

# class _MaskDocMappingView(collections.abc.Mapping):
#     """
#     stuff
#     """
#     def __init__(self, c_maskDict):
#         self._c_maskDict = c_maskDict

#     def __iter__(self):
#         return iter(self._c_maskDict.getPlaneNames())

#     def __len__(self):
#         return self._c_maskDict.size()

#     def __getitem__(self, name):
#         if self._c_maskDict.getPlaneId(name) < 0:
#             raise KeyError(name)
#         return self._c_maskDict.getPlaneDoc(name)


Mask.register(MaskPixel, MaskX)
Mask.alias("X", MaskX)

for cls in (MaskX, ):
    supportSlicing(cls)
    disableMaskArithmetic(cls)

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

__all__ = ["MaskedImage"]

from future.utils import with_metaclass
import numpy as np

from lsst.utils import TemplateMeta

from ..slicing import supportSlicing
from .maskedImage import MaskedImageI, MaskedImageF, MaskedImageD, MaskedImageU, MaskedImageL


class MaskedImage(with_metaclass(TemplateMeta, object)):

    def set(self, x, y=None, values=None):
        """Set the point (x, y) to a triple (value, mask, variance)"""

        if values is None:
            assert (y is None)
            values = x
            try:
                self.getImage().set(values[0])
                self.getMask().set(values[1])
                self.getVariance().set(values[2])
            except TypeError:
                self.getImage().set(values)
                self.getMask().set(0)
                self.getVariance().set(0)
        else:
            try:
                self.getImage().set(x, y, values[0])
                if len(values) > 1:
                    self.getMask().set(x, y, values[1])
                if len(values) > 2:
                    self.getVariance().set(x, y, values[2])
            except TypeError:
                self.getImage().set(x)
                self.getMask().set(y)
                self.getVariance().set(values)

    def get(self, x, y):
        """Return a triple (value, mask, variance) at the point (x, y)"""
        return (self.getImage().get(x, y),
                self.getMask().get(x, y),
                self.getVariance().get(x, y))

    def getArrays(self):
        """Return a tuple (value, mask, variance) numpy arrays."""
        return (self.getImage().getArray() if self.getImage() else None,
                self.getMask().getArray() if self.getMask() else None,
                self.getVariance().getArray() if self.getVariance() else None)

    def convertF(self):
        return MaskedImageF(self, True)

    def convertD(self):
        return MaskedImageD(self, True)

    def __reduce__(self):
        from lsst.afw.fits import reduceToFits
        return reduceToFits(self)


MaskedImage.register(np.int32, MaskedImageI)
MaskedImage.register(np.float32, MaskedImageF)
MaskedImage.register(np.float64, MaskedImageD)
MaskedImage.register(np.uint16, MaskedImageU)
MaskedImage.register(np.int64, MaskedImageL)
MaskedImage.alias("I", MaskedImageI)
MaskedImage.alias("F", MaskedImageF)
MaskedImage.alias("D", MaskedImageD)
MaskedImage.alias("U", MaskedImageU)
MaskedImage.alias("L", MaskedImageL)

for cls in set(MaskedImage.values()):
    supportSlicing(cls)

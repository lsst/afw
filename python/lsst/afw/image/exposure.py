from __future__ import absolute_import, division, print_function
from future.utils import with_metaclass
import numpy as np

from lsst.utils import TemplateMeta

from ._exposure import ExposureI, ExposureF, ExposureD, ExposureU, ExposureL, makeExposure

__all__ = ["Exposure", "ExposureI", "ExposureF", "ExposureD", "ExposureU", "ExposureL", "makeExposure"]


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

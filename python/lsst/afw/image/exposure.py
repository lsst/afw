from __future__ import absolute_import, division, print_function

from .slicing import supportSlicing
from ._exposure import ExposureI, ExposureF, ExposureD, ExposureU, ExposureL, makeExposure

__all__ = ["ExposureI", "ExposureF", "ExposureD", "ExposureU", "ExposureL", "makeExposure"]

for cls in (ExposureI, ExposureF, ExposureD, ExposureU, ExposureL):
    supportSlicing(cls)

    def __reduce__(self):
        from lsst.afw.fits import reduceToFits
        return reduceToFits(self)
    cls.__reduce__ = __reduce__

    def convertF(self):
        return ExposureF(self, True)
    cls.convertF = convertF

    def convertD(self):
        return ExposureD(self, True)
    cls.convertD = convertD

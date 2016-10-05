from __future__ import absolute_import

from ._exposure import *

def __reduce__(self):
    from lsst.afw.fits import reduceToFits
    return reduceToFits(self)

ExposureI.__reduce__ = __reduce__
ExposureF.__reduce__ = __reduce__
ExposureD.__reduce__ = __reduce__
ExposureU.__reduce__ = __reduce__
ExposureL.__reduce__ = __reduce__
del __reduce__

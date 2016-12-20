from __future__ import absolute_import
from future.utils import with_metaclass

from abc import ABCMeta

from ._mask import *
from .slicing import supportSlicing

supportSlicing(MaskU)

class Mask(with_metaclass(ABCMeta, object)):
    pass

for cls in (MaskU, ):
    Mask.register(cls)
    supportSlicing(cls)

    def __reduce__(self):
        from lsst.afw.fits import reduceToFits
        return reduceToFits(self)
    cls.__reduce__ = __reduce__


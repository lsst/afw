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


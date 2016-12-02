from __future__ import absolute_import
from future.utils import with_metaclass

from abc import ABCMeta

from .slicing import supportSlicing
from ._image import ImageI, ImageF, ImageD, ImageU, ImageL
from ._image import DecoratedImageI, DecoratedImageF, DecoratedImageD, DecoratedImageU, DecoratedImageL

__all__ = ["Image", "DecoratedImage"]

class Image(with_metaclass(ABCMeta, object)):
    pass

for cls in (ImageI, ImageF, ImageD, ImageU, ImageL):
    Image.register(cls)
    supportSlicing(cls)

class DecoratedImage(with_metaclass(ABCMeta, object)):
    pass

for cls in (DecoratedImageI, DecoratedImageF, DecoratedImageD, DecoratedImageU, DecoratedImageL):
    DecoratedImage.register(cls)
    supportSlicing(cls)

    def convertF(self):
        return ImageF(self, True)
    cls.convertF = convertF

    def convertD(self):
        return ImageD(self, True)
    cls.convertD = convertD

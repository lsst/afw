from __future__ import absolute_import

from .slicing import supportSlicing
from ._image import ImageI, ImageF, ImageD, ImageU, ImageL

__all__ = []  # import this module only for its side effects

for cls in (ImageI, ImageF, ImageD, ImageU, ImageL):
    supportSlicing(cls)

    def convertF(self):
        return ImageF(self, True)
    cls.convertF = convertF

    def convertD(self):
        return ImageD(self, True)
    cls.convertD = convertD

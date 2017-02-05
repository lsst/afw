from __future__ import absolute_import
from future.utils import with_metaclass

import numpy as np
from lsst.utils import TemplateMeta

from .slicing import supportSlicing
from ._image import ImageI, ImageF, ImageD, ImageU, ImageL
from ._image import DecoratedImageI, DecoratedImageF, DecoratedImageD, DecoratedImageU, DecoratedImageL

__all__ = ["Image", "DecoratedImage"]


class Image(with_metaclass(TemplateMeta, object)):

    def __reduce__(self):
        from lsst.afw.fits import reduceToFits
        return reduceToFits(self)


Image.register(np.int32, ImageI)
Image.register(np.float32, ImageF)
Image.register(np.float64, ImageD)
Image.register(np.uint16, ImageU)
Image.register(np.int64, ImageL)
Image.alias("I", ImageI)
Image.alias("F", ImageF)
Image.alias("D", ImageD)
Image.alias("U", ImageU)
Image.alias("L", ImageL)


class DecoratedImage(with_metaclass(TemplateMeta, object)):

    def convertF(self):
        return ImageF(self, deep=True)

    def convertD(self):
        return ImageD(self, deep=True)


DecoratedImage.register(np.int32, DecoratedImageI)
DecoratedImage.register(np.float32, DecoratedImageF)
DecoratedImage.register(np.float64, DecoratedImageD)
DecoratedImage.register(np.uint16, DecoratedImageU)
DecoratedImage.register(np.int64, DecoratedImageL)
DecoratedImage.alias("I", DecoratedImageI)
DecoratedImage.alias("F", DecoratedImageF)
DecoratedImage.alias("D", DecoratedImageD)
DecoratedImage.alias("U", DecoratedImageU)
DecoratedImage.alias("L", DecoratedImageL)


for suffix in "IFDUL":
    supportSlicing(Image[suffix])
    supportSlicing(DecoratedImage[suffix])

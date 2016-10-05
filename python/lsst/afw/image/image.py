from __future__ import absolute_import

from ._image import *
from .slicing import supportSlicing

for cls in (ImageI, ImageF, ImageD, ImageU, ImageL):
    supportSlicing(cls)


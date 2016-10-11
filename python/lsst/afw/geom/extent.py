from __future__ import absolute_import

from ._extent import *

Extent2D.truncate = lambda self : truncate(self)
Extent2D.floor = lambda self : floor(self)
Extent2D.ceil = lambda self : ceil(self)
Extent3D.truncate = lambda self : truncate(self)
Extent3D.floor = lambda self : floor(self)
Extent3D.ceil = lambda self : ceil(self)

ExtentI = Extent2I
ExtentD = Extent2D
Extent = {(int, 2):Extent2I, (float, 2):Extent2D, (int, 3):Extent3I, (float, 3):Extent3D}

from __future__ import absolute_import

from ._extent import Extent2I, Extent2D, Extent3I, Extent3D, truncate, floor, ceil

__all__ = ["Extent2I", "Extent2D", "Extent3I", "Extent3D", "ExtentI", "ExtentD", "Extent",
           "truncate", "floor", "ceil"]


def addRepr(cls):
    """Add __str__ and __repr__ to a Point or Extent
    """
    def __str__(self):
        return "({})".format(", ".join("%0.5g" % v for v in self))

    def __repr__(self):
        return "{}({})".format(type(self).__name__, ", ".join("%0.10g" % v for v in self))

    cls.__str__ = __str__
    cls.__repr__ = __repr__

for cls in (Extent2I, Extent2D, Extent3I, Extent3D):
    addRepr(cls)

for floatCls in (Extent2D, Extent3D):
    floatCls.truncate = lambda self: truncate(self)
    floatCls.floor = lambda self: floor(self)
    floatCls.ceil = lambda self: ceil(self)

ExtentI = Extent2I
ExtentD = Extent2D
Extent = {(int, 2): Extent2I, (float, 2): Extent2D, (int, 3): Extent3I, (float, 3): Extent3D}

def __reduce__(self):
    return (Extent2D, (self.getX(), self.getY()))
Extent2D.__reduce__ = __reduce__

def __reduce__(self):
    return (Extent3D, (self.getX(), self.getY(), self.getZ()))
Extent3D.__reduce__ = __reduce__

def __reduce__(self):
    return (Extent2I, (self.getX(), self.getY()))
Extent2I.__reduce__ = __reduce__

def __reduce__(self):
    return (Extent3I, (self.getX(), self.getY(), self.getZ()))
Extent3I.__reduce__ = __reduce__


from __future__ import absolute_import

from .point import Point2I, Point2D
from .extent import Extent2I, Extent2D
from ._box import *

def _Box2I__repr__(self):
    return "Box2I(%r, %r)" % (self.getMin(), self.getDimensions())
def _Box2I__reduce__(self):
    return (Box2I, (self.getMin(), self.getMax()))
def _Box2I__str__(self):
    return "Box2I(%s, %s)" % (self.getMin(), self.getMax())
def _Box2I_getSlices(self):
    return (slice(self.getBeginY(), self.getEndY()), slice(self.getBeginX(), self.getEndX()))
def _Box2I_getCorners(self):
    return (
        self.getMin(),
        self.Point2I(self.getMaxX(), self.getMinY()),
        self.getMax(),
        self.Point2I(self.getMinX(), self.getMaxY())
    )

Box2I.__repr__ = _Box2I__repr__
Box2I.__reduce__ = _Box2I__reduce__
Box2I.__str__ = _Box2I__str__
Box2I._getSlices = _Box2I_getSlices
Box2I._getCorners = _Box2I_getCorners
Box2I.Point = Point2I
Box2I.Extent = Extent2I

BoxI = Box2I


del _Box2I__repr__
del _Box2I__reduce__
del _Box2I__str__
del _Box2I_getSlices
del _Box2I_getCorners

def _Box2D__repr__(self):
    return "Box2D(%r, %r)" % (self.getMin(), self.getDimensions())
def _Box2D__reduce__(self):
    return (Box2D, (self.getMin(), self.getMax()))
def _Box2D__str__(self):
    return "Box2D(%s, %s)" % (self.getMin(), self.getMax())
def _Box2D_getSlices(self):
    return (slice(self.getBeginY(), self.getEndY()), slice(self.getBeginX(), self.getEndX()))
def _Box2D_getCorners(self):
    return (
        self.getMin(),
        self.Point2D(self.getMaxX(), self.getMinY()),
        self.getMax(),
        self.Point2D(self.getMinX(), self.getMaxY())
    )

Box2D.__repr__ = _Box2D__repr__
Box2D.__reduce__ = _Box2D__reduce__
Box2D.__str__ = _Box2D__str__
Box2D._getSlices = _Box2D_getSlices
Box2D._getCorners = _Box2D_getCorners
Box2D.Point = Point2D
Box2D.Extent = Extent2D

del _Box2D__repr__
del _Box2D__reduce__
del _Box2D__str__
del _Box2D_getSlices
del _Box2D_getCorners


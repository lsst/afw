from __future__ import absolute_import

from .extent import addRepr
from ._point import Point2I, Point2D, Point3I, Point3D

__all__ = ["Point2I", "Point2D", "Point3I", "Point3D", "PointI", "PointD", "Point"]


for cls in (Point2I, Point2D, Point3I, Point3D):
    addRepr(cls)

PointI = Point2I
PointD = Point2D
Point = {(int, 2): Point2I, (float, 2): Point2D, (int, 3): Point3I, (float, 3): Point3D}

def __reduce__(self):
    return (Point2I, (self.getX(), self.getY()))
Point2I.__reduce__ = __reduce__

def __reduce__(self):
    return (Point3I, (self.getX(), self.getY(), self.getZ()))
Point3I.__reduce__ = __reduce__

def __reduce__(self):
    return (Point2D, (self.getX(), self.getY()))
Point2D.__reduce__ = __reduce__

def __reduce__(self):
    return (Point3D, (self.getX(), self.getY(), self.getZ()))
Point3D.__reduce__ = __reduce__


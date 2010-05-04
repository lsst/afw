"""Application Framework geometry code including Point, Extent, and ellipses
"""
from .geomLib import (
    version,
    LinearTransform,
    AffineTransform,
    CoordinateExpr2,
    CoordinateExpr3,
    Extent2I,
    Extent3I,
    Extent2D,
    Extent3D,
    makeExtentI,
    makeExtentD,
    Point2I,
    Point3I,
    Point2D,
    Point3D,
    makePointI,
    makePointD,
    Box2I,
    Box2D,
)

PointI = Point2I
PointD = Point2D

ExtentI = Extent2I
ExtentD = Extent2D

BoxI = Box2I
BoxD = Box2D

import ellipses

Point = {(int, 2):Point2I, (float, 2):Point2D, (int, 3):Point3I, (float, 3):Point3D}
Extent = {(int, 2):Extent2I, (float, 2):Extent2D, (int, 3):Extent3I, (float, 3):Extent3D}
CoordinateExpr = {2:CoordinateExpr2, 3:CoordinateExpr3}

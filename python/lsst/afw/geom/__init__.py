"""Application Framework geometry code including Point, Extent, and ellipses
"""
from .geomLib import (
    version,
    AffineTransform,
    CoordinateExpr2,
    CoordinateExpr3,
    Extent2I,
    Extent3I,
    Extent2D,
    Extent3D,
    Point2I,
    Point3I,
    Point2D,
    Point3D,
)

PointI = Point2I
PointD = Point2D

ExtentI = Extent2I
ExtentD = Extent2D

import ellipses

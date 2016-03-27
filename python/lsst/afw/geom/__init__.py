#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

"""Application Framework geometry code including Point, Extent, and ellipses
"""
from .geomLib import *
from .xyTransformFactory import *
from .transformConfig import *
from .utils import *

BoxI = Box2I
BoxD = Box2D

PointI = Point2I
PointD = Point2D

ExtentI = Extent2I
ExtentD = Extent2D

Point = {(int, 2):Point2I, (float, 2):Point2D, (int, 3):Point3I, (float, 3):Point3D}
Extent = {(int, 2):Extent2I, (float, 2):Extent2D, (int, 3):Extent3I, (float, 3):Extent3D}
CoordinateExpr = {2:CoordinateExpr2, 3:CoordinateExpr3}

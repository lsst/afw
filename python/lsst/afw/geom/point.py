from __future__ import absolute_import

from ._point import *

PointI = Point2I
PointD = Point2D
Point = {(int, 2):Point2I, (float, 2):Point2D, (int, 3):Point3I, (float, 3):Point3D}


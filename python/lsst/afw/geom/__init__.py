# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
# 
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the LSST License Statement and 
# the GNU General Public License along with this program.  If not, 
# see <http://www.lsstcorp.org/LegalNotices/>.
#

"""Application Framework geometry code including Point, Extent, and ellipses
"""
from .geomLib import *

BoxI = Box2I
BoxD = Box2D

PointI = Point2I
PointD = Point2D

ExtentI = Extent2I
ExtentD = Extent2D

Point = {(int, 2):Point2I, (float, 2):Point2D, (int, 3):Point3I, (float, 3):Point3D}
Extent = {(int, 2):Extent2I, (float, 2):Extent2D, (int, 3):Extent3I, (float, 3):Extent3D}
CoordinateExpr = {2:CoordinateExpr2, 3:CoordinateExpr3}

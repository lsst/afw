#
# LSST Data Management System
# Copyright 2008-2017 LSST/AURA.
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

from lsst.utils import deprecate_pybind11

# for backwards compatibility make lsst.geom public symbols available in lsst.afw.geom
from lsst.geom import *

# But we deprecate the usages of these aliases
# Constants (like geom.PI) and units (geom.arcseconds) can not be wrapped
# by deprecate_pybind11.
AffineTransform = deprecate_pybind11(AffineTransform,
                                     reason="Replaced by lsst.geom.AffineTransform (will be removed before the release of v20.0)")
Angle = deprecate_pybind11(Angle,
                           reason="Replaced by lsst.geom.Angle (will be removed before the release of v20.0)")
AngleUnit = deprecate_pybind11(AngleUnit,
                               reason="Replaced by lsst.geom.AngleUnit (will be removed before the release of v20.0)")
Box2D = deprecate_pybind11(Box2D,
                           reason="Replaced by lsst.geom.Box2D (will be removed before the release of v20.0)")
Box2I = deprecate_pybind11(Box2I,
                           reason="Replaced by lsst.geom.Box2I (will be removed before the release of v20.0)")
BoxD = deprecate_pybind11(BoxD,
                          reason="Replaced by lsst.geom.BoxD (will be removed before the release of v20.0)")
BoxI = deprecate_pybind11(BoxI,
                          reason="Replaced by lsst.geom.BoxI (will be removed before the release of v20.0)")

CoordinateExpr = deprecate_pybind11(CoordinateExpr,
                                    reason="Replaced by lsst.geom.CoordinateExpr (will be removed before the release of v20.0)")
CoordinateExpr2 = deprecate_pybind11(CoordinateExpr2,
                                     reason="Replaced by lsst.geom.CoordinateExpr2 (will be removed before the release of v20.0)")
CoordinateExpr3 = deprecate_pybind11(CoordinateExpr3,
                                     reason="Replaced by lsst.geom.CoordinateExpr3 (will be removed before the release of v20.0)")
Extent = deprecate_pybind11(Extent,
                            reason="Replaced by lsst.geom.Extent (will be removed before the release of v20.0)")
Extent2D = deprecate_pybind11(Extent2D,
                              reason="Replaced by lsst.geom.Extent2D (will be removed before the release of v20.0)")
Extent2I = deprecate_pybind11(Extent2I,
                              reason="Replaced by lsst.geom.Extent2I (will be removed before the release of v20.0)")
Extent3D = deprecate_pybind11(Extent3D,
                              reason="Replaced by lsst.geom.Extent3D (will be removed before the release of v20.0)")
Extent3I = deprecate_pybind11(Extent3I,
                              reason="Replaced by lsst.geom.Extent3I (will be removed before the release of v20.0)")
ExtentBase2D = deprecate_pybind11(ExtentBase2D,
                                  reason="Replaced by lsst.geom.ExtentBase2D (will be removed before the release of v20.0)")
ExtentBase2I = deprecate_pybind11(ExtentBase2I,
                                  reason="Replaced by lsst.geom.ExtentBase2I (will be removed before the release of v20.0)")
ExtentBase3D = deprecate_pybind11(ExtentBase3D,
                                  reason="Replaced by lsst.geom.ExtentBase3D (will be removed before the release of v20.0)")
ExtentBase3I = deprecate_pybind11(ExtentBase3I, reason="Replaced by lsst.geom.ExtentBase3I (will be removed before the release of v20.0)")
ExtentD = deprecate_pybind11(ExtentD,
                             reason="Replaced by lsst.geom.ExtentD (will be removed before the release of v20.0)")
ExtentI = deprecate_pybind11(ExtentI,
                             reason="Replaced by lsst.geom.ExtentI (will be removed before the release of v20.0)")
LinearTransform = deprecate_pybind11(LinearTransform,
                                     reason="Replaced by lsst.geom.LinearTransform (will be removed before the release of v20.0)")
Point = deprecate_pybind11(Point,
                           reason="Replaced by lsst.geom.Point (will be removed before the release of v20.0)")
Point2D = deprecate_pybind11(Point2D,
                             reason="Replaced by lsst.geom.Point2D (will be removed before the release of v20.0)")
Point2I = deprecate_pybind11(Point2I,
                             reason="Replaced by lsst.geom.Point2I (will be removed before the release of v20.0)")
Point3D = deprecate_pybind11(Point3D,
                             reason="Replaced by lsst.geom.Point3D (will be removed before the release of v20.0)")
Point3I = deprecate_pybind11(Point3I,
                             reason="Replaced by lsst.geom.Point3I (will be removed before the release of v20.0)")
PointBase2D = deprecate_pybind11(PointBase2D,
                                 reason="Replaced by lsst.geom.PointBase2D (will be removed before the release of v20.0)")
PointBase2I = deprecate_pybind11(PointBase2I,
                                 reason="Replaced by lsst.geom.PointBase2I (will be removed before the release of v20.0)")
PointBase3D = deprecate_pybind11(PointBase3D, reason="Replaced by lsst.geom.PointBase3D (will be removed before the release of v20.0)")
PointBase3I = deprecate_pybind11(PointBase3I,
                                 reason="Replaced by lsst.geom.PointBase3I (will be removed before the release of v20.0)")
PointD = deprecate_pybind11(PointD,
                            reason="Replaced by lsst.geom.PointD (will be removed before the release of v20.0)")
PointI = deprecate_pybind11(PointI,
                            reason="Replaced by lsst.geom.PointI (will be removed before the release of v20.0)")
SpherePoint = deprecate_pybind11(SpherePoint,
                                 reason="Replaced by lsst.geom.SpherePoint (will be removed before the release of v20.0)")
arcsecToRad = deprecate_pybind11(arcsecToRad,
                                 reason="Replaced by lsst.geom.arcsecToRad (will be removed before the release of v20.0)")
averageSpherePoint = deprecate_pybind11(averageSpherePoint,
                                        reason="Replaced by lsst.geom.averageSpherePoint (will be removed before the release of v20.0)")
degToRad = deprecate_pybind11(degToRad,
                              reason="Replaced by lsst.geom.degToRad (will be removed before the release of v20.0)")
isAngle = deprecate_pybind11(isAngle,
                             reason="Replaced by lsst.geom.isAngle (will be removed before the release of v20.0)")
makeAffineTransformFromTriple = deprecate_pybind11(makeAffineTransformFromTriple,
                                                   reason="Replaced by lsst.geom.makeAffineTransformFromTriple (will be removed before the release of v20.0)")
masToRad = deprecate_pybind11(masToRad,
                              reason="Replaced by lsst.geom.masToRad (will be removed before the release of v20.0)")
radToArcsec = deprecate_pybind11(radToArcsec,
                                 reason="Replaced by lsst.geom.radToArcsec (will be removed before the release of v20.0)")
radToDeg = deprecate_pybind11(radToDeg,
                              reason="Replaced by lsst.geom.radToDeg (will be removed before the release of v20.0)")
radToMas = deprecate_pybind11(radToMas,
                              reason="Replaced by lsst.geom.radToMas (will be removed before the release of v20.0)")

del deprecate_pybind11

from .ellipses import Ellipse, Quadrupole
from .polygon import *
from .span import *
from .spanSet import *

from . import python
from .transformConfig import *
from .utils import *
from .endpoint import *
from .transform import *
from .transformFactory import *
from .transformConfig import *
from .skyWcs import *
from .transformFromString import *
from . import wcsUtils
from .sipApproximation import *

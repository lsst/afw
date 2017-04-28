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

from __future__ import absolute_import, division, print_function

__all__ = ["CoordinateExpr", "Extent", "ExtentI", "ExtentD",
           "Point", "PointI", "PointD"]

from future.utils import with_metaclass
from lsst.utils import TemplateMeta

from . import coordinates


def _coordinateStr(self):
    return "({})".format(", ".join("%0.5g" % v for v in self))


def _coordinateRepr(self):
    return "{}({})".format(type(self).__name__,
                           ", ".join("%0.10g" % v for v in self))


def _coordinateReduce(self):
    return (type(self), tuple(self))


class CoordinateExpr(with_metaclass(TemplateMeta, object)):
    """Abstract base class and factory for CoordinateExpr objects.
    """
    TEMPLATE_PARAMS = ("dimensions", )

    __str__ = _coordinateStr
    __repr__ = _coordinateRepr
    __reduce__ = _coordinateReduce


CoordinateExpr.register(2, coordinates.CoordinateExpr2)
CoordinateExpr.register(3, coordinates.CoordinateExpr3)


class Extent(with_metaclass(TemplateMeta, object)):
    """Abstract base class and factory for Extent objects.
    """
    TEMPLATE_PARAMS = ("dtype", "dimensions")
    TEMPLATE_DEFAULTS = (None, 2)

    __str__ = _coordinateStr
    __repr__ = _coordinateRepr
    __reduce__ = _coordinateReduce


Extent.register((int, 2), coordinates.Extent2I)
Extent.register((float, 2), coordinates.Extent2D)
Extent.register((int, 3), coordinates.Extent3I)
Extent.register((float, 3), coordinates.Extent3D)
ExtentI = coordinates.Extent2I
ExtentD = coordinates.Extent2D


class Point(with_metaclass(TemplateMeta, object)):
    """Abstract base class and factory for Point objects.
    """
    TEMPLATE_PARAMS = ("dtype", "dimensions")
    TEMPLATE_DEFAULTS = (None, 2)

    __str__ = _coordinateStr
    __repr__ = _coordinateRepr
    __reduce__ = _coordinateReduce


Point.register((int, 2), coordinates.Point2I)
Point.register((float, 2), coordinates.Point2D)
Point.register((int, 3), coordinates.Point3I)
Point.register((float, 3), coordinates.Point3D)
PointI = coordinates.Point2I
PointD = coordinates.Point2D

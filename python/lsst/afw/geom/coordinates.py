from __future__ import absolute_import

from future.utils import with_metaclass
from lsst.utils import TemplateMeta
from ._coordinates import * # noqa
from . import _coordinates


__all__ = dir(_coordinates) + ["CoordinateExpr",
                               "Extent", "ExtentI", "ExtentD",
                               "Point", "PointI", "PointD"]


def _coordinateStr(self):
    return "({})".format(", ".join("%0.5g" % v for v in self))


def _coordinateRepr(self):
    return "{}({})".format(type(self).__name__, ", ".join("%0.10g" % v for v in self))


def _coordinateReduce(self):
    return (type(self), tuple(self))


class CoordinateExpr(with_metaclass(TemplateMeta, object)):
    """Abstract base class and factory for CoordinateExpr objects.
    """
    TEMPLATE_PARAMS = ("dimensions", )

    __str__ = _coordinateStr
    __repr__ = _coordinateRepr
    __reduce__ = _coordinateReduce


CoordinateExpr.register(2, _coordinates.CoordinateExpr2)
CoordinateExpr.register(3, _coordinates.CoordinateExpr3)


class Extent(with_metaclass(TemplateMeta, object)):
    """Abstract base class and factory for Extent objects.
    """
    TEMPLATE_PARAMS = ("dtype", "dimensions")
    TEMPLATE_DEFAULTS = (None, 2)

    __str__ = _coordinateStr
    __repr__ = _coordinateRepr
    __reduce__ = _coordinateReduce


Extent.register((int, 2), _coordinates.Extent2I)
Extent.register((float, 2), _coordinates.Extent2D)
Extent.register((int, 3), _coordinates.Extent3I)
Extent.register((float, 3), _coordinates.Extent3D)
ExtentI = _coordinates.Extent2I
ExtentD = _coordinates.Extent2D


class Point(with_metaclass(TemplateMeta, object)):
    """Abstract base class and factory for Point objects.
    """
    TEMPLATE_PARAMS = ("dtype", "dimensions")
    TEMPLATE_DEFAULTS = (None, 2)

    __str__ = _coordinateStr
    __repr__ = _coordinateRepr
    __reduce__ = _coordinateReduce


Point.register((int, 2), _coordinates.Point2I)
Point.register((float, 2), _coordinates.Point2D)
Point.register((int, 3), _coordinates.Point3I)
Point.register((float, 3), _coordinates.Point3D)
PointI = _coordinates.Point2I
PointD = _coordinates.Point2D

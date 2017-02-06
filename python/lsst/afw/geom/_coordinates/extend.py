from __future__ import absolute_import

from future.utils import with_metaclass
from lsst.utils import TemplateMeta

from . import wrap

__all__ = ["CoordinateExpr", "Extent", "ExtentI", "ExtentD",
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


CoordinateExpr.register(2, wrap.CoordinateExpr2)
CoordinateExpr.register(3, wrap.CoordinateExpr3)


class Extent(with_metaclass(TemplateMeta, object)):
    """Abstract base class and factory for Extent objects.
    """
    TEMPLATE_PARAMS = ("dtype", "dimensions")
    TEMPLATE_DEFAULTS = (None, 2)

    __str__ = _coordinateStr
    __repr__ = _coordinateRepr
    __reduce__ = _coordinateReduce


Extent.register((int, 2), wrap.Extent2I)
Extent.register((float, 2), wrap.Extent2D)
Extent.register((int, 3), wrap.Extent3I)
Extent.register((float, 3), wrap.Extent3D)
ExtentI = wrap.Extent2I
ExtentD = wrap.Extent2D


class Point(with_metaclass(TemplateMeta, object)):
    """Abstract base class and factory for Point objects.
    """
    TEMPLATE_PARAMS = ("dtype", "dimensions")
    TEMPLATE_DEFAULTS = (None, 2)

    __str__ = _coordinateStr
    __repr__ = _coordinateRepr
    __reduce__ = _coordinateReduce


Point.register((int, 2), wrap.Point2I)
Point.register((float, 2), wrap.Point2D)
Point.register((int, 3), wrap.Point3I)
Point.register((float, 3), wrap.Point3D)
PointI = wrap.Point2I
PointD = wrap.Point2D

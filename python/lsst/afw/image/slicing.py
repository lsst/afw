from __future__ import absolute_import, division, print_function

import lsst.afw.geom as afwGeom
from .image import LOCAL

__all__ = ["supportSlicing"]


def _getBBoxFromSliceTuple(img, imageSlice):
    """Given a slice specification return the proper Box2I
    This is the worker routine behind __getitem__ and __setitem__

    The imageSlice may be:
       lsst.afw.geom.Box2I
       slice, slice
       :
    Only the first one or two parts of the slice are recognised (no stride), a single int is
    interpreted as n:n+1, and negative indices are interpreted relative to the end of the array,
    so supported slices include:
       2
       -1
       1:10
       :-2
       : (equivalent to ... (python's Ellipsis) which is also supported)

    E.g.
     im[-1, :]
     im[..., 18]
     im[4,  10]
     im[-3:, -2:]
     im[-2, -2]
     im[1:4, 6:10]
     im[:]
    """
    if isinstance(imageSlice, afwGeom.Box2I):
        return imageSlice

    if isinstance(imageSlice, slice) and imageSlice.start is None and imageSlice.stop is None:
        imageSlice = (Ellipsis, Ellipsis,)

    if not (isinstance(imageSlice, tuple) and len(imageSlice) == 2 and
            sum([isinstance(_, (slice, type(Ellipsis), int)) for _ in imageSlice]) == 2):
        raise IndexError("Images may only be indexed as a 2-D slice not %s", imageSlice)

    imageSlice, _imageSlice = [], imageSlice
    for s, wh in zip(_imageSlice, img.getDimensions()):
        if isinstance(s, slice):
            pass
        elif isinstance(s, int):
            if s < 0:
                s += wh
            s = slice(s, s + 1)
        else:
            s = slice(0, wh)

        imageSlice.append(s)

    x, y = [_.indices(wh) for _, wh in zip(imageSlice, img.getDimensions())]
    return afwGeom.Box2I(afwGeom.Point2I(x[0], y[0]), afwGeom.Point2I(x[1] - 1, y[1] - 1))


def supportSlicing(cls):
    """Support image slicing
    """
    def Factory(self, *args):
        """Return an object of this type
        """
        return cls(*args)
    cls.Factory = Factory

    def clone(self):
        """Return a deep copy of self"""
        return cls(self, True)
    cls.clone = clone

    def __getitem__(self, imageSlice):
        return self.Factory(self, _getBBoxFromSliceTuple(self, imageSlice), LOCAL)
    cls.__getitem__ = __getitem__

    def __setitem__(self, imageSlice, rhs):
        bbox = _getBBoxFromSliceTuple(self, imageSlice)

        if self.assign(rhs, bbox, LOCAL) is NotImplemented:
            lhs = self.Factory(self, bbox, LOCAL)
            lhs.set(rhs)
    cls.__setitem__ = __setitem__

    def __float__(self):
        """Convert a 1x1 image to a floating scalar"""
        if self.getDimensions() != afwGeom.Extent2I(1, 1):
            raise TypeError("Only single-pixel images may be converted to python scalars")

        try:
            return float(self.get(0, 0))
        except AttributeError:
            raise TypeError("Unable to extract a single pixel from a {}".format(type(self).__name__))
        except TypeError:
            raise TypeError("Unable to convert a {} pixel to a scalar".format(type(self).__name__))
    cls.__float__ = __float__

    def __int__(self):
        """Convert a 1x1 image to a integral scalar"""
        return int(float(self))
    cls.__int__ = __int__

import lsst.afw.geom as afwGeom
from .image import LOCAL, PARENT, ImageOrigin

__all__ = ["supportSlicing"]


def _getBBoxFromSliceTuple(img, inputSlice):
    """Given a slice specification return the proper Box2I and origin
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

    You may also add an extra argument, afwImage.PARENT or afwImage.LOCAL.  The default is PARENT
    as before, but if you specify LOCAL the bounding box is interpreted in LOCAL coordinates
    (this includes slices; e.g.
     im[3:, 2:, afwImage.LOCAL]
    still means the last few rows and columns, and
     im[1001:1004, 2006:2010, afwImage.PARENT]
    with xy0 = (1000, 2000) refers to the same pixels as
     im[1:4, 6:10, afwImage.LOCAL]
    )
    """
    origin = NotImplemented                      # this sets the default value

    try:
        if isinstance(inputSlice[-1], ImageOrigin):
            origin = inputSlice[-1]
            # imageSlice should only be a tuple if it has more than one element,
            # both before we pull out the origin and after.
            assert len(inputSlice) > 1, "This function should only be called from __getitem__/__setitem__"
            inputSlice = inputSlice[0] if len(inputSlice) == 2 else inputSlice[:-1]
    except TypeError:
        pass

    if isinstance(inputSlice, afwGeom.Box2I):
        return inputSlice, origin

    if isinstance(inputSlice, slice) and inputSlice.start is None and inputSlice.stop is None:
        inputSlice = (Ellipsis, Ellipsis,)

    if not (isinstance(inputSlice, tuple) and len(inputSlice) == 2 and
            sum([isinstance(_, (slice, type(Ellipsis), int)) for _ in inputSlice]) == 2):
        raise IndexError(
            "Images may only be indexed as a 2-D slice not %s" % inputSlice)

    # Because we're going to use slice.indices(...) to construct our ranges, and
    # python doesn't understand PARENT coordinate systems,  we need
    # to convert slices specified in PARENT coords to LOCAL

    outputSlice = []
    for s, wh, z0 in zip(inputSlice, img.getDimensions(), img.getXY0()):

        if origin is not NotImplemented and origin == LOCAL:
            z0 = 0                      # ignore image's xy0

        if isinstance(s, slice):
            if origin is NotImplemented and (slice.start is not None or slice.end is not None):
                    raise NotImplementedError("Origin defaults have been temporarily disabled.")
            if z0 != 0:
                if s.start < 0 or s.end < 0:
                    raise IndexError("Negative slice indices are not supported on images "
                                     "when using origin=PARENT.")
                start = s.start if s.start is None or s.start < 0 else s.start - z0
                stop = s.stop if s.stop is None or s.stop < 0 else s.stop - z0
                s = slice(start, stop, s.step)
        elif isinstance(s, int):
            if origin is NotImplemented:
                raise NotImplementedError("Origin defaults have been temporarily disabled.")
            if s < 0:
                if z0 != 0:
                    raise IndexError("Negative indices are not supported on images "
                                     "when using origin=PARENT.")
                s += z0 + wh
            s = slice(s - z0, s - z0 + 1)
        else:
            s = slice(0, wh)

        outputSlice.append(s)

    x, y = [_.indices(wh) for _, wh in zip(outputSlice, img.getDimensions())]
    return afwGeom.Box2I(afwGeom.Point2I(x[0], y[0]), afwGeom.Point2I(x[1] - 1, y[1] - 1)), LOCAL


def supportSlicing(cls):
    """Support image slicing
    """

    def _checkOrigin(origin):
        if origin not in (LOCAL, PARENT):
            raise RuntimeError("keyword origin must be afwImage.ORIGIN or afwImage.PARENT, not %s" % origin)

    def Factory(self, *args, **kwargs):
        """Return an object of this type
        """
        return cls(*args, **kwargs)
    cls.Factory = Factory

    def clone(self):
        """Return a deep copy of self"""
        return cls(self, True)
    cls.clone = clone

    def __getitem__(self, imageSlice):
        bbox, origin = _getBBoxFromSliceTuple(self, imageSlice)

        _checkOrigin(origin)
        return self.Factory(self, bbox, origin)
    cls.__getitem__ = __getitem__

    def __setitem__(self, imageSlice, rhs):
        bbox, origin = _getBBoxFromSliceTuple(self, imageSlice)

        _checkOrigin(origin)
        if self.assign(rhs, bbox, origin) is NotImplemented:
            lhs = self.Factory(self, bbox, origin=origin)
            lhs.set(rhs)
    cls.__setitem__ = __setitem__

    def __float__(self):
        """Convert a 1x1 image to a floating scalar"""
        if self.getDimensions() != afwGeom.Extent2I(1, 1):
            raise TypeError(
                "Only single-pixel images may be converted to python scalars")

        try:
            return float(self.get(0, 0))
        except AttributeError:
            raise TypeError(
                "Unable to extract a single pixel from a {}".format(type(self).__name__))
        except TypeError:
            raise TypeError(
                "Unable to convert a {} pixel to a scalar".format(type(self).__name__))
    cls.__float__ = __float__

    def __int__(self):
        """Convert a 1x1 image to a integral scalar"""
        return int(float(self))
    cls.__int__ = __int__

from lsst.geom import Point2I, Box2I
from .image import LOCAL, ImageOrigin

__all__ = ["supportSlicing"]


def splitSliceArgs(sliceArgs):
    """Separate the actual slice from an origin arguments to __getitem__ or
    __setitem__, using a default for the origin if it is not provided.

    See interpretSliceArgs for more information.
    """
    defaultOrigin = NotImplemented  # TODO: change default origin to PARENT.
    try:
        if isinstance(sliceArgs[-1], ImageOrigin):
            # Args are already a tuple that includes the origin.
            if len(sliceArgs) == 2:
                return sliceArgs[0], sliceArgs[-1]
            else:
                return sliceArgs[:-1], sliceArgs[-1]
        else:
            # Args are a tuple that does not include the origin; add it to make origin explicit.
            return sliceArgs, defaultOrigin
    except TypeError:  # Arg is a scalar; return it along with the default origin.
        return sliceArgs, defaultOrigin


def handleNegativeIndex(index, size, origin, default):
    """Handle negative indices passed to image accessors.

    When negative indices are used in LOCAL coordinates, we interpret them as
    relative to the upper bounds of the array, as in regular negative indexing
    in Python.

    When negative indices are used in PARENT coordinates, we interpret them as
    actual negative pixel values.
    """
    if index is None:
        assert default is not None
        return default
    if index < 0 and origin == LOCAL:
        index = size + index
    return index


def makePointFromIndices(x, y, origin, parent):
    """Create a Point2I from an x, y pair, correctly handling negative indices.
    """
    return Point2I(
        handleNegativeIndex(x, parent.getWidth(), origin, default=None),
        handleNegativeIndex(y, parent.getHeight(), origin, default=None)
    )


def makeBoxFromSlices(x, y, origin, parent):
    """Transform a tuple of slice objects into a Box2I, correctly handling negative indices.
    """
    if x.step is not None or y.step is not None:
        raise ValueError("Slices with steps are not supported in image indexing.")
    begin = Point2I(
        handleNegativeIndex(x.start, parent.getWidth(), origin, default=parent.getBeginX()),
        handleNegativeIndex(y.start, parent.getHeight(), origin, default=parent.getBeginY())
    )
    end = Point2I(
        handleNegativeIndex(x.stop, parent.getWidth(), origin, default=parent.getEndX()),
        handleNegativeIndex(y.stop, parent.getHeight(), origin, default=parent.getEndY())
    )
    return Box2I(begin, end - begin)


def interpretSliceArgs(sliceArgs, bboxGetter):
    """Transform arguments to __getitem__ or __setitem__ to a standard form.

    Parameters
    ----------
    sliceArgs : `tuple`, `Box2I`, or `Point2I`
        Slice arguments passed directly to `__getitem__` or `__setitem__`.
    bboxGetter : callable
        Callable that accepts an ImageOrigin enum value and returns the
        appropriate image bounding box.  Usually the bound getBBox method
        of an Image, Mask, or MaskedImage object.

    Returns
    -------
    box : `Box2I` or `None`
        A box to use to create a subimage, or None if the slice refers to a
        scalar.
    index: `tuple` or `None`
        An ``(x, y)`` tuple of integers, or None if the slice refers to a
        box.
    origin : `ImageOrigin`
        Enum indicating whether to account for xy0.
    """
    slices, origin = splitSliceArgs(sliceArgs)
    if isinstance(slices, Point2I):
        return None, slices, origin
    elif isinstance(slices, Box2I):
        return slices, None, origin
    elif isinstance(slices, slice):
        if slices.start is not None or slices.stop is not None or slices.step is not None:
            raise TypeError("Single-dimension slices must not have bounds.")
        x = slices
        y = slices
        origin = LOCAL  # doesn't matter, as the slices cover the full image
    else:
        x, y = slices
    if isinstance(x, slice):
        if isinstance(y, slice):
            return makeBoxFromSlices(x, y, origin=origin, parent=bboxGetter(origin)), None, origin
        raise TypeError("Mixed indices of the form (slice, int) are not supported for images.")
    else:
        if isinstance(y, slice):
            raise TypeError("Mixed indices of the form (int, slice) are not supported for images.")
        return None, makePointFromIndices(x, y, origin=origin, parent=bboxGetter(origin)), origin


def supportSlicing(cls):
    """Support image slicing
    """

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
        box, index, origin = interpretSliceArgs(imageSlice, self.getBBox)
        if box is not None:
            return self.subset(box, origin=origin)
        return self._get(index, origin=origin)
    cls.__getitem__ = __getitem__

    def __setitem__(self, imageSlice, rhs):
        box, index, origin = interpretSliceArgs(imageSlice, self.getBBox)
        if box is not None:
            if self.assign(rhs, box, origin) is NotImplemented:
                lhs = self.subset(box, origin=origin)
                lhs.set(rhs)
        else:
            self._set(index, origin=origin, value=rhs)
    cls.__setitem__ = __setitem__

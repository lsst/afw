from lsst.geom import Point2I, Box2I
from .image import LOCAL, PARENT, ImageOrigin

__all__ = ["supportSlicing"]


def splitSliceArgs(sliceArgs):
    """Separate the actual slice from origin arguments to __getitem__ or
    __setitem__, using PARENT for the origin if it is not provided.

    Parameter
    ---------
    sliceArgs : `tuple`, `Box2I`, or `Point2I`
        The first argument passed to an image-like object's
        ``__getitem__`` or ``__setitem__``.

    Returns
    -------
    sliceArgs : `tuple`, `Box2I`, or `Point2I`
        The original sliceArgs passed in, with any ImageOrigin argument
        stripped off.
    origin : `ImageOrigin`
        Enum value that sets whether or not to consider xy0 in positions.

    See interpretSliceArgs for more information.

    Intended primarily for internal use by `supportSlicing()`.
    """
    defaultOrigin = PARENT
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

    Using negative indices in PARENT coordinates is not allowed unless passed
    via a `Point2I` or `Box2I`; the potential for confusion between actual
    negative indices (when ``xy0 < 0``) and offsets relative to the upper
    bounds of the array is too great.

    Parameters
    ----------
    index : `int` or `None`
        1-d pixel index to interpret, as given by a caller to an image-like
        object's ``__getitem__`` or ``__setitem__``.
    size : `int`
        Size of the image in the dimension corresponding to ``index``.
    origin : `ImageOrigin`
        Enum value that sets whether or not to consider xy0 in positions.
    default : `int`
        Index to return if `index` is None.

    Returns
    -------
    index : `int`
        If ``origin==PARENT``, either the given ``index`` or ``default``.
        If ``origin==LOCAL``, an equivalent index guaranteed to be nonnegative.

    Intended primarily for internal use by `supportSlicing()`.
    """
    if index is None:
        assert default is not None
        return default
    if index < 0:
        if origin == LOCAL:
            index = size + index
        else:
            raise IndexError("Negative indices are not permitted with the PARENT origin. "
                             "Use LOCAL to use negative to index relative to the end, "
                             "and Point2I or Box2I indexing to access negative pixels "
                             "in PARENT coordinates.")
    return index


def translateSliceArgs(sliceArgs, bboxGetter):
    """Transform a tuple of slice objects into a Box2I, correctly handling negative indices.

    see `interpretSliceArgs` for a description of parameters

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

    x, y, origin = interpretSliceArgs(sliceArgs, bboxGetter)

    if isinstance(x, slice):
        assert isinstance(y, slice)
        if x.step is not None or y.step is not None:
            raise ValueError("Slices with steps are not supported in image indexing.")
        begin = Point2I(x.start, y.start)
        end = Point2I(x.stop, y.stop)
        return Box2I(begin, end - begin), None, origin

    assert not isinstance(y, slice)
    return None, Point2I(x, y), origin


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
    x : int or slice
        Index or slice in the x dimension
    y : int or slice
        Index or slice in the y dimension
    origin : `ImageOrigin`
        Either `PARENT` (coordinates respect XY0) or LOCAL
        (coordinates do not respect XY0)
    """
    slices, origin = splitSliceArgs(sliceArgs)
    if isinstance(slices, Point2I):
        return slices.getX(), slices.getY(), origin
    elif isinstance(slices, Box2I):
        x0 = slices.getMinX()
        y0 = slices.getMinY()
        return slice(x0, x0 + slices.getWidth()), slice(y0, y0 + slices.getHeight()), origin
    elif isinstance(slices, slice):
        if slices.start is not None or slices.stop is not None or slices.step is not None:
            raise TypeError("Single-dimension slices must not have bounds.")
        x = slices
        y = slices
        origin = LOCAL  # doesn't matter, as the slices cover the full image
    else:
        x, y = slices

    bbox = bboxGetter(origin)
    if isinstance(x, slice):
        if isinstance(y, slice):
            xSlice = slice(handleNegativeIndex(x.start, bbox.getWidth(), origin, default=bbox.getBeginX()),
                           handleNegativeIndex(x.stop, bbox.getWidth(), origin, default=bbox.getEndX()))
            ySlice = slice(handleNegativeIndex(y.start, bbox.getHeight(), origin, default=bbox.getBeginY()),
                           handleNegativeIndex(y.stop, bbox.getHeight(), origin, default=bbox.getEndY()))
            return xSlice, ySlice, origin
        raise TypeError("Mixed indices of the form (slice, int) are not supported for images.")

    if isinstance(y, slice):
        raise TypeError("Mixed indices of the form (int, slice) are not supported for images.")
    x = handleNegativeIndex(x, bbox.getWidth(), origin, default=None)
    y = handleNegativeIndex(y, bbox.getHeight(), origin, default=None)
    return x, y, origin


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
        box, index, origin = translateSliceArgs(imageSlice, self.getBBox)
        if box is not None:
            return self.subset(box, origin=origin)
        return self._get(index, origin=origin)
    cls.__getitem__ = __getitem__

    def __setitem__(self, imageSlice, rhs):
        box, index, origin = translateSliceArgs(imageSlice, self.getBBox)
        if box is not None:
            if self.assign(rhs, box, origin) is NotImplemented:
                lhs = self.subset(box, origin=origin)
                lhs.set(rhs)
        else:
            self._set(index, origin=origin, value=rhs)
    cls.__setitem__ = __setitem__

from lsst.afw.image.imageSlice import ImageSliceF, ImageSliceD

__all__ = ("disableImageArithmetic", "disableMaskArithmetic")


def wrapNotImplemented(cls, attr):
    """Wrap a method providing a helpful error message about image arithmetic

    Parameters
    ----------
    cls : `type`
        Class in which the method is to be defined.
    attr : `str`
        Name of the method.

    Returns
    -------
    method : callable
        Wrapped method.
    """
    existing = getattr(cls, attr, None)

    def notImplemented(self, other):
        """Provide a helpful error message about image arithmetic

        Unless we're operating on an ImageSlice, in which case it might be
        defined.

        Parameters
        ----------
        self : subclass of `lsst.afw.image.ImageBase`
            Image someone's attempting to do arithmetic with.
        other : anything
            The operand of the arithmetic operation.
        """
        if existing is not None and isinstance(other, (ImageSliceF, ImageSliceD)):
            return existing(self, other)
        raise NotImplementedError("This arithmetic operation is not implemented, in order to prevent the "
                                  "accidental proliferation of temporaries. Please use the in-place "
                                  "arithmetic operations (e.g., += instead of +) or operate on the "
                                  "underlying arrays.")
    return notImplemented


def disableImageArithmetic(cls):
    """Add helpful error messages about image arithmetic"""
    for attr in ("__add__", "__sub__", "__mul__", "__truediv__",
                 "__radd__", "__rsub__", "__rmul__", "__rtruediv__"):
        setattr(cls, attr, wrapNotImplemented(cls, attr))


def disableMaskArithmetic(cls):
    """Add helpful error messages about mask arithmetic"""
    for attr in ("__or__", "__and__", "__xor__",
                 "__ror__", "__rand__", "__rxor__"):
        setattr(cls, attr, wrapNotImplemented(cls, attr))

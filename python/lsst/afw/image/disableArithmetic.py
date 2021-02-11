# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

from __future__ import absolute_import, division, print_function

from ._arrays import ArrayFKey, ArrayDKey

__all__ = []  # import this module only for its side effects


def __getitem__(self, index):
    """
    operator[] in C++ only returns a single item, but `Array` has a method to get a slice of the
    array. To make the code more python we automatically check for a slice and return either
    a single item or slice as requested by the user.
    """
    if isinstance(index, slice):
        start, stop, stride = index.indices(self.getSize())
        if stride != 1:
            raise IndexError("Non-unit stride not supported")
        return self.slice(start, stop)
    return self._get_(index)

ArrayFKey.__getitem__ = __getitem__
ArrayDKey.__getitem__ = __getitem__

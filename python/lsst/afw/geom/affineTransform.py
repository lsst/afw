from __future__ import absolute_import

from ._affineTransform import AffineTransform

__all__ = []  # import this module only for its side effects


def addAffineTransformMethods(cls):
    def __setitem__(self, k, v):
        if k < 0 or k > 5:
            raise IndexError
        self._setitem_nochecking(k, v)
    AffineTransform.__setitem__ = __setitem__

    def __getitem__(self, k):
        try:
            i, j = k
            if i < 0 or i > 2:
                raise IndexError
            if j < 0 or j > 2:
                raise IndexError
            return self._getitem_nochecking(i, j)
        except TypeError:
            if k < 0 or k > 5:
                raise IndexError
            return self._getitem_nochecking(k)
    AffineTransform.__getitem__ = __getitem__

    def __str__(self):
        return str(self.getMatrix())
    AffineTransform.__str__ = __str__

    def __reduce__(self):
        return (AffineTransform, (self.getMatrix(),))
    AffineTransform.__reduce__ = __reduce__

    def __repr__(self):
        return "AffineTransform(\n%r\n)" % (self.getMatrix(),)
    AffineTransform.__repr__ = __repr__

addAffineTransformMethods(AffineTransform)

from __future__ import absolute_import

from ._affineTransform import *

def AffineTransform__setitem__(self, k, v):
    if k < 0 or k > 5: raise IndexError
    self._setitem_nochecking(k, v)
AffineTransform.__setitem__ = AffineTransform__setitem__
del AffineTransform__setitem__

def AffineTransform__getitem__(self, k):
    try:
        i,j = k
        if i < 0 or i > 2: raise IndexError
        if j < 0 or j > 2: raise IndexError
        return self._getitem_nochecking(i, j)
    except TypeError:
        if k < 0 or k > 5: raise IndexError
        return self._getitem_nochecking(k)
AffineTransform.__getitem__ = AffineTransform__getitem__
del AffineTransform__getitem__

def AffineTransform__str__(self):
    return str(self.getMatrix())
AffineTransform.__str__ = AffineTransform__str__
del AffineTransform__str__

def AffineTransform__reduce__(self):
    return (AffineTransform, (self.getMatrix(),))
AffineTransform.__reduce__ = AffineTransform__reduce__
del AffineTransform__reduce__

def AffineTransform__repr__(self):
    return "AffineTransform(\n%r\n)" % (self.getMatrix(),)
AffineTransform.__repr__ = AffineTransform__repr__
del AffineTransform__repr__


from __future__ import absolute_import

from ._linearTransform import *

__all__ = []

def __str__(self):
    return str(self.getMatrix())

def __reduce__(self):
    return (LinearTransform, (self.getMatrix(),))

def __repr__(self):
    return "LinearTransform(\n%r\n)" % (self.getMatrix(),)

LinearTransform.__str__ = __str__
LinearTransform.__reduce__ = __reduce__
LinearTransform.__repr__ = __repr__

from __future__ import absolute_import

from ._maskedImage import *
from .slicing import supportSlicing

for cls in (MaskedImageI, MaskedImageF, MaskedImageD, MaskedImageU, MaskedImageL):
    def set(self, x, y=None, values=None):
        """Set the point (x, y) to a triple (value, mask, variance)"""
    
        if values is None:
            assert (y is None)
            values = x
            try:
                self.getImage().set(values[0])
                self.getMask().set(values[1])
                self.getVariance().set(values[2])
            except TypeError:
                self.getImage().set(values)
                self.getMask().set(0)
                self.getVariance().set(0)
        else:
            try:
                self.getImage().set(x, y, values[0])
                if len(values) > 1:
                    self.getMask().set(x, y, values[1])
                if len(values) > 2:
                   self.getVariance().set(x, y, values[2])
            except TypeError:
                self.getImage().set(x)
                self.getMask().set(y)
                self.getVariance().set(values)
    cls.set = set
    del set
    
    def get(self, x, y):
        """Return a triple (value, mask, variance) at the point (x, y)"""
        return (self.getImage().get(x, y),
                self.getMask().get(x, y),
                self.getVariance().get(x, y))
    cls.get = get
    del get
    
    def getArrays(self):
        """Return a tuple (value, mask, variance) numpy arrays."""
        return (self.getImage().getArray() if self.getImage() else None,
                self.getMask().getArray() if self.getMask() else None,
                self.getVariance().getArray() if self.getVariance() else None)
    cls.getArrays = getArrays
    del getArrays

    supportSlicing(cls)

from __future__ import absolute_import

from ._angle import *

__all__ = []

def _Angle__abs__(self): 
    return abs(self.asRadians())*radians
Angle.__abs__ = _Angle__abs__


def _AngleUnit__mul__(self, other):
    if isinstance(other, (Angle, AngleUnit)):
        raise NotImplementedError
    return AngleUnit._mul(self, other)
AngleUnit.__mul__ = _AngleUnit__mul__


def _AngleUnit__rmul__(self, other):
    if isinstance(other, (Angle, AngleUnit)):
        raise NotImplementedError
    return AngleUnit._rmul(self, other)
AngleUnit.__rmul__ = _AngleUnit__rmul__

 
def _Angle__reduce__(self): 
    return (Angle, (self.asRadians(),))
Angle.__reduce__ = _Angle__reduce__


def _AngleUnit__reduce__(self):
    return (AngleUnit, (1.0*self,))
AngleUnit.__reduce__ = _AngleUnit__reduce__


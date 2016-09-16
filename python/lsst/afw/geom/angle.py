from __future__ import absolute_import

from ._angle import *

def _Angle__reduce__(self): 
    return (Angle, (self.asRadians(),))
def _Angle__abs__(self): 
    return abs(self.asRadians())*radians
def _Angle__eq__(self, rhs):
    try:
        return float(self) == float(rhs)
    except Exception:
        return NotImplemented
def _Angle__ne__(self, rhs):
    return not self == rhs
def _Angle__mul__(self, rhs):
    try:
        return Angle_mul(self, rhs)
    except Exception:
        raise NotImplementedError
def _Angle__rmul__(self, lhs):
    try:
        return Angle_mul(lhs, self)
    except Exception:
        raise NotImplementedError

Angle.__reduce__ = _Angle__reduce__
Angle.__abs__ = _Angle__abs__
Angle.__eq__ = _Angle__eq__
Angle.__ne__ = _Angle__ne__
Angle.__mul__ = _Angle__mul__
Angle.__rmul__ = _Angle__rmul__
Angle.__truediv__ = Angle.__div__

del _Angle__reduce__
del _Angle__abs__
del _Angle__eq__
del _Angle__ne__
del _Angle__mul__
del _Angle__rmul__
 
def _AngleUnit__reduce__(self):
    return (AngleUnit, (1.0*self,))
def _AngleUnit__mul__(self, rhs):
    return AngleUnit_mul(self, rhs)
def _AngleUnit__rmul__(self, lhs):
    return AngleUnit_mul(lhs, self)

AngleUnit.__reduce__ = _AngleUnit__reduce__
AngleUnit.__mul__ = _AngleUnit__mul__
AngleUnit.__rmul__ = _AngleUnit__rmul__

del _AngleUnit__reduce__
del _AngleUnit__mul__
del _AngleUnit__rmul__

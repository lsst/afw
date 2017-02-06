from __future__ import absolute_import

from lsst.utils import continueClass
from .wrap import Angle, AngleUnit, radians

__all__ = []


@continueClass
class Angle:

    def __abs__(self):
        return abs(self.asRadians())*radians

    def __reduce__(self):
        return (Angle, (self.asRadians(),))


@continueClass
class AngleUnit:

    def __mul__(self, other):
        if isinstance(other, (Angle, AngleUnit)):
            raise NotImplementedError
        return AngleUnit._mul(self, other)

    def __rmul__(self, other):
        if isinstance(other, (Angle, AngleUnit)):
            raise NotImplementedError
        return AngleUnit._rmul(self, other)

    def __reduce__(self):
        return (AngleUnit, (1.0*self,))

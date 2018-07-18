__all__ = []  # import this module only for its side effects

from lsst.utils import continueClass
from .axes import Axes


@continueClass  # noqa F811
class Axes:
    def __repr__(self):
        return "Axes(a=%r, b=%r, theta=%r)" % (self.getA(), self.getB(), self.getTheta())

    def __str__(self):
        return "(a=%s, b=%s, theta=%s)" % (self.getA(), self.getB(), self.getTheta())

    def __reduce__(self):
        return (Axes, (self.getA(), self.getB(), self.getTheta()))

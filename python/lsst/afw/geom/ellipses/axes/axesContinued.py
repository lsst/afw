__all__ = []  # import this module only for its side effects

from lsst.utils import continueClass
from .axes import Axes


@continueClass  # noqa: F811
class Axes:
    def __repr__(self):
        return f"Axes(a={self.getA()!r}, b={self.getB()!r}, theta={self.getTheta!r})"

    def __str__(self):
        return f"(a={self.getA()}, b={self.getB()}, theta={self.getTheta})"

    def __reduce__(self):
        return (Axes, (self.getA(), self.getB(), self.getTheta()))

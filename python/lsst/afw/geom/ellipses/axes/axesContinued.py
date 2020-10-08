__all__ = []  # import this module only for its side effects

from lsst.utils import continueClass
from .axes import Axes


@continueClass  # noqa: F811 (FIXME: remove for py 3.8+)
class Axes:  # noqa: F811
    def __repr__(self):
        return f"Axes(a={self.getA()!r}, b={self.getB()!r}, theta={self.getTheta!r})"

    def __str__(self):
        return f"(a={self.getA()}, b={self.getB()}, theta={self.getTheta})"

    def __reduce__(self):
        return (Axes, (self.getA(), self.getB(), self.getTheta()))

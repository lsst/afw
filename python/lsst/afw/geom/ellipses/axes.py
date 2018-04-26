from ._axes import Axes

__all__ = []  # import this module only for its side effects


def __repr__(self):
    return "Axes(a=%r, b=%r, theta=%r)" % (self.getA(), self.getB(), self.getTheta())


Axes.__repr__ = __repr__


def __str__(self):
    return "(a=%s, b=%s, theta=%s)" % (self.getA(), self.getB(), self.getTheta())


Axes.__str__ = __str__


def __reduce__(self):
    return (Axes, (self.getA(), self.getB(), self.getTheta()))


Axes.__reduce__ = __reduce__

__all__ = []  # import this module only for its side effects

from lsst.utils import continueClass
from .quadrupole import Quadrupole


@continueClass  # noqa: F811
class Quadrupole:
    def __repr__(self):
        return f"Quadrupole(ixx={self.getIxx()!r}, iyy={self.getIyy()!r}, ixy={self.getIxy()!r})"

    def __reduce__(self):
        return (Quadrupole, (self.getIxx(), self.getIyy(), self.getIxy()))

    def __str__(self):
        return f"(ixx={self.getIxx()}, iyy={self.getIyy()}, ixy={self.getIxy()})"

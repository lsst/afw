__all__ = []  # import this module only for its side effects

from lsst.utils import continueClass
from .quadrupole import Quadrupole


@continueClass  # noqa F811
class Quadrupole:
    def __repr__(self):
        return "Quadrupole(ixx=%r, iyy=%r, ixy=%r)" % (self.getIxx(), self.getIyy(), self.getIxy())

    def __reduce__(self):
        return (Quadrupole, (self.getIxx(), self.getIyy(), self.getIxy()))

    def __str__(self):
        return "(ixx=%s, iyy=%s, ixy=%s)" % (self.getIxx(), self.getIyy(), self.getIxy())

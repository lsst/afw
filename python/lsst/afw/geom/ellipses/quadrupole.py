from __future__ import absolute_import, division, print_function

from ._quadrupole import Quadrupole

__all__ = []  # import this module only for its side effects


def __repr__(self):
    return "Quadrupole(ixx=%r, iyy=%r, ixy=%r)" % (self.getIxx(), self.getIyy(), self.getIxy())


def __reduce__(self):
    return (Quadrupole, (self.getIxx(), self.getIyy(), self.getIxy()))


def __str__(self):
    return "(ixx=%s, iyy=%s, ixy=%s)" % (self.getIxx(), self.getIyy(), self.getIxy())

Quadrupole.__repr__ = __repr__
Quadrupole.__reduce__ = __reduce__
Quadrupole.__str__ = __str__

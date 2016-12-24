#
# LSST Data Management System
# Copyright 2016 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

from ._spherePoint import SpherePoint

__all__ = []  # import this module only for its side effects


def addSpherePointMethods(cls):
    """Add methods to the pybind11-wrapped SpherePoint class
    """
    def __iter__(self):
        for i in (0, 1):
            yield self[i]
    cls.__iter__ = __iter__

    def __len__(self):
        return 2
    cls.__len__ = __len__

    def __repr__(self):
        argList = ["%r*afwGeom.degrees" % (pos.asDegrees(),) for pos in self]
        return "SpherePoint(%s)" % (", ".join(argList))
    cls.__repr__ = __repr__

    def __reduce__(self):
        return (SpherePoint, (self.getLongitude(), self.getLatitude()))
    cls.__reduce__ = __reduce__

addSpherePointMethods(SpherePoint)

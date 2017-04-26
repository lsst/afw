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
from __future__ import absolute_import, division, print_function

__all__ = ["addTransformMapMethods"]


def addTransformMapMethods(cls):
    """Add pure python methods to an instantiation of a TransformMap<CoordSys> class

    @param[in] cls  The class to which to add the methods, e.g. lsst::afw::cameraGeom::CameraTransformMap
    """
    def __iter__(self):
        """Get an iterator over coordinate systems"""
        return iter(self.getCoordSysList())

    def get(self, coordSys, default=None):
        """Get an XYTransform that transforms from `coordSys` to the native coordinate system
        in the forward direction.

        @parameter[in] coordSys  Coordinate system of desired transform
        @param[in] default  Value to return if `coordSys` is not found in this TransformMap
        """
        if coordSys in self:
            return self[coordSys]
        return default

    cls.__iter__ = __iter__
    cls.get = get

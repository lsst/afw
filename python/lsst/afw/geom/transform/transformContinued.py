#
# LSST Data Management System
# Copyright 2017 LSST/AURA.
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

from lsst.pex.exceptions import InvalidParameterError

from . import transform

__all__ = []


def getJacobian(self, x):
    # Force 2D matrix over numpy's protests
    matrix = self._getJacobian(x)
    matrix.shape = (self.getToEndpoint().getNAxes(),
                    self.getFromEndpoint().getNAxes())
    return matrix


def of(self, first):
    if first.getToEndpoint() == self.getFromEndpoint():
        return self._of(first)
    else:
        raise InvalidParameterError(
            "Cannot concatenate %r and %r: endpoints do not match."
            % (first, self))


endpoints = ("Generic", "Point2", "Point3", "SpherePoint")

for fromPoint in endpoints:
    for toPoint in endpoints:
        name = "Transform" + fromPoint + "To" + toPoint
        cls = getattr(transform, name)
        cls.getJacobian = getJacobian
        cls.of = of

#
# LSST Data Management System
# Copyright 2017 LSST Corporation.
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
"""pybind11 utilities for Transform classes

@anchor pybind11_transform Transforms are instantiations of
lsst::afw::geom::Transform<FromEndpoint, ToEndpoint> and
or subclasses, such as lsst::afw::geom::SkyWcs.
In Python the Transform classes have names such as TransformSpherePointToPoint3.
"""

from __future__ import absolute_import, division, print_function

__all__ = ["addTransformMethods"]

from lsst.pex.exceptions import InvalidParameterError


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


def addTransformMethods(cls):
    """Add pure python methods to the specified @ref pybind11_transform "Transform" class
    """
    cls.getJacobian = getJacobian
    cls.of = of

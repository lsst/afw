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
"""Python helpers for pybind11 wrapping of Transform classes and subclasses

.. _pybind11_transform_classes:

Transform Classes and Subclasses
--------------------------------

Transforms are instances of
lsst::afw::geom::Transform<FromEndpoint, ToEndpoint>
and subclasses, such as lsst::afw::geom::SkyWcs.

In Python the templated Transform classes have names such as
`lsst.afw.geom.TransformSpherePointToPoint3` for
`lsst::afw::geom::Transform<SpherePointEndpoint, Point3Endpoint>`
"""

from __future__ import absolute_import, division, print_function

__all__ = ["addTransformMethods"]

import lsst.pex.exceptions


def getJacobian(self, x):
    # Force 2D matrix over numpy's protests
    matrix = self._getJacobian(x)
    matrix.shape = (self.getToEndpoint().getNAxes(),
                    self.getFromEndpoint().getNAxes())
    return matrix


def of(self, first):
    """Concatenate two transforms

    The result of B.of(A) is C(x) = B(A(x))
    """
    if first.getToEndpoint() == self.getFromEndpoint():
        return self._of(first)
    else:
        raise lsst.pex.exceptions.InvalidParameterError(
            "Cannot concatenate %r and %r: endpoints do not match."
            % (first, self))


def addTransformMethods(cls):
    """Add pure python methods to the specified Transform class

    All :ref:`_pybind11_transform_classes` must call this function.

    Parameters
    ----------
    cls : :ref:`_pybind11_transform_classes`
        A Transform class or subclass, e.g.
        `lsst.afw.geom.TransformPoint2ToSpherePoint`
    """
    cls.getJacobian = getJacobian
    cls.of = of

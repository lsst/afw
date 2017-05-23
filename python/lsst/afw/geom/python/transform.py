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
`lsst.afw.geom.TransformSpherePointToPoint2` for
`lsst::afw::geom::Transform<SpherePointEndpoint, Point2Endpoint>`
"""

from __future__ import absolute_import, division, print_function

__all__ = ["addTransformMethods", "transformRegistry"]

import lsst.pex.exceptions

# registry of transform classes; a dict of class name: transform class
transformRegistry = {}


def getJacobian(self, x):
    # Force 2D matrix over numpy's protests
    matrix = self._getJacobian(x)
    matrix.shape = (self.toEndpoint.nAxes,
                    self.fromEndpoint.nAxes)
    return matrix


def of(self, first):
    """Concatenate two transforms

    The result of B.of(A) is C(x) = B(A(x))
    """
    if first.toEndpoint == self.fromEndpoint:
        return self._of(first)
    else:
        raise lsst.pex.exceptions.InvalidParameterError(
            "Cannot concatenate %r and %r: endpoints do not match."
            % (first, self))


def saveToFile(self, path):
    """Save this @ref pybind11_transform "Transform" to the specified file
    """
    className = type(self).__name__
    bodyText = self.getFrameSet().show()
    with open(path, "w") as outFile:
        outFile.write(className + "\n")
        outFile.write(bodyText)


def addTransformMethods(cls):
    """Add pure python methods to the specified Transform class, and register
    the class in `transformRegistry`

    All :ref:`_pybind11_transform_classes` must call this function.

    Parameters
    ----------
    cls : :ref:`_pybind11_transform_classes`
    A Transform class or subclass, e.g.
    `lsst.afw.geom.TransformPoint2ToSpherePoint`
    """
    global transformRegistry
    className = cls.__name__
    if className in transformRegistry:
        raise RuntimeError("Class %r=%s already registered; cannot register class %s" %
                           (className, transformRegistry[className], cls))
    transformRegistry[cls.__name__] = cls
    cls.getJacobian = getJacobian
    cls.of = of
    cls.saveToFile = saveToFile

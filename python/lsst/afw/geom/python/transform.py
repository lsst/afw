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
`lsst.afw.geom.TransformIcrsCoordToPoint2` for
`lsst::afw::geom::Transform<IcrsCoordEndpoint, Point2Endpoint>`
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


def then(self, next):
    """Concatenate two transforms

    The result of A.then(B) is is C(x) = B(A(x))
    """
    if self.toEndpoint == next.fromEndpoint:
        return self._then(next)
    else:
        raise lsst.pex.exceptions.InvalidParameterError(
            "Cannot concatenate %r and %r: endpoints do not match."
            % (self, next))


def unpickleTransform(cls, state):
    """Unpickle a Transform object

    Parameters
    ----------
    cls : `type`
        A `Transform` class.
    state : `str`
        Pickled state.

    Returns
    -------
    transform : `cls`
        The unpickled Transform.
    """
    return cls.readString(state)


def reduceTransform(transform):
    """Pickle a Transform object

    This provides the `__reduce__` implementation for a Transform.
    """
    return unpickleTransform, (type(transform), transform.writeString())


def addTransformMethods(cls):
    """Add pure python methods to the specified Transform class, and register
    the class in `transformRegistry`

    All :ref:`_pybind11_transform_classes` must call this function.

    Parameters
    ----------
    cls : :ref:`_pybind11_transform_classes`
    A Transform class or subclass, e.g.
    `lsst.afw.geom.TransformPoint2ToIcrsCoord`
    """
    global transformRegistry
    className = cls.__name__
    if className in transformRegistry:
        raise RuntimeError("Class %r=%s already registered; cannot register class %s" %
                           (className, transformRegistry[className], cls))
    transformRegistry[cls.__name__] = cls
    cls.getJacobian = getJacobian
    cls.then = then
    cls.__reduce__ = reduceTransform

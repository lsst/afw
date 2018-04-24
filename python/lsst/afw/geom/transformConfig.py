#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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

import functools
import numpy
from lsst.pex.config import Config, ListField, makeRegistry, \
    ConfigDictField, ConfigurableField
from .transformFactory import makeTransform, makeIdentityTransform, \
    makeRadialTransform
from .affineTransform import AffineTransform

__all__ = ["transformRegistry", "OneTransformConfig", "TransformConfig",
           "IdentityTransformConfig", "AffineTransformConfig", "RadialTransformConfig",
           "MultiTransformConfig"]

transformRegistry = makeRegistry(
    """"A registry of ``Transform`` factories

    A ``Transform`` factory is a function that obeys these rules:
    - has an attribute ``ConfigClass``
    - takes one argument, ``config`` (an instance of ``ConfigClass``) by name
    - returns a ``Transform``
    """
)


class IdentityTransformConfig(Config):
    """A Config representing a ``Transform`` that does nothing.

    See Also
    --------
    lsst.afw.geom.makeIdentityTransform
    """
    pass


def identityFactory(config):
    """Make an identity ``Transform``
    """
    return makeIdentityTransform()


identityFactory.ConfigClass = IdentityTransformConfig
transformRegistry.register("identity", identityFactory)


class OneTransformConfig(Config):
    """A Config representing a single ``Transform`` in a compound ``Transform``.

    See Also
    --------
    lsst.afw.geom.MultiTransformConfig
    """
    transform = ConfigurableField(
        doc="Transform factory",
        target=identityFactory,
    )


def invertingFactory(config):
    """Invert a ``Transform`` specified by config.
    """
    return config.transform.apply().getInverse()


invertingFactory.ConfigClass = OneTransformConfig
transformRegistry.register("inverted", invertingFactory)


class AffineTransformConfig(Config):
    """A Config representing an affine ``Transform``.

    See Also
    --------
    lsst.afw.geom.makeTransform
    """
    linear = ListField(
        doc="2x2 linear matrix in the usual numpy order; "
            "to rotate a vector by theta use: cos(theta), sin(theta), "
            "-sin(theta), cos(theta)",
        dtype=float,
        length=4,
        default=(1, 0, 0, 1),
    )
    translation = ListField(
        doc="x, y translation vector",
        dtype=float,
        length=2,
        default=(0, 0),
    )


def affineFactory(config):
    """Make an affine ``Transform``
    """
    linear = numpy.array(config.linear)
    linear.shape = (2, 2)
    translation = numpy.array(config.translation)
    return makeTransform(AffineTransform(linear, translation))


affineFactory.ConfigClass = AffineTransformConfig
transformRegistry.register("affine", affineFactory)


class RadialTransformConfig(Config):
    """A Config representing a radially symmetric ``Transform``.

    See Also
    --------
    lsst.afw.geom.makeRadialTransform
    """
    coeffs = ListField(
        doc="Coefficients for the radial polynomial; coeff[0] must be 0",
        dtype=float,
        minLength=1,
        optional=False,
    )

    def validate(self):
        if len(self.coeffs) == 0:
            return
        if len(self.coeffs) == 1 or self.coeffs[0] != 0 or self.coeffs[1] == 0:
            raise RuntimeError(
                "invalid radial transform coeffs %s: " % (self.coeffs,) +
                "need len(coeffs)=0 or len(coeffs)>1, coeffs[0]==0, "
                "and coeffs[1]!=0")


def radialFactory(config):
    """Make a radial ``Transform``
    """
    return makeRadialTransform(config.coeffs._list)


radialFactory.ConfigClass = RadialTransformConfig
transformRegistry.register("radial", radialFactory)


class MultiTransformConfig(Config):
    """A Config representing a chain of consecutive ``Transforms``.
    """
    transformDict = ConfigDictField(
        doc="Dict of index: OneTransformConfig (a transform wrapper); "
            "key order is transform order",
        keytype=int,
        itemtype=OneTransformConfig,
    )


def multiFactory(config):
    """Concatenate multiple ``Transforms``
    """
    transformKeys = sorted(config.transformDict.keys())
    transformList = [config.transformDict[key].transform.apply()
                     for key in transformKeys]

    # Can't use then(self, other) directly because no single Transform class
    def concat(transform1, transform2):
        return transform1.then(transform2)

    return functools.reduce(concat, transformList)


multiFactory.ConfigClass = MultiTransformConfig
transformRegistry.register("multi", multiFactory)


class TransformConfig(Config):
    """Config that identifies ``Transforms`` by keyword.

    Supported configs:

    ``"identity"``
        `IdentityTransformConfig`
    ``"inverted"``
        `OneTransformConfig`
    ``"affine"``
        `AffineTransformConfig`
    ``"radial"``
        `RadialTransformConfig`
    ``"multi"``
        `MultiTransformConfig`
    """
    transform = transformRegistry.makeField(
        doc="a Transform from the registry"
    )

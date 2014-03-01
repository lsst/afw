from __future__ import absolute_import, division
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
import numpy
from lsst.pex.config import Config, ListField, makeRegistry, ConfigDictField, ConfigurableField
from .geomLib import IdentityXYTransform, InvertedXYTransform, \
    AffineTransform, AffineXYTransform, RadialXYTransform, MultiXYTransform

__all__ = ["xyTransformRegistry", "OneXYTransformConfig"]

xyTransformRegistry = makeRegistry(
    '''A registry of XYTransform factories

        An XYTransform factory is a function that obeys these rules:
        - has an attribute ConfigClass
        - takes one argument, config (an instance of ConfigClass) by name
        - returns an XYTransform
        '''
)

def makeIdentityTransform(config=None):
    """Make an IdentityXYTransform (which has no config parameters)
    """
    return IdentityXYTransform()
makeIdentityTransform.ConfigClass = Config
xyTransformRegistry.register("identity", makeIdentityTransform)


class InvertedXYTransformConfig(Config):
    transform = xyTransformRegistry.makeField("Transform to invert")
def makeInvertedTransform(config):
    """Make an InvertedXYTransform
    """
    return InvertedXYTransform(config.transform.apply())
makeInvertedTransform.ConfigClass = InvertedXYTransformConfig
xyTransformRegistry.register("inverted", makeInvertedTransform)


class AffineXYTransformConfig(Config):
    linear = ListField(
        doc = """2x2 linear matrix in the usual numpy order;
            to rotate a vector by theta use: cos(theta), sin(theta), -sin(theta), cos(theta)""",
        dtype = float,
        length = 4,
        default = (1, 0, 0, 1),
    )
    translation = ListField(
        doc = "x, y translation vector",
        dtype = float,
        length = 2,
        default = (0, 0),
    )
def makeAffineXYTransform(config):
    """Make an AffineXYTransform
    """
    linear = numpy.array(config.linear)
    linear.shape = (2,2)
    translation = numpy.array(config.translation)
    return AffineXYTransform(AffineTransform(linear, translation))
makeAffineXYTransform.ConfigClass = AffineXYTransformConfig
xyTransformRegistry.register("affine", makeAffineXYTransform)


class RadialXYTransformConfig(Config):
    coeffs = ListField(
        doc = "Coefficients for the radial polynomial; coeff[0] must be 0",
        dtype = float,
        minLength = 1,
        optional = False,
    )
    def validate(self):
        if len(self.coeffs) == 0:
            return
        if len(self.coeffs) == 1 or self.coeffs[0] != 0 or self.coeffs[1] == 0:
            raise RuntimeError(
                "invalid RadialXYTransform coeffs %s: " % (self.coeffs,) \
                + " need len(coeffs)=0 or len(coeffs)>1, coeffs[0]==0, and coeffs[1]!=0")
def makeRadialXYTransform(config):
    """Make a RadialXYTransform
    """
    return RadialXYTransform(config.coeffs)
makeRadialXYTransform.ConfigClass = RadialXYTransformConfig
xyTransformRegistry.register("radial", makeRadialXYTransform)


class OneXYTransformConfig(Config):
    tr = ConfigurableField(
        doc = "XYTransform factory",
        target = makeIdentityTransform,
    )
class MultiXYTransformConfig(Config):
    transformDict = ConfigDictField(
        doc = "Dict of index: OneXYTransformConfig (a transform wrapper); key order is transform order",
        keytype = int,
        itemtype = OneXYTransformConfig,
    )
def makeMultiTransform(config):
    """Make an MultiXYTransform
    """
    transformKeys = sorted(config.transformDict.iterkeys())
    transformList = [config.transformDict[key].tr.apply() for key in transformKeys]
    return MultiXYTransform(transformList)
makeMultiTransform.ConfigClass = MultiXYTransformConfig
xyTransformRegistry.register("multi", makeMultiTransform)


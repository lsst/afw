from __future__ import absolute_import, division
#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
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

class OneXYTransformConfig(Config):
    transform = ConfigurableField(
        doc = "XYTransform factory",
        target = makeIdentityTransform,
    )

def makeInvertedTransform(config):
    """Make an InvertedXYTransform
    """
    return InvertedXYTransform(config.transform.apply())
makeInvertedTransform.ConfigClass = OneXYTransformConfig
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
    transformList = [config.transformDict[key].transform.apply() for key in transformKeys]
    return MultiXYTransform(transformList)
makeMultiTransform.ConfigClass = MultiXYTransformConfig
xyTransformRegistry.register("multi", makeMultiTransform)


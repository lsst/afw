from __future__ import absolute_import, division, print_function

import lsst.pex.config as pexConfig
from lsst.afw.geom import TransformConfig


class TransformMapConfig(pexConfig.Config):
    transforms = pexConfig.ConfigDictField(
        doc = "Dict of coordinate system name: TransformConfig",
        keytype = str,
        itemtype = TransformConfig,
    )
    nativeSys = pexConfig.Field(
        doc = "Name of reference coordinate system",
        dtype = str,
        optional = False,
    )

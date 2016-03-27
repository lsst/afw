#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#
from __future__ import absolute_import, division
from . import cameraGeomLib

DetectorTypeValNameDict = {
    cameraGeomLib.SCIENCE:   "SCIENCE",
    cameraGeomLib.FOCUS:     "FOCUS",
    cameraGeomLib.GUIDER:    "GUIDER",
    cameraGeomLib.WAVEFRONT: "WAVEFRONT",
}
DetectorTypeNameValDict = dict((val, key) for key, val in DetectorTypeValNameDict.iteritems())

#
# LSST Data Management System
# Copyright 2016 LSST Corporation.
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

__all__ = ["DetectorBase"]

import lsst.geom
import lsst.afw.cameraGeom as afwGeom
from lsst.utils import continueClass
from .detector import DetectorBase, DetectorType


DetectorTypeValNameDict = {
    0: DetectorType.SCIENCE,
    1: DetectorType.FOCUS,
    2: DetectorType.GUIDER,
    3: DetectorType.WAVEFRONT,
}


@continueClass  # noqa: F811
class DetectorBase:
    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def fromDict(self, inputDict, translationDict=None):
        if translationDict is not None:
            for key in translationDict.keys():
                if key in inputDict:
                    alias = translationDict[key]
                    inputDict[alias] = inputDict[key]
                    #        import pdb
                    #        pdb.set_trace()
        # CZW: These need to be set when constructing the det from the camera.
        #      I.e. the camera needs to know about the new instance.
        #        self.setName(inputDict.get('name', "Undefined Detector"))
        #        self.setId(inputDict.get('id', 0))
        self.setSerial(inputDict.get('serial', "Undefined Serial"))
        self.setDetectorType(DetectorTypeValNameDict[inputDict.get('detectorType', DetectorType.SCIENCE)])
        self.setPhysicalType(inputDict.get('physicalType', "Undefined Physical Type"))

        if 'crosstalk' in inputDict:
            self.setCrosstalk(inputDict.get('crosstalk'))

        self.setPixelSize(lsst.geom.Extent2D(inputDict.get('pixelSize', (1.0, 1.0))))
        #        self.setNativeSys(afwGeom.FOCAL_PLANE)

        self.setBBox(makeBBoxFromList(inputDict.get('bbox', None)))

        # CZW: How should this be handled?
        # self.setTranspose(inputDict.get('transposeDetector', False))

        offset = lsst.geom.Point2D(inputDict.get('offset', (0, 0)))
        refPos = lsst.geom.Point2D(inputDict.get('refpos', (0, 0)))
        yaw = lsst.geom.Angle(inputDict.get('yaw', 0.0), lsst.geom.degrees)
        pitch = lsst.geom.Angle(inputDict.get('pitch', 0.0), lsst.geom.degrees)
        roll = lsst.geom.Angle(inputDict.get('roll', 0.0), lsst.geom.degrees)
        orientation = afwGeom.Orientation(offset, refPos, yaw, pitch, roll)
        self.setOrientation(orientation)

        # CZW: addTransforms.
        #        self.setTransforms(inputDict('transformDict', None))
        print(inputDict.get('transformDict'))

        if 'amplifiers' in inputDict:
            for name, amp in inputDict['amplifiers'].items():
                amp['name'] = name
                ampBuilder = afwGeom.Amplifier.Builder()
                ampBuilder.fromDict(amp)
                self.append(ampBuilder)


def makeBBoxFromList(inlist, x0=0, y0=0):
    """Given a list [(x0, y0), (xsize, ysize)], probably from a yaml file,
    return a BoxI.
    """
    if inlist is None:
        return lsst.geom.BoxI(lsst.geom.PointI(0, 0), lsst.geom.ExtentI(0, 0))
    else:
        (xs, ys), (xsize, ysize) = inlist
        box = lsst.geom.BoxI(lsst.geom.PointI(xs, ys), lsst.geom.ExtentI(xsize, ysize))
        box.shift(lsst.geom.Extent2I(x0, y0))
        return box

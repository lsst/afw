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

    def __repr__(self):
        return f"DetBuilder: {self.getName()} {self.getDetectorType()} {self.getBBox()}"

    def fromDict(self, inputDict, translationDict=None):
        if translationDict is not None:
            for key in translationDict.keys():
                if key in inputDict:
                    alias = translationDict[key]
                    inputDict[alias] = inputDict[key]

                    #        import pdb
                    #        pdb.set_trace()
        print("Detector Level Transforms here.")

        self.setSerial(inputDict.get('serial', "Undefined Serial"))
        self.setDetectorType(DetectorTypeValNameDict[inputDict.get('detectorType', DetectorType.SCIENCE)])
        self.setPhysicalType(inputDict.get('physicalType', "Undefined Physical Type"))
        if 'crosstalk' in inputDict:  # For type reasons.
            self.setCrosstalk(inputDict.get('crosstalk'))
        # self.setTranspose(inputDict.get('transposeDetector', False))

        # Box like keys may be split into 4 elements:
        if 'bbox' in inputDict.keys():
            self.setBBox(makeBBoxFromList(inputDict.get('bbox', None), corners=True))
        elif 'bbox_x0' in inputDict.keys():
            bboxTuple = ((inputDict.get("bbox_x0", None), inputDict.get("bbox_y0", None)),
                         (inputDict.get("bbox_x1", None), inputDict.get("bbox_y1", None)))
            self.setBBox(makeBBoxFromList(bboxTuple, corners=True))

        # Extent like keys may be split into 2 elements:
        if 'pixelSize' in inputDict.keys():
            self.setPixelSize(lsst.geom.Extent2D(inputDict.get('pixelSize', (1.0, 1.0))))
        elif 'pixelSize_x' in inputDict.keys():
            pixelSizeTuple = (inputDict.get('pixelSize_x', None), inputDict.get('pixelSize_y', None))
            self.setPixelSize(lsst.geom.Extent2D(pixelSizeTuple))

        if 'orientation' in inputDict.keys():
            self.setOrientation(inputDict.get('orientation', None))
        elif 'yaw' in inputDict.keys() or 'yawDeg' in inputDict.keys():
            if 'offset' in inputDict.keys():
                offset = lsst.geom.Point2D(inputDict.get('offset', (0, 0)))
            elif 'offset_x' in inputDict.keys():
                offsetTuple = (inputDict.get('offset_x', None), inputDict.get('offset_y', None))
                offset = lsst.geom.Point2D(offsetTuple)

            if 'refpos' in inputDict.keys():
                refPos = lsst.geom.Point2D(inputDict.get('refpos', (0, 0)))
            elif 'refpos_x' in inputDict.keys():
                refPosTuple = (inputDict.get('refpos_x', None), inputDict.get('refpos_y', None))
                refPos = lsst.geom.Point2D(refPosTuple)

            if 'yawDeg' in inputDict.keys():
                yaw = lsst.geom.Angle(inputDict.get('yawDeg', 0.0), lsst.geom.degrees)
            else:
                yaw = lsst.geom.Angle(inputDict.get('yaw', 0.0), lsst.geom.degrees)
            if 'pitchDeg' in inputDict.keys():
                pitch = lsst.geom.Angle(inputDict.get('pitchDeg', 0.0), lsst.geom.degrees)
            else:
                pitch = lsst.geom.Angle(inputDict.get('pitch', 0.0), lsst.geom.degrees)
            if 'rollDeg' in inputDict.keys():
                roll = lsst.geom.Angle(inputDict.get('rollDeg', 0.0), lsst.geom.degrees)
            else:
                roll = lsst.geom.Angle(inputDict.get('roll', 0.0), lsst.geom.degrees)

            orientation = afwGeom.Orientation(offset, refPos, yaw, pitch, roll)
            self.setOrientation(orientation)

        if 'amplifiers' in inputDict:
            for name, amp in inputDict['amplifiers'].items():
                amp['name'] = name
                ampBuilder = afwGeom.Amplifier.Builder()
                ampBuilder.fromDict(amp)
                self.append(ampBuilder)

    def fromConfig(self, config):
        """Convert the elements of a detectorConfig object to an inputDict.
        """
        inputDict = dict()
        keysToConvert = ['serial', 'detectorType', 'physicalType',
                         'crosstalk', 'transformDict', 'transforms',
                         'yaw', 'pitch', 'roll',
                         'yawDeg', 'pitchDeg', 'rollDeg',
                         'orientation', 'amplifiers',
                         'bbox', 'bbox_x0', 'bbox_x1', 'bbox_y0', 'bbox_y1',
                         'offset', 'offset_x', 'offset_y',
                         'refpos', 'refpos_x', 'refpos_y',
                         'pixelSize', 'pixelSize_x', 'pixelSize_y']
        for key in keysToConvert:
            value = getattr(config, key)
            if value is not None:
                inputDict[key] = value

        self.fromDict(inputDict)

    def makeTransformsToParent(self, transformDict, parent):
        """Ensure all expected transforms from this detector to the parent system exist.
        """
        resMap = dict()
        if transformDict is not None:
            assert transformDict['nativeSys'] == afwGeom.PIXELS, \
                "Detectors with nativeSys != PIXELS are not supported."

            for key in transformDict:
                transform = transformDict[key].transform.apply()
                resMap[afwGeom.CameraSys(key)] = transform

        # Implied:
        # resMap['nativeSys'] = PIXELS
        for toSys, transform in resMap.items:
            self.setTransformFromPixlesTo(toSys, transform)


def makeBBoxFromList(inlist, corners=False, x0=0, y0=0):
    """Given a list [(x0, y0), (xsize, ysize)], probably from a yaml file,
    return a BoxI.
    """
    if inlist is None:
        return lsst.geom.BoxI(lsst.geom.PointI(0, 0), lsst.geom.ExtentI(0, 0))
    elif corners is False:
        (xs, ys), (xsize, ysize) = inlist
        box = lsst.geom.BoxI(lsst.geom.PointI(xs, ys), lsst.geom.ExtentI(xsize, ysize))
        box.shift(lsst.geom.Extent2I(x0, y0))
        return box
    elif corners is True:
        (xmin, ymin), (xmax, ymax) = inlist

        box = lsst.geom.BoxI(minimum=lsst.geom.Point2I(xmin, ymin),
                             maximum=lsst.geom.Point2I(xmax, ymax))
        box.shift(lsst.geom.Extent2I(x0, y0))
        return box

#!/usr/bin/env python
# 
# LSST Data Management System
# Copyright 2014 LSST Corporation.
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
"""
Tests for lsst.afw.cameraGeom.Detector
"""
import unittest

import lsst.utils.tests
from lsst.pex.exceptions import LsstCppException
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom

class DetectorWrapper(object):
    """Construct a detector, with various errors possible
    """
    def __init__(self, numAmps=3, tryDuplicateAmpNames=False, tryBadCameraSys=False):
        self.name = "detector 1"
        self.type = cameraGeom.SCIENCE
        self.serial = "xkcd722"
        self.ampList = []
        for i in range(numAmps):
            ampName = "amp %d" % (i + 1,)
            if i == 1 and tryDuplicateAmpNames:
                ampName = self.ampList[0].getName()
            bbox = afwGeom.Box2I(afwGeom.Point2I(-1, 1), afwGeom.Extent2I(5, 6))
            gain = 1.71234e3
            readNoise = 0.521237e2
            self.ampList.append(cameraGeom.Amplifier(ampName, bbox, gain, readNoise, None))
        self.orientation = cameraGeom.Orientation(2)
        self.pixelSize = 0.02
        self.transMap = {
            cameraGeom.FOCAL_PLANE: afwGeom.RadialXYTransform([0, self.pixelSize]),
            cameraGeom.CameraSys(cameraGeom.ACTUAL_PIXELS, self.name): afwGeom.RadialXYTransform([0, 0.95, 0.01]),
        }
        if tryBadCameraSys:
            self.transMap[cameraGeom.CameraSys("foo", "wrong detector")] = afwGeom.IdentityXYTransform(False)
        self.detector = cameraGeom.Detector(
            self.name,
            self.type,
            self.serial,
            self.ampList,
            self.orientation,
            self.pixelSize,
            self.transMap,
        )


class DetectorTestCase(unittest.TestCase):
    def testBasics(self):
        """Test getters and other basics
        """
        dw = DetectorWrapper()
        detector = dw.detector
        for methodName in ("begin", "end", "size"):
            if hasattr(detector, methodName):
                self.assertFalse(hasattr(detector, methodName))
        self.assertEquals(dw.name,   detector.getName())
        self.assertEquals(dw.type,   detector.getType())
        self.assertEquals(dw.serial, detector.getSerial())
        self.assertAlmostEquals(dw.pixelSize, detector.getPixelSize())
        self.assertEquals(len(detector), len(dw.ampList))

        orientation = detector.getOrientation()

        transformRegistry = detector.getTransformRegistry()
        self.assertEquals(len(transformRegistry), len(dw.transMap) + 1) # add 1 for null transform
        for cameraSys in dw.transMap:
            self.assertTrue(cameraSys in transformRegistry)

        # make sure some complex objects stick around after detector is deleted

        detectorName = detector.getName()
        orientNQuarter = dw.orientation.getNQuarter()
        del detector
        del dw
        self.assertEquals(orientation.getNQuarter(), orientNQuarter)
        nativeCoordSys = transformRegistry.getNativeCoordSys()
        self.assertEquals(nativeCoordSys,
            cameraGeom.CameraSys(cameraGeom.PIXELS.getSysName(), detectorName))

    def testConstructorErrors(self):
        """Test constructor errors
        """
        self.assertRaises(LsstCppException, DetectorWrapper, tryDuplicateAmpNames=True)
        self.assertRaises(LsstCppException, DetectorWrapper, tryBadCameraSys=True)

    def testConvert(self):
        """Test the convert method
        """
        dw = DetectorWrapper()
        for xyMM in ((25.6, -31.07), (0, 0), (-1.234e5, 3.123e4)):
            fpPoint = afwGeom.Point2D(*xyMM)
            fpCamPoint = cameraGeom.CameraPoint(fpPoint, cameraGeom.FOCAL_PLANE)
            pixCamPoint = dw.detector.convert(fpCamPoint, cameraGeom.PIXELS)
            pixPoint = pixCamPoint.getPoint()
            for i in range(2):
                self.assertAlmostEquals(fpPoint[i] * dw.pixelSize, pixPoint[i])
            fpCamPoint2 = dw.detector.convert(pixCamPoint, cameraGeom.FOCAL_PLANE)
            fpPoint2 = fpCamPoint2.getPoint()
            for i in range(2):
                self.assertAlmostEquals(fpPoint[i], fpPoint2[i])

    def testIteration(self):
        """Test iteration over amplifiers and __getitem__
        """
        dw = DetectorWrapper()
        ampList = [amp for amp in dw.detector]
        self.assertEquals(len(ampList), len(dw.ampList))
        for i, amp in enumerate(ampList):
            self.assertEquals(amp.getName(), dw.detector[i].getName())
            self.assertEquals(amp.getName(), dw.ampList[i].getName())
            self.assertEquals(amp.getName(), dw.detector[amp.getName()].getName())

    def testMakeCameraPoint(self):
        """Test the makeCameraPoint method
        """
        dw = DetectorWrapper()
        for xyMM in ((25.6, -31.07), (0, 0)):
            point = afwGeom.Point2D(*xyMM)
            for sysName in ("csys1", "csys2"):
                for detectorName in ("", dw.name, "a different detector"):
                    cameraSys1 = cameraGeom.CameraSys(sysName, detectorName)
                    cameraPoint1 = dw.detector.makeCameraPoint(point, cameraSys1)

                    self.assertEquals(cameraPoint1.getPoint(), point)
                    self.assertEquals(cameraPoint1.getCameraSys(), cameraSys1)

                cameraSysPrefix = cameraGeom.CameraSysPrefix(sysName)
                cameraPoint2 = dw.detector.makeCameraPoint(point, cameraSysPrefix)
                predCameraSys2 = cameraGeom.CameraSys(sysName, dw.name)
                self.assertEquals(cameraPoint2.getPoint(), point)
                self.assertEquals(cameraPoint2.getCameraSys(), predCameraSys2)

    def testMakeCameraSys(self):
        """Test the makeCameraSys method
        """
        dw = DetectorWrapper()
        for sysName in ("csys1", "csys2"):
            for detectorName in ("", dw.name, "a different detector"):
                inCamSys = cameraGeom.CameraSys(sysName, detectorName)
                outCamSys = dw.detector.makeCameraSys(inCamSys)
                self.assertEquals(inCamSys, outCamSys)

            inCamSysPrefix = cameraGeom.CameraSysPrefix(sysName)
            outCamSys2 = dw.detector.makeCameraSys(inCamSysPrefix)
            self.assertEquals(outCamSys2, cameraGeom.CameraSys(sysName, dw.name))


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(DetectorTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

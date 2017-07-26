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
from __future__ import absolute_import, division, print_function
import unittest

from builtins import range
from builtins import zip

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom
from lsst.afw.cameraGeom.testUtils import DetectorWrapper


class DetectorTestCase(lsst.utils.tests.TestCase):

    def testBasics(self):
        """Test getters and other basics
        """
        dw = DetectorWrapper()
        detector = dw.detector
        for methodName in ("begin", "end", "size"):
            if hasattr(detector, methodName):
                self.assertFalse(hasattr(detector, methodName))
        self.assertEqual(dw.name, detector.getName())
        self.assertEqual(dw.id, detector.getId())
        self.assertEqual(dw.type, detector.getType())
        self.assertEqual(dw.serial, detector.getSerial())
        bbox = detector.getBBox()
        for i in range(2):
            self.assertEqual(bbox.getMin()[i], dw.bbox.getMin()[i])
            self.assertEqual(bbox.getMax()[i], dw.bbox.getMax()[i])
        self.assertAlmostEqual(dw.pixelSize, detector.getPixelSize())
        self.assertEqual(len(detector), len(dw.ampInfo))

        orientation = detector.getOrientation()

        transformMap = detector.getTransformMap()
        # add 1 for null transform
        self.assertEqual(len(transformMap), len(dw.transMap) + 1)
        for cameraSys in dw.transMap:
            self.assertTrue(cameraSys in transformMap)

        # make sure some complex objects stick around after detector is deleted

        detectorName = detector.getName()
        nativeCoordSys = detector.getNativeCoordSys()
        offset = dw.orientation.getFpPosition()
        del detector
        del dw
        self.assertEqual(orientation.getFpPosition(), offset)
        self.assertEqual(nativeCoordSys,
                         cameraGeom.CameraSys(cameraGeom.PIXELS.getSysName(), detectorName))

    def testConstructorErrors(self):
        """Test constructor errors
        """
        def duplicateAmpName(dw):
            """Set two amplifiers to the same name"""
            dw.ampInfo[1].setName(dw.ampInfo[0].getName())
        with self.assertRaises(lsst.pex.exceptions.Exception):
            DetectorWrapper(modFunc=duplicateAmpName)

        def addBadCameraSys(dw):
            """Add an invalid camera system"""
            dw.transMap[cameraGeom.CameraSys("foo", "wrong detector")] = \
                afwGeom.makeIdentityTransform()
        with self.assertRaises(lsst.pex.exceptions.Exception):
            DetectorWrapper(modFunc=addBadCameraSys)

        # These break in the pybind layer
        for crosstalk in ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],  # 1D and not numpy
                          np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),  # 1D, wrong numpy type
                          np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32),  # 1D
                          ):
            self.assertRaises(TypeError, DetectorWrapper, crosstalk=crosstalk)
        # These break in the Detector ctor: wrong shape
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError,
                          DetectorWrapper, crosstalk=np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError,
                          DetectorWrapper, crosstalk=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    def testTransform(self):
        """Test the transform method
        """
        dw = DetectorWrapper()
        pixOffset = dw.orientation.getReferencePoint()
        for xyMM in ((25.6, -31.07), (0, 0), (-1.234e5, 3.123e4)):
            fpPoint = afwGeom.Point2D(*xyMM)
            fpCamPoint = cameraGeom.CameraPoint(
                fpPoint, cameraGeom.FOCAL_PLANE)
            pixCamPoint = dw.detector.transform(fpCamPoint, cameraGeom.PIXELS)
            pixPoint = pixCamPoint.getPoint()
            for i in range(2):
                self.assertAlmostEqual(
                    fpPoint[i]/dw.pixelSize[i] + pixOffset[i], pixPoint[i])
            fpCamPoint2 = dw.detector.transform(
                pixCamPoint, cameraGeom.FOCAL_PLANE)
            fpPoint2 = fpCamPoint2.getPoint()
            for i in range(2):
                self.assertAlmostEqual(fpPoint[i], fpPoint2[i])

            # test pix to pix
            pixCamPoint2 = dw.detector.transform(
                pixCamPoint, cameraGeom.PIXELS)
            for i in range(2):
                self.assertAlmostEqual(pixCamPoint.getPoint()[
                                       i], pixCamPoint2.getPoint()[i])

        # make sure you cannot transform to a different detector
        pixCamPoint = dw.detector.makeCameraPoint(
            afwGeom.Point2D(1, 1), cameraGeom.PIXELS)
        otherCamSys = cameraGeom.CameraSys(cameraGeom.PIXELS, "other detector")
        with self.assertRaises(lsst.pex.exceptions.Exception):
            dw.detector.transform(pixCamPoint, otherCamSys)

    def testIteration(self):
        """Test iteration over amplifiers and __getitem__
        """
        dw = DetectorWrapper()
        ampList = [amp for amp in dw.detector]
        self.assertEqual(len(ampList), len(dw.ampInfo))
        for i, amp in enumerate(ampList):
            self.assertEqual(amp.getName(), dw.detector[i].getName())
            self.assertEqual(amp.getName(), dw.ampInfo[i].getName())
            self.assertEqual(
                amp.getName(), dw.detector[amp.getName()].getName())

    def testTransformAccess(self):
        """Test hasTransform and getTransform
        """
        detector = DetectorWrapper().detector
        for fromSys in (cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS, cameraGeom.TAN_PIXELS):
            fullFromSys = detector.makeCameraSys(fromSys)
            for toSys in (cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS, cameraGeom.TAN_PIXELS):
                fullToSys = detector.makeCameraSys(toSys)
                self.assertTrue(detector.hasTransform(fromSys))
                self.assertTrue(detector.hasTransform(fullFromSys))
                self.assertTrue(detector.hasTransform(toSys))
                self.assertTrue(detector.hasTransform(fullToSys))
                detector.getTransform(fromSys, toSys)
                detector.getTransform(fromSys, fullToSys)
                detector.getTransform(fullFromSys, toSys)
                detector.getTransform(fullFromSys, fullToSys)

        for badCamSys in (
            cameraGeom.CameraSys("badName"),
            cameraGeom.CameraSys("pixels", "badDetectorName")
        ):
            self.assertFalse(detector.hasTransform(badCamSys))
            self.assertTrue(detector.hasTransform(cameraGeom.PIXELS))
            with self.assertRaises(lsst.pex.exceptions.Exception):
                detector.getTransform(cameraGeom.PIXELS, badCamSys)

    def testMakeCameraPoint(self):
        """Test the makeCameraPoint method
        """
        dw = DetectorWrapper()
        for xyMM in ((25.6, -31.07), (0, 0)):
            point = afwGeom.Point2D(*xyMM)
            for sysName in ("csys1", "csys2"):
                for detectorName in ("", dw.name, "a different detector"):
                    cameraSys1 = cameraGeom.CameraSys(sysName, detectorName)
                    cameraPoint1 = dw.detector.makeCameraPoint(
                        point, cameraSys1)

                    self.assertEqual(cameraPoint1.getPoint(), point)
                    self.assertEqual(cameraPoint1.getCameraSys(), cameraSys1)

                cameraSysPrefix = cameraGeom.CameraSysPrefix(sysName)
                cameraPoint2 = dw.detector.makeCameraPoint(
                    point, cameraSysPrefix)
                predCameraSys2 = cameraGeom.CameraSys(sysName, dw.name)
                self.assertEqual(cameraPoint2.getPoint(), point)
                self.assertEqual(cameraPoint2.getCameraSys(), predCameraSys2)

    def testMakeCameraSys(self):
        """Test the makeCameraSys method
        """
        dw = DetectorWrapper()
        for sysName in ("csys1", "csys2"):
            for detectorName in ("", dw.name, "a different detector"):
                inCamSys = cameraGeom.CameraSys(sysName, detectorName)
                outCamSys = dw.detector.makeCameraSys(inCamSys)
                self.assertEqual(inCamSys, outCamSys)

            inCamSysPrefix = cameraGeom.CameraSysPrefix(sysName)
            outCamSys2 = dw.detector.makeCameraSys(inCamSysPrefix)
            self.assertEqual(
                outCamSys2, cameraGeom.CameraSys(sysName, dw.name))

    def testGetCorners(self):
        """Test the getCorners method
        """
        dw = DetectorWrapper()
        for cameraSys in (cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS):
            cornerList = dw.detector.getCorners(cameraSys)
            for fromPoint, toPoint in zip(afwGeom.Box2D(dw.bbox).getCorners(), cornerList):
                predToCameraPoint = dw.detector.transform(
                    dw.detector.makeCameraPoint(fromPoint, cameraGeom.PIXELS),
                    cameraSys,
                )
                predToPoint = predToCameraPoint.getPoint()
                self.assertEqual(
                    predToCameraPoint.getCameraSys().getSysName(),
                    cameraSys.getSysName())
                for i in range(2):
                    self.assertAlmostEqual(predToPoint[i], toPoint[i])
                    if cameraSys == cameraGeom.PIXELS:
                        self.assertAlmostEqual(fromPoint[i], toPoint[i])

    def testGetCenter(self):
        """Test the getCenter method
        """
        dw = DetectorWrapper()
        ctrPixPoint = afwGeom.Box2D(dw.detector.getBBox()).getCenter()
        ctrPixCameraPoint = dw.detector.makeCameraPoint(
            ctrPixPoint, cameraGeom.PIXELS)
        for cameraSys in (cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS):
            ctrCameraPoint = dw.detector.getCenter(cameraSys)
            self.assertEqual(
                ctrCameraPoint.getCameraSys().getSysName(),
                cameraSys.getSysName())
            ctrPoint = ctrCameraPoint.getPoint()
            predCtrCameraPoint = dw.detector.transform(
                ctrPixCameraPoint, cameraSys)
            predCtrPoint = predCtrCameraPoint.getPoint()
            for i in range(2):
                self.assertAlmostEqual(ctrPoint[i], predCtrPoint[i])
                if cameraSys == cameraGeom.PIXELS:
                    self.assertAlmostEqual(ctrPixPoint[i], ctrPoint[i])

    def testCopyDetector(self):
        """Test copyDetector() method
        """
        #
        # Make a copy without any modifications
        #
        detector = DetectorWrapper().detector
        ndetector = cameraGeom.copyDetector(detector)

        self.assertEqual(detector.getName(), ndetector.getName())
        self.assertEqual(detector.getBBox(), ndetector.getBBox())
        for amp, namp in zip(detector, ndetector):
            self.assertEqual(amp.getBBox(), namp.getBBox())
            self.assertEqual(amp.getRawXYOffset(), namp.getRawXYOffset())
        #
        # Now make a copy with a hacked-up set of amps
        #
        ampInfoCatalog = detector.getAmpInfoCatalog().copy(deep=True)
        for i, amp in enumerate(ampInfoCatalog, 1):
            amp.setRawXYOffset(i*afwGeom.ExtentI(1, 1))

        ndetector = cameraGeom.copyDetector(
            detector, ampInfoCatalog=ampInfoCatalog)

        self.assertEqual(detector.getName(), ndetector.getName())
        self.assertEqual(detector.getBBox(), ndetector.getBBox())
        for i, (amp, namp) in enumerate(zip(detector, ndetector), 1):
            self.assertEqual(amp.getBBox(), namp.getBBox())
            self.assertNotEqual(amp.getRawXYOffset(), namp.getRawXYOffset())
            self.assertEqual(namp.getRawXYOffset()[0], i)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

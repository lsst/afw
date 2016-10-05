#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import range
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2014 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#"""
#pybind11#Tests for lsst.afw.cameraGeom.Detector
#pybind11#"""
#pybind11#import itertools
#pybind11#import unittest
#pybind11#from builtins import zip
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.cameraGeom as cameraGeom
#pybind11#from lsst.afw.cameraGeom.testUtils import DetectorWrapper
#pybind11#
#pybind11#
#pybind11#class DetectorTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def testBasics(self):
#pybind11#        """Test getters and other basics
#pybind11#        """
#pybind11#        dw = DetectorWrapper()
#pybind11#        detector = dw.detector
#pybind11#        for methodName in ("begin", "end", "size"):
#pybind11#            if hasattr(detector, methodName):
#pybind11#                self.assertFalse(hasattr(detector, methodName))
#pybind11#        self.assertEquals(dw.name, detector.getName())
#pybind11#        self.assertEquals(dw.id, detector.getId())
#pybind11#        self.assertEquals(dw.type, detector.getType())
#pybind11#        self.assertEquals(dw.serial, detector.getSerial())
#pybind11#        bbox = detector.getBBox()
#pybind11#        for i in range(2):
#pybind11#            self.assertEquals(bbox.getMin()[i], dw.bbox.getMin()[i])
#pybind11#            self.assertEquals(bbox.getMax()[i], dw.bbox.getMax()[i])
#pybind11#        self.assertAlmostEquals(dw.pixelSize, detector.getPixelSize())
#pybind11#        self.assertEquals(len(detector), len(dw.ampInfo))
#pybind11#
#pybind11#        orientation = detector.getOrientation()
#pybind11#
#pybind11#        transformMap = detector.getTransformMap()
#pybind11#        self.assertEquals(len(transformMap), len(dw.transMap) + 1)  # add 1 for null transform
#pybind11#        for cameraSys in dw.transMap:
#pybind11#            self.assertTrue(cameraSys in transformMap)
#pybind11#
#pybind11#        # make sure some complex objects stick around after detector is deleted
#pybind11#
#pybind11#        detectorName = detector.getName()
#pybind11#        offset = dw.orientation.getFpPosition()
#pybind11#        del detector
#pybind11#        del dw
#pybind11#        self.assertEquals(orientation.getFpPosition(), offset)
#pybind11#        nativeCoordSys = transformMap.getNativeCoordSys()
#pybind11#        self.assertEquals(nativeCoordSys,
#pybind11#                          cameraGeom.CameraSys(cameraGeom.PIXELS.getSysName(), detectorName))
#pybind11#
#pybind11#    def testConstructorErrors(self):
#pybind11#        """Test constructor errors
#pybind11#        """
#pybind11#        def duplicateAmpName(dw):
#pybind11#            """Set two amplifiers to the same name"""
#pybind11#            dw.ampInfo[1].setName(dw.ampInfo[0].getName())
#pybind11#        with self.assertRaises(lsst.pex.exceptions.Exception):
#pybind11#            DetectorWrapper(modFunc=duplicateAmpName)
#pybind11#
#pybind11#        def addBadCameraSys(dw):
#pybind11#            """Add an invalid camera system"""
#pybind11#            dw.transMap[cameraGeom.CameraSys("foo", "wrong detector")] = afwGeom.IdentityXYTransform()
#pybind11#        with self.assertRaises(lsst.pex.exceptions.Exception):
#pybind11#            DetectorWrapper(modFunc=addBadCameraSys)
#pybind11#
#pybind11#    def testTransform(self):
#pybind11#        """Test the transform method
#pybind11#        """
#pybind11#        dw = DetectorWrapper()
#pybind11#        pixOffset = dw.orientation.getReferencePoint()
#pybind11#        for xyMM in ((25.6, -31.07), (0, 0), (-1.234e5, 3.123e4)):
#pybind11#            fpPoint = afwGeom.Point2D(*xyMM)
#pybind11#            fpCamPoint = cameraGeom.CameraPoint(fpPoint, cameraGeom.FOCAL_PLANE)
#pybind11#            pixCamPoint = dw.detector.transform(fpCamPoint, cameraGeom.PIXELS)
#pybind11#            pixPoint = pixCamPoint.getPoint()
#pybind11#            for i in range(2):
#pybind11#                self.assertAlmostEquals(fpPoint[i]/dw.pixelSize[i] + pixOffset[i], pixPoint[i])
#pybind11#            fpCamPoint2 = dw.detector.transform(pixCamPoint, cameraGeom.FOCAL_PLANE)
#pybind11#            fpPoint2 = fpCamPoint2.getPoint()
#pybind11#            for i in range(2):
#pybind11#                self.assertAlmostEquals(fpPoint[i], fpPoint2[i])
#pybind11#
#pybind11#            # test pix to pix
#pybind11#            pixCamPoint2 = dw.detector.transform(pixCamPoint, cameraGeom.PIXELS)
#pybind11#            for i in range(2):
#pybind11#                self.assertAlmostEquals(pixCamPoint.getPoint()[i], pixCamPoint2.getPoint()[i])
#pybind11#
#pybind11#        # make sure you cannot transform to a different detector
#pybind11#        pixCamPoint = dw.detector.makeCameraPoint(afwGeom.Point2D(1, 1), cameraGeom.PIXELS)
#pybind11#        otherCamSys = cameraGeom.CameraSys(cameraGeom.PIXELS, "other detector")
#pybind11#        with self.assertRaises(lsst.pex.exceptions.Exception):
#pybind11#            dw.detector.transform(pixCamPoint, otherCamSys)
#pybind11#
#pybind11#    def testIteration(self):
#pybind11#        """Test iteration over amplifiers and __getitem__
#pybind11#        """
#pybind11#        dw = DetectorWrapper()
#pybind11#        ampList = [amp for amp in dw.detector]
#pybind11#        self.assertEquals(len(ampList), len(dw.ampInfo))
#pybind11#        for i, amp in enumerate(ampList):
#pybind11#            self.assertEquals(amp.getName(), dw.detector[i].getName())
#pybind11#            self.assertEquals(amp.getName(), dw.ampInfo[i].getName())
#pybind11#            self.assertEquals(amp.getName(), dw.detector[amp.getName()].getName())
#pybind11#
#pybind11#    def testTransformAccess(self):
#pybind11#        """Test hasTransform and getTransform
#pybind11#        """
#pybind11#        detector = DetectorWrapper().detector
#pybind11#        for camSys in (cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS, cameraGeom.TAN_PIXELS):
#pybind11#            # camSys may be a CameraSys or a CameraSysPrefix
#pybind11#            fullCamSys = detector.makeCameraSys(camSys)
#pybind11#            self.assertTrue(detector.hasTransform(camSys))
#pybind11#            self.assertTrue(detector.hasTransform(fullCamSys))
#pybind11#            detector.getTransform(camSys)
#pybind11#            detector.getTransform(fullCamSys)
#pybind11#
#pybind11#        for badCamSys in (
#pybind11#            cameraGeom.CameraSys("badName"),
#pybind11#            cameraGeom.CameraSys("pixels", "badDetectorName")
#pybind11#        ):
#pybind11#            self.assertFalse(detector.hasTransform(badCamSys))
#pybind11#            with self.assertRaises(lsst.pex.exceptions.Exception):
#pybind11#                detector.getTransform(badCamSys)
#pybind11#
#pybind11#    def testMakeCameraPoint(self):
#pybind11#        """Test the makeCameraPoint method
#pybind11#        """
#pybind11#        dw = DetectorWrapper()
#pybind11#        for xyMM in ((25.6, -31.07), (0, 0)):
#pybind11#            point = afwGeom.Point2D(*xyMM)
#pybind11#            for sysName in ("csys1", "csys2"):
#pybind11#                for detectorName in ("", dw.name, "a different detector"):
#pybind11#                    cameraSys1 = cameraGeom.CameraSys(sysName, detectorName)
#pybind11#                    cameraPoint1 = dw.detector.makeCameraPoint(point, cameraSys1)
#pybind11#
#pybind11#                    self.assertEquals(cameraPoint1.getPoint(), point)
#pybind11#                    self.assertEquals(cameraPoint1.getCameraSys(), cameraSys1)
#pybind11#
#pybind11#                cameraSysPrefix = cameraGeom.CameraSysPrefix(sysName)
#pybind11#                cameraPoint2 = dw.detector.makeCameraPoint(point, cameraSysPrefix)
#pybind11#                predCameraSys2 = cameraGeom.CameraSys(sysName, dw.name)
#pybind11#                self.assertEquals(cameraPoint2.getPoint(), point)
#pybind11#                self.assertEquals(cameraPoint2.getCameraSys(), predCameraSys2)
#pybind11#
#pybind11#    def testMakeCameraSys(self):
#pybind11#        """Test the makeCameraSys method
#pybind11#        """
#pybind11#        dw = DetectorWrapper()
#pybind11#        for sysName in ("csys1", "csys2"):
#pybind11#            for detectorName in ("", dw.name, "a different detector"):
#pybind11#                inCamSys = cameraGeom.CameraSys(sysName, detectorName)
#pybind11#                outCamSys = dw.detector.makeCameraSys(inCamSys)
#pybind11#                self.assertEquals(inCamSys, outCamSys)
#pybind11#
#pybind11#            inCamSysPrefix = cameraGeom.CameraSysPrefix(sysName)
#pybind11#            outCamSys2 = dw.detector.makeCameraSys(inCamSysPrefix)
#pybind11#            self.assertEquals(outCamSys2, cameraGeom.CameraSys(sysName, dw.name))
#pybind11#
#pybind11#    def testGetCorners(self):
#pybind11#        """Test the getCorners method
#pybind11#        """
#pybind11#        dw = DetectorWrapper()
#pybind11#        for cameraSys in (cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS):
#pybind11#            cornerList = dw.detector.getCorners(cameraSys)
#pybind11#            for fromPoint, toPoint in zip(afwGeom.Box2D(dw.bbox).getCorners(), cornerList):
#pybind11#                predToCameraPoint = dw.detector.transform(
#pybind11#                    dw.detector.makeCameraPoint(fromPoint, cameraGeom.PIXELS),
#pybind11#                    cameraSys,
#pybind11#                )
#pybind11#                predToPoint = predToCameraPoint.getPoint()
#pybind11#                self.assertEquals(predToCameraPoint.getCameraSys().getSysName(), cameraSys.getSysName())
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEquals(predToPoint[i], toPoint[i])
#pybind11#                    if cameraSys == cameraGeom.PIXELS:
#pybind11#                        self.assertAlmostEquals(fromPoint[i], toPoint[i])
#pybind11#
#pybind11#    def testGetCenter(self):
#pybind11#        """Test the getCenter method
#pybind11#        """
#pybind11#        dw = DetectorWrapper()
#pybind11#        ctrPixPoint = afwGeom.Box2D(dw.detector.getBBox()).getCenter()
#pybind11#        ctrPixCameraPoint = dw.detector.makeCameraPoint(ctrPixPoint, cameraGeom.PIXELS)
#pybind11#        for cameraSys in (cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS):
#pybind11#            ctrCameraPoint = dw.detector.getCenter(cameraSys)
#pybind11#            self.assertEquals(ctrCameraPoint.getCameraSys().getSysName(), cameraSys.getSysName())
#pybind11#            ctrPoint = ctrCameraPoint.getPoint()
#pybind11#            predCtrCameraPoint = dw.detector.transform(ctrPixCameraPoint, cameraSys)
#pybind11#            predCtrPoint = predCtrCameraPoint.getPoint()
#pybind11#            for i in range(2):
#pybind11#                self.assertAlmostEquals(ctrPoint[i], predCtrPoint[i])
#pybind11#                if cameraSys == cameraGeom.PIXELS:
#pybind11#                    self.assertAlmostEquals(ctrPixPoint[i], ctrPoint[i])
#pybind11#
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

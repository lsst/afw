#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import next
#pybind11#from builtins import zip
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
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
#pybind11#
#pybind11#import unittest
#pybind11#import numpy
#pybind11#import os
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#
#pybind11#from lsst.afw.cameraGeom import PIXELS, PUPIL, FOCAL_PLANE, CameraSys, CameraSysPrefix, \
#pybind11#    CameraPoint, Camera, Detector, assembleAmplifierImage, assembleAmplifierRawImage
#pybind11#import lsst.afw.cameraGeom.testUtils as testUtils
#pybind11#import lsst.afw.cameraGeom.utils as cameraGeomUtils
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11#
#pybind11#testPath = os.path.abspath(os.path.dirname(__file__))
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class CameraGeomTestCase(unittest.TestCase):
#pybind11#    """A test case for camera geometry"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.lsstCamWrapper = testUtils.CameraWrapper(isLsstLike=True)
#pybind11#        self.scCamWrapper = testUtils.CameraWrapper(isLsstLike=False)
#pybind11#        self.cameraList = (self.lsstCamWrapper, self.scCamWrapper)
#pybind11#        self.assemblyList = {}
#pybind11#        self.assemblyList[self.lsstCamWrapper.camera.getName()] =\
#pybind11#            [afwImage.ImageU(os.path.join(testPath, 'test_amp.fits.gz')) for i in range(8)]
#pybind11#        self.assemblyList[self.scCamWrapper.camera.getName()] =\
#pybind11#            [afwImage.ImageU(os.path.join(testPath, 'test.fits.gz'))]
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.lsstCamWrapper
#pybind11#        del self.scCamWrapper
#pybind11#        del self.cameraList
#pybind11#        del self.assemblyList
#pybind11#
#pybind11#    def testConstructor(self):
#pybind11#        for cw in self.cameraList:
#pybind11#            self.assertIsInstance(cw.camera, Camera)
#pybind11#            self.assertEqual(cw.nDetectors, len(cw.camera))
#pybind11#            self.assertEqual(cw.nDetectors, len(cw.ampInfoDict))
#pybind11#            self.assertEqual(sorted(cw.detectorNameList), sorted(cw.camera.getNameIter()))
#pybind11#            self.assertEqual(sorted(cw.detectorIdList), sorted(cw.camera.getIdIter()))
#pybind11#            for det in cw.camera:
#pybind11#                self.assertIsInstance(det, Detector)
#pybind11#                self.assertEqual(cw.ampInfoDict[det.getName()]['namps'], len(det))
#pybind11#
#pybind11#    def testMakeCameraPoint(self):
#pybind11#        point = afwGeom.Point2D(0, 0)
#pybind11#        for cw in self.cameraList:
#pybind11#            camera = cw.camera
#pybind11#            for coordSys in (PUPIL, FOCAL_PLANE):
#pybind11#                pt1 = afwGeom.Point2D(0.1, 0.3)
#pybind11#                pt2 = afwGeom.Point2D(0., 0.)
#pybind11#                pt3 = afwGeom.Point2D(-0.2, 0.2)
#pybind11#                pt4 = afwGeom.Point2D(0.02, -0.2)
#pybind11#                pt5 = afwGeom.Point2D(-0.2, -0.03)
#pybind11#                for pt in (pt1, pt2, pt3, pt4, pt5):
#pybind11#                    cp = camera.makeCameraPoint(pt, coordSys)
#pybind11#                    self.assertEquals(cp.getPoint(), pt)
#pybind11#                    self.assertEquals(cp.getCameraSys().getSysName(), coordSys.getSysName())
#pybind11#
#pybind11#                    # test == and !=
#pybind11#                    cp2 = camera.makeCameraPoint(pt, coordSys)
#pybind11#                    self.assertTrue(cp == cp2)
#pybind11#                    self.assertFalse(cp != cp2)
#pybind11#
#pybind11#            det = camera[next(camera.getNameIter())]
#pybind11#            cp = camera.makeCameraPoint(point, FOCAL_PLANE)
#pybind11#            self.checkCamPoint(cp, point, FOCAL_PLANE)
#pybind11#            cp = camera.makeCameraPoint(point, det.makeCameraSys(PIXELS))
#pybind11#            self.checkCamPoint(cp, point, det.makeCameraSys(PIXELS))
#pybind11#            # non-existant camera sys in makeCameraPoint
#pybind11#            self.assertRaises(RuntimeError, camera.makeCameraPoint, point, CameraSys('abcd'))
#pybind11#            # CameraSysPrefix camera sys in makeCameraPoint
#pybind11#            self.assertRaises(TypeError, camera.makeCameraPoint, point, PIXELS)
#pybind11#
#pybind11#    def testCameraSysRepr(self):
#pybind11#        """Test CameraSys.__repr__ and CameraSysPrefix.__repr__
#pybind11#        """
#pybind11#        for sysName in ("FocalPlane", "Pupil", "Pixels", "foo"):
#pybind11#            cameraSys = CameraSys(sysName)
#pybind11#            predRepr = "CameraSys(%s)" % (sysName)
#pybind11#            self.assertEqual(repr(cameraSys), predRepr)
#pybind11#
#pybind11#            cameraSysPrefix = CameraSysPrefix(sysName)
#pybind11#            predCSPRepr = "CameraSysPrefix(%s)" % (sysName)
#pybind11#            self.assertEqual(repr(cameraSysPrefix), predCSPRepr)
#pybind11#            for detectorName in ("Detector 1", "bar"):
#pybind11#                cameraSys2 = CameraSys(sysName, detectorName)
#pybind11#                predRepr2 = "CameraSys(%s, %s)" % (sysName, detectorName)
#pybind11#                self.assertEqual(repr(cameraSys2), predRepr2)
#pybind11#
#pybind11#    def testCameraPointRepr(self):
#pybind11#        """Test CameraPoint.__repr__
#pybind11#        """
#pybind11#        point = afwGeom.Point2D(1.5, -23.4)
#pybind11#        cameraSys = FOCAL_PLANE
#pybind11#        cameraPoint = CameraPoint(point, cameraSys)
#pybind11#        predRepr = "CameraPoint(%s, %s)" % (point, cameraSys)
#pybind11#        self.assertEqual(repr(cameraPoint), predRepr)
#pybind11#
#pybind11#    def testAccessor(self):
#pybind11#        for cw in self.cameraList:
#pybind11#            camera = cw.camera
#pybind11#            for name in cw.detectorNameList:
#pybind11#                self.assertIsInstance(camera[name], Detector)
#pybind11#            for detId in cw.detectorIdList:
#pybind11#                self.assertIsInstance(camera[detId], Detector)
#pybind11#
#pybind11#    def testTransformSlalib(self):
#pybind11#        """Test Camera.transform against data computed using SLALIB
#pybind11#
#pybind11#        These test data come from SLALIB using SLA_PCD with 0.925 and
#pybind11#        a plate scale of 20 arcsec/mm
#pybind11#        """
#pybind11#        testData = [(-1.84000000, 1.04000000, -331.61689069, 187.43563387),
#pybind11#                    (-1.64000000, 0.12000000, -295.42491556, 21.61645724),
#pybind11#                    (-1.44000000, -0.80000000, -259.39818797, -144.11010443),
#pybind11#                    (-1.24000000, -1.72000000, -223.48275934, -309.99221457),
#pybind11#                    (-1.08000000, 1.36000000, -194.56520533, 245.00803635),
#pybind11#                    (-0.88000000, 0.44000000, -158.44320430, 79.22160215),
#pybind11#                    (-0.68000000, -0.48000000, -122.42389383, -86.41686623),
#pybind11#                    (-0.48000000, -1.40000000, -86.45332534, -252.15553224),
#pybind11#                    (-0.32000000, 1.68000000, -57.64746955, 302.64921514),
#pybind11#                    (-0.12000000, 0.76000000, -21.60360306, 136.82281940),
#pybind11#                    (0.08000000, -0.16000000, 14.40012984, -28.80025968),
#pybind11#                    (0.28000000, -1.08000000, 50.41767773, -194.46818554),
#pybind11#                    (0.48000000, -2.00000000, 86.50298919, -360.42912163),
#pybind11#                    (0.64000000, 1.08000000, 115.25115701, 194.48632746),
#pybind11#                    (0.84000000, 0.16000000, 151.23115189, 28.80593369),
#pybind11#                    (1.04000000, -0.76000000, 187.28751874, -136.86395600),
#pybind11#                    (1.24000000, -1.68000000, 223.47420612, -302.77150507),
#pybind11#                    (1.40000000, 1.40000000, 252.27834478, 252.27834478),
#pybind11#                    (1.60000000, 0.48000000, 288.22644118, 86.46793236),
#pybind11#                    (1.80000000, -0.44000000, 324.31346653, -79.27662515), ]
#pybind11#
#pybind11#        for cw in self.cameraList:
#pybind11#            camera = cw.camera
#pybind11#            for point in testData:
#pybind11#                fpGivenPos = afwGeom.Point2D(point[2], point[3])
#pybind11#                fpGivenCP = camera.makeCameraPoint(fpGivenPos, FOCAL_PLANE)
#pybind11#                pupilGivenPos = afwGeom.Point2D(afwGeom.degToRad(point[0]), afwGeom.degToRad(point[1]))
#pybind11#                pupilGivenCP = camera.makeCameraPoint(pupilGivenPos, PUPIL)
#pybind11#
#pybind11#                fpComputedCP = camera.transform(pupilGivenCP, FOCAL_PLANE)
#pybind11#                self.assertCamPointAlmostEquals(fpComputedCP, fpGivenCP)
#pybind11#
#pybind11#                pupilComputedCP = camera.transform(fpGivenCP, PUPIL)
#pybind11#                self.assertCamPointAlmostEquals(pupilComputedCP, pupilGivenCP)
#pybind11#
#pybind11#    def testTransformDet(self):
#pybind11#        """Test Camera.transform with detector-based coordinate systems (PIXELS)
#pybind11#        """
#pybind11#        for cw in self.cameraList:
#pybind11#            numOffUsable = 0  # number of points off one detector but on another
#pybind11#            camera = cw.camera
#pybind11#            detNameList = list(camera.getNameIter())
#pybind11#            for detName in detNameList:
#pybind11#                det = camera[detName]
#pybind11#
#pybind11#                # test transforms using a point on the detector
#pybind11#                pixCP = det.makeCameraPoint(afwGeom.Point2D(10, 10), PIXELS)
#pybind11#                fpCP = camera.transform(pixCP, FOCAL_PLANE)
#pybind11#                pupilCP = camera.transform(pixCP, PUPIL)
#pybind11#
#pybind11#                pupilCP2 = camera.transform(fpCP, PUPIL)
#pybind11#                self.assertCamPointAlmostEquals(pupilCP, pupilCP2)
#pybind11#
#pybind11#                for intermedCP in (pixCP, fpCP, pupilCP):
#pybind11#                    pixRoundTripCP = camera.transform(intermedCP, det.makeCameraSys(PIXELS))
#pybind11#                    self.assertCamPointAlmostEquals(pixCP, pixRoundTripCP)
#pybind11#
#pybind11#                    pixFindRoundTripCP = camera.transform(intermedCP, PIXELS)
#pybind11#                    self.assertCamPointAlmostEquals(pixCP, pixFindRoundTripCP)
#pybind11#
#pybind11#                # test transforms using a point off the detector
#pybind11#                pixOffDetCP = det.makeCameraPoint(afwGeom.Point2D(0, -10), PIXELS)
#pybind11#                pixOffDetRoundTripCP = camera.transform(pixOffDetCP, det.makeCameraSys(PIXELS))
#pybind11#                self.assertCamPointAlmostEquals(pixOffDetCP, pixOffDetRoundTripCP)
#pybind11#
#pybind11#                # the point off the detector MAY be on another detector
#pybind11#                # (depending if the detector has neighbor on the correct edge)
#pybind11#                detList = camera.findDetectors(pixOffDetCP)
#pybind11#                if len(detList) == 1:
#pybind11#                    numOffUsable += 1
#pybind11#                    pixFindOffCP = camera.transform(pixOffDetCP, PIXELS)
#pybind11#                    self.assertNotEqual(pixCP.getCameraSys(), pixFindOffCP.getCameraSys())
#pybind11#
#pybind11#                    # convert point on other detector to pixels on the main detector
#pybind11#                    # the result should not be on the main detector
#pybind11#                    pixToPixCP = camera.transform(pixFindOffCP, det.makeCameraSys(PIXELS))
#pybind11#                    self.assertFalse(afwGeom.Box2D(det.getBBox()).contains(pixToPixCP.getPoint()))
#pybind11#            self.assertGreater(numOffUsable, 0)
#pybind11#            print("numOffUsable=", numOffUsable)
#pybind11#
#pybind11#    def testFindDetectors(self):
#pybind11#        for cw in self.cameraList:
#pybind11#            detPointsList = []
#pybind11#            for det in cw.camera:
#pybind11#                # This currently assumes there is only one detector at the center
#pybind11#                # position of any detector.  That is not enforced and multiple detectors
#pybind11#                # at a given PUPIL position is supported.  Change this if the default
#pybind11#                # camera changes.
#pybind11#                #cp = cw.camera.makeCameraPoint(det.getCenter(), PUPIL)
#pybind11#                cp = det.getCenter(FOCAL_PLANE)
#pybind11#                detPointsList.append(cp.getPoint())
#pybind11#                detList = cw.camera.findDetectors(cp)
#pybind11#                self.assertEquals(len(detList), 1)
#pybind11#                self.assertEquals(det.getName(), detList[0].getName())
#pybind11#            detList = cw.camera.findDetectorsList(detPointsList, FOCAL_PLANE)
#pybind11#            self.assertEquals(len(cw.camera), len(detList))
#pybind11#            for dets in detList:
#pybind11#                self.assertEquals(len(dets), 1)
#pybind11#
#pybind11#    def testFpBbox(self):
#pybind11#        for cw in self.cameraList:
#pybind11#            camera = cw.camera
#pybind11#            bbox = afwGeom.Box2D()
#pybind11#            for name in cw.detectorNameList:
#pybind11#                for corner in camera[name].getCorners(FOCAL_PLANE):
#pybind11#                    bbox.include(corner)
#pybind11#            self.assertEqual(bbox.getMin(), camera.getFpBBox().getMin())
#pybind11#            self.assertEqual(bbox.getMax(), camera.getFpBBox().getMax())
#pybind11#
#pybind11#    def testLinearity(self):
#pybind11#        """Test if we can set/get Linearity parameters"""
#pybind11#        for cw in self.cameraList:
#pybind11#            camera = cw.camera
#pybind11#            for det in camera:
#pybind11#                for amp in det:
#pybind11#                    self.assertEquals(cw.ampInfoDict[det.getName()]['linInfo'][amp.getName()]['linthresh'],
#pybind11#                                      amp.get('linearityThreshold'))
#pybind11#                    self.assertEquals(cw.ampInfoDict[det.getName()]['linInfo'][amp.getName()]['linmax'],
#pybind11#                                      amp.get('linearityMaximum'))
#pybind11#                    self.assertEquals(cw.ampInfoDict[det.getName()]['linInfo'][amp.getName()]['linunits'],
#pybind11#                                      amp.get('linearityUnits'))
#pybind11#                    self.assertEquals(cw.ampInfoDict[det.getName()]['linInfo'][amp.getName()]['lintype'],
#pybind11#                                      amp.getLinearityType())
#pybind11#                    for c1, c2 in zip(cw.ampInfoDict[det.getName()]['linInfo'][amp.getName()]['lincoeffs'],
#pybind11#                                      amp.getLinearityCoeffs()):
#pybind11#                        if numpy.isfinite(c1) and numpy.isfinite(c2):
#pybind11#                            self.assertEquals(c1, c2)
#pybind11#
#pybind11#    def testAssembly(self):
#pybind11#        ccdNames = ('R:0,0 S:1,0', 'R:0,0 S:0,1')
#pybind11#        compMap = {True: afwImage.ImageU(os.path.join(testPath, 'test_comp_trimmed.fits.gz')),
#pybind11#                   False: afwImage.ImageU(os.path.join(testPath, 'test_comp.fits.gz'))}
#pybind11#        for cw in self.cameraList:
#pybind11#            camera = cw.camera
#pybind11#            imList = self.assemblyList[camera.getName()]
#pybind11#            for ccdName in ccdNames:
#pybind11#                for trim, assemble in ((False, assembleAmplifierRawImage), (True, assembleAmplifierImage)):
#pybind11#                    det = camera[ccdName]
#pybind11#                    if not trim:
#pybind11#                        outBbox = cameraGeomUtils.calcRawCcdBBox(det)
#pybind11#                    else:
#pybind11#                        outBbox = det.getBBox()
#pybind11#                    outImage = afwImage.ImageU(outBbox)
#pybind11#                    if len(imList) == 1:
#pybind11#                        for amp in det:
#pybind11#                            assemble(outImage, imList[0], amp)
#pybind11#                    else:
#pybind11#                        for amp, im in zip(det, imList):
#pybind11#                            assemble(outImage, im, amp)
#pybind11#                    self.assertListEqual(outImage.getArray().flatten().tolist(),
#pybind11#                                         compMap[trim].getArray().flatten().tolist())
#pybind11#
#pybind11#    @unittest.skipIf(not display, "display variable not set; skipping cameraGeomUtils test")
#pybind11#    def testCameraGeomUtils(self):
#pybind11#        for cw in self.cameraList:
#pybind11#            camera = cw.camera
#pybind11#            cameraGeomUtils.showCamera(camera, referenceDetectorName=camera[0].getName())
#pybind11#            ds9.incrDefaultFrame()
#pybind11#            for det in (camera[0], camera[1]):
#pybind11#                cameraGeomUtils.showCcd(det, isTrimmed=True, inCameraCoords=False)
#pybind11#                ds9.incrDefaultFrame()
#pybind11#                cameraGeomUtils.showCcd(det, isTrimmed=True, inCameraCoords=True)
#pybind11#                ds9.incrDefaultFrame()
#pybind11#                cameraGeomUtils.showCcd(det, isTrimmed=False, inCameraCoords=False)
#pybind11#                ds9.incrDefaultFrame()
#pybind11#                cameraGeomUtils.showCcd(det, isTrimmed=False, inCameraCoords=True)
#pybind11#                ds9.incrDefaultFrame()
#pybind11#                for amp in det:
#pybind11#                    cameraGeomUtils.showAmp(amp)
#pybind11#                    ds9.incrDefaultFrame()
#pybind11#
#pybind11#    def testCameraRaises(self):
#pybind11#        for cw in self.cameraList:
#pybind11#            camera = cw.camera
#pybind11#            cp = camera.makeCameraPoint(afwGeom.Point2D(1e6, 1e6), FOCAL_PLANE)
#pybind11#            # Way off the focal plane
#pybind11#            self.assertRaises(RuntimeError, camera.transform, cp, PIXELS)
#pybind11#            # non-existant destination camera system
#pybind11#            cp = camera.makeCameraPoint(afwGeom.Point2D(0, 0), FOCAL_PLANE)
#pybind11#            self.assertRaises(RuntimeError, camera.transform, cp, CameraSys('abcd'))
#pybind11#
#pybind11#    def checkCamPoint(self, cp, testPt, testSys):
#pybind11#        """Assert that a CameraPoint contains the specified Point2D and CameraSys"""
#pybind11#        self.assertEquals(cp.getCameraSys(), testSys)
#pybind11#        self.assertEquals(cp.getPoint(), testPt)
#pybind11#
#pybind11#    def assertCamPointAlmostEquals(self, cp1, cp2, ndig=6):
#pybind11#        """Assert that two CameraPoints are nearly equal
#pybind11#        """
#pybind11#        self.assertEquals(cp1.getCameraSys(), cp2.getCameraSys())
#pybind11#        for i in range(2):
#pybind11#            self.assertAlmostEquals(cp1.getPoint()[i], cp2.getPoint()[i], 6)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

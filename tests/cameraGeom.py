#!/usr/bin/env python
"""
Tests for SpatialCell

Run with:
   python SpatialCell.py
or
   python
   >>> import SpatialCell; SpatialCell.run()
"""

import math
import os
import sys
import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.pex.policy as pexPolicy
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.display.ds9 as ds9

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.cameraGeom.utils as cameraGeomUtils

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def trimCcd(ccd, ccdImage=""):
    """Trim a Ccd and maybe the image of the untrimmed Ccd"""
    
    if ccdImage == "":
        ccdImage = cameraGeomUtils.makeImageFromCcd(ccd)

    if ccd.isTrimmed():
        return ccdImage

    ccd.setTrimmed(True)

    if ccdImage is not None:
        trimmedImage = ccdImage.Factory(ccd.getAllPixels().getDimensions())
        for a in ccd:
            data =      ccdImage.Factory(ccdImage, a.getDataSec(False))
            tdata = trimmedImage.Factory(trimmedImage, a.getDataSec())
            tdata <<= data
    else:
        trimmedImage = None

    return trimmedImage

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class CameraGeomTestCase(unittest.TestCase):
    """A test case for camera geometry"""

    def assertEqualPoint(self, lhs, rhs):
        """Workaround the Point2D logical operator returning a Point not a bool"""
        if (lhs != rhs)[0] or (lhs != rhs)[1]:
            self.assertTrue(False, "%s != %s" % (lhs, rhs))

    def setUp(self):
        CameraGeomTestCase.ampSerial = [0] # an array so we pass the value by reference

        policyFile = pexPolicy.DefaultPolicyFile("afw", "CameraGeomDictionary.paf", "policy")
        defPolicy = pexPolicy.Policy.createPolicy(policyFile, policyFile.getRepositoryPath(), True)

        polFile = pexPolicy.DefaultPolicyFile("afw", "TestCameraGeom.paf", "tests")
        self.geomPolicy = pexPolicy.Policy.createPolicy(polFile)
        try:
            self.geomPolicy.mergeDefaults(defPolicy.getDictionary())
        except Exception, e:
            global warnedAboutDict
            try:
                type(warnedAboutDict)
            except:
                print >> sys.stderr, "Not validating dict due to old pexPolicy:", e
                warnedAboutDict = True

    def tearDown(self):
        pass

    def testDictionary(self):
        """Test the camera geometry dictionary"""

        if False:
            for r in self.geomPolicy.getArray("Raft"):
                print "raft", r
            for c in self.geomPolicy.getArray("Ccd"):
                print "ccd", c
            for a in self.geomPolicy.getArray("Amp"):
                print "amp", a

    def testId(self):
        """Test cameraGeom.Id"""

        self.assertTrue(cameraGeom.Id(1) == cameraGeom.Id(1))
        self.assertFalse(cameraGeom.Id(1) == cameraGeom.Id(100))
        
        self.assertTrue(cameraGeom.Id("AA") == cameraGeom.Id("AA"))
        self.assertFalse(cameraGeom.Id("AA") == cameraGeom.Id("BB"))
        
        self.assertTrue(cameraGeom.Id(1, "AA") == cameraGeom.Id(1, "AA"))
        self.assertFalse(cameraGeom.Id(1, "AA") == cameraGeom.Id(2, "AA"))
        self.assertFalse(cameraGeom.Id(1, "AA") == cameraGeom.Id(1, "BB"))

    def testCcd(self):
        """Test if we can build a Ccd out of Amps"""

        #print >> sys.stderr, "Skipping testCcd"; return

        ccdId = cameraGeom.Id("CCD")
        ccdInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        ccd = cameraGeomUtils.makeCcd(self.geomPolicy, ccdId, ccdInfo=ccdInfo)
        if display:
            cameraGeomUtils.showCcd(ccd)
            ds9.incrDefaultFrame()

        for i in range(2):
            self.assertEqual(ccd.getSize()[i], ccdInfo["pixelSize"]*ccd.getAllPixels(True).getDimensions()[i])

        self.assertEqual(ccd.getId().getName(), ccdInfo["name"])
        self.assertEqual(ccd.getAllPixels().getWidth(), ccdInfo["width"])
        self.assertEqual(ccd.getAllPixels().getHeight(), ccdInfo["height"])
        self.assertEqual([a.getId().getSerial() for a in ccd],
                         range(ccdInfo["ampIdMin"], ccdInfo["ampIdMax"] + 1))

        id = cameraGeom.Id("ID%d" % ccdInfo["ampIdMax"])
        self.assertTrue(ccd.findAmp(id), id)

        self.assertEqual(ccd.findAmp(afwGeom.Point2I.makeXY(10, 10)).getId().getSerial(), ccdInfo["ampIdMin"])

        self.assertEqual(ccd.getAllPixels().getLLC(),
                         ccd.findAmp(afwGeom.Point2I.makeXY(10, 10)).getAllPixels().getLLC())

        self.assertEqual(ccd.getAllPixels().getURC(),
                         ccd.findAmp(afwGeom.Point2I.makeXY(ccdInfo["width"] - 1,
                                                            ccdInfo["height"] - 1)).getAllPixels().getURC())
        #
        # Test mapping pixel <--> mm
        #
        pix = afwGeom.Point2I.makeXY(100, 204) # wrt bottom left
        pos = afwGeom.Point2D.makeXY(0.0, 1.02) # wrt CCD center
        #
        # Map pix into untrimmed coordinates
        #
        amp = ccd.findAmp(pix)
        corr = amp.getDataSec(False).getLLC() - amp.getDataSec(True).getLLC()
        corr = afwGeom.Extent2I(afwGeom.Point2I.makeXY(corr[0], corr[1]))
        pix += corr
        
        self.assertEqualPoint(ccd.getPixelFromPosition(pos) + corr, pix)
        self.assertEqualPoint(ccd.getPositionFromPixel(pix), pos)
        #
        # Trim the CCD and try again
        #
        trimmedImage = trimCcd(ccd)

        if display:
            ds9.mtv(trimmedImage, title='Trimmed')
            cameraGeomUtils.showCcd(ccd, trimmedImage)
            ds9.incrDefaultFrame()

        a = ccd.findAmp(cameraGeom.Id("ID%d" % ccdInfo["ampIdMin"]))
        self.assertEqual(a.getDataSec(), afwImage.BBox(afwImage.PointI(0, 0),
                                                       ccdInfo["ampWidth"], ccdInfo["ampHeight"]))

        self.assertEqual(ccd.getSize()[0], ccdInfo["pixelSize"]*ccdInfo["trimmedWidth"])
        self.assertEqual(ccd.getSize()[1], ccdInfo["pixelSize"]*ccdInfo["trimmedHeight"])
        #
        # Test mapping pixel <--> mm
        #
        pix = afwGeom.Point2I.makeXY(100, 204) # wrt LLC
        pos = afwGeom.Point2D.makeXY(0.0, 1.02) # wrt chip centre
        
        self.assertEqualPoint(ccd.getPixelFromPosition(pos), pix)
        self.assertEqualPoint(ccd.getPositionFromPixel(pix), pos)

    def testRotatedCcd(self):
        """Test if we can build a Ccd out of Amps"""

        #print >> sys.stderr, "Skipping testRotatedCcd"; return

        ccdId = cameraGeom.Id("Rotated CCD")
        ccdInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        ccd = cameraGeomUtils.makeCcd(self.geomPolicy, ccdId, ccdInfo=ccdInfo)
        ccd.setOrientation(cameraGeom.Orientation(1, 0.0, 0.0, 0.0))
        if display:
            cameraGeomUtils.showCcd(ccd)
            ds9.incrDefaultFrame()
        #
        # Trim the CCD and try again
        #
        trimmedImage = trimCcd(ccd)

        if display:
            ds9.mtv(trimmedImage, title='Rotated trimmed')
            cameraGeomUtils.showCcd(ccd, trimmedImage)
            ds9.incrDefaultFrame()

    def testRaft(self):
        """Test if we can build a Raft out of Ccds"""

        #print >> sys.stderr, "Skipping testRaft"; return
        raftId = cameraGeom.Id("Raft")
        raftInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        raft = cameraGeomUtils.makeRaft(self.geomPolicy, raftId, raftInfo=raftInfo)

        if display:
            cameraGeomUtils.showRaft(raft)
            ds9.incrDefaultFrame()

        if False:
            print cameraGeomUtils.describeRaft(raft)

        self.assertEqual(raft.getAllPixels().getWidth(), raftInfo["width"])
        self.assertEqual(raft.getAllPixels().getHeight(), raftInfo["height"])

        for x, y, serial, cen in [(  0,   0,  5, (-1.01, -2.02)),
                                  (150, 250, 21, (-1.01,  0.0 )),
                                  (250, 250, 29, ( 1.01,  0.0 )),
                                  (300, 500, 42, ( 1.01,  2.02))]:
            det = raft.findDetector(afwGeom.Point2I.makeXY(x, y))
            ccd = cameraGeom.cast_Ccd(det)
            if False:
                print x, y, det.getId().getName(), \
                      ccd.findAmp(afwGeom.Point2I.makeXY(150, 152), True).getId().getSerial()
            self.assertEqual(ccd.findAmp(afwGeom.Point2I.makeXY(150, 152), True).getId().getSerial(), serial)
            for i in range(2):
                self.assertAlmostEqual(ccd.getCenter()[i], cen[i])

        name = "C:0,2"
        self.assertEqual(raft.findDetector(cameraGeom.Id(name)).getId().getName(), name)

        self.assertEqual(raft.getSize()[0], raftInfo["widthMm"])
        self.assertEqual(raft.getSize()[1], raftInfo["heightMm"])
        #
        # Test mapping pixel <--> mm
        #
        for ix, iy, x, y in [(102, 500, -1.01,  2.02),
                             (306, 100,  1.01, -2.02),
                             (306, 500,  1.01,  2.02),
                             (356, 525,  1.51,  2.27),
                             ]:
            pix = afwGeom.Point2I.makeXY(ix, iy) # wrt raft LLC
            pos = afwGeom.Point2D.makeXY(x, y) # wrt raft center

            self.assertEqualPoint(raft.getPixelFromPosition(pos), pix)
            self.assertEqualPoint(raft.getPositionFromPixel(pix), pos)
        
    def testCamera(self):
        """Test if we can build a Camera out of Rafts"""

        #print >> sys.stderr, "Skipping testCamera"; return

        cameraInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        camera = cameraGeomUtils.makeCamera(self.geomPolicy, cameraInfo=cameraInfo)

        if display:
            cameraGeomUtils.showCamera(camera, )
            ds9.incrDefaultFrame()

        if False:
            print cameraGeomUtils.describeCamera(camera)

        self.assertEqual(camera.getAllPixels().getWidth(), cameraInfo["width"])
        self.assertEqual(camera.getAllPixels().getHeight(), cameraInfo["height"])

        for rx, ry, cx, cy, serial, cen in [(0, 0,     0,   0,   4,  (-3.12, -2.02)),
                                            (0,   0,   150, 250, 20, (-3.12,  0.00)),
                                            (600, 300, 0,   0,   52, ( 1.1,  -2.02)),
                                            (600, 300, 150, 250, 68, ( 1.1,  0.00)),
                                            ]:
            raft = cameraGeom.cast_Raft(camera.findDetector(afwGeom.Point2I.makeXY(rx, ry)))

            ccd = cameraGeom.cast_Ccd(raft.findDetector(afwGeom.Point2I.makeXY(cx, cy)))
            self.assertEqual(ccd.findAmp(afwGeom.Point2I.makeXY(153, 152), True).getId().getSerial(), serial)
            for i in range(2):
                self.assertAlmostEqual(ccd.getCenter()[i], cen[i])

        name = "R:1,0"
        self.assertEqual(camera.findDetector(cameraGeom.Id(name)).getId().getName(), name)

        self.assertEqual(camera.getSize()[0], cameraInfo["widthMm"])
        self.assertEqual(camera.getSize()[1], cameraInfo["heightMm"])
        #
        # Test mapping pixel <--> mm
        #
        for ix, iy, x, y in [(102, 500, -3.12, 2.02),
                             (152, 525, -2.62, 2.27),
                             (714, 500,  3.12, 2.02),
                             ]:
            pix = afwGeom.Point2I.makeXY(ix, iy) # wrt raft LLC
            pos = afwGeom.Point2D.makeXY(x, y) # wrt raft center
            
            self.assertEqualPoint(camera.getPixelFromPosition(pos), pix)
            self.assertEqualPoint(camera.getPositionFromPixel(pix), pos)

    def testDefect(self):
        """Test Defects"""

        #print >> sys.stderr, "Skipping testDefect"; return

        ccd = cameraGeomUtils.makeCcd(self.geomPolicy, cameraGeom.Id("Defects"))

        ccdImage = cameraGeomUtils.makeImageFromCcd(ccd)
        #
        # Insert some defects into the Ccd
        #
        defects = afwImage.DefectSet()
        for x0, y0, x1, y1 in [
            (34,  0,   35,  80 ),
            (34,  81,  34,  100),
            (180, 100, 182, 130),
            ]:
            bbox = afwImage.BBox(afwImage.PointI(x0, y0), afwImage.PointI(x1, y1))
            bad = ccdImage.Factory(ccdImage, bbox)
            bad.set(100)

            defects.push_back(afwImage.Defect(bbox))

        if display:
            ds9.mtv(ccdImage, title="Defects")
            cameraGeomUtils.showCcd(ccd, None)
            ds9.incrDefaultFrame()

        ccd.setDefects(defects)

        if False:
            print [str(d.getBBox()) for d in ccd.getDefects()]
            
            for a in ccd:
                print "    ", a.getId(), [str(d.getBBox()) for d in a.getDefects()]


        for id, i, x0, y0, x1, y1 in [
            (0, 0, 34,  0,   35,  57 ),
            (1, 0, 34,  58,  35,  80 ),
            (1, 1, 34,  81,  34,  100),
            (2, -1, 0, 0, 0, 0),
            (3, -1, 0, 0, 0, 0),
            (4, -1, 0, 0, 0, 0),
            (5, 0, 180, 100, 182, 115),
            (6, 0, 180, 116, 182, 130),
            (7, -1, 0, 0, 0, 0),
            ]:
            defects = ccd.findAmp(cameraGeom.Id(id)).getDefects()
            if i < 0:
                self.assertEqual(len(defects), 0)
            else:
                bbox = defects[i].getBBox()
                self.assertEqual(bbox, afwImage.BBox(afwImage.PointI(x0, y0), afwImage.PointI(x1, y1)))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(CameraGeomTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""

    ds9.setDefaultFrame(0)
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

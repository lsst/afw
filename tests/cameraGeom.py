#!/usr/bin/env python
"""
Tests for SpatialCell

Run with:
   python SpatialCell.py
or
   python
   >>> import SpatialCell; SpatialCell.run()
"""

import os
import sys
import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
#import lsst.afw.math.mathLib as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

import lsst.afw.cameraGeom as cameraGeom

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def showCcd(ccd, ccdImage, origin=None, frame=None):
    if ccdImage:
        ds9.mtv(ccdImage, frame=frame, title=ccd.getId().getName())

    for a in ccd:
        if a.getBiasSec().getWidth():
            displayUtils.drawBBox(a.getBiasSec(), origin=origin, ctype=ds9.RED, frame=frame)
        displayUtils.drawBBox(a.getDataSec(), origin=origin, ctype=ds9.BLUE, frame=frame)
        displayUtils.drawBBox(a.getAllPixels(), origin=origin, borderWidth=0.25, frame=frame)
        # Label each Amp
        ap = a.getAllPixels()
        xc, yc = (ap.getX0() + ap.getX1())//2, (ap.getY0() + ap.getY1())//2
        cen = afwGeom.Point2I.makeXY(xc, yc)
        if origin:
            xc += origin[0]
            yc += origin[1]
        ds9.dot(str(ccd.findAmp(cen).getId().getSerial()), xc, yc, frame=frame)

    displayUtils.drawBBox(ccd.getAllPixels(), borderWidth=0.49, ctype=ds9.MAGENTA, frame=frame)

def showRaft(raft, raftImage, frame=None):
    ds9.mtv(raftImage, frame=frame, title=raft.getId().getName())

    for dl in raft:
        ccd = cameraGeom.cast_Ccd(dl.getDetector())
        ccd.setTrimmed(True)
        
        bbox = ccd.getAllPixels(True)
        ds9.dot(ccd.getId().getName(),
                dl.getOrigin()[0] + bbox.getWidth()/2, dl.getOrigin()[1] + bbox.getHeight()/2, frame=frame)

        showCcd(ccd, None, frame=frame, origin=dl.getOrigin())

def trimCcd(ccd, ccdImage):
    ccd.setTrimmed(True)

    if ccdImage:
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

    ampSerial = ccdSerial = raftSerial = 0

    def assertEqualPoint(self, lhs, rhs):
        """Workaround the Point2D logical operator returning a Point not a bool"""
        if (lhs != rhs)[0] or (lhs != rhs)[1]:
            self.assertTrue(False, "%s != %s" % (lhs, rhs))

    def makeCcd(self, ccdName, makeImage=False):
        """Build a Ccd from a set of amplifiers"""
        self.pixelSize = 10e-3          # 10 microns

        nCol = 2                        # number of columns of amps; 2 == Left and Right
        nRow = 4                        # number of rows of amps
        width, height = 100, 50         # size of physical device
        extended = 10                   # length of extended register
        preRows = 2                     # extra rows before first real serial transfer -- always 0?
        overclockH = 15                 # number of serial overclock pixels
        overclockV = 5                  # number of parallel overclock pixels

        eWidth = extended + width + overclockH
        eHeight = preRows + height + overclockV

        ccdInfo = {}
        ccdInfo["name"] = ccdName
        ccdInfo["ampWidth"], ccdInfo["ampHeight"] = width, height
        ccdInfo["width"], ccdInfo["height"] = nCol*eWidth, nRow*eHeight
        ccdInfo["trimmedWidth"], ccdInfo["trimmedHeight"] = nCol*width, nRow*height
        ccdInfo["ampIdMin"] = CameraGeomTestCase.ampSerial

        ccd = cameraGeom.Ccd(cameraGeom.Id(CameraGeomTestCase.ccdSerial, ccdInfo["name"]), self.pixelSize)
        CameraGeomTestCase.ccdSerial += 1
        
        for Col in range(nCol):
            allPixels = afwImage.BBox(afwImage.PointI(0, 0), eWidth, eHeight)

            if Col == 0:
                c = cameraGeom.Amp.LLC
                biasSec = afwImage.BBox(afwImage.PointI(extended + width, preRows), overclockH, height)
                dataSec = afwImage.BBox(afwImage.PointI(extended, preRows), width, height)
            else:
                c = cameraGeom.Amp.LRC
                biasSec = afwImage.BBox(afwImage.PointI(0, preRows), overclockH, height)
                dataSec = afwImage.BBox(afwImage.PointI(overclockH, preRows), width, height)

            for Row in range(nRow):
                amp = cameraGeom.Amp(cameraGeom.Id(CameraGeomTestCase.ampSerial,
                                                   "ID%d" % CameraGeomTestCase.ampSerial),
                                     allPixels, biasSec, dataSec, c, 1.0, 10.0, 65535)
                CameraGeomTestCase.ampSerial += 1

                ccd.addAmp(Col, Row, amp)

        ccdInfo["ampIdMax"] = CameraGeomTestCase.ampSerial - 1
        #
        # Make an Image of that CCD?
        #
        if makeImage:
            ccdImage = afwImage.ImageU(ccd.getAllPixels().getDimensions())
            for a in ccd:
                im = ccdImage.Factory(ccdImage, a.getAllPixels())
                im += 1 + (a.getId().getSerial() - ccdInfo["ampIdMin"])
                im = ccdImage.Factory(ccdImage, a.getDataSec())
                im += 20
        else:
            ccdImage = None

        return ccd, ccdImage, ccdInfo

    def makeRaft(self, raftName, makeImage=False):
        """Build a Raft from a set of Ccd"""
        nCol = 2                        # number of columns of CCDs
        nRow = 3                        # number of rows of CCDs

        raftInfo = {}
        raftInfo["name"] = raftName

        raft = cameraGeom.Raft(cameraGeom.Id(CameraGeomTestCase.raftSerial, raftInfo["name"]))
        CameraGeomTestCase.raftSerial += 1

        for Col in range(nCol):
            for Row in range(nRow):
                ccd = self.makeCcd("R:%d,%d" % (Col, Row))[0]
                raft.addDetector(Col, Row, ccd)
        raftInfo["width"] =  nCol*ccd.getAllPixels(True).getWidth()
        raftInfo["height"] = nRow*ccd.getAllPixels(True).getHeight()
        #
        # Make an Image of that CCD?
        #
        if makeImage:
            raftImage = afwImage.ImageU(raft.getAllPixels().getDimensions())
            for dl in raft:
                det = dl.getDetector();
                bbox = det.getAllPixels(True)
                bbox.shift(dl.getOrigin()[0], dl.getOrigin()[1])
                im = raftImage.Factory(raftImage, bbox)
                im += 1 + (det.getId().getSerial())
        else:
            raftImage = None

        return raft, raftImage, raftInfo

    def setUp(self):
        CameraGeomTestCase.ampSerial = 0
        CameraGeomTestCase.ccdSerial = 1000
        CameraGeomTestCase.raftSerial = 2000
    
    def tearDown(self):
        pass

    def testId(self):
        """Test cameraGeom.Id"""
        try:
            cameraGeom.Id(-1)
        except pexExcept.LsstCppException:
            pass
        else:
            self.assertFalse("Should raise an exception on -ve serials")

        self.assertTrue(cameraGeom.Id(1) == cameraGeom.Id(1))
        self.assertFalse(cameraGeom.Id(1) == cameraGeom.Id(100))
        
        self.assertTrue(cameraGeom.Id("AA") == cameraGeom.Id("AA"))
        self.assertFalse(cameraGeom.Id("AA") == cameraGeom.Id("BB"))
        
        self.assertTrue(cameraGeom.Id(1, "AA") == cameraGeom.Id(1, "AA"))
        self.assertFalse(cameraGeom.Id(1, "AA") == cameraGeom.Id(2, "AA"))
        self.assertFalse(cameraGeom.Id(1, "AA") == cameraGeom.Id(1, "BB"))

    def testCcd(self):
        """Test if we can build a Ccd out of Amps"""

        print >> sys.stderr, "Skipping testCcd"; return

        ccd, ccdImage, ccdInfo = self.makeCcd("The Beast", makeImage=display)
        
        if display:
            showCcd(ccd, ccdImage, frame=0)

        for i in range(2):
            self.assertEqual(ccd.getSize()[i], self.pixelSize*ccd.getAllPixels().getDimensions()[i])

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
        pix = afwGeom.Point2I.makeXY(100, 200)
        pos = afwGeom.Point2D.makeXY(1.0, 2.0)
        #
        # Map pix into untrimmed coordinates
        #
        amp = ccd.findAmp(pix)
        corr = amp.getDataSec(False).getLLC() - amp.getDataSec(True).getLLC()
        pix += afwGeom.Extent2I(afwGeom.Point2I.makeXY(corr[0], corr[1]))
        
        self.assertEqual(ccd.getIndexFromPosition(pos), pix)

        self.assertEqualPoint(ccd.getPositionFromIndex(pix), pos)
        #
        # Trim the CCD and try again
        #
        trimmedImage = trimCcd(ccd, ccdImage)

        if display:
            showCcd(ccd, trimmedImage, frame=1)

        a = ccd.findAmp(cameraGeom.Id("ID%d" % ccdInfo["ampIdMin"]))
        self.assertEqual(a.getDataSec(), afwImage.BBox(afwImage.PointI(0, 0),
                                                       ccdInfo["ampWidth"], ccdInfo["ampHeight"]))

        self.assertEqual(ccd.getSize()[0], self.pixelSize*ccdInfo["trimmedWidth"])
        self.assertEqual(ccd.getSize()[1], self.pixelSize*ccdInfo["trimmedHeight"])
        #
        # Test mapping pixel <--> mm
        #
        pix = afwGeom.Point2I.makeXY(100, 200)
        pos = afwGeom.Point2D.makeXY(1.0, 2.0)
        
        self.assertEqualPoint(ccd.getIndexFromPosition(pos), pix)
        self.assertEqualPoint(ccd.getPositionFromIndex(pix), pos)

    def testRaft(self):
        """Test if we can build a Ccd out of Amps"""

        #print >> sys.stderr, "Skipping testRaft"; return

        raft, raftImage, raftInfo = self.makeRaft("Robinson", makeImage=True)

        if display:
            showRaft(raft, raftImage, frame=2)

        if False:
            print "Raft Name \"%s\", serial %d,  BBox %s" % \
                  (raft.getId().getName(), raft.getId().getSerial(), raft.getAllPixels())

            for d in raft:
                print d.getOrigin(), d.getDetector().getAllPixels(True)

            print "Size =", raft.getSize()

        self.assertEqual(raft.getAllPixels().getWidth(), raftInfo["width"])
        self.assertEqual(raft.getAllPixels().getHeight(), raftInfo["height"])

        for x, y, serial in [(0, 0, 7), (150, 250, 15), (250, 250, 39)]:
            det = raft.findDetector(afwGeom.Point2I.makeXY(x, y)).getDetector()
            ccd = cameraGeom.cast_Ccd(det)
            if False:
                print x, y, det.getId().getName(), \
                      ccd.findAmp(afwGeom.Point2I.makeXY(150, 152), True).getId().getSerial()
            self.assertEqual(ccd.findAmp(afwGeom.Point2I.makeXY(150, 152), True).getId().getSerial(), serial)

        name = "R:0,2"
        self.assertEqual(raft.findDetector(cameraGeom.Id(name)).getDetector().getId().getName(), name)
        #
        # This test isn't really right as we don't allow for e.g. gaps between Detectors.  Well, the
        # test is right but getSize() isn't
        #
        for i in range(2):
            self.assertEqual(raft.getSize()[i], self.pixelSize*raft.getAllPixels().getDimensions()[i])
        
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
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

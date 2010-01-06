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

def showCcd(ccd, ccdImage, frame=None):
    ds9.mtv(ccdImage, frame=frame)

    for a in ccd:
        if a.getBiasSec().getWidth():
            displayUtils.drawBBox(a.getBiasSec(), ctype=ds9.RED, frame=frame)
        displayUtils.drawBBox(a.getDataSec(), ctype=ds9.BLUE, frame=frame)
        displayUtils.drawBBox(a.getAllPixels(), borderWidth=0.25, frame=frame)
        # Label each Amp
        ap = a.getAllPixels()
        xc, yc = (ap.getX0() + ap.getX1())//2, (ap.getY0() + ap.getY1())//2
        cen = afwGeom.Point2I.makeXY(xc, yc)
        ds9.dot(str(ccd.getAmp(cen).getId().getSerial()), xc, yc, frame=frame)

    displayUtils.drawBBox(ccd.getAllPixels(), borderWidth=0.49, ctype=ds9.MAGENTA, frame=frame)

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

    def assertEqualPoint(self, lhs, rhs):
        """Workaround the Point2D logical operator returning a Point not a bool"""
        if (lhs != rhs)[0] or (lhs != rhs)[1]:
            self.assertTrue(False, "%s != %s" % (lhs, rhs))

    def makeCcd(self, serial0, ccdName, makeImage=False):
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
        ccdInfo["ccdWidth"], ccdInfo["ccdHeight"] = nCol*eWidth, nRow*eHeight
        ccdInfo["ccdTrimmedWidth"], ccdInfo["ccdTrimmedHeight"] = nCol*width, nRow*height
        ccdInfo["ampIdMin"] = serial0

        serial = serial0
        ccd = cameraGeom.Ccd(cameraGeom.Id(666, ccdInfo["name"]), self.pixelSize)
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
                amp = cameraGeom.Amp(cameraGeom.Id(serial, "ID%d" % serial), allPixels, biasSec, dataSec, \
                                     c, 1.0, 10.0, 65535)

                ccd.addAmp(Col, Row, amp)
                serial += 1

        ccdInfo["ampIdMax"] = serial - 1
        #
        # Make an Image of that CCD?
        #
        if makeImage:
            ccdImage = afwImage.ImageU(ccdInfo["ccdWidth"], ccdInfo["ccdHeight"])
            for a in ccd:
                im = ccdImage.Factory(ccdImage, a.getAllPixels())
                im += 1 + (a.getId().getSerial() - ccdInfo["ampIdMin"])
                im = ccdImage.Factory(ccdImage, a.getDataSec())
                im += 20
        else:
            ccdImage = None

        return ccd, ccdImage, ccdInfo

    def makeRaft(self, serial0, raftName):
        """Build a Raft from a set of Ccd"""
        nCol = 2                        # number of columns of CCDs
        nRow = 3                        # number of rows of CCDs

        raftInfo = {}
        raftInfo["name"] = raftName

        raft = cameraGeom.Raft(cameraGeom.Id(serial0, raftInfo["name"]))

        for Col in range(nCol):
            for Row in range(nRow):
                ccd = self.makeCcd(1234, "XX")[0]
                raft.addDetector(Col, Row, ccd)
                
        return raft, None, raftInfo

    def setUp(self):
        pass
    
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

        ccd, ccdImage, ccdInfo = self.makeCcd(666, "The Beast", makeImage=display)
        
        if display:
            showCcd(ccd, ccdImage, frame=0)

        for i in range(2):
            self.assertEqual(ccd.getSize()[i], self.pixelSize*ccd.getAllPixels().getDimensions()[i])

        self.assertEqual(ccd.getId().getName(), ccdInfo["name"])
        self.assertEqual(ccd.getAllPixels().getWidth(), ccdInfo["ccdWidth"])
        self.assertEqual([a.getId().getSerial() for a in ccd],
                         range(ccdInfo["ampIdMin"], ccdInfo["ampIdMax"] + 1))

        id = cameraGeom.Id("ID%d" % ccdInfo["ampIdMax"])
        self.assertTrue(ccd.getAmp(id), id)

        self.assertEqual(ccd.getAmp(afwGeom.Point2I.makeXY(10, 10)).getId().getSerial(), ccdInfo["ampIdMin"])

        self.assertEqual(ccd.getAllPixels().getLLC(),
                         ccd.getAmp(afwGeom.Point2I.makeXY(10, 10)).getAllPixels().getLLC())

        self.assertEqual(ccd.getAllPixels().getURC(),
                         ccd.getAmp(afwGeom.Point2I.makeXY(ccdInfo["ccdWidth"] - 1,
                                                           ccdInfo["ccdHeight"] - 1)).getAllPixels().getURC())
        #
        # Test mapping pixel <--> mm
        #
        pix = afwGeom.Point2I.makeXY(100, 200)
        pos = afwGeom.Point2D.makeXY(1.0, 2.0)
        #
        # Map pix into untrimmed coordinates
        #
        amp = ccd.getAmp(pix)
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

        a = ccd.getAmp(cameraGeom.Id("ID%d" % ccdInfo["ampIdMin"]))
        self.assertEqual(a.getDataSec(), afwImage.BBox(afwImage.PointI(0, 0),
                                                       ccdInfo["ampWidth"], ccdInfo["ampHeight"]))

        self.assertEqual(ccd.getSize()[0], self.pixelSize*ccdInfo["ccdTrimmedWidth"])
        self.assertEqual(ccd.getSize()[1], self.pixelSize*ccdInfo["ccdTrimmedHeight"])
        #
        # Test mapping pixel <--> mm
        #
        pix = afwGeom.Point2I.makeXY(100, 200)
        pos = afwGeom.Point2D.makeXY(1.0, 2.0)
        
        self.assertEqualPoint(ccd.getIndexFromPosition(pos), pix)
        self.assertEqualPoint(ccd.getPositionFromIndex(pix), pos)

    def testRaft(self):
        """Test if we can build a Ccd out of Amps"""

        raft, raftImage, raftInfo = self.makeRaft(1, "Robinson")

        if False:
            print raft.getAllPixels()
            for d in raft:
                print d.getAllPixels(True), d.getAllPixels()

        print "XXXXXXXXXXXXXXXX"
        print "[repr(d.get())][0]", [repr(d.getAllPixels(False)) for d in raft][0]
        print "repr([d.get()][0])", repr([d.getAllPixels()  for d in raft][0])
        print "[repr([][0].get())", repr([d for d in raft][0].getAllPixels(False))
        #print [d.getAllPixels() for d in raft]
        
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

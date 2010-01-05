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

import lsst.ip.isr.cameraGeomLib as cameraGeom

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
        cen = afwImage.PointI(xc, yc)
        ds9.dot(str(ccd.getAmp(cen).getId().getSerial()), xc, yc, frame=frame)

    displayUtils.drawBBox(ccd.getAllPixels(), borderWidth=0.49, ctype=ds9.MAGENTA, frame=frame)

def trimCcd(ccd, ccdImage):
    ccd.setTrimmed(True)

    trimmedImage = ccdImage.Factory(ccd.getAllPixels().getDimensions())
    for a in ccd:
        data =      ccdImage.Factory(ccdImage, a.getDataSec(False))
        tdata = trimmedImage.Factory(trimmedImage, a.getDataSec())
        tdata <<= data

    return trimmedImage

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class CameraGeomTestCase(unittest.TestCase):
    """A test case for camera geometry"""

    def assertEqualPoint(self, lhs, rhs):
        """Workaround the Point2D logical operator returning a Point not a bool"""
        if (lhs != rhs)[0] or (lhs != rhs)[1]:
            self.assertTrue(False, "%s != %s" % (lhs, rhs))

    def setUp(self):
        # Build a set of amplifiers
        serial = 666
        nCol = 2                        # number of columns of amps; 2 == Left and Right
        nRow = 4                        # number of rows of amps
        width, height = 100, 50         # size of physical device
        extended = 10                   # length of extended register
        preRows = 2                     # extra rows before first real serial transfer -- always 0?
        overclockH = 15                 # number of serial overclock pixels
        overclockV = 5                  # number of parallel overclock pixels

        eWidth = extended + width + overclockH
        eHeight = preRows + height + overclockV

        self.ccdName = "The Beast"
        self.ampWidth, self.ampHeight = width, height
        self.ccdWidth, self.ccdHeight = nCol*eWidth, nRow*eHeight
        self.ccdTrimmedWidth, self.ccdTrimmedHeight = nCol*width, nRow*height
        self.ampIdMin = serial

        self.pixelSize = 10e-3          # 10 microns
        self.ccd = cameraGeom.Ccd(cameraGeom.Id(666, self.ccdName), self.pixelSize)
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

                self.ccd.addAmp(Col, Row, amp)
                serial += 1

        self.ampIdMax = serial - 1
        #
        # Make an Image of that CCD
        #
        self.ccdImage = afwImage.ImageU(self.ccdWidth, self.ccdHeight)
        for a in self.ccd:
            im = self.ccdImage.Factory(self.ccdImage, a.getAllPixels())
            im += 1 + (a.getId().getSerial() - self.ampIdMin)
            im = self.ccdImage.Factory(self.ccdImage, a.getDataSec())
            im += 20

    def tearDown(self):
        pass

    def testCcd(self):
        """Test if we can build a Ccd out of Amps"""

        if display:
            showCcd(self.ccd, self.ccdImage, frame=0)
        #
        # OK, to work
        #
        for i in range(2):
            self.assertEqual(self.ccd.getSize()[i], self.pixelSize*self.ccd.getAllPixels().getDimensions()[i])

        self.assertEqual(self.ccd.getId().getName(), self.ccdName)
        self.assertEqual(self.ccd.getAllPixels().getWidth(), self.ccdWidth)
        self.assertEqual([a.getId().getSerial() for a in self.ccd], range(self.ampIdMin, self.ampIdMax + 1))

        id = cameraGeom.Id("ID%d" % self.ampIdMax)
        self.assertTrue(self.ccd.getAmp(id), id)

        self.assertEqual(self.ccd.getAmp(afwImage.PointI(10, 10)).getId().getSerial(), self.ampIdMin)

        self.assertEqual(self.ccd.getAllPixels().getLLC(),
                         self.ccd.getAmp(afwImage.PointI(10, 10)).getAllPixels().getLLC())

        self.assertEqual(self.ccd.getAllPixels().getURC(),
                         self.ccd.getAmp(afwImage.PointI(self.ccdWidth - 1,
                                                         self.ccdHeight - 1)).getAllPixels().getURC())
        #
        # Test mapping pixel <--> mm
        #
        pix = afwGeom.Point2I.makeXY(100, 200)
        pos = afwGeom.Point2D.makeXY(1.0, 2.0)
        #
        # Map pix into untrimmed coordinates
        #
        #amp = self.ccd.getAmp(pix)
        
        self.ccd.getPositionFromIndex
        self.assertEqual(self.ccd.getIndexFromPosition(pos), pix)

        if False:
            self.assertEqualPoint(self.ccd.getPositionFromIndex(pix), pos)
        #
        # Trim the CCD and try again
        #
        trimmedImage = trimCcd(self.ccd, self.ccdImage)

        if display:
            showCcd(self.ccd, trimmedImage, frame=1)

        a = self.ccd.getAmp(cameraGeom.Id("ID%d" % self.ampIdMin))
        self.assertEqual(a.getDataSec(), afwImage.BBox(afwImage.PointI(0, 0), self.ampWidth, self.ampHeight))

        self.assertEqual(self.ccd.getSize()[0], self.pixelSize*self.ccdTrimmedWidth)
        self.assertEqual(self.ccd.getSize()[1], self.pixelSize*self.ccdTrimmedHeight)
        #
        # Test mapping pixel <--> mm
        #
        pix = afwGeom.Point2I.makeXY(100, 200)
        pos = afwGeom.Point2D.makeXY(1.0, 2.0)
        
        self.assertEqualPoint(self.ccd.getIndexFromPosition(pos), pix)
        self.assertEqualPoint(self.ccd.getPositionFromIndex(pix), pos)

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

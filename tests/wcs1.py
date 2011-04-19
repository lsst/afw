#!/usr/bin/env python

# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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

import os
import math
import pdb                          # we may want to say pdb.set_trace()
import unittest

import eups
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.utils.tests as utilsTests
import lsst.afw.display.ds9 as ds9
import lsst.pex.exceptions.exceptionsLib as exceptions
import lsst

try:
    type(verbose)
except NameError:
    verbose = 0

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests")
InputImagePath = os.path.join(dataDir, "871034p_1_MI")
InputSmallImagePath = os.path.join(dataDir, "small_img.fits")
InputCorruptMaskedImageName = "small_MI_corrupt"
currDir = os.path.abspath(os.path.dirname(__file__))
InputCorruptFilePath = os.path.join(currDir, "data", InputCorruptMaskedImageName)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class WCSTestCaseSDSS(unittest.TestCase):
    """A test case for WCS using a small (SDSS) image with a slightly weird WCS"""

    def setUp(self):
        self.im = afwImage.DecoratedImageD(InputSmallImagePath)

        self.wcs = afwImage.makeWcs(self.im.getMetadata())

        if False:
            ds9.mtv(im, wcs=self.wcs)

    def tearDown(self):
        del self.wcs
        del self.im

    def testValidWcs(self):
        """Test operator bool() (== isValid)"""
        pass

    def testInvalidWcs(self):
        """Test operator bool() (== isValid)
        This test has been improved by deleting some essential
        metadata (in this case, CRPIX1, and CRPIX2) from the
        MaskedImage's metadata and using that.
        """
        wcs = afwImage.Wcs()
        self.assertFalse(wcs)

        # Using MaskedImage with corrupt metadata
        infile = afwImage.MaskedImageF_imageFileName(InputCorruptFilePath)
        decoratedImage = afwImage.DecoratedImageF(infile)
        metadata = decoratedImage.getMetadata()

        
        self.assertRaises(exceptions.LsstCppException, afwImage.makeWcs, metadata)

    def testCrpix(self):
        metadata = self.im.getMetadata()
        crpix0 = metadata.getAsDouble("CRPIX1")
        crpix1 = metadata.getAsDouble("CRPIX2")
        
        lsstCrpix = self.wcs.getPixelOrigin()
        
        self.assertEqual(lsstCrpix[0], crpix0-1)
        self.assertEqual(lsstCrpix[1], crpix1-1)
        
    def testXyToRaDecArguments(self):
        """Check that conversion of xy to ra dec (and back again) works"""
        xy = afwGeom.Point2D(110, 123)
        raDec = self.wcs.pixelToSky(xy)
        xy2 = self.wcs.skyToPixel(raDec)

        self.assertAlmostEqual(xy.getX(), xy2.getX())
        self.assertAlmostEqual(xy.getY(), xy2.getY())

        raDec = afwCoord.makeCoord(afwCoord.ICRS, 245.167400, +19.1976583)
        
        xy = self.wcs.skyToPixel(raDec)
        raDec2 = self.wcs.pixelToSky(xy)
        
        self.assertAlmostEqual(raDec[0], raDec2[0])
        self.assertAlmostEqual(raDec[1], raDec2[1])

    def test_RaTan_DecTan(self):
        """Check the RA---TAN, DEC--TAN WCS conversion"""
        # values from wcstools xy2sky (v3.8.1). Confirmed by ds9
        raDec0 = afwGeom.Point2D(245.15984167, +19.1960472) 
        raDec = self.wcs.pixelToSky(0.0, 0.0).getPosition()
        

        self.assertAlmostEqual(raDec.getX(), raDec0.getX(), 5)
        self.assertAlmostEqual(raDec.getY(), raDec0.getY(), 5) 

    def testIdentity(self):
        """Convert from ra, dec to col, row and back again"""
        raDec = afwCoord.makeCoord(afwCoord.ICRS, 244, 20)
        rowCol = self.wcs.skyToPixel(raDec)
        raDec2 = self.wcs.pixelToSky(rowCol)

        p1 = raDec.getPosition()
        p2 = raDec.getPosition()
        self.assertAlmostEqual(p1[0], p2[0])
        self.assertAlmostEqual(p1[1], p2[1])

    def testInvalidRaDec(self):
        """Test a conversion for an invalid position.  Well, "test" isn't
        quite right as the result is invalid, but make sure that it still is"""
        raDec = afwCoord.makeCoord(afwCoord.ICRS, 1, 2)

        self.assertRaises(lsst.pex.exceptions.exceptionsLib.LsstCppException, self.wcs.skyToPixel, raDec)

    def testCD(self):
        self.wcs.getCDMatrix()

    def testStripKeywords(self):
        """Test that we can strip WCS keywords from metadata when constructing a Wcs"""
        metadata = self.im.getMetadata()
        self.wcs = afwImage.makeWcs(metadata)

        self.assertTrue(metadata.exists("CRPIX1"))

        strip = True
        self.wcs = afwImage.makeWcs(metadata, strip)
        self.assertFalse(metadata.exists("CRPIX1"))

    def testAffineTransform(self):
        a = self.wcs.getLinearTransform()
        l = self.wcs.getCDMatrix()
        #print print a[a.XX], a[a.XY], a[a.YX], a[a.YY]
        print a, l

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class WCSTestCaseCFHT(unittest.TestCase):
    """A test case for WCS"""

    def setUp(self):
        path = InputImagePath + "_img.fits"
        self.metadata = afwImage.readMetadata(path)
        self.wcs = afwImage.makeWcs(self.metadata)
        if False:
            ds9.mtv(e)

    def tearDown(self):
        del self.wcs
        del self.metadata

    def test_RaTan_DecTan(self):
        """Check the RA---TAN, DEC--TAN WCS conversion"""
        raDec = self.wcs.pixelToSky(0.0, 0.0).getPosition() # position read off ds9

        self.assertAlmostEqual(raDec[0], 17.87673, 5) # ra from ds9
        self.assertAlmostEqual(raDec[1],  7.72231, 5) # dec from ds9

    def testPlateScale(self):
        """Test that we can measure the area of a pixel"""

        p00 = afwGeom.Point2D(10, 10)
        p00 = afwGeom.Point2D(self.metadata.getAsDouble("CRPIX1"), self.metadata.getAsDouble("CRPIX2"))

        sky00 = self.wcs.pixelToSky(p00).getPosition()
        cosdec = math.cos(math.pi/180*sky00[1])

        side = 1e-3
        icrs = afwCoord.ICRS
        degrees = afwCoord.DEGREES
        sky10 = afwCoord.makeCoord(icrs, sky00 + afwGeom.Extent2D(side/cosdec, 0), degrees)
        sky01 = afwCoord.makeCoord(icrs, sky00 + afwGeom.Extent2D(0,side),         degrees)
        p10 = self.wcs.skyToPixel(sky10) - p00
        p01 = self.wcs.skyToPixel(sky01) - p00

        area = side*side/abs(p10.getX()*p01.getY() - p01.getX()*p10.getY())

        self.assertAlmostEqual(math.sqrt(self.wcs.pixArea(p00)), math.sqrt(area))
        #
        # Now check that the area's the same as the CD matrix gives.
        #
        cd = [self.metadata.get("CD1_1"), self.metadata.get("CD1_2"),
              self.metadata.get("CD2_1"), self.metadata.get("CD2_2")]
        area = math.fabs(cd[0]*cd[3] - cd[1]*cd[2])

        self.assertAlmostEqual(math.sqrt(self.wcs.pixArea(p00)), math.sqrt(area))

    def testReadWcs(self):
        """Test reading a Wcs directly from a fits file"""

        meta = afwImage.readMetadata(InputImagePath + "_img.fits")
        wcs = afwImage.makeWcs(meta)

        sky0 = wcs.pixelToSky(0.0, 0.0).getPosition()
        sky1 = self.wcs.pixelToSky(0.0, 0.0).getPosition()
        self.assertEqual(sky0, sky1)

    def testShiftWcs(self):
        """Test shifting the reference pixel"""
        sky10_10 = self.wcs.pixelToSky(afwGeom.Point2D(10, 10))

        self.wcs.shiftReferencePixel(-10, -10)
        sky00 = self.wcs.pixelToSky(afwGeom.Point2D(0, 0))
        self.assertEqual((sky00[0], sky00[1]), (sky10_10[0], sky10_10[1]))

    def testCloneWcs(self):
        """Test Cloning a Wcs"""
        sky00 = self.wcs.pixelToSky(afwGeom.Point2D(0, 0)).getPosition()

        new = self.wcs.clone()
        self.wcs.pixelToSky(afwGeom.Point2D(10, 10)) # shouldn't affect new

        nsky00 = new.pixelToSky(afwGeom.Point2D(0, 0)).getPosition()
        self.assertEqual((sky00[0], sky00[1]), (nsky00[0], nsky00[1]))

    def testCD(self):
        cd = self.wcs.getCDMatrix()
        self.assertAlmostEqual(cd[0,0], self.metadata.getAsDouble("CD1_1"))
        self.assertAlmostEqual(cd[0,1], self.metadata.getAsDouble("CD1_2"))
        self.assertAlmostEqual(cd[1,0], self.metadata.getAsDouble("CD2_1"))
        self.assertAlmostEqual(cd[1,1], self.metadata.getAsDouble("CD2_2"))

    def testConstructor(self):
        copy = afwImage.Wcs(self.wcs.getSkyOrigin(), self.wcs.getPixelOrigin(), 
                            self.wcs.getCDMatrix())

    def testAffineTransform(self):
        a = self.wcs.getLinearTransform()
        l = self.wcs.getCDMatrix()
        #print print a[a.XX], a[a.XY], a[a.YX], a[a.YY]

        sky00g = afwGeom.Point2D(10, 10)
        sky00i = afwGeom.Point2D(sky00g.getX(), sky00g.getY())
        sky00c = afwCoord.makeCoord(afwCoord.ICRS, sky00i, afwCoord.DEGREES)
        a = self.wcs.linearizeSkyToPixel(sky00c)
        pix00i = self.wcs.skyToPixel(sky00c)
        pix00g = afwGeom.Point2D(pix00i.getX(), pix00i.getY())
        sky00gApprox = a(pix00g);
        self.assertAlmostEqual(sky00g.getX(), sky00gApprox.getX())
        self.assertAlmostEqual(sky00g.getY(), sky00gApprox.getY())
        self.assertAlmostEqual(self.wcs.pixArea(sky00i), abs(a[a.XX]* a[a.YY] - a[a.XY]*a[a.YX]))
        a.invert()

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(WCSTestCaseSDSS)
#    suites += unittest.makeSuite(WCSTestCaseCFHT)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

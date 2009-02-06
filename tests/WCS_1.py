#!/usr/bin/env python
import os
import math
import pdb                          # we may want to say pdb.set_trace()
import unittest

import eups
import lsst.afw.image as afwImage
import lsst.utils.tests as utilsTests

try:
    type(verbose)
except NameError:
    verbose = 0

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests")
InputImagePath = os.path.join(dataDir, "871034p_1_MI_img.fits")
InputSmallImagePath = os.path.join(dataDir, "small_img.fits")
InputCorruptMaskedImageName = "small_MI_corrupt"
currDir = os.path.abspath(os.path.dirname(__file__))
InputCorruptFilePath = os.path.join(currDir, "data", InputCorruptMaskedImageName)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class WCSTestCaseSDSS(unittest.TestCase):
    """A test case for WCS using a small (SDSS) image with a slightly weird WCS"""

    def setUp(self):
        im = afwImage.DecoratedImageD(InputSmallImagePath)

        self.wcs = afwImage.Wcs(im.getMetadata())

        if False:
            import lsst.afw.display.ds9 as ds9; ds9.mtv(im, wcs=self.wcs)

    def tearDown(self):
        del self.wcs

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

        corruptWcs = afwImage.Wcs(metadata)
        if False:
            self.assertTrue(not corruptWcs)
        else:
            print "Ignoring failure to detect corrupt WCS from", infile

    def testraDecToXYArguments(self):
        """Check that all the expected forms of raDecToXY/xyToRaDec work"""
        raDec = afwImage.PointD(1,2)
        self.wcs.raDecToXY(raDec)
        self.wcs.raDecToXY(1, 2)

    def test_RaTan_DecTan(self):
        """Check the RA---TAN, DEC--TAN WCS conversion"""
        raDec = self.wcs.xyToRaDec(0.0, 0.0)
        raDec0 = afwImage.PointD(19.1960467992, 245.1598413385) # values from wcstools' xy2sky, transposed

        self.assertAlmostEqual(raDec.getX(), raDec0.getX(), 5)
        self.assertAlmostEqual(raDec.getY(), raDec0.getY(), 5) # dec from ds9

    def testIdentity(self):
        """Convert from ra, dec to col, row and back again"""
        raDec = afwImage.PointD(20, 150)
        rowCol = self.wcs.raDecToXY(raDec)
        raDec2 = self.wcs.xyToRaDec(rowCol)

        self.assertAlmostEqual(raDec.getX(), raDec2.getX())
        self.assertAlmostEqual(raDec.getY(), raDec2.getY())

    def testInvalidRaDec(self):
        """Test a conversion for an invalid position.  Well, "test" isn't
        quite right as the result is invalid, but make sure that it still is"""
        raDec = afwImage.PointD(1, 2)
        rowCol = self.wcs.raDecToXY(raDec)
        raDec2 = self.wcs.xyToRaDec(rowCol)

        self.assertAlmostEqual(raDec2.getX(), -raDec.getX())
        self.assertAlmostEqual(raDec2.getY(), 180 + raDec.getY())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class WCSTestCaseCFHT(unittest.TestCase):
    """A test case for WCS"""

    def setUp(self):
        im = afwImage.DecoratedImageD(InputImagePath)

        self.wcs = afwImage.Wcs(im.getMetadata())

    def tearDown(self):
        del self.wcs

    def test_RaTan_DecTan(self):
        """Check the RA---TAN, DEC--TAN WCS conversion"""
        raDec = self.wcs.xyToRaDec(0.0, 0.0) # position read off ds9

        self.assertAlmostEqual(raDec.getX(), 17.87673, 5) # ra from ds9
        self.assertAlmostEqual(raDec.getY(),  7.72231, 5) # dec from ds9

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(WCSTestCaseSDSS)
    suites += unittest.makeSuite(WCSTestCaseCFHT)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

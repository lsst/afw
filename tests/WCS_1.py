import os
import math
import pdb                          # we may want to say pdb.set_trace()
import unittest

import eups
import lsst.afw.image as afwImage
import lsst.daf.tests as dafTests

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
        im = afwImage.ImageD()
        im.readFits(InputSmallImagePath)

        self.wcs = afw.WCS(im.getMetaData())

        if False:
            import lsst.afw.display.ds9 as ds9; ds9.mtv(im, WCS=self.wcs)

    def tearDown(self):
        del self.wcs

    def testValidWCS(self):
        """Test operator bool() (== isValid)"""
        self.assertTrue(self.wcs.isValid())

    def testInvalidWCS(self):
        """Test operator bool() (== isValid)
        This test has been improved by deleting some essential
        metadata (in this case, CRPIX1, and CRPIX2) from the
        MaskedImage's metadata and using that.
        """
        wcs = afw.WCS()
        self.assertFalse(wcs.isValid())

        # Using MaskedImage with corrupt metadata 
        maskedImage = afwImage.MaskedImageF()
        maskedImage.readFits(InputCorruptFilePath)
        metadata = maskedImage.getImage().getMetaData()
        corruptWcs = afw.WCS(metadata)
        self.assertTrue(corruptWcs.isValid())
        

    def testraDecToColRowArguments(self):
        """Check that all the expected forms of raDecToColRow/colRowToRaDec work"""
        raDec = afw.Coord2D(1,2)
        self.wcs.raDecToColRow(raDec)
        self.wcs.raDecToColRow(1, 2)
        rowCol = afw.Coord2D()
        self.wcs.raDecToColRow(raDec, rowCol)

    def test_RaTan_DecTan(self):
        """Check the RA---TAN, DEC--TAN WCS conversion"""
        raDec = self.wcs.colRowToRaDec(1.0, 1.0)
        raDec0 = afw.Coord2D(19.1960467992, 245.1598413385) # values from wcstools' xy2sky, transposed

        self.assertAlmostEqual(raDec.x(), raDec0.x(), 5)
        self.assertAlmostEqual(raDec.y(), raDec0.y(), 5) # dec from ds9

    def testIdentity(self):
        """Convert from ra, dec to col, row and back again"""
        raDec = afw.Coord2D(20, 150)
        rowCol = self.wcs.raDecToColRow(raDec)
        raDec2 = self.wcs.colRowToRaDec(rowCol)

        self.assertAlmostEqual(raDec.x(), raDec2.x())
        self.assertAlmostEqual(raDec.y(), raDec2.y())

    def testInvalidRaDec(self):
        """Test a conversion for an invalid position.  Well, "test" isn't
        quite right as the result is invalid, but make sure that it still is"""
        raDec = afw.Coord2D(1, 2)
        rowCol = self.wcs.raDecToColRow(raDec)
        raDec2 = self.wcs.colRowToRaDec(rowCol)

        self.assertAlmostEqual(raDec2.x(), -raDec.x())
        self.assertAlmostEqual(raDec2.y(), 180 + raDec.y())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class WCSTestCaseCFHT(unittest.TestCase):
    """A test case for WCS"""

    def setUp(self):
        im = afwImage.ImageD()

        im.readFits(InputImagePath)

        self.wcs = afw.WCS(im.getMetaData())

    def tearDown(self):
        del self.wcs

    def test_RaTan_DecTan(self):
        """Check the RA---TAN, DEC--TAN WCS conversion"""
        raDec = self.wcs.colRowToRaDec(33.0, 1.0 ) # position read off ds9

        self.assertAlmostEqual(raDec.x(), 17.87840, 5) # ra from ds9
        self.assertAlmostEqual(raDec.y(), 7.72231, 5) # dec from ds9

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    dafTests.init()

    suites = []
    suites += unittest.makeSuite(WCSTestCaseSDSS)
    suites += unittest.makeSuite(WCSTestCaseCFHT)
    suites += unittest.makeSuite(dafTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    dafTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

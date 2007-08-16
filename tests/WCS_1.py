import pdb                              # we may want to say pdb.set_trace()
import unittest
import os.path

import eups
import lsst.fw.Core.fwLib as fw
import lsst.mwi.tests as tests
import lsst.mwi.utils as mwiu

try:
    type(verbose)
except NameError:
    verbose = 0
    mwiu.Trace_setVerbosity("fw.DataProperty", verbose)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class WCSTestCaseSDSS(unittest.TestCase):
    """A test case for WCS using a small (SDSS) image with a slightly weird WCS"""

    def setUp(self):
        im = fw.ImageD()
        im.readFits(tests.findFileFromRoot("tests/data/small_img.fits"))

        self.wcs = fw.WCS(im.getMetaData())

        if False:
            import lsst.fw.Display.ds9 as ds9; ds9.mtv(im, WCS=self.wcs)

    def tearDown(self):
        del self.wcs

    def testValidWCS(self):
        """Test operator bool() (== isValid)"""
        self.assertTrue(self.wcs.isValid())

    def testInvalidWCS(self):
        """Test operator bool() (== isValid)"""
        wcs = fw.WCS(fw.ImageD().getMetaData())
        self.assertFalse(wcs.isValid())

    def testraDecToColRowArguments(self):
        """Check that all the expected forms of raDecToColRow/colRowToRaDec work"""
        raDec = fw.Coord2D(1,2)
        self.wcs.raDecToColRow(raDec)
        self.wcs.raDecToColRow(1, 2)
        rowCol = fw.Coord2D()
        self.wcs.raDecToColRow(raDec, rowCol)

    def test_RaTan_DecTan(self):
        """Check the RA---TAN, DEC--TAN WCS conversion"""
        raDec = self.wcs.colRowToRaDec(1.0, 1.0)
        raDec0 = fw.Coord2D(19.1960467992, 245.1598413385) # values from wcstools' xy2sky, transposed

        self.assertAlmostEqual(raDec.x(), raDec0.x(), 5)
        self.assertAlmostEqual(raDec.y(), raDec0.y(), 5) # dec from ds9

    def testIdentity(self):
        """Convert from ra, dec to col, row and back again"""
        raDec = fw.Coord2D(20, 150)
        rowCol = self.wcs.raDecToColRow(raDec)
        raDec2 = self.wcs.colRowToRaDec(rowCol)

        self.assertAlmostEqual(raDec.x(), raDec2.x())
        self.assertAlmostEqual(raDec.y(), raDec2.y())

    def testInvalidRaDec(self):
        """Test a conversion for an invalid position.  Well, "test" isn't
        quite right as the result is invalid, but make sure that it still is"""
        raDec = fw.Coord2D(1, 2)
        rowCol = self.wcs.raDecToColRow(raDec)
        raDec2 = self.wcs.colRowToRaDec(rowCol)

        self.assertAlmostEqual(raDec2.x(), -raDec.x())
        self.assertAlmostEqual(raDec2.y(), 180 + raDec.y())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class WCSTestCaseCFHT(unittest.TestCase):
    """A test case for WCS"""

    def setUp(self):
        im = fw.ImageD()

        im.readFits(tests.findFileFromRoot(os.path.join(dataDir, "871034p_1_img.fits")))

        self.wcs = fw.WCS(im.getMetaData())

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

    global dataDir

    try:                                # eups version >= v0_7_33
        dataDir = eups.directory("fwData", "setup")
    except AttributeError:
        dataDir = eups.list("fwData", eups.current("fwData")) # should be "setup", but "current" is available
        if dataDir:
            dataDir = dataDir[2]

    suites = []

    if dataDir:
        suites += unittest.makeSuite(WCSTestCaseSDSS)
        suites += unittest.makeSuite(WCSTestCaseCFHT)

    suites += unittest.makeSuite(tests.MemoryTestCase)
        
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    tests.run(suite())

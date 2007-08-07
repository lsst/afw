import pdb                              # we may want to say pdb.set_trace()
import unittest
import lsst.fw.Core.fwLib as fw
import lsst.mwi.tests as tests
import lsst.mwi.utils as mwiu

try:
    type(verbose)
except NameError:
    verbose = 0
    mwiu.Trace_setVerbosity("fw.DataProperty", verbose)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class WCSTestCase(unittest.TestCase):
    """A test case for WCS"""

    def setUp(self):
        im = fw.ImageD()
        im.readFits(tests.findFileFromRoot("tests/data/871034p_1_img.fits"))

        self.wcs = fw.WCS(im.getMetaData())

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
        raDec = self.wcs.colRowToRaDec(33.0, 1.0 ) # position read off ds9

        self.assertAlmostEqual(raDec.x(), 17.87840, 5) # ra from ds9
        self.assertAlmostEqual(raDec.y(), 7.72231, 5) # dec from ds9

        if False:
            print raDec.x(), raDec.y()

            colRow = self.wcs.raDecToColRow(17.87840, 7.77231)
            print colRow.x(), colRow.y()
            colRow = self.wcs.raDecToColRow(raDec)
            print colRow.x(), colRow.y()
        

    def testIdentity(self):
        """Convert from ra, dec to col, row and back again"""
        raDec = fw.Coord2D(1,2)
        rowCol = self.wcs.raDecToColRow(raDec)
        raDec2 = self.wcs.colRowToRaDec(rowCol)

        self.assertAlmostEqual(raDec.x(), raDec2.x())
        self.assertAlmostEqual(raDec.y(), raDec2.y())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    suites = []
    suites += unittest.makeSuite(WCSTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    tests.run(suite())

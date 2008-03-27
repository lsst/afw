"""
Test cases to test image I/O
"""
import os
import pdb                          # we may want to say pdb.set_trace()
import unittest

import lsst.afw.image as afwImage
import lsst.daf.tests as dafTests

try:
    type(verbose)
except NameError:
    verbose = 0

try:
    import eups; dataDir = eups.productDir("fwData") # needs eups >= v0_7_40
except:
    dataDir = os.environ.get("FWDATA_DIR")

if not dataDir:
    raise RuntimeError("Must set up fwData to run these tests")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ReadFitsTestCase(unittest.TestCase):
    """A test case for reading FITS images"""

    def setUp(self):
        self.im = afwImage.ImageD()

    def tearDown(self):
        del self.im

    def testU16(self):
        """Test reading U16 image"""
        self.im.readFits(os.path.join(dataDir, "small_img.fits"))
        
        col, row, val = 0, 0, 1154
        self.assertEqual(self.im.getVal(col, row), val)

    def testS16(self):
        """Test reading S16 image"""
        self.im.readFits(os.path.join(dataDir, "871034p_1_img.fits"))

        if False:
            import lsst.afw.display.ds9 as ds9; ds9.mtv(self.im)
        
        col, row, val = 32, 1, 62
        self.assertEqual(self.im.getVal(col, row), val)

    def testF32(self):
        """Test reading F32 image"""
        self.im.readFits(os.path.join(dataDir, "871034p_1_MI_var.fits"))
        
        col, row, val = 32, 1, 39.11672
        self.assertAlmostEqual(self.im.getVal(col, row), val, 5)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    dafTests.init()

    suites = []
    suites += unittest.makeSuite(ReadFitsTestCase)
    suites += unittest.makeSuite(dafTests.MemoryTestCase)

    return unittest.TestSuite(suites)

if __name__ == "__main__":
    dafTests.run(suite())

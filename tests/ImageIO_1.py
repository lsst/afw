#!/usr/bin/env python
"""
Test cases to test image I/O
"""
import os
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

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ReadFitsTestCase(unittest.TestCase):
    """A test case for reading FITS images"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testU16(self):
        """Test reading U16 image"""

        im = afwImage.ImageD(os.path.join(dataDir, "small_img.fits"))
        
        col, row, val = 0, 0, 1154
        self.assertEqual(im.get(col, row), val)

    def testS16(self):
        """Test reading S16 image"""
        im = afwImage.ImageD(os.path.join(dataDir, "871034p_1_img.fits"))

        if False:
            import lsst.afw.display.ds9 as ds9; ds9.mtv(im)
        
        col, row, val = 32, 1, 62
        self.assertEqual(im.get(col, row), val)

    def testF32(self):
        """Test reading F32 image"""
        im = afwImage.ImageD(os.path.join(dataDir, "871034p_1_MI_var.fits"))
        
        col, row, val = 32, 1, 39.11672
        self.assertAlmostEqual(im.get(col, row), val, 5)

    def testF64(self):
        """Test reading an F64 image"""
        im = afwImage.ImageD(os.path.join(dataDir, "small_img.fits"))
        col, row, val = 0, 0, 1154
        self.assertEqual(im.get(col, row), val)
        
        #print "IM = ", im
    def testWriteF64(self):
        """Test writing an F64 image"""

        im = afwImage.ImageD(100, 100); im.set(666)
        im.writeFits("smallD.fits")
        newIm = afwImage.ImageD("smallD.fits")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ReadFitsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)
 
if __name__ == "__main__":
    run(True)

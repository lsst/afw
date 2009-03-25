#!/usr/bin/env python
"""
Tests for MaskedImages

Run with:
   python MaskedImageIO.py
or
   python
   >>> import MaskedImageIO; MaskedImageIO.run()
"""

import pdb                              # we may want to say pdb.set_trace()
import os
import unittest

import eups

import lsst.utils.tests as utilsTests
import lsst.afw.image as afwImage
import lsst.afw.display.ds9 as ds9
import lsst.pex.exceptions as pexEx

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("You must set up afwdata to run these tests")

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class MaskedImageTestCase(unittest.TestCase):
    """A test case for MaskedImage"""
    def setUp(self):
        # Set a (non-standard) initial Mask plane definition
        #
        # Ideally we'd use the standard dictionary and a non-standard file, but
        # a standard file's what we have
        #
        mask = afwImage.MaskU()

        mask.clearMaskPlaneDict()
        for p in ("ZERO", "BAD", "SAT", "INTRP", "CR", "EDGE"):
            mask.addMaskPlane(p)

        if False:
            self.baseName = os.path.join(dataDir, "Small_MI")
        else:
            self.baseName = os.path.join(dataDir,"CFHT", "D4", "cal-53535-i-797722_1")
        self.mi = afwImage.MaskedImageF(self.baseName)

    def tearDown(self):
        del self.mi

    def testFitsRead(self):
        """Check if we read MaskedImages"""

        image = self.mi.getImage()
        mask = self.mi.getMask()

        if display:
            ds9.mtv(self.mi)

        self.assertEqual(image.get(32, 1), 3728);
        self.assertEqual(mask.get(0,0), 2); # == BAD
            
    def testFitsReadConform(self):
        """Check if we read MaskedImages and make them replace Mask's plane dictionary"""

        hdu, metadata, bbox, conformMasks = 0, None, afwImage.BBox(), True
        self.mi = afwImage.MaskedImageF(self.baseName, hdu, metadata, bbox, conformMasks)

        image = self.mi.getImage()
        mask = self.mi.getMask()

        self.assertEqual(image.get(32, 1), 3728);
        self.assertEqual(mask.get(0,0), 1); # i.e. not shifted 1 place to the right

        self.assertEqual(mask.getMaskPlane("CR"), 3, "Plane CR has value specified in FITS file")

    def testFitsReadNoConform2(self):
        """Check that reading a mask doesn't invalidate the plane dictionary"""

        testMask = afwImage.MaskU(afwImage.MaskedImageF_maskFileName(self.baseName))

        mask = self.mi.getMask()
        mask |= testMask

    def testFitsReadConform2(self):
        """Check that conforming a mask invalidates the plane dictionary"""

        hdu, metadata, bbox, conformMasks = 0, None, afwImage.BBox(), True
        testMask = afwImage.MaskU(afwImage.MaskedImageF_maskFileName(self.baseName), hdu, metadata, bbox, conformMasks)

        mask = self.mi.getMask()
        def tst(mask=mask):
            mask |= testMask

        utilsTests.assertRaisesLsstCpp(self, pexEx.RuntimeErrorException, tst)

    def testTicket617(self):
        """Test reading an F64 image and converting it to a MaskedImage"""
        im = afwImage.ImageD(100, 100); im.set(666)
        mi = afwImage.MaskedImageD(im)

    def testReadWriteXY0(self):
        """Test that we read and write (X0, Y0) correctly"""
        im = afwImage.MaskedImageF(10, 20)

        x0, y0 = 1, 2
        im.setXY0(x0, y0)
        tmpFile = "foo"
        im.writeFits(tmpFile)

        im2 = im.Factory(tmpFile)
        self.assertEqual(im2.getX0(), x0)
        self.assertEqual(im2.getY0(), y0)

        self.assertEqual(im2.getImage().getX0(), x0)
        self.assertEqual(im2.getImage().getY0(), y0)

        self.assertEqual(im2.getMask().getX0(), x0)
        self.assertEqual(im2.getMask().getY0(), y0)
        
        self.assertEqual(im2.getVariance().getX0(), x0)
        self.assertEqual(im2.getVariance().getY0(), y0)
        
        os.remove(afwImage.MaskedImageF.imageFileName(tmpFile))
        os.remove(afwImage.MaskedImageF.maskFileName(tmpFile))
        os.remove(afwImage.MaskedImageF.varianceFileName(tmpFile))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(MaskedImageTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

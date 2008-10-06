#!/usr/bin/env python
"""
Tests for MaskedImages

Run with:
   python MaskedImage_1.py
or
   python
   >>> import MaskedImage_1; MaskedImage_1.run()
"""

import pdb  # we may want to say pdb.set_trace()
import unittest

import lsst.utils.tests as utilsTests
import lsst.afw.image as afwImage
import lsst.afw.display.ds9 as ds9

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class MaskedImageTestCase(unittest.TestCase):
    """A test case for MaskedImage"""
    def setUp(self):
        self.maskedImage1 = afwImage.MaskedImageF(272, 1037)
        self.maskedImage2 = afwImage.MaskedImageF(272, 1037)

        for m in (self.maskedImage1, self.maskedImage2):
            m.getMask().addMaskPlane("CR")
            m.getMask().addMaskPlane("INTERP")

    def tearDown(self):
        del self.maskedImage1
        del self.maskedImage2

    def testGC(self):
        """Check that Peaks are automatically garbage collected (when MemoryTestCase runs)"""
        pass

    def testAddMaskPlane(self):
        """Check if we can add mask planes"""
        mask = self.maskedImage1.getMask()

        name = "XXX"
        assert mask.getMaskPlane(name) == -1, "There is no plane named %s" %name

        name = "FOO"
        mask.addMaskPlane(name)
        try:
            mask.getMaskPlane(name)
        except IndexError:
            self.fail("Failed to add maskPlane %s" % name)

    def testAddMaskedImages(self):
        self.maskedImage2 += self.maskedImage1
    
    def testCopyConstructors(self):
        image = afwImage.ImageF(afwImage.ImageF(100, 100))
        mask = afwImage.MaskU(afwImage.MaskU(100, 100))
        maskedImage = afwTests.MaskedImageF(self.maskedImage1)

    def testDisplay(self):
        """Test decomposing a mask into its bit planes"""

        mask = self.maskedImage1.getMask()

        pixelList = afwImage.listPixelCoord()
        for x in range(0, mask.getCols()):
            for y in range(300, 400, 20):
                pixelList.push_back(afwImage.PixelCoord(x, y))
        mask.setMaskPlaneValues(mask.getMaskPlane('CR'), pixelList)

        pixelList = afwImage.listPixelCoord()
        for x in range(300, 400, 20):
            for y in range(0, mask.getRows()):
                pixelList.push_back(afwImage.PixelCoord(x, y))
        mask.setMaskPlaneValues(mask.getMaskPlane('INTERP'), pixelList)

        if display:
            ds9.mtv(self.maskedImage1.getImage(), isMask=False)

            ds9.setMaskColor(ds9.RED)
            CR = mask; mask.removeMaskPlane("INTERP")
            ds9.mtv(CR, isMask=True)

            ds9.setMaskColor(ds9.GREEN)
            INTERP = mask; mask.removeMaskPlane("CR")
            ds9.mtv(INTERP, isMask=True)
            
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

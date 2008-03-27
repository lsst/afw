"""
Tests for MaskedImages

Run with:
   python MaskedImage_1.py
or
   python
   >>> import MaskedImage_1; MaskedImage_1.run()
"""

import pdb                              # we may want to say pdb.set_trace()
import unittest
import lsst.mwi.tests as tests
import lsst.afw as afw
import lsst.afw.display.ds9 as ds9
import fwTests

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class MaskedImageTestCase(unittest.TestCase):
    """A test case for MaskedImage"""
    def setUp(self):
        self.maskedImage1 = afw.image.MaskedImageF(272, 1037)
        self.maskedImage2 = afw.image.MaskedImageF(272, 1037)

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
        image = fwTests.copyImageF(afw.image.ImageF(100, 100))
        mask = fwTests.copyMaskU(afw.image.MaskU(100, 100))
        maskedImage = fwTests.copyMaskedImageF(self.maskedImage1)

    def testPixelProc(self):
        fooFunc = fwTests.testPixProcFuncF(self.maskedImage1)

        fooFunc.init()
        self.maskedImage1.processPixels(fooFunc)

    def testDisplay(self):
        """Test decomposing a mask into its bit planes"""

        mask = self.maskedImage1.getMask()

        pixelList = afw.listPixelCoord()
        for x in range(0, mask.getCols()):
            for y in range(300, 400, 20):
                pixelList.push_back(afw.PixelCoord(x, y))
        mask.setMaskPlaneValues(mask.getMaskPlane('CR'), pixelList)

        pixelList = afw.listPixelCoord()
        for x in range(300, 400, 20):
            for y in range(0, mask.getRows()):
                pixelList.push_back(afw.PixelCoord(x, y))
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

    tests.init()

    suites = []
    suites += unittest.makeSuite(MaskedImageTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    try:
        tests.run(suite(), exit)        # mwi 2.0
    except:
        tests.run(suite())

if __name__ == "__main__":
    run(True)

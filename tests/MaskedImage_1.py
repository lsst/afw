"""
Tests for Peaks, Footprints, and DetectionSets

Run with:
   python Footprint_1.py
or
   python
   >>> import unittest; T=load("Footprint_1"); unittest.TextTestRunner(verbosity=1).run(T.suite())
"""

import pdb                              # we may want to say pdb.set_trace()
import unittest
import lsst.fw.Core.tests as tests
import lsst.fw.Core.fwLib as fw
import fwTests

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class MaskedImageTestCase(unittest.TestCase):
    """A test case for MaskedImage"""
    def setUp(self):
        self.maskedImage1 = fw.MaskedImageD(272, 1037)
        self.maskedImage2 = fw.MaskedImageD(272, 1037)

        for m in (self.maskedImage1, self.maskedImage2):
            m.getMask().addMaskPlane("CR")

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

    def testPixelProc(self):
        fooFunc = fwTests.testPixProcFuncD(self.maskedImage1)

        fooFunc.init()
        self.maskedImage1.processPixels(fooFunc)
            
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    tests.init()

    suites = []
    suites += unittest.makeSuite(MaskedImageTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    unittest.main()

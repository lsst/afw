"""
Tests for Peaks

Run with:
   python Peak_1.py
or
   python
   >>> import unittest; T=load("Peak_1"); unittest.TextTestRunner(verbosity=1).run(T.suite())
"""

import pdb                              # we may want to say pdb.set_trace()
import unittest
import lsst.utils.tests as tests
import lsst.pex.logging as logging
import lsst.afw.image.imageLib as imageLib
import lsst.afw.detection.detectionLib as detection

try:
    type(verbose)
except NameError:
    verbose = 0
    logging.Trace_setVerbosity("detection.Footprint", verbose)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class PeakTestCase(unittest.TestCase):
    """A test case for Peak"""
    def setUp(self):
        self.peak = detection.Peak()

    def tearDown(self):
        del self.peak

    def testGC(self):
        """Check that Peaks are automatically garbage collected (when MemoryTestCase runs)"""
        
        f = detection.Peak()

    def testToString(self):
        assert self.peak.toString() != None
        
    def testCentroidInt(self):
        x, y = 10, -10
        peak = detection.Peak(x, y)
        self.assertEqual(peak.getIx(), x)
        self.assertEqual(peak.getIy(), y)

        self.assertEqual(peak.getFx(), x)
        self.assertEqual(peak.getFy(), y)

    def testCentroidFloat(self):
        for x, y in [(5, 6), (10.5, -10.5)]:
            peak = detection.Peak(x, y)
            self.assertEqual(peak.getFx(), x)
            self.assertEqual(peak.getFy(), y)

            self.assertEqual(peak.getIx(), int(x) if x > 0 else -int(-x) - 1)
            self.assertEqual(peak.getIy(), int(y) if y > 0 else -int(-y) - 1)

    def testId(self):
        """Test uniqueness of IDs"""
        
        self.assertNotEqual(self.peak.getId(), detection.Peak().getId())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(PeakTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    tests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

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
import eups
import unittest
import lsst.mwi.tests as tests
import lsst.fw.Core.fwLib as fw
import lsst.fw.Display.ds9 as ds9
import fwTests

dataDir = eups.productDir("fwData")
if not dataDir:
    raise RuntimeError("You must set up fwData to run these tests")

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class MaskedImageTestCase(unittest.TestCase):
    """A test case for MaskedImage"""
    def setUp(self):
        self.mi = fw.MaskedImageF()

        file = os.path.join(dataDir, "CFHT", "D4", "cal-53535-i-797722_1")
        self.mi.readFits(file)

    def tearDown(self):
        del self.mi

    def testFitsRead(self):
        """Check if we read MaskedImages"""

        if display:
            ds9.mtv(self.mi)

        self.assertEqual(self.mi.getImage().getVal(32, 1), 3728);
        self.assertEqual(self.mi.getMask().getVal(0,0), 1);
        
            
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
    tests.run(suite(), exit)        # mwi 2.0

if __name__ == "__main__":
    run(True)

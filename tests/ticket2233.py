#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

"""
Sogo Mineo writes:

'''
If I read Wcs from, e.g., the following file:
   master:/data1a/Subaru/SUPA/rerun/mineo-Abell1689/03430/W-S-I+/corr/wcs01098593.fits

then Wcs::_nWcsInfo becomes 2.

But WcsFormatter assumes that Wcs::_nWcsInfo is 1.

When the stacking program tries bcasting Wcs:
    - In serializing Wcs, the value _nWcsInfo = 2 is recorded and so read in 
deserialization.
    - But in the deserialization, the formatter allocates only a single 
element of _wcsInfo.

It causes inconsistency at the destructor, and SEGV arrises.
'''

The example file above has been copied and is used in the below test.
"""

import os, os.path
import unittest
import pickle

import lsst.afw.image as afwImage
import lsst.utils.tests as utilsTests

DATA = os.path.join("tests", "data", "ticket2233.fits")


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
class WcsFormatterTest(unittest.TestCase):
    """Test the WCS formatter, by round-trip pickling."""
    def setUp(self):
        exposure = afwImage.ExposureF(DATA)
        self.wcs = exposure.getWcs()        

    def tearDown(self):
        del self.wcs

    def testFormat(self):
        dumped = pickle.dumps(self.wcs)
        wcs = pickle.loads(dumped)
        self.assertEqual(wcs.getFitsMetadata().toString(), self.wcs.getFitsMetadata().toString())


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(WcsFormatterTest)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

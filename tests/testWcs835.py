#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

import unittest
import lsst.utils.tests as tests
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
import lsst.pex.exceptions as pexExcept


class TanSipTestCases(unittest.TestCase):
    """Tests for the existence of the bug reported in #835
       (Wcs class doesn't gracefully handle the case of ctypes
       having -SIP appended to them).
    """
    
    def setUp(self):
        #metadata taken from CFHT data
        #v695856-e0/v695856-e0-c000-a00.sci_img.fits

        metadata = dafBase.PropertySet()

        metadata.set("SIMPLE",                    "T") 
        metadata.set("BITPIX",                  -32) 
        metadata.set("NAXIS",                    2) 
        metadata.set("NAXIS1",                 1024) 
        metadata.set("NAXIS2",                 1153) 
        metadata.set("RADECSYS", 'FK5')
        metadata.set("EQUINOX",                2000.)


        metadata.setDouble("CRVAL1",     215.604025685476)
        metadata.setDouble("CRVAL2",     53.1595451514076)
        metadata.setDouble("CRPIX1",     1109.99981456774)
        metadata.setDouble("CRPIX2",     560.018167811613)
        metadata.set("CTYPE1", 'RA---TAN-SIP')
        metadata.set("CTYPE2", 'DEC--TAN-SIP')

        metadata.setDouble("CD1_1", 5.10808596133527E-05)
        metadata.setDouble("CD1_2", 1.85579539217196E-07)
        metadata.setDouble("CD2_2", -5.10281493481982E-05)
        metadata.setDouble("CD2_1", -8.27440751733828E-07)


        self.metadata = metadata


    def tearDown(self):
        del self.metadata

    def testExcept(self):
        self.assertRaises(pexExcept.Exception, afwImage.makeWcs, self.metadata)

#####
        
def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(TanSipTestCases)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

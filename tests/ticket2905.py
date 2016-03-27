#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

import os, os.path
import unittest

import lsst.afw.image as afwImage
import lsst.utils.tests as utilsTests


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
class Ticket2905Test(unittest.TestCase):
    """Test reading a FITS header that contains:

    INR-STR =                2E-05
    """

    def test(self):
        path = os.path.join("tests", "data", "ticket2905.fits")
        md = afwImage.readMetadata(path)
        value = md.get("INR-STR")
        self.assertEqual(type(value), float)
        self.assertEqual(value, 2.0e-5)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(Ticket2905Test)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

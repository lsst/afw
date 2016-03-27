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

DATA = os.path.join("tests", "data", "ticket2352.fits")


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
class ReadMefTest(unittest.TestCase):
    """Test the reading of a multi-extension FITS (MEF) file"""
    def checkExtName(self, name, value, extNum):
        filename = DATA + "[%s]" % name

        header = afwImage.readMetadata(filename)
        self.assertEqual(header.get("EXT_NUM"), extNum)
        self.assertEqual(header.get("EXTNAME").strip(), name)

        image = afwImage.ImageI(filename)
        self.assertEqual(image.get(0,0), value)

    def testExtName(self):
        self.checkExtName("ONE", 1, 2)
        self.checkExtName("TWO", 2, 3)
        self.checkExtName("THREE", 3, 4)

    def checkExtNum(self, hdu, extNum):
        header = afwImage.readMetadata(DATA, hdu)
        self.assertEqual(header.get("EXT_NUM"), extNum)

    def testExtNum(self):
        self.checkExtNum(0, 2) # Should skip PHU
        self.checkExtNum(1, 1)
        self.checkExtNum(2, 2)
        self.checkExtNum(3, 3)
        self.checkExtNum(4, 4)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ReadMefTest)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

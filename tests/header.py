#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

import numpy
import unittest

import lsst.afw.image as afwImage
import lsst.utils.tests as utilsTests


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
class HeaderTestCase(unittest.TestCase):
    """Test that headers round-trip"""
    def testHeaders(self):
        filename = "tests/header.fits"
        header = {"STR": "String",
                  "INT": 12345,
                  "FLOAT": 678.9,
                  "NAN": numpy.nan,
                  "PLUSINF": numpy.inf,
                  "MINUSINF": -numpy.inf,
                  "LONG": long(987654321),
                  }

        exp = afwImage.ExposureI(0,0)
        metadata = exp.getMetadata()
        for k,v in header.items():
            metadata.add(k, v)

        exp.writeFits(filename)

        exp = afwImage.ExposureI(filename)
        metadata = exp.getMetadata()
        for k,v in header.items():
            self.assertTrue(metadata.exists(k))
            if isinstance(v, float) and numpy.isnan(v):
                self.assertTrue(isinstance(metadata.get(k), float))
                self.assertTrue(numpy.isnan(metadata.get(k)))
            else:
                self.assertEqual(metadata.get(k), v)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(HeaderTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

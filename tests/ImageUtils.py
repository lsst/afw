#!/usr/bin/env python
"""
Tests for ImageUtils

Run with:
   ./ImageUtils.py
or
   python
   >>> import Image; Image.run()
"""

import os
import pdb  # we may want to say pdb.set_trace()
import sys
import numpy
import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.daf.base
import lsst.afw.image as afwImage
import eups
import lsst.afw.display.ds9 as ds9

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ImageUtilsTestCase(unittest.TestCase):
    """A test case for Image"""
    def testIntIndexToPosition(self):
        for ind in range(-10, 10):
            pos = afwImage.indexToPosition(ind)
            self.assertEqual(pos, float(ind + afwImage.PixelZeroPos))
            ind2 = afwImage.positionToIndex(pos)
            self.assertEqual(float(ind), ind2)

    def testFloatIndexToPosition(self):
        for floatInd in numpy.arange(-10.0, 10.0):
            pos = afwImage.indexToPosition(floatInd)
            self.assertEqual(pos, floatInd + afwImage.PixelZeroPos)
            ind2 = afwImage.positionToIndex(pos)
            self.assertEqual(floatInd, ind2)

    def disabledTestPairPositionToIndex(self):
        """Test version of PositionToIndex that returns a pair
        only it cannot be called from Python
        """
        for basicFloatPos in numpy.arange(-10.0, 10.0):
            for offset in [-0.00000000001, 0, 0.00000000001]:
                floatPos = basicFloatPos + offset
                indResidPair = positionToIndex(floatPos, True)
                # this cannot be unpacked so forget it

        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ImageUtilsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

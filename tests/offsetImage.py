#!/usr/bin/env python
"""
Tests for offsetting images in (dx, dy)

Run with:
   python offsetImage.py
or
   python
   >>> import offsetImage; offsetImage.run()
"""

import os
import pdb  # we may want to say pdb.set_trace()
import sys
import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.daf.base
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math.mathLib as afwMath
import lsst.afw.display.ds9 as ds9

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class offsetImageTestCase(unittest.TestCase):
    """A test case for offsetImage"""

    def setUp(self):
        self.inImage = afwImage.ImageF(200, 100)
        self.background = 200
        self.inImage.set(self.background);
        self.algorithm = "lanczos5"

    def tearDown(self):
        del self.inImage

    def testSetFluxConvervation(self):
        """Test that flux is preserved"""

        outImage = afwMath.offsetImage(self.inImage, 0, 0, self.algorithm)
        self.assertEqual(outImage.get(50, 50), self.background)

        outImage = afwMath.offsetImage(self.inImage, 0.5, 0, self.algorithm)
        self.assertAlmostEqual(outImage.get(50, 50), self.background, 4)

        outImage = afwMath.offsetImage(self.inImage, 0.5, 0.5, self.algorithm)
        self.assertAlmostEqual(outImage.get(50, 50), self.background, 4)

    def testSetIntegerOffset(self):
        """Test that we can offset by positive and negative amounts"""
        
        self.inImage.set(50, 50, 400);

        if display:
            frame = 0
            ds9.mtv(self.inImage, frame=frame)
            ds9.pan(50, 50, frame=frame);
            ds9.dot("+", 50, 50, frame=frame)

        for delta in [-0.49, 0.51]:
            for dx, dy in [(2, 3), (-2, 3), (-2, -3), (2, -3)]:
                outImage = afwMath.offsetImage(self.inImage, dx + delta, dy + delta, self.algorithm)
                
                if display:
                    frame += 1
                    ds9.mtv(outImage, frame=frame)
                    ds9.pan(50, 50, frame=frame);
                    ds9.dot("+", 50 + dx + delta - outImage.getX0(), 50 + dy + delta - outImage.getY0(),
                            frame=frame)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(offsetImageTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

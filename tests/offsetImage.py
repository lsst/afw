#!/usr/bin/env python
"""
Tests for offsetting images in (dx, dy)

Run with:
   python offsetImage.py
or
   python
   >>> import offsetImage; offsetImage.run()
"""
import math

import unittest
import numpy

import lsst.utils.tests as utilsTests
import lsst.daf.base
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math.mathLib as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.afw.image.testUtils as imTestUtils

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
        self.inImage.set(self.background)
        self.algorithm = "lanczos5"

    def tearDown(self):
        del self.inImage

    def XXXtestSetFluxConvervation(self):
        """Test that flux is preserved"""

        outImage = afwMath.offsetImage(self.inImage, 0, 0, self.algorithm)
        self.assertEqual(outImage.get(50, 50), self.background)

        outImage = afwMath.offsetImage(self.inImage, 0.5, 0, self.algorithm)
        self.assertAlmostEqual(outImage.get(50, 50), self.background, 4)

        outImage = afwMath.offsetImage(self.inImage, 0.5, 0.5, self.algorithm)
        self.assertAlmostEqual(outImage.get(50, 50), self.background, 4)

    def XXXtestSetIntegerOffset(self):
        """Test that we can offset by positive and negative amounts"""
        
        self.inImage.set(50, 50, 400)

        if False and display:
            frame = 0
            ds9.mtv(self.inImage, frame=frame)
            ds9.pan(50, 50, frame=frame)
            ds9.dot("+", 50, 50, frame=frame)

        for delta in [-0.49, 0.51]:
            for dx, dy in [(2, 3), (-2, 3), (-2, -3), (2, -3)]:
                outImage = afwMath.offsetImage(self.inImage, dx + delta, dy + delta, self.algorithm)
                
                if False and display:
                    frame += 1
                    ds9.mtv(outImage, frame=frame)
                    ds9.pan(50, 50, frame=frame)
                    ds9.dot("+", 50 + dx + delta - outImage.getX0(), 50 + dy + delta - outImage.getY0(),
                            frame=frame)

    def calcGaussian(self, im, x, y, amp, sigma1):
        """Insert a Gaussian into the image centered at (x, y)"""

        x = x - im.getX0()
        y = y - im.getY0()

        for ix in range(im.getWidth()):
            for iy in range(im.getHeight()):
                r2 = math.pow(x - ix, 2) + math.pow(y - iy, 2)
                val = math.exp(-r2/(2.0*pow(sigma1, 2)))
                im.set(ix, iy, amp*val)

    def testOffsetGaussian(self):
        """Insert a Gaussian, offset, and check the residuals"""
        size = 100
        im = afwImage.ImageF(size, size)

        xc, yc = size/2.0, size/2.0

        amp, sigma1 = 1.0, 3
        #
        # Calculate an image with a Gaussian at (xc -dx, yc - dy) and then shift it to (xc, yc)
        #
        dx, dy = 0.5, -0.5
        self.calcGaussian(im, xc - dx, yc - dy, amp, sigma1)
        im2 = afwMath.offsetImage(im, dx, dy, "lanczos5")
        #
        # Calculate Gaussian directly at (xc, yc)
        #
        self.calcGaussian(im, xc, yc, amp, sigma1)
        #
        # See how they differ
        #
        if display:
            ds9.mtv(im, frame=0)

        im -= im2

        if display:
            ds9.mtv(im, frame=1)

        imArr = imTestUtils.arrayFromImage(im)
        imGoodVals = numpy.ma.array(imArr, copy=False, mask=numpy.isnan(imArr)).compressed()
        imMean = imGoodVals.mean()
        imMax = imGoodVals.max()
        imMin = imGoodVals.min()

        if not False:
            print "mean = %g, min = %g, max = %g" % (imMean, imMin, imMax)
            
        self.assertTrue(abs(imMean) < 1e-7)
        self.assertTrue(abs(imMin) < 1.2e-3*amp)
        self.assertTrue(abs(imMax) < 1.2e-3*amp)

# the following would be preferable if there was an easy way to NaN pixels
#
#         stats = afwMath.makeStatistics(im, afwMath.MEAN | afwMath.MAX | afwMath.MIN)
# 
#         if not False:
#             print "mean = %g, min = %g, max = %g" % (stats.getValue(afwMath.MEAN),
#                                                      stats.getValue(afwMath.MIN),
#                                                      stats.getValue(afwMath.MAX))
#             
#         self.assertTrue(abs(stats.getValue(afwMath.MEAN)) < 1e-7)
#         self.assertTrue(abs(stats.getValue(afwMath.MIN)) < 1.2e-3*amp)
#         self.assertTrue(abs(stats.getValue(afwMath.MAX)) < 1.2e-3*amp)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(offsetImageTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

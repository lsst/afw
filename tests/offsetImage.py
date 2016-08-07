#!/usr/bin/env python2
from __future__ import absolute_import, division
from __future__ import print_function
from builtins import range

#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

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
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
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
        self.inImage.set(self.background)

    def tearDown(self):
        del self.inImage

    def testSetFluxConvervation(self):
        """Test that flux is preserved"""

        for algorithm in ("lanczos5", "bilinear", "nearest"):
            outImage = afwMath.offsetImage(self.inImage, 0, 0, algorithm)
            self.assertEqual(outImage.get(50, 50), self.background)

            outImage = afwMath.offsetImage(self.inImage, 0.5, 0, algorithm)
            self.assertAlmostEqual(outImage.get(50, 50), self.background, 4)

            outImage = afwMath.offsetImage(self.inImage, 0.5, 0.5, algorithm)
            self.assertAlmostEqual(outImage.get(50, 50), self.background, 4)

    def testSetIntegerOffset(self):
        """Test that we can offset by positive and negative amounts"""

        self.inImage.set(50, 50, 400)

        if False and display:
            frame = 0
            ds9.mtv(self.inImage, frame=frame)
            ds9.pan(50, 50, frame=frame)
            ds9.dot("+", 50, 50, frame=frame)

        for algorithm in ("lanczos5", "bilinear", "nearest"):
            for delta in [-0.49, 0.51]:
                for dx, dy in [(2, 3), (-2, 3), (-2, -3), (2, -3)]:
                    outImage = afwMath.offsetImage(self.inImage, dx + delta, dy + delta, algorithm)

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

        size = 50
        refIm = afwImage.ImageF(size, size)
        unshiftedIm = afwImage.ImageF(size, size)

        xc, yc = size/2.0, size/2.0

        amp, sigma1 = 1.0, 3

        #
        # Calculate Gaussian directly at (xc, yc)
        #
        self.calcGaussian(refIm, xc, yc, amp, sigma1)

        for dx in (-55.5, -1.500001, -1.5, -1.499999, -1.00001, -1.0, -0.99999, -0.5,
            0.0, 0.5, 0.99999, 1.0, 1.00001, 1.499999, 1.5, 1.500001, 99.3):
            for dy in (-3.7, -1.500001, -1.5, -1.499999, -1.00001, -1.0, -0.99999, -0.5,
                0.0, 0.5, 0.99999, 1.0, 1.00001, 1.499999, 1.5, 1.500001, 2.99999):
                dOrigX, dOrigY, dFracX, dFracY = getOrigFracShift(dx, dy)
                self.calcGaussian(unshiftedIm, xc - dFracX, yc - dFracY, amp, sigma1)

                for algorithm, maxMean, maxLim in (
                    ("lanczos5", 1e-8, 0.0015),
                    ("bilinear", 1e-8, 0.03),
                    ("nearest",  1e-8, 0.2),
                ):
                    im = afwImage.ImageF(size, size)
                    im = afwMath.offsetImage(unshiftedIm, dx, dy, algorithm)


                    if display:
                        ds9.mtv(im, frame=0)

                    im -= refIm

                    if display:
                        ds9.mtv(im, frame=1)

                    imArr = im.getArray()
                    imGoodVals = numpy.ma.array(imArr, copy=False, mask=numpy.isnan(imArr)).compressed()

                    try:
                        imXY0 = tuple(im.getXY0())
                        self.assertEqual(imXY0, (dOrigX, dOrigY))
                        self.assertLess(abs(imGoodVals.mean()), maxMean*amp)
                        self.assertLess(abs(imGoodVals.max()), maxLim*amp)
                        self.assertLess(abs(imGoodVals.min()), maxLim*amp)
                    except:
                        print("failed on algorithm=%s; dx = %s; dy = %s" % (algorithm, dx, dy))
                        raise

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


def getOrigFracShift(dx, dy):
    """Return the predicted integer shift to XY0 and the fractional shift that offsetImage will use

    offsetImage preserves the origin if dx and dy both < 1 pixel; larger shifts are to the nearest pixel.
    """
    if (abs(dx) < 1) and (abs(dy) < 1):
        return (0, 0, dx, dy)

    dOrigX = math.floor(dx + 0.5)
    dOrigY = math.floor(dy + 0.5)
    dFracX = dx - dOrigX
    dFracY = dy - dOrigY
    return (int(dOrigX), int(dOrigY), dFracX, dFracY)


class transformImageTestCase(unittest.TestCase):
    """A test case for rotating images"""

    def setUp(self):
        self.inImage = afwImage.ImageF(20, 10)
        self.inImage.set(0, 0, 100)
        self.inImage.set(10, 0, 50)

    def tearDown(self):
        del self.inImage

    def testRotate(self):
        """Test that we end up with the correct image after rotating by 90 degrees"""

        for nQuarter, x, y in [(0, 0, 0),
                               (1, 9, 0),
                               (2, 19, 9),
                               (3, 0, 19)]:
            outImage = afwMath.rotateImageBy90(self.inImage, nQuarter)
            if display:
                ds9.mtv(outImage, frame=nQuarter, title="out %d" % nQuarter)
            self.assertEqual(self.inImage.get(0, 0), outImage.get(x, y))

    def testFlip(self):
        """Test that we end up with the correct image after flipping it"""

        frame = 2
        for flipLR, flipTB, x, y in [(True, False, 19, 0),
                                     (True, True,  19, 9),
                                     (False, True, 0,  9),
                                     (False, False, 0, 0)]:
            outImage = afwMath.flipImage(self.inImage, flipLR, flipTB)
            if display:
                ds9.mtv(outImage, frame=frame, title="%s %s" % (flipLR, flipTB))
                frame += 1
            self.assertEqual(self.inImage.get(0, 0), outImage.get(x, y))

    def testMask(self):
        """Test that we can flip a Mask"""
        mask = afwImage.MaskU(10, 20)
        afwMath.flipImage(mask, True, False) # for a while, swig couldn't handle the resulting Mask::Ptr

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class binImageTestCase(unittest.TestCase):
    """A test case for binning images"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testBin(self):
        """Test that we can bin images"""

        inImage = afwImage.ImageF(203, 131)
        inImage.set(1)
        bin = 4

        outImage = afwMath.binImage(inImage, bin)

        self.assertEqual(outImage.getWidth(), inImage.getWidth()//bin)
        self.assertEqual(outImage.getHeight(), inImage.getHeight()//bin)

        stats = afwMath.makeStatistics(outImage, afwMath.MAX | afwMath.MIN)
        self.assertEqual(stats.getValue(afwMath.MIN), 1)
        self.assertEqual(stats.getValue(afwMath.MAX), 1)

    def testBin2(self):
        """Test that we can bin images anisotropically"""

        inImage = afwImage.ImageF(203, 131)
        val = 1
        inImage.set(val)
        binX, binY = 2, 4

        outImage = afwMath.binImage(inImage, binX, binY)

        self.assertEqual(outImage.getWidth(), inImage.getWidth()//binX)
        self.assertEqual(outImage.getHeight(), inImage.getHeight()//binY)

        stats = afwMath.makeStatistics(outImage, afwMath.MAX | afwMath.MIN)
        self.assertEqual(stats.getValue(afwMath.MIN), val)
        self.assertEqual(stats.getValue(afwMath.MAX), val)

        inImage.set(0)
        subImg = inImage.Factory(inImage, afwGeom.BoxI(afwGeom.PointI(4, 4), afwGeom.ExtentI(4, 8)),
                                 afwImage.LOCAL)
        subImg.set(100)
        del subImg
        outImage = afwMath.binImage(inImage, binX, binY)

        if display:
            ds9.mtv(inImage, frame=2, title="unbinned")
            ds9.mtv(outImage, frame=3, title="binned %dx%d" % (binX, binY))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(offsetImageTestCase)
    suites += unittest.makeSuite(transformImageTestCase)
    suites += unittest.makeSuite(binImageTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

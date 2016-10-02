#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11#"""
#pybind11#Tests for offsetting images in (dx, dy)
#pybind11#
#pybind11#Run with:
#pybind11#   python offsetImage.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import offsetImage; offsetImage.run()
#pybind11#"""
#pybind11#import math
#pybind11#
#pybind11#import unittest
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class offsetImageTestCase(unittest.TestCase):
#pybind11#    """A test case for offsetImage"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.inImage = afwImage.ImageF(200, 100)
#pybind11#        self.background = 200
#pybind11#        self.inImage.set(self.background)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.inImage
#pybind11#
#pybind11#    def testSetFluxConvervation(self):
#pybind11#        """Test that flux is preserved"""
#pybind11#
#pybind11#        for algorithm in ("lanczos5", "bilinear", "nearest"):
#pybind11#            outImage = afwMath.offsetImage(self.inImage, 0, 0, algorithm)
#pybind11#            self.assertEqual(outImage.get(50, 50), self.background)
#pybind11#
#pybind11#            outImage = afwMath.offsetImage(self.inImage, 0.5, 0, algorithm)
#pybind11#            self.assertAlmostEqual(outImage.get(50, 50), self.background, 4)
#pybind11#
#pybind11#            outImage = afwMath.offsetImage(self.inImage, 0.5, 0.5, algorithm)
#pybind11#            self.assertAlmostEqual(outImage.get(50, 50), self.background, 4)
#pybind11#
#pybind11#    def testSetIntegerOffset(self):
#pybind11#        """Test that we can offset by positive and negative amounts"""
#pybind11#
#pybind11#        self.inImage.set(50, 50, 400)
#pybind11#
#pybind11#        if False and display:
#pybind11#            frame = 0
#pybind11#            ds9.mtv(self.inImage, frame=frame)
#pybind11#            ds9.pan(50, 50, frame=frame)
#pybind11#            ds9.dot("+", 50, 50, frame=frame)
#pybind11#
#pybind11#        for algorithm in ("lanczos5", "bilinear", "nearest"):
#pybind11#            for delta in [-0.49, 0.51]:
#pybind11#                for dx, dy in [(2, 3), (-2, 3), (-2, -3), (2, -3)]:
#pybind11#                    outImage = afwMath.offsetImage(self.inImage, dx + delta, dy + delta, algorithm)
#pybind11#
#pybind11#                    if False and display:
#pybind11#                        frame += 1
#pybind11#                        ds9.mtv(outImage, frame=frame)
#pybind11#                        ds9.pan(50, 50, frame=frame)
#pybind11#                        ds9.dot("+", 50 + dx + delta - outImage.getX0(), 50 + dy + delta - outImage.getY0(),
#pybind11#                                frame=frame)
#pybind11#
#pybind11#    def calcGaussian(self, im, x, y, amp, sigma1):
#pybind11#        """Insert a Gaussian into the image centered at (x, y)"""
#pybind11#
#pybind11#        x = x - im.getX0()
#pybind11#        y = y - im.getY0()
#pybind11#
#pybind11#        for ix in range(im.getWidth()):
#pybind11#            for iy in range(im.getHeight()):
#pybind11#                r2 = math.pow(x - ix, 2) + math.pow(y - iy, 2)
#pybind11#                val = math.exp(-r2/(2.0*pow(sigma1, 2)))
#pybind11#                im.set(ix, iy, amp*val)
#pybind11#
#pybind11#    def testOffsetGaussian(self):
#pybind11#        """Insert a Gaussian, offset, and check the residuals"""
#pybind11#
#pybind11#        size = 50
#pybind11#        refIm = afwImage.ImageF(size, size)
#pybind11#        unshiftedIm = afwImage.ImageF(size, size)
#pybind11#
#pybind11#        xc, yc = size/2.0, size/2.0
#pybind11#
#pybind11#        amp, sigma1 = 1.0, 3
#pybind11#
#pybind11#        #
#pybind11#        # Calculate Gaussian directly at (xc, yc)
#pybind11#        #
#pybind11#        self.calcGaussian(refIm, xc, yc, amp, sigma1)
#pybind11#
#pybind11#        for dx in (-55.5, -1.500001, -1.5, -1.499999, -1.00001, -1.0, -0.99999, -0.5,
#pybind11#                   0.0, 0.5, 0.99999, 1.0, 1.00001, 1.499999, 1.5, 1.500001, 99.3):
#pybind11#            for dy in (-3.7, -1.500001, -1.5, -1.499999, -1.00001, -1.0, -0.99999, -0.5,
#pybind11#                       0.0, 0.5, 0.99999, 1.0, 1.00001, 1.499999, 1.5, 1.500001, 2.99999):
#pybind11#                dOrigX, dOrigY, dFracX, dFracY = getOrigFracShift(dx, dy)
#pybind11#                self.calcGaussian(unshiftedIm, xc - dFracX, yc - dFracY, amp, sigma1)
#pybind11#
#pybind11#                for algorithm, maxMean, maxLim in (
#pybind11#                    ("lanczos5", 1e-8, 0.0015),
#pybind11#                    ("bilinear", 1e-8, 0.03),
#pybind11#                    ("nearest", 1e-8, 0.2),
#pybind11#                ):
#pybind11#                    im = afwImage.ImageF(size, size)
#pybind11#                    im = afwMath.offsetImage(unshiftedIm, dx, dy, algorithm)
#pybind11#
#pybind11#                    if display:
#pybind11#                        ds9.mtv(im, frame=0)
#pybind11#
#pybind11#                    im -= refIm
#pybind11#
#pybind11#                    if display:
#pybind11#                        ds9.mtv(im, frame=1)
#pybind11#
#pybind11#                    imArr = im.getArray()
#pybind11#                    imGoodVals = numpy.ma.array(imArr, copy=False, mask=numpy.isnan(imArr)).compressed()
#pybind11#
#pybind11#                    try:
#pybind11#                        imXY0 = tuple(im.getXY0())
#pybind11#                        self.assertEqual(imXY0, (dOrigX, dOrigY))
#pybind11#                        self.assertLess(abs(imGoodVals.mean()), maxMean*amp)
#pybind11#                        self.assertLess(abs(imGoodVals.max()), maxLim*amp)
#pybind11#                        self.assertLess(abs(imGoodVals.min()), maxLim*amp)
#pybind11#                    except:
#pybind11#                        print("failed on algorithm=%s; dx = %s; dy = %s" % (algorithm, dx, dy))
#pybind11#                        raise
#pybind11#
#pybind11## the following would be preferable if there was an easy way to NaN pixels
#pybind11##
#pybind11##         stats = afwMath.makeStatistics(im, afwMath.MEAN | afwMath.MAX | afwMath.MIN)
#pybind11##
#pybind11##         if not False:
#pybind11##             print "mean = %g, min = %g, max = %g" % (stats.getValue(afwMath.MEAN),
#pybind11##                                                      stats.getValue(afwMath.MIN),
#pybind11##                                                      stats.getValue(afwMath.MAX))
#pybind11##
#pybind11##         self.assertTrue(abs(stats.getValue(afwMath.MEAN)) < 1e-7)
#pybind11##         self.assertTrue(abs(stats.getValue(afwMath.MIN)) < 1.2e-3*amp)
#pybind11##         self.assertTrue(abs(stats.getValue(afwMath.MAX)) < 1.2e-3*amp)
#pybind11#
#pybind11#
#pybind11#def getOrigFracShift(dx, dy):
#pybind11#    """Return the predicted integer shift to XY0 and the fractional shift that offsetImage will use
#pybind11#
#pybind11#    offsetImage preserves the origin if dx and dy both < 1 pixel; larger shifts are to the nearest pixel.
#pybind11#    """
#pybind11#    if (abs(dx) < 1) and (abs(dy) < 1):
#pybind11#        return (0, 0, dx, dy)
#pybind11#
#pybind11#    dOrigX = math.floor(dx + 0.5)
#pybind11#    dOrigY = math.floor(dy + 0.5)
#pybind11#    dFracX = dx - dOrigX
#pybind11#    dFracY = dy - dOrigY
#pybind11#    return (int(dOrigX), int(dOrigY), dFracX, dFracY)
#pybind11#
#pybind11#
#pybind11#class transformImageTestCase(unittest.TestCase):
#pybind11#    """A test case for rotating images"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.inImage = afwImage.ImageF(20, 10)
#pybind11#        self.inImage.set(0, 0, 100)
#pybind11#        self.inImage.set(10, 0, 50)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.inImage
#pybind11#
#pybind11#    def testRotate(self):
#pybind11#        """Test that we end up with the correct image after rotating by 90 degrees"""
#pybind11#
#pybind11#        for nQuarter, x, y in [(0, 0, 0),
#pybind11#                               (1, 9, 0),
#pybind11#                               (2, 19, 9),
#pybind11#                               (3, 0, 19)]:
#pybind11#            outImage = afwMath.rotateImageBy90(self.inImage, nQuarter)
#pybind11#            if display:
#pybind11#                ds9.mtv(outImage, frame=nQuarter, title="out %d" % nQuarter)
#pybind11#            self.assertEqual(self.inImage.get(0, 0), outImage.get(x, y))
#pybind11#
#pybind11#    def testFlip(self):
#pybind11#        """Test that we end up with the correct image after flipping it"""
#pybind11#
#pybind11#        frame = 2
#pybind11#        for flipLR, flipTB, x, y in [(True, False, 19, 0),
#pybind11#                                     (True, True, 19, 9),
#pybind11#                                     (False, True, 0, 9),
#pybind11#                                     (False, False, 0, 0)]:
#pybind11#            outImage = afwMath.flipImage(self.inImage, flipLR, flipTB)
#pybind11#            if display:
#pybind11#                ds9.mtv(outImage, frame=frame, title="%s %s" % (flipLR, flipTB))
#pybind11#                frame += 1
#pybind11#            self.assertEqual(self.inImage.get(0, 0), outImage.get(x, y))
#pybind11#
#pybind11#    def testMask(self):
#pybind11#        """Test that we can flip a Mask"""
#pybind11#        mask = afwImage.MaskU(10, 20)
#pybind11#        afwMath.flipImage(mask, True, False)  # for a while, swig couldn't handle the resulting Mask::Ptr
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class binImageTestCase(unittest.TestCase):
#pybind11#    """A test case for binning images"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        pass
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        pass
#pybind11#
#pybind11#    def testBin(self):
#pybind11#        """Test that we can bin images"""
#pybind11#
#pybind11#        inImage = afwImage.ImageF(203, 131)
#pybind11#        inImage.set(1)
#pybind11#        bin = 4
#pybind11#
#pybind11#        outImage = afwMath.binImage(inImage, bin)
#pybind11#
#pybind11#        self.assertEqual(outImage.getWidth(), inImage.getWidth()//bin)
#pybind11#        self.assertEqual(outImage.getHeight(), inImage.getHeight()//bin)
#pybind11#
#pybind11#        stats = afwMath.makeStatistics(outImage, afwMath.MAX | afwMath.MIN)
#pybind11#        self.assertEqual(stats.getValue(afwMath.MIN), 1)
#pybind11#        self.assertEqual(stats.getValue(afwMath.MAX), 1)
#pybind11#
#pybind11#    def testBin2(self):
#pybind11#        """Test that we can bin images anisotropically"""
#pybind11#
#pybind11#        inImage = afwImage.ImageF(203, 131)
#pybind11#        val = 1
#pybind11#        inImage.set(val)
#pybind11#        binX, binY = 2, 4
#pybind11#
#pybind11#        outImage = afwMath.binImage(inImage, binX, binY)
#pybind11#
#pybind11#        self.assertEqual(outImage.getWidth(), inImage.getWidth()//binX)
#pybind11#        self.assertEqual(outImage.getHeight(), inImage.getHeight()//binY)
#pybind11#
#pybind11#        stats = afwMath.makeStatistics(outImage, afwMath.MAX | afwMath.MIN)
#pybind11#        self.assertEqual(stats.getValue(afwMath.MIN), val)
#pybind11#        self.assertEqual(stats.getValue(afwMath.MAX), val)
#pybind11#
#pybind11#        inImage.set(0)
#pybind11#        subImg = inImage.Factory(inImage, afwGeom.BoxI(afwGeom.PointI(4, 4), afwGeom.ExtentI(4, 8)),
#pybind11#                                 afwImage.LOCAL)
#pybind11#        subImg.set(100)
#pybind11#        del subImg
#pybind11#        outImage = afwMath.binImage(inImage, binX, binY)
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(inImage, frame=2, title="unbinned")
#pybind11#            ds9.mtv(outImage, frame=3, title="binned %dx%d" % (binX, binY))
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

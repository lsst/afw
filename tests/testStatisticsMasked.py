#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
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
#pybind11#Tests for statisticsMasked
#pybind11#
#pybind11#Run with:
#pybind11#   ./statisticsMasked.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import statisticsMasked; statisticsMasked.run()
#pybind11#"""
#pybind11#
#pybind11#
#pybind11#import unittest
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.math as afwMath
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#class StatisticsTestCase(unittest.TestCase):
#pybind11#
#pybind11#    """A test case to check that special values (NaN and Masks) are begin handled in Statistics"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.valL, self.valR = 10, 20
#pybind11#        self.nRow, self.nCol = 100, 200
#pybind11#        self.n = self.nRow*self.nCol
#pybind11#
#pybind11#        self.bboxL = afwGeom.Box2I(afwGeom.Point2I(0, 0),
#pybind11#                                   afwGeom.Point2I(self.nRow//2 - 1, self.nCol - 1))
#pybind11#        self.bboxR = afwGeom.Box2I(afwGeom.Point2I(self.nRow//2, 0),
#pybind11#                                   afwGeom.Point2I(self.nRow - 1, self.nCol - 1))
#pybind11#
#pybind11#        # create masked images and set the left side to valL, and right to valR
#pybind11#        self.mimg = afwImage.MaskedImageF(afwGeom.Extent2I(self.nRow, self.nCol))
#pybind11#        self.mimg.set(0.0, 0x0, 0.0)
#pybind11#        self.mimgL = afwImage.MaskedImageF(self.mimg, self.bboxL, afwImage.LOCAL)
#pybind11#        self.mimgL.set(self.valL, 0x0, self.valL)
#pybind11#        self.mimgR = afwImage.MaskedImageF(self.mimg, self.bboxR, afwImage.LOCAL)
#pybind11#        self.mimgR.set(self.valR, 0x0, self.valR)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.mimg
#pybind11#        del self.mimgL
#pybind11#        del self.mimgR
#pybind11#
#pybind11#    # Verify that NaN values are being ignored
#pybind11#    # (by default, StatisticsControl.useNanSafe = True)
#pybind11#    # We'll set the L and R sides of an image to two different values and verify mean and stdev
#pybind11#    # ... then set R-side to NaN and try again ... we should get mean,stdev for L-side
#pybind11#    def testNaN(self):
#pybind11#
#pybind11#        # get the stats for the image with two values
#pybind11#        stats = afwMath.makeStatistics(self.mimg, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV)
#pybind11#        mean = 0.5*(self.valL + self.valR)
#pybind11#        nL, nR = self.mimgL.getWidth()*self.mimgL.getHeight(), self.mimgR.getWidth()*self.mimgR.getHeight()
#pybind11#        stdev = ((nL*(self.valL - mean)**2 + nR*(self.valR - mean)**2)/(nL + nR - 1))**0.5
#pybind11#
#pybind11#        self.assertEqual(stats.getValue(afwMath.NPOINT), self.n)
#pybind11#        self.assertEqual(stats.getValue(afwMath.MEAN), mean)
#pybind11#        self.assertEqual(stats.getValue(afwMath.STDEV), stdev)
#pybind11#
#pybind11#        # set the right side to NaN and stats should be just for the left side
#pybind11#        self.mimgR.set(numpy.nan, 0x0, self.valR)
#pybind11#
#pybind11#        statsNaN = afwMath.makeStatistics(self.mimg, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV)
#pybind11#        mean = self.valL
#pybind11#        stdev = 0.0
#pybind11#        self.assertEqual(statsNaN.getValue(afwMath.NPOINT), nL)
#pybind11#        self.assertEqual(statsNaN.getValue(afwMath.MEAN), mean)
#pybind11#        self.assertEqual(statsNaN.getValue(afwMath.STDEV), stdev)
#pybind11#
#pybind11#    # Verify that Masked pixels are being ignored according to the andMask
#pybind11#    # (by default, StatisticsControl.andMask = 0x0)
#pybind11#    # We'll set the L and R sides of an image to two different values and verify mean and stdev
#pybind11#    # ... then set R-side Mask and the andMask to 0x1 and try again ... we should get mean,stdev for L-side
#pybind11#    def testMasked(self):
#pybind11#
#pybind11#        # get the stats for the image with two values
#pybind11#        self.mimgR.set(self.valR, 0x0, self.valR)
#pybind11#        stats = afwMath.makeStatistics(self.mimg, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV)
#pybind11#        mean = 0.5*(self.valL + self.valR)
#pybind11#        nL, nR = self.mimgL.getWidth()*self.mimgL.getHeight(), self.mimgR.getWidth()*self.mimgR.getHeight()
#pybind11#        stdev = ((nL*(self.valL - mean)**2 + nR*(self.valR - mean)**2)/(nL + nR - 1))**0.5
#pybind11#
#pybind11#        self.assertEqual(stats.getValue(afwMath.NPOINT), self.n)
#pybind11#        self.assertEqual(stats.getValue(afwMath.MEAN), mean)
#pybind11#        self.assertEqual(stats.getValue(afwMath.STDEV), stdev)
#pybind11#
#pybind11#        # set the right side Mask and the StatisticsControl andMask to 0x1
#pybind11#        #  Stats should be just for the left side!
#pybind11#        maskBit = 0x1
#pybind11#        self.mimgR.getMask().set(maskBit)
#pybind11#
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#        sctrl.setAndMask(maskBit)
#pybind11#        statsNaN = afwMath.makeStatistics(self.mimg, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV, sctrl)
#pybind11#
#pybind11#        mean = self.valL
#pybind11#        stdev = 0.0
#pybind11#
#pybind11#        self.assertEqual(statsNaN.getValue(afwMath.NPOINT), nL)
#pybind11#        self.assertEqual(statsNaN.getValue(afwMath.MEAN), mean)
#pybind11#        self.assertEqual(statsNaN.getValue(afwMath.STDEV), stdev)
#pybind11#
#pybind11#    # Verify that pixels are being weighted according to the variance plane (1/var)
#pybind11#    # We'll set the L and R sides of an image to two different values and verify mean
#pybind11#    # ... then set R-side Variance to equal the Image value, and set 'weighted' and try again ...
#pybind11#    def testWeighted(self):
#pybind11#
#pybind11#        self.mimgR.set(self.valR, 0x0, self.valR)
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#        sctrl.setWeighted(True)
#pybind11#        stats = afwMath.makeStatistics(self.mimg, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV, sctrl)
#pybind11#        nL, nR = self.mimgL.getWidth()*self.mimgL.getHeight(), self.mimgR.getWidth()*self.mimgR.getHeight()
#pybind11#
#pybind11#        mean = 1.0*(nL + nR)/(nL/self.valL + nR/self.valR)
#pybind11#
#pybind11#        # get the stats for the image with two values
#pybind11#        self.assertEqual(stats.getValue(afwMath.NPOINT), self.n)
#pybind11#        self.assertAlmostEqual(stats.getValue(afwMath.MEAN), mean, 10)
#pybind11#
#pybind11#    def testWeightedSimple(self):
#pybind11#        mimg = afwImage.MaskedImageF(afwGeom.Extent2I(1, 2))
#pybind11#        mimg.set(0, 0, (self.valR, 0x0, self.valR))
#pybind11#        mimg.set(0, 1, (self.valL, 0x0, self.valL))
#pybind11#
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#        sctrl.setWeighted(True)
#pybind11#        stats = afwMath.makeStatistics(mimg, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV, sctrl)
#pybind11#        vsum = 2.0
#pybind11#        vsum2 = self.valR + self.valL
#pybind11#        wsum = 1.0/self.valR + 1.0/self.valL
#pybind11#        wwsum = 1.0/self.valR**2 + 1.0/self.valL**2
#pybind11#        mean = vsum/wsum
#pybind11#        variance = vsum2/wsum - mean**2  # biased variance
#pybind11#
#pybind11#        n = 2
#pybind11#        # original estimate; just a rewrite of the usual n/(n - 1) correction
#pybind11#        stddev = (1.0*(vsum2)/(wsum*(1.0-1.0/n)) - (vsum**2)/(wsum**2*(1.0-1.0/n)))**0.5
#pybind11#        self.assertAlmostEqual(stddev, numpy.sqrt(variance*n/(n - 1)))
#pybind11#        #
#pybind11#        # The correct formula:
#pybind11#        stddev = numpy.sqrt(variance*wsum**2/(wsum**2 - wwsum))
#pybind11#
#pybind11#        # get the stats for the image with two values
#pybind11#        self.assertAlmostEqual(stats.getValue(afwMath.MEAN), mean, 10)
#pybind11#        self.assertAlmostEqual(stats.getValue(afwMath.STDEV), stddev, 10)
#pybind11#
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

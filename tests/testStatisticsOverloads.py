#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
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
#pybind11#Tests for Statistics
#pybind11#
#pybind11#Run with:
#pybind11#   ./statisticsOverloads.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import statisticsOverloads; statisticsOverloads.run()
#pybind11#"""
#pybind11#
#pybind11#
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.math as afwMath
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class StatisticsTestCase(unittest.TestCase):
#pybind11#
#pybind11#    """A test case to check all overloaded makeStatistics() factories for Statistics"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.val = 10
#pybind11#        self.nRow, self.nCol = 100, 200
#pybind11#        self.sctrl = afwMath.StatisticsControl()
#pybind11#
#pybind11#        # Integers
#pybind11#        self.mimgI = afwImage.MaskedImageI(afwGeom.Extent2I(self.nRow, self.nCol))
#pybind11#        self.mimgI.set(self.val, 0x0, self.val)
#pybind11#        self.imgI = afwImage.ImageI(afwGeom.Extent2I(self.nRow, self.nCol), self.val)
#pybind11#        self.vecI = afwMath.vectorI(self.nRow*self.nCol, self.val)
#pybind11#
#pybind11#        # floats
#pybind11#        self.mimgF = afwImage.MaskedImageF(afwGeom.Extent2I(self.nRow, self.nCol))
#pybind11#        self.mimgF.set(self.val, 0x0, self.val)
#pybind11#        self.imgF = afwImage.ImageF(afwGeom.Extent2I(self.nRow, self.nCol), self.val)
#pybind11#        self.vecF = afwMath.vectorF(self.nRow*self.nCol, self.val)
#pybind11#
#pybind11#        # doubles
#pybind11#        self.mimgD = afwImage.MaskedImageD(afwGeom.Extent2I(self.nRow, self.nCol))
#pybind11#        self.mimgD.set(self.val, 0x0, self.val)
#pybind11#        self.imgD = afwImage.ImageD(afwGeom.Extent2I(self.nRow, self.nCol), self.val)
#pybind11#        self.vecD = afwMath.vectorD(self.nRow*self.nCol, self.val)
#pybind11#
#pybind11#        self.imgList = [self.imgI, self.imgF, self.imgD]
#pybind11#        self.mimgList = [self.mimgI, self.mimgF, self.mimgD]
#pybind11#        self.vecList = [self.vecI, self.vecF, self.vecD]
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.mimgI
#pybind11#        del self.mimgF
#pybind11#        del self.mimgD
#pybind11#        del self.imgI
#pybind11#        del self.imgF
#pybind11#        del self.imgD
#pybind11#        del self.vecI
#pybind11#        del self.vecF
#pybind11#        del self.vecD
#pybind11#
#pybind11#        del self.mimgList
#pybind11#        del self.imgList
#pybind11#        del self.vecList
#pybind11#
#pybind11#    # The guts of the testing: grab a mean, stddev, and sum for whatever you're called with
#pybind11#    def compareMakeStatistics(self, image, n):
#pybind11#        stats = afwMath.makeStatistics(image, afwMath.NPOINT | afwMath.STDEV |
#pybind11#                                       afwMath.MEAN | afwMath.SUM, self.sctrl)
#pybind11#
#pybind11#        self.assertEqual(stats.getValue(afwMath.NPOINT), n)
#pybind11#        self.assertEqual(stats.getValue(afwMath.NPOINT)*stats.getValue(afwMath.MEAN),
#pybind11#                         stats.getValue(afwMath.SUM))
#pybind11#        self.assertEqual(stats.getValue(afwMath.MEAN), self.val)
#pybind11#        self.assertEqual(stats.getValue(afwMath.STDEV), 0)
#pybind11#
#pybind11#    # same as compareMakeStatistics but calls constructor directly (only for masked image)
#pybind11#    def compareStatistics(self, stats, n):
#pybind11#        self.assertEqual(stats.getValue(afwMath.NPOINT), n)
#pybind11#        self.assertEqual(stats.getValue(afwMath.NPOINT)*stats.getValue(afwMath.MEAN),
#pybind11#                         stats.getValue(afwMath.SUM))
#pybind11#        self.assertEqual(stats.getValue(afwMath.MEAN), self.val)
#pybind11#        self.assertEqual(stats.getValue(afwMath.STDEV), 0)
#pybind11#
#pybind11#    # Test regular image::Image
#pybind11#    def testImage(self):
#pybind11#        for img in self.imgList:
#pybind11#            self.compareMakeStatistics(img, img.getWidth()*img.getHeight())
#pybind11#
#pybind11#    # Test the image::MaskedImages
#pybind11#    def testMaskedImage(self):
#pybind11#        for mimg in self.mimgList:
#pybind11#            self.compareMakeStatistics(mimg, mimg.getWidth()*mimg.getHeight())
#pybind11#
#pybind11#    # Test the std::vectors
#pybind11#    def testVector(self):
#pybind11#        for vec in self.vecList:
#pybind11#            self.compareMakeStatistics(vec, vec.size())
#pybind11#
#pybind11#    def testWeightedVector(self):
#pybind11#        """Test std::vector, but with weights"""
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#
#pybind11#        nval = len(self.vecList[0])
#pybind11#        weight = 10
#pybind11#        weights = [i*weight/float(nval - 1) for i in range(nval)]
#pybind11#
#pybind11#        for vec in self.vecList:
#pybind11#            stats = afwMath.makeStatistics(vec, weights,
#pybind11#                                           afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM, sctrl)
#pybind11#
#pybind11#            self.assertAlmostEqual(0.5*weight*sum(vec)/stats.getValue(afwMath.SUM), 1.0)
#pybind11#            self.assertAlmostEqual(sum(vec)/vec.size(), stats.getValue(afwMath.MEAN))
#pybind11#
#pybind11#    # Try calling the Statistics constructor directly
#pybind11#    def testStatisticsConstructor(self):
#pybind11#        if False:
#pybind11#            statsI = afwMath.StatisticsI(self.mimgI.getImage(), self.mimgI.getMask(),
#pybind11#                                         afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM,
#pybind11#                                         self.sctrl)
#pybind11#            statsF = afwMath.StatisticsF(self.mimgF.getImage(), self.mimgF.getMask(),
#pybind11#                                         afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM,
#pybind11#                                         self.sctrl)
#pybind11#            statsD = afwMath.StatisticsD(self.mimgD.getImage(), self.mimgD.getMask(),
#pybind11#                                         afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM,
#pybind11#                                         self.sctrl)
#pybind11#
#pybind11#            self.compareStatistics(statsI, self.mimgI.getWidth()*self.mimgI.getHeight())
#pybind11#            self.compareStatistics(statsF, self.mimgF.getWidth()*self.mimgF.getHeight())
#pybind11#            self.compareStatistics(statsD, self.mimgD.getWidth()*self.mimgD.getHeight())
#pybind11#
#pybind11#    # Test the Mask specialization
#pybind11#    def testMask(self):
#pybind11#        mask = afwImage.MaskU(afwGeom.Extent2I(10, 10))
#pybind11#        mask.set(0x0)
#pybind11#
#pybind11#        mask.set(1, 1, 0x10)
#pybind11#        mask.set(3, 1, 0x08)
#pybind11#        mask.set(5, 4, 0x08)
#pybind11#        mask.set(4, 5, 0x02)
#pybind11#
#pybind11#        stats = afwMath.makeStatistics(mask, afwMath.SUM | afwMath.NPOINT)
#pybind11#        self.assertEqual(mask.getWidth()*mask.getHeight(), stats.getValue(afwMath.NPOINT))
#pybind11#        self.assertEqual(0x1a, stats.getValue(afwMath.SUM))
#pybind11#
#pybind11#        def tst():
#pybind11#            afwMath.makeStatistics(mask, afwMath.MEAN)
#pybind11#        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, tst)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
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

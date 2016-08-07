#!/usr/bin/env python2
from __future__ import absolute_import, division

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
Tests for Statistics

Run with:
   ./statisticsOverloads.py
or
   python
   >>> import statisticsOverloads; statisticsOverloads.run()
"""


import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class StatisticsTestCase(unittest.TestCase):

    """A test case to check all overloaded makeStatistics() factories for Statistics"""
    def setUp(self):
        self.val = 10
        self.nRow, self.nCol = 100, 200
        self.sctrl = afwMath.StatisticsControl()

        # Integers
        self.mimgI = afwImage.MaskedImageI(afwGeom.Extent2I(self.nRow, self.nCol))
        self.mimgI.set(self.val, 0x0, self.val)
        self.imgI = afwImage.ImageI(afwGeom.Extent2I(self.nRow, self.nCol), self.val)
        self.vecI = afwMath.vectorI(self.nRow*self.nCol, self.val)

        # floats
        self.mimgF = afwImage.MaskedImageF(afwGeom.Extent2I(self.nRow, self.nCol))
        self.mimgF.set(self.val, 0x0, self.val)
        self.imgF = afwImage.ImageF(afwGeom.Extent2I(self.nRow, self.nCol), self.val)
        self.vecF = afwMath.vectorF(self.nRow*self.nCol, self.val)

        # doubles
        self.mimgD = afwImage.MaskedImageD(afwGeom.Extent2I(self.nRow, self.nCol))
        self.mimgD.set(self.val, 0x0, self.val)
        self.imgD = afwImage.ImageD(afwGeom.Extent2I(self.nRow, self.nCol), self.val)
        self.vecD = afwMath.vectorD(self.nRow*self.nCol, self.val)

        self.imgList  = [self.imgI,  self.imgF,  self.imgD]
        self.mimgList = [self.mimgI, self.mimgF, self.mimgD]
        self.vecList  = [self.vecI,  self.vecF,  self.vecD]

    def tearDown(self):
        del self.mimgI; del self.mimgF; del self.mimgD
        del self.imgI; del self.imgF; del self.imgD
        del self.vecI; del self.vecF; del self.vecD

        del self.mimgList
        del self.imgList
        del self.vecList

    # The guts of the testing: grab a mean, stddev, and sum for whatever you're called with
    def compareMakeStatistics(self, image, n):
        stats = afwMath.makeStatistics(image, afwMath.NPOINT | afwMath.STDEV |
                                       afwMath.MEAN | afwMath.SUM, self.sctrl)

        self.assertEqual(stats.getValue(afwMath.NPOINT), n)
        self.assertEqual(stats.getValue(afwMath.NPOINT)*stats.getValue(afwMath.MEAN),
                         stats.getValue(afwMath.SUM))
        self.assertEqual(stats.getValue(afwMath.MEAN), self.val)
        self.assertEqual(stats.getValue(afwMath.STDEV), 0)

    # same as compareMakeStatistics but calls constructor directly (only for masked image)
    def compareStatistics(self, stats, n):
        self.assertEqual(stats.getValue(afwMath.NPOINT), n)
        self.assertEqual(stats.getValue(afwMath.NPOINT)*stats.getValue(afwMath.MEAN),
                         stats.getValue(afwMath.SUM))
        self.assertEqual(stats.getValue(afwMath.MEAN), self.val)
        self.assertEqual(stats.getValue(afwMath.STDEV), 0)

    # Test regular image::Image
    def testImage(self):
        for img in self.imgList:
            self.compareMakeStatistics(img, img.getWidth()*img.getHeight())

    # Test the image::MaskedImages
    def testMaskedImage(self):
        for mimg in self.mimgList:
            self.compareMakeStatistics(mimg, mimg.getWidth()*mimg.getHeight())

    # Test the std::vectors
    def testVector(self):
        for vec in self.vecList:
            self.compareMakeStatistics(vec, vec.size())

    def testWeightedVector(self):
        """Test std::vector, but with weights"""
        sctrl = afwMath.StatisticsControl()

        nval = len(self.vecList[0])
        weight = 10
        weights = [i*weight/float(nval - 1) for i in range(nval)]

        for vec in self.vecList:
            stats = afwMath.makeStatistics(vec, weights,
                                           afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM, sctrl)

            self.assertAlmostEqual(0.5*weight*sum(vec)/stats.getValue(afwMath.SUM), 1.0)
            self.assertAlmostEqual(sum(vec)/vec.size(), stats.getValue(afwMath.MEAN))

    # Try calling the Statistics constructor directly
    def testStatisticsConstructor(self):
        if False:
            statsI = afwMath.StatisticsI(self.mimgI.getImage(), self.mimgI.getMask(),
                                         afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM,
                                         self.sctrl)
            statsF = afwMath.StatisticsF(self.mimgF.getImage(), self.mimgF.getMask(),
                                        afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM,
                                         self.sctrl)
            statsD = afwMath.StatisticsD(self.mimgD.getImage(), self.mimgD.getMask(),
                                        afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM,
                                         self.sctrl)

            self.compareStatistics(statsI, self.mimgI.getWidth()*self.mimgI.getHeight())
            self.compareStatistics(statsF, self.mimgF.getWidth()*self.mimgF.getHeight())
            self.compareStatistics(statsD, self.mimgD.getWidth()*self.mimgD.getHeight())



    # Test the Mask specialization
    def testMask(self):
        mask = afwImage.MaskU(afwGeom.Extent2I(10, 10))
        mask.set(0x0)

        mask.set(1, 1, 0x10)
        mask.set(3, 1, 0x08)
        mask.set(5, 4, 0x08)
        mask.set(4, 5, 0x02)

        stats = afwMath.makeStatistics(mask, afwMath.SUM | afwMath.NPOINT)
        self.assertEqual(mask.getWidth()*mask.getHeight(), stats.getValue(afwMath.NPOINT))
        self.assertEqual(0x1a, stats.getValue(afwMath.SUM))

        def tst():
            afwMath.makeStatistics(mask, afwMath.MEAN)
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, tst)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(StatisticsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

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
Tests for statisticsMasked

Run with:
   ./statisticsMasked.py
or
   python
   >>> import statisticsMasked; statisticsMasked.run()
"""


import unittest
import numpy

import lsst.utils.tests as utilsTests
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class StatisticsTestCase(unittest.TestCase):

    """A test case to check that special values (NaN and Masks) are begin handled in Statistics"""
    def setUp(self):
        self.valL, self.valR = 10, 20
        self.nRow, self.nCol = 100, 200
        self.n = self.nRow*self.nCol

        self.bboxL = afwGeom.Box2I(afwGeom.Point2I(0, 0),
                                   afwGeom.Point2I(self.nRow//2 - 1, self.nCol - 1))
        self.bboxR = afwGeom.Box2I(afwGeom.Point2I(self.nRow//2, 0),
                                   afwGeom.Point2I(self.nRow - 1, self.nCol - 1))

        # create masked images and set the left side to valL, and right to valR
        self.mimg = afwImage.MaskedImageF(afwGeom.Extent2I(self.nRow, self.nCol))
        self.mimg.set(0.0, 0x0, 0.0)
        self.mimgL = afwImage.MaskedImageF(self.mimg, self.bboxL, afwImage.LOCAL)
        self.mimgL.set(self.valL, 0x0, self.valL)
        self.mimgR = afwImage.MaskedImageF(self.mimg, self.bboxR, afwImage.LOCAL)
        self.mimgR.set(self.valR, 0x0, self.valR)


    def tearDown(self):
        del self.mimg
        del self.mimgL
        del self.mimgR

    # Verify that NaN values are being ignored
    # (by default, StatisticsControl.useNanSafe = True)
    # We'll set the L and R sides of an image to two different values and verify mean and stdev
    # ... then set R-side to NaN and try again ... we should get mean,stdev for L-side
    def testNaN(self):

        # get the stats for the image with two values
        stats = afwMath.makeStatistics(self.mimg, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV)
        mean = 0.5*(self.valL + self.valR)
        nL, nR = self.mimgL.getWidth()*self.mimgL.getHeight(), self.mimgR.getWidth()*self.mimgR.getHeight()
        stdev = ((nL*(self.valL - mean)**2 + nR*(self.valR - mean)**2)/(nL + nR - 1))**0.5

        self.assertEqual(stats.getValue(afwMath.NPOINT), self.n)
        self.assertEqual(stats.getValue(afwMath.MEAN), mean)
        self.assertEqual(stats.getValue(afwMath.STDEV), stdev)

        # set the right side to NaN and stats should be just for the left side
        self.mimgR.set(numpy.nan, 0x0, self.valR)

        statsNaN = afwMath.makeStatistics(self.mimg, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV)
        mean = self.valL
        stdev = 0.0
        self.assertEqual(statsNaN.getValue(afwMath.NPOINT), nL)
        self.assertEqual(statsNaN.getValue(afwMath.MEAN), mean)
        self.assertEqual(statsNaN.getValue(afwMath.STDEV), stdev)


    # Verify that Masked pixels are being ignored according to the andMask
    # (by default, StatisticsControl.andMask = 0x0)
    # We'll set the L and R sides of an image to two different values and verify mean and stdev
    # ... then set R-side Mask and the andMask to 0x1 and try again ... we should get mean,stdev for L-side
    def testMasked(self):

        # get the stats for the image with two values
        self.mimgR.set(self.valR, 0x0, self.valR)
        stats = afwMath.makeStatistics(self.mimg, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV)
        mean = 0.5*(self.valL + self.valR)
        nL, nR = self.mimgL.getWidth()*self.mimgL.getHeight(), self.mimgR.getWidth()*self.mimgR.getHeight()
        stdev = ((nL*(self.valL - mean)**2 + nR*(self.valR - mean)**2)/(nL + nR - 1))**0.5

        self.assertEqual(stats.getValue(afwMath.NPOINT), self.n)
        self.assertEqual(stats.getValue(afwMath.MEAN), mean)
        self.assertEqual(stats.getValue(afwMath.STDEV), stdev)

        # set the right side Mask and the StatisticsControl andMask to 0x1
        #  Stats should be just for the left side!
        maskBit = 0x1
        self.mimgR.getMask().set(maskBit)

        sctrl = afwMath.StatisticsControl()
        sctrl.setAndMask(maskBit)
        statsNaN = afwMath.makeStatistics(self.mimg, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV, sctrl)

        mean = self.valL
        stdev = 0.0

        self.assertEqual(statsNaN.getValue(afwMath.NPOINT), nL)
        self.assertEqual(statsNaN.getValue(afwMath.MEAN), mean)
        self.assertEqual(statsNaN.getValue(afwMath.STDEV), stdev)



    # Verify that pixels are being weighted according to the variance plane (1/var)
    # We'll set the L and R sides of an image to two different values and verify mean
    # ... then set R-side Variance to equal the Image value, and set 'weighted' and try again ...
    def testWeighted(self):

        self.mimgR.set(self.valR, 0x0, self.valR)
        sctrl = afwMath.StatisticsControl()
        sctrl.setWeighted(True)
        stats = afwMath.makeStatistics(self.mimg, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV, sctrl)
        nL, nR = self.mimgL.getWidth()*self.mimgL.getHeight(), self.mimgR.getWidth()*self.mimgR.getHeight()

        mean = 1.0*(nL + nR)/(nL/self.valL + nR/self.valR)

        # get the stats for the image with two values
        self.assertEqual(stats.getValue(afwMath.NPOINT), self.n)
        self.assertAlmostEqual(stats.getValue(afwMath.MEAN), mean, 10)

    def testWeightedSimple(self):
        mimg = afwImage.MaskedImageF(afwGeom.Extent2I(1, 2))
        mimg.set(0, 0, (self.valR, 0x0, self.valR))
        mimg.set(0, 1, (self.valL, 0x0, self.valL))

        sctrl = afwMath.StatisticsControl()
        sctrl.setWeighted(True)
        stats = afwMath.makeStatistics(mimg, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV, sctrl)
        vsum = 2.0
        vsum2 = self.valR + self.valL
        wsum = 1.0/self.valR + 1.0/self.valL
        wwsum = 1.0/self.valR**2 + 1.0/self.valL**2
        mean = vsum/wsum
        variance = vsum2/wsum - mean**2 # biased variance

        n = 2
        # original estimate; just a rewrite of the usual n/(n - 1) correction
        stddev = (1.0*(vsum2)/(wsum*(1.0-1.0/n)) - (vsum**2)/(wsum**2*(1.0-1.0/n)))**0.5
        self.assertAlmostEqual(stddev, numpy.sqrt(variance*n/(n - 1)))
        #
        # The correct formula:
        stddev = numpy.sqrt(variance*wsum**2/(wsum**2 - wwsum))

        # get the stats for the image with two values
        self.assertAlmostEqual(stats.getValue(afwMath.MEAN), mean, 10)
        self.assertAlmostEqual(stats.getValue(afwMath.STDEV), stddev, 10)


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

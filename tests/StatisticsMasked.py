#!/usr/bin/env python
"""
Tests for Statistics

Run with:
   ./Statistics.py
or
   python
   >>> import Statistics; Statistics.run()
"""

import math
import os
import pdb  # we may want to say pdb.set_trace()
import sys
import unittest
import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class StatisticsTestCase(unittest.TestCase):
    
    """A test case to check that special values (NaN and Masks) are begin handled in Statistics"""
    def setUp(self):
        self.valL, self.valR = 10, 20
        self.nRow, self.nCol = 100, 200
        self.N = self.nRow*self.nCol

        self.bboxL = afwImage.BBox(afwImage.PointI(0, 0),
                                   afwImage.PointI(self.nRow/2 - 1, self.nCol - 1))
        self.bboxR = afwImage.BBox(afwImage.PointI(self.nRow/2, 0),
                                   afwImage.PointI(self.nRow - 1, self.nCol - 1))

        # create masked images and set the left side to valL, and right to valR
        self.mimg = afwImage.MaskedImageF(self.nRow, self.nCol)
        self.mimg.set(0.0, 0x0, 0.0)
        self.mimgL = afwImage.MaskedImageF(self.mimg, self.bboxL)
        self.mimgL.set(self.valL, 0x0, self.valL)
        self.mimgR = afwImage.MaskedImageF(self.mimg, self.bboxR)
        self.mimgR.set(self.valR, 0x0, self.valR)
        
        
    def tearDown(self):
        del self.mimg


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

        self.assertEqual(stats.getValue(afwMath.NPOINT), self.N)
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

        self.assertEqual(stats.getValue(afwMath.NPOINT), self.N)
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

        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(StatisticsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit = False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

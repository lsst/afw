#!/usr/bin/env python
"""
Tests for Statistics

Run with:
   ./Statistics.py
or
   python
   >>> import Statistics; Statistics.run()
"""

import pdb  # we may want to say pdb.set_trace()
import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math as afwMath
#import lsst.afw.display.ds9 as ds9

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
        self.mimgI = afwImage.MaskedImageI(self.nRow, self.nCol)
        self.mimgI.set(self.val, 0x0, self.val)
        self.imgI = afwImage.ImageI(self.nRow, self.nCol, self.val)
        self.vecI = afwMath.vectorI(self.nRow*self.nCol, self.val)

        # floats
        self.mimgF = afwImage.MaskedImageF(self.nRow, self.nCol)
        self.mimgF.set(self.val, 0x0, self.val)
        self.imgF = afwImage.ImageF(self.nRow, self.nCol, self.val)
        self.vecF = afwMath.vectorF(self.nRow*self.nCol, self.val)

        # doubles
        self.mimgD = afwImage.MaskedImageD(self.nRow, self.nCol)
        self.mimgD.set(self.val, 0x0, self.val)
        self.imgD = afwImage.ImageD(self.nRow, self.nCol, self.val)
        self.vecD = afwMath.vectorD(self.nRow*self.nCol, self.val)


        self.imgList  = [self.imgI,  self.imgF,  self.imgD]
        self.mimgList = [self.mimgI, self.mimgF, self.mimgD]
        self.vecList  = [self.vecI,  self.vecF,  self.vecD]
        
    def tearDown(self):
        for img in self.imgList:
            del img
        for mimg in self.mimgList:
            del mimg
        for vec in self.vecList:
            del vec


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
        mask = afwImage.MaskU(10, 10)
        mask.set(0x0)

        mask.set(1, 1, 0x10)
        mask.set(3, 1, 0x08)
        mask.set(5, 4, 0x08)
        mask.set(4, 5, 0x02)

        stats = afwMath.makeStatistics(mask, afwMath.SUM | afwMath.NPOINT)
        self.assertEqual(mask.getWidth()*mask.getHeight(), stats.getValue(afwMath.NPOINT))
        self.assertEqual(0x1a, stats.getValue(afwMath.SUM))

        def tst():
            stats = afwMath.makeStatistics(mask, afwMath.MEAN)
        utilsTests.assertRaisesLsstCpp(self, lsst.pex.exceptions.InvalidParameterException, tst)

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

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
    """A test case for Statistics"""
    def setUp(self):
        self.val = 10
        self.image = afwImage.ImageF(100, 200); self.image.set(self.val)

    def tearDown(self):
        del self.image

    def testStats1(self):
        stats = afwMath.StatisticsF(self.image, afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN)

        self.assertEqual(stats.getValue(afwMath.NPOINT), self.image.getWidth()*self.image.getHeight())
        self.assertEqual(stats.getValue(afwMath.MEAN), self.val)
        #BOOST_CHECK(std::isnan(stats.getError(afwMath.MEAN))) // we didn't ask for the error, so it's a NaN
        self.assertEqual(stats.getValue(afwMath.STDEV), 0)

    def testStats2(self):
        stats = afwMath.StatisticsF(self.image, afwMath.STDEV | afwMath.MEAN | afwMath.ERRORS)
        mean = stats.getResult(afwMath.MEAN)
        sd = stats.getValue(afwMath.STDEV)
        
        self.assertEqual(mean[0],  self.image.get(0,0))
        self.assertEqual(mean[1], sd/math.sqrt(self.image.getWidth()*self.image.getHeight()))

    def testStats3(self):
        stats = afwMath.StatisticsF(self.image, afwMath.NPOINT)

        def getMean():
            stats.getValue(afwMath.MEAN)

        self.assertRaises(lsst.pex.exceptions.LsstInvalidParameter, getMean)

    def testStatsZebra(self):
	"""Add 1 to every other row"""
        image2 = self.image.Factory(self.image, True)
        #
        # Add 1 to every other row, so the variance is 1/4
        #
        self.assertEqual(image2.getHeight()%2, 0)
        width = image2.getWidth()
        for y in range(1, image2.getHeight(), 2):
            sim = image2.Factory(image2, afwImage.BBox(afwImage.PointI(0, y), width, 1))
            sim += 1

	if display:
	    ds9.mtv(self.image, frame=0)
	    ds9.mtv(image2, frame=1)

        stats = afwMath.StatisticsF(image2, afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.ERRORS)
        mean = stats.getResult(afwMath.MEAN)
        n = stats.getValue(afwMath.NPOINT)
        sd = stats.getValue(afwMath.STDEV)

        self.assertEqual(mean[0],  image2.get(0,0) + 0.5)
        self.assertEqual(sd, 1/math.sqrt(4.0)*math.sqrt(n/(n - 1)))
        self.assertAlmostEqual(mean[1], sd/math.sqrt(image2.getWidth()*image2.getHeight()), 10)

    def testMedian(self):
	"""Test the median code"""
        stats = afwMath.StatisticsF(self.image, afwMath.MEDIAN)

        self.assertEqual(stats.getValue(afwMath.MEDIAN), self.val)

    def testIqrange(self):
	"""Test the inter-quartile range"""
	stats = afwMath.StatisticsF(self.image, afwMath.IQRANGE)

	self.assertEqual(stats.getValue(afwMath.IQRANGE), 0)

    def testMeanClip(self):
	"""Test the 3-sigma clipped mean"""
	stats = afwMath.StatisticsF(self.image, afwMath.MEANCLIP)

	self.assertEqual(stats.getValue(afwMath.MEANCLIP), self.val)

    def testVarianceClip(self):
	"""Test the 3-sigma clipped variance"""
	stats = afwMath.StatisticsF(self.image, afwMath.VARIANCECLIP)

	self.assertEqual(stats.getValue(afwMath.VARIANCECLIP), 0)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(StatisticsTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

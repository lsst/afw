#!/usr/bin/env python
# -*- lsst-python -*-
"""
Tests for ticket 1043 - Photometry fails when no PSF is provided
"""

import lsst.afw.math as afwMath
import numpy as num

import math
import unittest
import lsst.utils.tests as utilsTests

# math.isnan() available in 2.6, but not 2.5.2
try:
    math.isnan()
except AttributeError:
    math.isnan = lambda x: x != x

class ticket1045TestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testTicket1045(self):
        values = [1.08192,1.08792,1.08774,1.09953,1.1122,1.09408,0.879792,1.12235,1.10115,1.08999]
        knownMean, knownStdev =  num.mean(values), 0.069903889977279199

        # this was reported to work
        dmean1 = afwMath.makeStatistics(values, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV)
        n1 = dmean1.getValue(afwMath.NPOINT)
        mean1 = dmean1.getValue(afwMath.MEAN)
        stdev1 = dmean1.getValue(afwMath.STDEV)
        self.assertAlmostEqual(mean1, knownMean, 8)
        self.assertEqual(stdev1, knownStdev)

        # this was reported to fail
        # (problem was due to error in median)
        knownMeanClip = 1.097431111111111
        knownStdevClip = 0.012984991763998597
        
        dmean2 = afwMath.makeStatistics(values, afwMath.NPOINT | afwMath.MEANCLIP | afwMath.STDEVCLIP)
        n2 = dmean2.getValue(afwMath.NPOINT)
        mean2 = dmean2.getValue(afwMath.MEANCLIP)
        stdev2 = dmean2.getValue(afwMath.STDEVCLIP)
        self.assertEqual(mean2, knownMeanClip)
        self.assertEqual(stdev2, knownStdevClip)

        # check the median, just for giggles
        knownMedian = num.median(values)
        stat = afwMath.makeStatistics(values, afwMath.MEDIAN)
        median = stat.getValue(afwMath.MEDIAN)
        self.assertEqual(median, knownMedian)

        # check the median with an odd number of values
        knownMedian = num.median(values[1:])
        stat = afwMath.makeStatistics(values[1:], afwMath.MEDIAN)
        median = stat.getValue(afwMath.MEDIAN)
        self.assertEqual(median, knownMedian)
        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ticket1045TestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(exit = False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
 

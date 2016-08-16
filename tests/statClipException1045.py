#!/usr/bin/env python

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

# -*- lsst-python -*-
"""
Tests for ticket 1043 - Photometry fails when no PSF is provided
"""

import lsst.afw.math as afwMath
import numpy as num

import math
import unittest
import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept

# math.isnan() available in 2.6, but not 2.5.2
try:
    math.isnan(1)
except AttributeError:
    math.isnan = lambda x: x != x


class Ticket1045TestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testTicket1045(self):
        values = [1.08192, 1.08792, 1.08774, 1.09953, 1.1122, 1.09408, 0.879792, 1.12235, 1.10115, 1.08999]
        knownMean, knownStdev = num.mean(values), 0.069903889977279199

        # this was reported to work
        dmean1 = afwMath.makeStatistics(values, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV)
        mean1 = dmean1.getValue(afwMath.MEAN)
        stdev1 = dmean1.getValue(afwMath.STDEV)
        self.assertAlmostEqual(mean1, knownMean, 8)
        self.assertAlmostEqual(stdev1, knownStdev, places=16)

        # this was reported to fail
        # (problem was due to error in median)
        knownMeanClip = 1.097431111111111
        knownStdevClip = 0.012984991763998597

        dmean2 = afwMath.makeStatistics(values, afwMath.NPOINT | afwMath.MEANCLIP | afwMath.STDEVCLIP)
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
        vals = values[1:]
        knownMedian = num.median(vals)
        stat = afwMath.makeStatistics(vals, afwMath.MEDIAN)
        median = stat.getValue(afwMath.MEDIAN)
        self.assertEqual(median, knownMedian)

        # check the median with only two values
        vals = values[0:2]
        knownMedian = num.median(vals)
        stat = afwMath.makeStatistics(vals, afwMath.MEDIAN)
        median = stat.getValue(afwMath.MEDIAN)
        self.assertEqual(median, knownMedian)

        # check the median with only 1 value
        vals = values[0:1]
        knownMedian = num.median(vals)
        stat = afwMath.makeStatistics(vals, afwMath.MEDIAN)
        median = stat.getValue(afwMath.MEDIAN)
        self.assertEqual(median, knownMedian)

        # check the median with no values
        vals = []

        def tst():
            stat = afwMath.makeStatistics(vals, afwMath.MEDIAN)
            median = stat.getValue(afwMath.MEDIAN)
            return median
        self.assertRaises(pexExcept.InvalidParameterError, tst)

    def testUnexpectedNan1051(self):

        values = [7824.0, 7803.0, 7871.0, 7567.0, 7813.0, 7809.0, 8011.0, 7807.0]
        npValues = num.array(values)

        meanClip = afwMath.makeStatistics(values, afwMath.MEANCLIP).getValue()
        iKept = num.array([0, 1, 2, 4, 5, 7])  # note ... it will clip indices 3 and 6
        knownMeanClip = num.mean(npValues[iKept])
        self.assertEqual(meanClip, knownMeanClip)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(Ticket1045TestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

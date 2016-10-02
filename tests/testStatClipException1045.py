#pybind11##!/usr/bin/env python
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
#pybind11## -*- lsst-python -*-
#pybind11#"""
#pybind11#Tests for ticket 1043 - Photometry fails when no PSF is provided
#pybind11#"""
#pybind11#
#pybind11#import lsst.afw.math as afwMath
#pybind11#import numpy as num
#pybind11#
#pybind11#import math
#pybind11#import unittest
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#
#pybind11## math.isnan() available in 2.6, but not 2.5.2
#pybind11#try:
#pybind11#    math.isnan(1)
#pybind11#except AttributeError:
#pybind11#    math.isnan = lambda x: x != x
#pybind11#
#pybind11#
#pybind11#class Ticket1045TestCase(unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        pass
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        pass
#pybind11#
#pybind11#    def testTicket1045(self):
#pybind11#        values = [1.08192, 1.08792, 1.08774, 1.09953, 1.1122, 1.09408, 0.879792, 1.12235, 1.10115, 1.08999]
#pybind11#        knownMean, knownStdev = num.mean(values), 0.069903889977279199
#pybind11#
#pybind11#        # this was reported to work
#pybind11#        dmean1 = afwMath.makeStatistics(values, afwMath.NPOINT | afwMath.MEAN | afwMath.STDEV)
#pybind11#        mean1 = dmean1.getValue(afwMath.MEAN)
#pybind11#        stdev1 = dmean1.getValue(afwMath.STDEV)
#pybind11#        self.assertAlmostEqual(mean1, knownMean, 8)
#pybind11#        self.assertAlmostEqual(stdev1, knownStdev, places=16)
#pybind11#
#pybind11#        # this was reported to fail
#pybind11#        # (problem was due to error in median)
#pybind11#        knownMeanClip = 1.097431111111111
#pybind11#        knownStdevClip = 0.012984991763998597
#pybind11#
#pybind11#        dmean2 = afwMath.makeStatistics(values, afwMath.NPOINT | afwMath.MEANCLIP | afwMath.STDEVCLIP)
#pybind11#        mean2 = dmean2.getValue(afwMath.MEANCLIP)
#pybind11#        stdev2 = dmean2.getValue(afwMath.STDEVCLIP)
#pybind11#        self.assertEqual(mean2, knownMeanClip)
#pybind11#        self.assertEqual(stdev2, knownStdevClip)
#pybind11#
#pybind11#        # check the median, just for giggles
#pybind11#        knownMedian = num.median(values)
#pybind11#        stat = afwMath.makeStatistics(values, afwMath.MEDIAN)
#pybind11#        median = stat.getValue(afwMath.MEDIAN)
#pybind11#        self.assertEqual(median, knownMedian)
#pybind11#
#pybind11#        # check the median with an odd number of values
#pybind11#        vals = values[1:]
#pybind11#        knownMedian = num.median(vals)
#pybind11#        stat = afwMath.makeStatistics(vals, afwMath.MEDIAN)
#pybind11#        median = stat.getValue(afwMath.MEDIAN)
#pybind11#        self.assertEqual(median, knownMedian)
#pybind11#
#pybind11#        # check the median with only two values
#pybind11#        vals = values[0:2]
#pybind11#        knownMedian = num.median(vals)
#pybind11#        stat = afwMath.makeStatistics(vals, afwMath.MEDIAN)
#pybind11#        median = stat.getValue(afwMath.MEDIAN)
#pybind11#        self.assertEqual(median, knownMedian)
#pybind11#
#pybind11#        # check the median with only 1 value
#pybind11#        vals = values[0:1]
#pybind11#        knownMedian = num.median(vals)
#pybind11#        stat = afwMath.makeStatistics(vals, afwMath.MEDIAN)
#pybind11#        median = stat.getValue(afwMath.MEDIAN)
#pybind11#        self.assertEqual(median, knownMedian)
#pybind11#
#pybind11#        # check the median with no values
#pybind11#        vals = []
#pybind11#
#pybind11#        def tst():
#pybind11#            stat = afwMath.makeStatistics(vals, afwMath.MEDIAN)
#pybind11#            median = stat.getValue(afwMath.MEDIAN)
#pybind11#            return median
#pybind11#        self.assertRaises(pexExcept.InvalidParameterError, tst)
#pybind11#
#pybind11#    def testUnexpectedNan1051(self):
#pybind11#
#pybind11#        values = [7824.0, 7803.0, 7871.0, 7567.0, 7813.0, 7809.0, 8011.0, 7807.0]
#pybind11#        npValues = num.array(values)
#pybind11#
#pybind11#        meanClip = afwMath.makeStatistics(values, afwMath.MEANCLIP).getValue()
#pybind11#        iKept = num.array([0, 1, 2, 4, 5, 7])  # note ... it will clip indices 3 and 6
#pybind11#        knownMeanClip = num.mean(npValues[iKept])
#pybind11#        self.assertEqual(meanClip, knownMeanClip)
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

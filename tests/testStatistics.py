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
#pybind11#   ./statistics.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import statistics; statistics.run()
#pybind11#"""
#pybind11#
#pybind11#import sys
#pybind11#import math
#pybind11#import os
#pybind11#import numpy as np
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.image.imageLib as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#
#pybind11#try:
#pybind11#    afwdataDir = lsst.utils.getPackageDir("afwdata")
#pybind11#except pexExcept.NotFoundError:
#pybind11#    afwdataDir = None
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class StatisticsTestCase(lsst.utils.tests.TestCase):
#pybind11#    """A test case for Statistics"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.val = 10
#pybind11#        self.image = afwImage.ImageF(afwGeom.Extent2I(100, 200))
#pybind11#        self.image.set(self.val)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.image
#pybind11#
#pybind11#    def testDefaultGet(self):
#pybind11#        """Test that we can get a single statistic without specifying it"""
#pybind11#        stats = afwMath.makeStatistics(self.image, afwMath.MEDIAN)
#pybind11#
#pybind11#        self.assertEqual(stats.getValue(), stats.getValue(afwMath.MEDIAN))
#pybind11#        self.assertEqual(stats.getResult()[0], stats.getResult(afwMath.MEDIAN)[0])
#pybind11#        #
#pybind11#        stats = afwMath.makeStatistics(self.image, afwMath.MEDIAN | afwMath.ERRORS)
#pybind11#
#pybind11#        self.assertEqual(stats.getValue(), stats.getValue(afwMath.MEDIAN))
#pybind11#        self.assertEqual(stats.getResult(), stats.getResult(afwMath.MEDIAN))
#pybind11#        self.assertEqual(stats.getError(), stats.getError(afwMath.MEDIAN))
#pybind11#
#pybind11#        def tst():
#pybind11#            stats.getValue()
#pybind11#        stats = afwMath.makeStatistics(self.image, afwMath.MEDIAN | afwMath.MEAN)
#pybind11#        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, tst)
#pybind11#
#pybind11#    def testStats1(self):
#pybind11#        stats = afwMath.makeStatistics(self.image,
#pybind11#                                       afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM)
#pybind11#
#pybind11#        self.assertEqual(stats.getValue(afwMath.NPOINT), self.image.getWidth()*self.image.getHeight())
#pybind11#        self.assertEqual(stats.getValue(afwMath.NPOINT)*stats.getValue(afwMath.MEAN),
#pybind11#                         stats.getValue(afwMath.SUM))
#pybind11#        self.assertEqual(stats.getValue(afwMath.MEAN), self.val)
#pybind11#        self.assertTrue(np.isnan(stats.getError(afwMath.MEAN)))  # didn't ask for error, so it's a NaN
#pybind11#        self.assertEqual(stats.getValue(afwMath.STDEV), 0)
#pybind11#
#pybind11#    def testStats2(self):
#pybind11#        stats = afwMath.makeStatistics(self.image, afwMath.STDEV | afwMath.MEAN | afwMath.ERRORS)
#pybind11#        mean = stats.getResult(afwMath.MEAN)
#pybind11#        sd = stats.getValue(afwMath.STDEV)
#pybind11#
#pybind11#        self.assertEqual(mean[0], self.image.get(0, 0))
#pybind11#        self.assertEqual(mean[1], sd/math.sqrt(self.image.getWidth()*self.image.getHeight()))
#pybind11#
#pybind11#    def testStats3(self):
#pybind11#        stats = afwMath.makeStatistics(self.image, afwMath.NPOINT)
#pybind11#
#pybind11#        def getMean():
#pybind11#            stats.getValue(afwMath.MEAN)
#pybind11#
#pybind11#        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, getMean)
#pybind11#
#pybind11#    def testStatsZebra(self):
#pybind11#        """Add 1 to every other row"""
#pybind11#        image2 = self.image.Factory(self.image, True)
#pybind11#        #
#pybind11#        # Add 1 to every other row, so the variance is 1/4
#pybind11#        #
#pybind11#        self.assertEqual(image2.getHeight() % 2, 0)
#pybind11#        width = image2.getWidth()
#pybind11#        for y in range(1, image2.getHeight(), 2):
#pybind11#            sim = image2.Factory(image2, afwGeom.Box2I(afwGeom.Point2I(0, y), afwGeom.Extent2I(width, 1)),
#pybind11#                                 afwImage.LOCAL)
#pybind11#            sim += 1
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(self.image, frame=0)
#pybind11#            ds9.mtv(image2, frame=1)
#pybind11#
#pybind11#        stats = afwMath.makeStatistics(image2,
#pybind11#                                       afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.ERRORS)
#pybind11#        mean = stats.getResult(afwMath.MEAN)
#pybind11#        n = stats.getValue(afwMath.NPOINT)
#pybind11#        sd = stats.getValue(afwMath.STDEV)
#pybind11#
#pybind11#        self.assertEqual(mean[0], image2.get(0, 0) + 0.5)
#pybind11#        self.assertEqual(sd, 1/math.sqrt(4.0)*math.sqrt(n/(n - 1)))
#pybind11#        self.assertAlmostEqual(mean[1], sd/math.sqrt(image2.getWidth()*image2.getHeight()), 10)
#pybind11#
#pybind11#        meanSquare = afwMath.makeStatistics(image2, afwMath.MEANSQUARE).getValue()
#pybind11#        self.assertEqual(meanSquare, 0.5*(image2.get(0, 0)**2 + image2.get(0, 1)**2))
#pybind11#
#pybind11#    def testStatsStdevclip(self):
#pybind11#        """Test STDEVCLIP; cf. #611"""
#pybind11#        image2 = self.image.Factory(self.image, True)
#pybind11#
#pybind11#        stats = afwMath.makeStatistics(image2, afwMath.STDEVCLIP | afwMath.NPOINT | afwMath.SUM)
#pybind11#        self.assertEqual(stats.getValue(afwMath.STDEVCLIP), 0)
#pybind11#        #
#pybind11#        # Check we get the correct sum even when clipping
#pybind11#        #
#pybind11#        self.assertEqual(stats.getValue(afwMath.NPOINT) *
#pybind11#                         afwMath.makeStatistics(image2, afwMath.MEAN).getValue(),
#pybind11#                         stats.getValue(afwMath.SUM))
#pybind11#
#pybind11#    def testMedian(self):
#pybind11#        """Test the median code"""
#pybind11#        stats = afwMath.makeStatistics(self.image, afwMath.MEDIAN)
#pybind11#
#pybind11#        self.assertEqual(stats.getValue(afwMath.MEDIAN), self.val)
#pybind11#
#pybind11#        values = [1.0, 2.0, 3.0, 2.0]
#pybind11#        self.assertEqual(afwMath.makeStatistics(values, afwMath.MEDIAN).getValue(), 2.0)
#pybind11#
#pybind11#    def testIqrange(self):
#pybind11#        """Test the inter-quartile range"""
#pybind11#        stats = afwMath.makeStatistics(self.image, afwMath.IQRANGE)
#pybind11#        self.assertEqual(stats.getValue(afwMath.IQRANGE), 0)
#pybind11#
#pybind11#    def testMeanClip(self):
#pybind11#        """Test the 3-sigma clipped mean"""
#pybind11#        stats = afwMath.makeStatistics(self.image, afwMath.MEANCLIP)
#pybind11#        self.assertEqual(stats.getValue(afwMath.MEANCLIP), self.val)
#pybind11#
#pybind11#    def testStdevClip(self):
#pybind11#        """Test the 3-sigma clipped standard deviation"""
#pybind11#        stats = afwMath.makeStatistics(self.image, afwMath.STDEVCLIP)
#pybind11#        self.assertEqual(stats.getValue(afwMath.STDEVCLIP), 0)
#pybind11#
#pybind11#    def testVarianceClip(self):
#pybind11#        """Test the 3-sigma clipped variance"""
#pybind11#        stats = afwMath.makeStatistics(self.image, afwMath.VARIANCECLIP)
#pybind11#        self.assertEqual(stats.getValue(afwMath.VARIANCECLIP), 0)
#pybind11#
#pybind11#    def _testBadValue(self, badVal):
#pybind11#        """Test that we can handle an instance of `badVal` in the data correctly"""
#pybind11#        x, y = 10, 10
#pybind11#        for useImage in [True, False]:
#pybind11#            if useImage:
#pybind11#                self.image = afwImage.ImageF(100, 100)
#pybind11#                self.image.set(self.val)
#pybind11#                self.image.set(x, y, badVal)
#pybind11#            else:
#pybind11#                self.image = afwImage.MaskedImageF(100, 100)
#pybind11#                self.image.set(self.val, 0x0, 1.0)
#pybind11#                self.image.set(x, y, (badVal, 0x0, 1.0))
#pybind11#
#pybind11#            self.assertEqual(afwMath.makeStatistics(self.image, afwMath.MAX).getValue(), self.val)
#pybind11#            self.assertEqual(afwMath.makeStatistics(self.image, afwMath.MEAN).getValue(), self.val)
#pybind11#
#pybind11#            sctrl = afwMath.StatisticsControl()
#pybind11#
#pybind11#            sctrl.setNanSafe(False)
#pybind11#            self.assertFalse(np.isfinite(afwMath.makeStatistics(self.image, afwMath.MAX, sctrl).getValue()))
#pybind11#            self.assertFalse(np.isfinite(afwMath.makeStatistics(self.image, afwMath.MEAN, sctrl).getValue()))
#pybind11#
#pybind11#    def testMaxWithNan(self):
#pybind11#        """Test that we can handle NaNs correctly"""
#pybind11#        self._testBadValue(np.nan)
#pybind11#
#pybind11#    def testMaxWithInf(self):
#pybind11#        """Test that we can handle infinities correctly"""
#pybind11#        self._testBadValue(np.inf)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testSampleImageStats(self):
#pybind11#        """ Compare our results to known values in test data """
#pybind11#
#pybind11#        imgfiles = []
#pybind11#        imgfiles.append("v1_i1_g_m400_s20_f.fits")
#pybind11#        imgfiles.append("v1_i1_g_m400_s20_u16.fits")
#pybind11#        imgfiles.append("v1_i2_g_m400_s20_f.fits")
#pybind11#        imgfiles.append("v1_i2_g_m400_s20_u16.fits")
#pybind11#        imgfiles.append("v2_i1_p_m9_f.fits")
#pybind11#        imgfiles.append("v2_i1_p_m9_u16.fits")
#pybind11#        imgfiles.append("v2_i2_p_m9_f.fits")
#pybind11#        imgfiles.append("v2_i2_p_m9_u16.fits")
#pybind11#
#pybind11#        afwdataDir = os.getenv("AFWDATA_DIR")
#pybind11#
#pybind11#        for imgfile in imgfiles:
#pybind11#
#pybind11#            imgPath = os.path.join(afwdataDir, "Statistics", imgfile)
#pybind11#
#pybind11#            # get the image and header
#pybind11#            dimg = afwImage.DecoratedImageF(imgPath)
#pybind11#            fitsHdr = dimg.getMetadata()
#pybind11#
#pybind11#            # get the true values of the mean and stdev
#pybind11#            trueMean = fitsHdr.getAsDouble("MEANCOMP")
#pybind11#            trueStdev = fitsHdr.getAsDouble("SIGCOMP")
#pybind11#
#pybind11#            # measure the mean and stdev with the Statistics class
#pybind11#            img = dimg.getImage()
#pybind11#            statobj = afwMath.makeStatistics(img, afwMath.MEAN | afwMath.STDEV)
#pybind11#            mean = statobj.getValue(afwMath.MEAN)
#pybind11#            stdev = statobj.getValue(afwMath.STDEV)
#pybind11#
#pybind11#            # print trueMean, mean, trueStdev, stdev
#pybind11#            self.assertAlmostEqual(mean, trueMean, 8)
#pybind11#            self.assertAlmostEqual(stdev, trueStdev, 8)
#pybind11#
#pybind11#    def testStatisticsRamp(self):
#pybind11#        """ Tests Statistics on a 'ramp' (image with constant gradient) """
#pybind11#
#pybind11#        nx = 101
#pybind11#        ny = 64
#pybind11#        img = afwImage.ImageF(afwGeom.Extent2I(nx, ny))
#pybind11#
#pybind11#        z0 = 10.0
#pybind11#        dzdx = 1.0
#pybind11#        mean = z0 + (nx//2)*dzdx
#pybind11#        stdev = 0.0
#pybind11#        for y in range(ny):
#pybind11#            for x in range(nx):
#pybind11#                z = z0 + dzdx*x
#pybind11#                img.set(x, y, z)
#pybind11#                stdev += (z - mean)*(z - mean)
#pybind11#
#pybind11#        stdev = math.sqrt(stdev/(nx*ny - 1))
#pybind11#
#pybind11#        stats = afwMath.makeStatistics(img, afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN)
#pybind11#        testmean = stats.getValue(afwMath.MEAN)
#pybind11#        teststdev = stats.getValue(afwMath.STDEV)
#pybind11#
#pybind11#        self.assertEqual(stats.getValue(afwMath.NPOINT), nx*ny)
#pybind11#        self.assertEqual(testmean, mean)
#pybind11#        self.assertAlmostEqual(teststdev, stdev)
#pybind11#
#pybind11#        stats = afwMath.makeStatistics(img, afwMath.STDEV | afwMath.MEAN | afwMath.ERRORS)
#pybind11#        mean, meanErr = stats.getResult(afwMath.MEAN)
#pybind11#        sd = stats.getValue(afwMath.STDEV)
#pybind11#
#pybind11#        self.assertEqual(mean, img.get(nx//2, ny//2))
#pybind11#        self.assertEqual(meanErr, sd/math.sqrt(img.getWidth()*img.getHeight()))
#pybind11#
#pybind11#        # ===============================================================================
#pybind11#        # sjb code for percentiles and clipped stats
#pybind11#
#pybind11#        stats = afwMath.makeStatistics(img, afwMath.MEDIAN)
#pybind11#        self.assertEqual(z0 + dzdx*(nx - 1)/2.0, stats.getValue(afwMath.MEDIAN))
#pybind11#
#pybind11#        stats = afwMath.makeStatistics(img, afwMath.IQRANGE)
#pybind11#        self.assertEqual(dzdx*(nx - 1)/2.0, stats.getValue(afwMath.IQRANGE))
#pybind11#
#pybind11#        stats = afwMath.makeStatistics(img, afwMath.MEANCLIP)
#pybind11#        self.assertEqual(z0 + dzdx*(nx - 1)/2.0, stats.getValue(afwMath.MEANCLIP))
#pybind11#
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
#pybind11#    def testTicket1025(self):
#pybind11#        """
#pybind11#        Ticket #1025 reported that the Statistics median was getting '3' as the median of [1,2,3,2]
#pybind11#        it was caused by an off-by-one error in the implementation
#pybind11#        """
#pybind11#
#pybind11#        # check the exact example in the ticket
#pybind11#        values = [1.0, 2.0, 3.0, 2.0]
#pybind11#        self.assertEqual(afwMath.makeStatistics(values, afwMath.MEDIAN).getValue(), 2)
#pybind11#        self.assertEqual(afwMath.makeStatistics(sorted(values), afwMath.MEDIAN).getValue(), 2)
#pybind11#
#pybind11#        # check some other possible ways it could show up
#pybind11#        values = list(range(10))
#pybind11#        self.assertEqual(afwMath.makeStatistics(values, afwMath.MEDIAN).getValue(), 4.5)
#pybind11#        values = list(range(11))
#pybind11#        self.assertEqual(afwMath.makeStatistics(values, afwMath.MEDIAN).getValue(), 5.0)
#pybind11#
#pybind11#    def testTicket1123(self):
#pybind11#        """
#pybind11#        Ticket #1123 reported that the Statistics stack routine throws an exception
#pybind11#        when all pixels in a stack are masked.  Returning a NaN pixel in the stack is preferred
#pybind11#        """
#pybind11#
#pybind11#        ctrl = afwMath.StatisticsControl()
#pybind11#        ctrl.setAndMask(~0x0)
#pybind11#
#pybind11#        mimg = afwImage.MaskedImageF(afwGeom.Extent2I(10, 10))
#pybind11#        mimg.set([self.val, 0x1, self.val])
#pybind11#
#pybind11#        # test the case with no valid pixels ... both mean and stdev should be nan
#pybind11#        stat = afwMath.makeStatistics(mimg, afwMath.MEAN | afwMath.STDEV, ctrl)
#pybind11#        mean = stat.getValue(afwMath.MEAN)
#pybind11#        stdev = stat.getValue(afwMath.STDEV)
#pybind11#        self.assertNotEqual(mean, mean)   # NaN does not equal itself
#pybind11#        self.assertNotEqual(stdev, stdev)  # NaN does not equal itself
#pybind11#
#pybind11#        # test the case with one valid pixel ... mean is ok, but stdev should still be nan
#pybind11#        mimg.getMask().set(1, 1, 0x0)
#pybind11#        stat = afwMath.makeStatistics(mimg, afwMath.MEAN | afwMath.STDEV, ctrl)
#pybind11#        mean = stat.getValue(afwMath.MEAN)
#pybind11#        stdev = stat.getValue(afwMath.STDEV)
#pybind11#        self.assertEqual(mean, self.val)
#pybind11#        self.assertNotEqual(stdev, stdev)  # NaN does not equal itself
#pybind11#
#pybind11#        # test the case with two valid pixels ... both mean and stdev are ok
#pybind11#        mimg.getMask().set(1, 2, 0x0)
#pybind11#        stat = afwMath.makeStatistics(mimg, afwMath.MEAN | afwMath.STDEV, ctrl)
#pybind11#        mean = stat.getValue(afwMath.MEAN)
#pybind11#        stdev = stat.getValue(afwMath.STDEV)
#pybind11#        self.assertEqual(mean, self.val)
#pybind11#        self.assertEqual(stdev, 0.0)
#pybind11#
#pybind11#    def testTicket1125(self):
#pybind11#        """Ticket 1125 reported that the clipped routines were aborting when called with no valid pixels. """
#pybind11#        mimg = afwImage.MaskedImageF(afwGeom.Extent2I(10, 10))
#pybind11#        mimg.set([self.val, 0x1, self.val])
#pybind11#
#pybind11#        ctrl = afwMath.StatisticsControl()
#pybind11#        ctrl.setAndMask(~0x0)
#pybind11#
#pybind11#        # test the case with no valid pixels ... try MEANCLIP and STDEVCLIP
#pybind11#        stat = afwMath.makeStatistics(mimg, afwMath.MEANCLIP | afwMath.STDEVCLIP, ctrl)
#pybind11#        mean = stat.getValue(afwMath.MEANCLIP)
#pybind11#        stdev = stat.getValue(afwMath.STDEVCLIP)
#pybind11#        self.assertNotEqual(mean, mean)   # NaN does not equal itself
#pybind11#        self.assertNotEqual(stdev, stdev)  # NaN does not equal itself
#pybind11#
#pybind11#    def testWeightedSum(self):
#pybind11#        ctrl = afwMath.StatisticsControl()
#pybind11#        mi = afwImage.MaskedImageF(afwGeom.Extent2I(10, 10))
#pybind11#        mi.getImage().set(1.0)
#pybind11#        mi.getVariance().set(0.1)
#pybind11#
#pybind11#        stats = afwMath.makeStatistics(mi, afwMath.SUM, ctrl)
#pybind11#        self.assertEqual(stats.getValue(afwMath.SUM), 100.0)
#pybind11#
#pybind11#        ctrl.setWeighted(True)
#pybind11#        weighted = afwMath.makeStatistics(mi, afwMath.SUM, ctrl)
#pybind11#        # precision at "4 places" as images are floats
#pybind11#        # ... variance = 0.1 is stored as 0.100000001
#pybind11#        self.assertAlmostEqual(weighted.getValue(afwMath.SUM), 1000.0, 4)
#pybind11#
#pybind11#    def testWeightedSum2(self):
#pybind11#        """Test using a weight image separate from the variance plane"""
#pybind11#        weight, mean = 0.1, 1.0
#pybind11#
#pybind11#        ctrl = afwMath.StatisticsControl()
#pybind11#        mi = afwImage.MaskedImageF(afwGeom.Extent2I(10, 10))
#pybind11#        npix = 10*10
#pybind11#        mi.getImage().set(mean)
#pybind11#        mi.getVariance().set(np.nan)
#pybind11#
#pybind11#        weights = afwImage.ImageF(mi.getDimensions())
#pybind11#        weights.set(weight)
#pybind11#
#pybind11#        stats = afwMath.makeStatistics(mi, afwMath.SUM, ctrl)
#pybind11#        self.assertEqual(stats.getValue(afwMath.SUM), mean*npix)
#pybind11#
#pybind11#        weighted = afwMath.makeStatistics(mi, weights, afwMath.SUM, ctrl)
#pybind11#        # precision at "4 places" as images are floats
#pybind11#        # ... variance = 0.1 is stored as 0.100000001
#pybind11#        self.assertAlmostEqual(weighted.getValue(afwMath.SUM), mean*npix*weight, 4)
#pybind11#
#pybind11#    def testErrorsFromVariance(self):
#pybind11#        """Test that we can estimate the errors from the incoming variances"""
#pybind11#        weight, mean, variance = 0.1, 1.0, 10.0
#pybind11#
#pybind11#        ctrl = afwMath.StatisticsControl()
#pybind11#        mi = afwImage.MaskedImageF(afwGeom.Extent2I(10, 10))
#pybind11#        npix = 10*10
#pybind11#        mi.getImage().set(mean)
#pybind11#        mi.getVariance().set(variance)
#pybind11#
#pybind11#        weights = afwImage.ImageF(mi.getDimensions())
#pybind11#        weights.set(weight)
#pybind11#
#pybind11#        ctrl.setCalcErrorFromInputVariance(True)
#pybind11#        weighted = afwMath.makeStatistics(mi, weights,
#pybind11#                                          afwMath.MEAN | afwMath.MEANCLIP | afwMath.SUM | afwMath.ERRORS, ctrl)
#pybind11#
#pybind11#        self.assertAlmostEqual(weighted.getValue(afwMath.SUM)/(npix*mean*weight), 1)
#pybind11#        self.assertAlmostEqual(weighted.getValue(afwMath.MEAN), mean)
#pybind11#        self.assertAlmostEqual(weighted.getError(afwMath.MEAN)**2, variance/npix)
#pybind11#        self.assertAlmostEqual(weighted.getError(afwMath.MEANCLIP)**2, variance/npix)
#pybind11#
#pybind11#    def testMeanClipSingleValue(self):
#pybind11#        """Verify that the 3-sigma clipped mean doesn't not return NaN for a single value."""
#pybind11#        stats = afwMath.makeStatistics(self.image, afwMath.MEANCLIP)
#pybind11#        self.assertEqual(stats.getValue(afwMath.MEANCLIP), self.val)
#pybind11#
#pybind11#        # this bug was caused by the iterative nature of the MEANCLIP.
#pybind11#        # With only one point, the sample variance returns NaN to avoid a divide by zero error
#pybind11#        # Thus, on the second iteration, the clip width (based on _variance) is NaN and corrupts
#pybind11#        #   all further calculations.
#pybind11#        img = afwImage.ImageF(afwGeom.Extent2I(1, 1))
#pybind11#        img.set(0)
#pybind11#        stats = afwMath.makeStatistics(img, afwMath.MEANCLIP)
#pybind11#        self.assertEqual(stats.getValue(), 0)
#pybind11#
#pybind11#    def testMismatch(self):
#pybind11#        """Test that we get an exception when there's a size mismatch"""
#pybind11#        scale = 5
#pybind11#        dims = self.image.getDimensions()
#pybind11#        mask = afwImage.MaskU(dims*scale)
#pybind11#        mask.set(0xFF)
#pybind11#        ctrl = afwMath.StatisticsControl()
#pybind11#        ctrl.setAndMask(0xFF)
#pybind11#        # If it didn't raise, this would result in a NaN (the image data is completely masked).
#pybind11#        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, afwMath.makeStatistics,
#pybind11#                          self.image, mask, afwMath.MEDIAN, ctrl)
#pybind11#        subMask = afwImage.MaskU(mask, afwGeom.Box2I(afwGeom.Point2I(dims*(scale - 1)), dims))
#pybind11#        subMask.set(0)
#pybind11#        # Using subMask is successful.
#pybind11#        self.assertEqual(afwMath.makeStatistics(self.image, subMask, afwMath.MEDIAN, ctrl).getValue(),
#pybind11#                         self.val)
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

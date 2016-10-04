#!/usr/bin/env python
from __future__ import absolute_import, division
from builtins import range

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
   ./statistics.py
or
   python
   >>> import statistics; statistics.run()
"""

import math
import os
import numpy as np
import unittest

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.afw.display.ds9 as ds9
import lsst.pex.exceptions as pexExcept

try:
    afwdataDir = lsst.utils.getPackageDir("afwdata")
except pexExcept.NotFoundError:
    afwdataDir = None

try:
    type(display)
except NameError:
    display = False


class StatisticsTestCase(lsst.utils.tests.TestCase):
    """A test case for Statistics"""

    def setUp(self):
        self.val = 10
        self.image = afwImage.ImageF(afwGeom.Extent2I(100, 200))
        self.image.set(self.val)

    def tearDown(self):
        del self.image

    def testDefaultGet(self):
        """Test that we can get a single statistic without specifying it"""
        stats = afwMath.makeStatistics(self.image, afwMath.MEDIAN)

        self.assertEqual(stats.getValue(), stats.getValue(afwMath.MEDIAN))
        self.assertEqual(stats.getResult()[0], stats.getResult(afwMath.MEDIAN)[0])
        #
        stats = afwMath.makeStatistics(self.image, afwMath.MEDIAN | afwMath.ERRORS)

        self.assertEqual(stats.getValue(), stats.getValue(afwMath.MEDIAN))
        self.assertEqual(stats.getResult(), stats.getResult(afwMath.MEDIAN))
        self.assertEqual(stats.getError(), stats.getError(afwMath.MEDIAN))

        def tst():
            stats.getValue()
        stats = afwMath.makeStatistics(self.image, afwMath.MEDIAN | afwMath.MEAN)
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, tst)

    def testStats1(self):
        stats = afwMath.makeStatistics(self.image,
                                       afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM)

        self.assertEqual(stats.getValue(afwMath.NPOINT), self.image.getWidth()*self.image.getHeight())
        self.assertEqual(stats.getValue(afwMath.NPOINT)*stats.getValue(afwMath.MEAN),
                         stats.getValue(afwMath.SUM))
        self.assertEqual(stats.getValue(afwMath.MEAN), self.val)
        self.assertTrue(np.isnan(stats.getError(afwMath.MEAN)))  # didn't ask for error, so it's a NaN
        self.assertEqual(stats.getValue(afwMath.STDEV), 0)

    def testStats2(self):
        stats = afwMath.makeStatistics(self.image, afwMath.STDEV | afwMath.MEAN | afwMath.ERRORS)
        mean = stats.getResult(afwMath.MEAN)
        sd = stats.getValue(afwMath.STDEV)

        self.assertEqual(mean[0], self.image.get(0, 0))
        self.assertEqual(mean[1], sd/math.sqrt(self.image.getWidth()*self.image.getHeight()))

    def testStats3(self):
        stats = afwMath.makeStatistics(self.image, afwMath.NPOINT)

        def getMean():
            stats.getValue(afwMath.MEAN)

        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, getMean)

    def testStatsZebra(self):
        """Add 1 to every other row"""
        image2 = self.image.Factory(self.image, True)
        #
        # Add 1 to every other row, so the variance is 1/4
        #
        self.assertEqual(image2.getHeight() % 2, 0)
        width = image2.getWidth()
        for y in range(1, image2.getHeight(), 2):
            sim = image2.Factory(image2, afwGeom.Box2I(afwGeom.Point2I(0, y), afwGeom.Extent2I(width, 1)),
                                 afwImage.LOCAL)
            sim += 1

        if display:
            ds9.mtv(self.image, frame=0)
            ds9.mtv(image2, frame=1)

        stats = afwMath.makeStatistics(image2,
                                       afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.ERRORS)
        mean = stats.getResult(afwMath.MEAN)
        n = stats.getValue(afwMath.NPOINT)
        sd = stats.getValue(afwMath.STDEV)

        self.assertEqual(mean[0], image2.get(0, 0) + 0.5)
        self.assertEqual(sd, 1/math.sqrt(4.0)*math.sqrt(n/(n - 1)))
        self.assertAlmostEqual(mean[1], sd/math.sqrt(image2.getWidth()*image2.getHeight()), 10)

        meanSquare = afwMath.makeStatistics(image2, afwMath.MEANSQUARE).getValue()
        self.assertEqual(meanSquare, 0.5*(image2.get(0, 0)**2 + image2.get(0, 1)**2))

    def testStatsStdevclip(self):
        """Test STDEVCLIP; cf. #611"""
        image2 = self.image.Factory(self.image, True)

        stats = afwMath.makeStatistics(image2, afwMath.STDEVCLIP | afwMath.NPOINT | afwMath.SUM)
        self.assertEqual(stats.getValue(afwMath.STDEVCLIP), 0)
        #
        # Check we get the correct sum even when clipping
        #
        self.assertEqual(stats.getValue(afwMath.NPOINT) *
                         afwMath.makeStatistics(image2, afwMath.MEAN).getValue(),
                         stats.getValue(afwMath.SUM))

    def testMedian(self):
        """Test the median code"""
        stats = afwMath.makeStatistics(self.image, afwMath.MEDIAN)

        self.assertEqual(stats.getValue(afwMath.MEDIAN), self.val)

        values = [1.0, 2.0, 3.0, 2.0]
        self.assertEqual(afwMath.makeStatistics(values, afwMath.MEDIAN).getValue(), 2.0)

    def testIqrange(self):
        """Test the inter-quartile range"""
        stats = afwMath.makeStatistics(self.image, afwMath.IQRANGE)
        self.assertEqual(stats.getValue(afwMath.IQRANGE), 0)

    def testMeanClip(self):
        """Test the 3-sigma clipped mean"""
        stats = afwMath.makeStatistics(self.image, afwMath.MEANCLIP)
        self.assertEqual(stats.getValue(afwMath.MEANCLIP), self.val)

    def testStdevClip(self):
        """Test the 3-sigma clipped standard deviation"""
        stats = afwMath.makeStatistics(self.image, afwMath.STDEVCLIP)
        self.assertEqual(stats.getValue(afwMath.STDEVCLIP), 0)

    def testVarianceClip(self):
        """Test the 3-sigma clipped variance"""
        stats = afwMath.makeStatistics(self.image, afwMath.VARIANCECLIP)
        self.assertEqual(stats.getValue(afwMath.VARIANCECLIP), 0)

    def _testBadValue(self, badVal):
        """Test that we can handle an instance of `badVal` in the data correctly"""
        x, y = 10, 10
        for useImage in [True, False]:
            if useImage:
                self.image = afwImage.ImageF(100, 100)
                self.image.set(self.val)
                self.image.set(x, y, badVal)
            else:
                self.image = afwImage.MaskedImageF(100, 100)
                self.image.set(self.val, 0x0, 1.0)
                self.image.set(x, y, (badVal, 0x0, 1.0))

            self.assertEqual(afwMath.makeStatistics(self.image, afwMath.MAX).getValue(), self.val)
            self.assertEqual(afwMath.makeStatistics(self.image, afwMath.MEAN).getValue(), self.val)

            sctrl = afwMath.StatisticsControl()

            sctrl.setNanSafe(False)
            self.assertFalse(np.isfinite(afwMath.makeStatistics(self.image, afwMath.MAX, sctrl).getValue()))
            self.assertFalse(np.isfinite(afwMath.makeStatistics(self.image, afwMath.MEAN, sctrl).getValue()))

    def testMaxWithNan(self):
        """Test that we can handle NaNs correctly"""
        self._testBadValue(np.nan)

    def testMaxWithInf(self):
        """Test that we can handle infinities correctly"""
        self._testBadValue(np.inf)

    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
    def testSampleImageStats(self):
        """ Compare our results to known values in test data """

        imgfiles = []
        imgfiles.append("v1_i1_g_m400_s20_f.fits")
        imgfiles.append("v1_i1_g_m400_s20_u16.fits")
        imgfiles.append("v1_i2_g_m400_s20_f.fits")
        imgfiles.append("v1_i2_g_m400_s20_u16.fits")
        imgfiles.append("v2_i1_p_m9_f.fits")
        imgfiles.append("v2_i1_p_m9_u16.fits")
        imgfiles.append("v2_i2_p_m9_f.fits")
        imgfiles.append("v2_i2_p_m9_u16.fits")

        afwdataDir = os.getenv("AFWDATA_DIR")

        for imgfile in imgfiles:

            imgPath = os.path.join(afwdataDir, "Statistics", imgfile)

            # get the image and header
            dimg = afwImage.DecoratedImageF(imgPath)
            fitsHdr = dimg.getMetadata()

            # get the true values of the mean and stdev
            trueMean = fitsHdr.getAsDouble("MEANCOMP")
            trueStdev = fitsHdr.getAsDouble("SIGCOMP")

            # measure the mean and stdev with the Statistics class
            img = dimg.getImage()
            statobj = afwMath.makeStatistics(img, afwMath.MEAN | afwMath.STDEV)
            mean = statobj.getValue(afwMath.MEAN)
            stdev = statobj.getValue(afwMath.STDEV)

            # print trueMean, mean, trueStdev, stdev
            self.assertAlmostEqual(mean, trueMean, 8)
            self.assertAlmostEqual(stdev, trueStdev, 8)

    def testStatisticsRamp(self):
        """ Tests Statistics on a 'ramp' (image with constant gradient) """

        nx = 101
        ny = 64
        img = afwImage.ImageF(afwGeom.Extent2I(nx, ny))

        z0 = 10.0
        dzdx = 1.0
        mean = z0 + (nx//2)*dzdx
        stdev = 0.0
        for y in range(ny):
            for x in range(nx):
                z = z0 + dzdx*x
                img.set(x, y, z)
                stdev += (z - mean)*(z - mean)

        stdev = math.sqrt(stdev/(nx*ny - 1))

        stats = afwMath.makeStatistics(img, afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN)
        testmean = stats.getValue(afwMath.MEAN)
        teststdev = stats.getValue(afwMath.STDEV)

        self.assertEqual(stats.getValue(afwMath.NPOINT), nx*ny)
        self.assertEqual(testmean, mean)
        self.assertAlmostEqual(teststdev, stdev)

        stats = afwMath.makeStatistics(img, afwMath.STDEV | afwMath.MEAN | afwMath.ERRORS)
        mean, meanErr = stats.getResult(afwMath.MEAN)
        sd = stats.getValue(afwMath.STDEV)

        self.assertEqual(mean, img.get(nx//2, ny//2))
        self.assertEqual(meanErr, sd/math.sqrt(img.getWidth()*img.getHeight()))

        # ===============================================================================
        # sjb code for percentiles and clipped stats

        stats = afwMath.makeStatistics(img, afwMath.MEDIAN)
        self.assertEqual(z0 + dzdx*(nx - 1)/2.0, stats.getValue(afwMath.MEDIAN))

        stats = afwMath.makeStatistics(img, afwMath.IQRANGE)
        self.assertEqual(dzdx*(nx - 1)/2.0, stats.getValue(afwMath.IQRANGE))

        stats = afwMath.makeStatistics(img, afwMath.MEANCLIP)
        self.assertEqual(z0 + dzdx*(nx - 1)/2.0, stats.getValue(afwMath.MEANCLIP))

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

    def testTicket1025(self):
        """
        Ticket #1025 reported that the Statistics median was getting '3' as the median of [1,2,3,2]
        it was caused by an off-by-one error in the implementation
        """

        # check the exact example in the ticket
        values = [1.0, 2.0, 3.0, 2.0]
        self.assertEqual(afwMath.makeStatistics(values, afwMath.MEDIAN).getValue(), 2)
        self.assertEqual(afwMath.makeStatistics(sorted(values), afwMath.MEDIAN).getValue(), 2)

        # check some other possible ways it could show up
        values = list(range(10))
        self.assertEqual(afwMath.makeStatistics(values, afwMath.MEDIAN).getValue(), 4.5)
        values = list(range(11))
        self.assertEqual(afwMath.makeStatistics(values, afwMath.MEDIAN).getValue(), 5.0)

    def testTicket1123(self):
        """
        Ticket #1123 reported that the Statistics stack routine throws an exception
        when all pixels in a stack are masked.  Returning a NaN pixel in the stack is preferred
        """

        ctrl = afwMath.StatisticsControl()
        ctrl.setAndMask(~0x0)

        mimg = afwImage.MaskedImageF(afwGeom.Extent2I(10, 10))
        mimg.set([self.val, 0x1, self.val])

        # test the case with no valid pixels ... both mean and stdev should be nan
        stat = afwMath.makeStatistics(mimg, afwMath.MEAN | afwMath.STDEV, ctrl)
        mean = stat.getValue(afwMath.MEAN)
        stdev = stat.getValue(afwMath.STDEV)
        self.assertNotEqual(mean, mean)   # NaN does not equal itself
        self.assertNotEqual(stdev, stdev)  # NaN does not equal itself

        # test the case with one valid pixel ... mean is ok, but stdev should still be nan
        mimg.getMask().set(1, 1, 0x0)
        stat = afwMath.makeStatistics(mimg, afwMath.MEAN | afwMath.STDEV, ctrl)
        mean = stat.getValue(afwMath.MEAN)
        stdev = stat.getValue(afwMath.STDEV)
        self.assertEqual(mean, self.val)
        self.assertNotEqual(stdev, stdev)  # NaN does not equal itself

        # test the case with two valid pixels ... both mean and stdev are ok
        mimg.getMask().set(1, 2, 0x0)
        stat = afwMath.makeStatistics(mimg, afwMath.MEAN | afwMath.STDEV, ctrl)
        mean = stat.getValue(afwMath.MEAN)
        stdev = stat.getValue(afwMath.STDEV)
        self.assertEqual(mean, self.val)
        self.assertEqual(stdev, 0.0)

    def testTicket1125(self):
        """Ticket 1125 reported that the clipped routines were aborting when called with no valid pixels. """
        mimg = afwImage.MaskedImageF(afwGeom.Extent2I(10, 10))
        mimg.set([self.val, 0x1, self.val])

        ctrl = afwMath.StatisticsControl()
        ctrl.setAndMask(~0x0)

        # test the case with no valid pixels ... try MEANCLIP and STDEVCLIP
        stat = afwMath.makeStatistics(mimg, afwMath.MEANCLIP | afwMath.STDEVCLIP, ctrl)
        mean = stat.getValue(afwMath.MEANCLIP)
        stdev = stat.getValue(afwMath.STDEVCLIP)
        self.assertNotEqual(mean, mean)   # NaN does not equal itself
        self.assertNotEqual(stdev, stdev)  # NaN does not equal itself

    def testWeightedSum(self):
        ctrl = afwMath.StatisticsControl()
        mi = afwImage.MaskedImageF(afwGeom.Extent2I(10, 10))
        mi.getImage().set(1.0)
        mi.getVariance().set(0.1)

        stats = afwMath.makeStatistics(mi, afwMath.SUM, ctrl)
        self.assertEqual(stats.getValue(afwMath.SUM), 100.0)

        ctrl.setWeighted(True)
        weighted = afwMath.makeStatistics(mi, afwMath.SUM, ctrl)
        # precision at "4 places" as images are floats
        # ... variance = 0.1 is stored as 0.100000001
        self.assertAlmostEqual(weighted.getValue(afwMath.SUM), 1000.0, 4)

    def testWeightedSum2(self):
        """Test using a weight image separate from the variance plane"""
        weight, mean = 0.1, 1.0

        ctrl = afwMath.StatisticsControl()
        mi = afwImage.MaskedImageF(afwGeom.Extent2I(10, 10))
        npix = 10*10
        mi.getImage().set(mean)
        mi.getVariance().set(np.nan)

        weights = afwImage.ImageF(mi.getDimensions())
        weights.set(weight)

        stats = afwMath.makeStatistics(mi, afwMath.SUM, ctrl)
        self.assertEqual(stats.getValue(afwMath.SUM), mean*npix)

        weighted = afwMath.makeStatistics(mi, weights, afwMath.SUM, ctrl)
        # precision at "4 places" as images are floats
        # ... variance = 0.1 is stored as 0.100000001
        self.assertAlmostEqual(weighted.getValue(afwMath.SUM), mean*npix*weight, 4)

    def testErrorsFromVariance(self):
        """Test that we can estimate the errors from the incoming variances"""
        weight, mean, variance = 0.1, 1.0, 10.0

        ctrl = afwMath.StatisticsControl()
        mi = afwImage.MaskedImageF(afwGeom.Extent2I(10, 10))
        npix = 10*10
        mi.getImage().set(mean)
        mi.getVariance().set(variance)

        weights = afwImage.ImageF(mi.getDimensions())
        weights.set(weight)

        ctrl.setCalcErrorFromInputVariance(True)
        weighted = afwMath.makeStatistics(mi, weights,
                                          afwMath.MEAN | afwMath.MEANCLIP | afwMath.SUM | afwMath.ERRORS,
                                          ctrl)

        self.assertAlmostEqual(weighted.getValue(afwMath.SUM)/(npix*mean*weight), 1)
        self.assertAlmostEqual(weighted.getValue(afwMath.MEAN), mean)
        self.assertAlmostEqual(weighted.getError(afwMath.MEAN)**2, variance/npix)
        self.assertAlmostEqual(weighted.getError(afwMath.MEANCLIP)**2, variance/npix)

    def testMeanClipSingleValue(self):
        """Verify that the 3-sigma clipped mean doesn't not return NaN for a single value."""
        stats = afwMath.makeStatistics(self.image, afwMath.MEANCLIP)
        self.assertEqual(stats.getValue(afwMath.MEANCLIP), self.val)

        # this bug was caused by the iterative nature of the MEANCLIP.
        # With only one point, the sample variance returns NaN to avoid a divide by zero error
        # Thus, on the second iteration, the clip width (based on _variance) is NaN and corrupts
        #   all further calculations.
        img = afwImage.ImageF(afwGeom.Extent2I(1, 1))
        img.set(0)
        stats = afwMath.makeStatistics(img, afwMath.MEANCLIP)
        self.assertEqual(stats.getValue(), 0)

    def testMismatch(self):
        """Test that we get an exception when there's a size mismatch"""
        scale = 5
        dims = self.image.getDimensions()
        mask = afwImage.MaskU(dims*scale)
        mask.set(0xFF)
        ctrl = afwMath.StatisticsControl()
        ctrl.setAndMask(0xFF)
        # If it didn't raise, this would result in a NaN (the image data is completely masked).
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, afwMath.makeStatistics,
                          self.image, mask, afwMath.MEDIAN, ctrl)
        subMask = afwImage.MaskU(mask, afwGeom.Box2I(afwGeom.Point2I(dims*(scale - 1)), dims))
        subMask.set(0)
        # Using subMask is successful.
        self.assertEqual(afwMath.makeStatistics(self.image, subMask, afwMath.MEDIAN, ctrl).getValue(),
                         self.val)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

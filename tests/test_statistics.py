# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
import unittest

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display as afwDisplay
import lsst.pex.exceptions as pexExcept

afwDisplay.setDefaultMaskTransparency(75)

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

    clippedVariance3 = 0.9733369     # variance of an N(0, 1) Gaussian clipped at 3 sigma

    def setUp(self):
        w, h = 900, 1500

        mean = 10.5                      # requested mean
        std = 1.0                        # and standard deviation

        self.images = []
        # ImageI
        np.random.seed(666)
        isInt = True
        image = afwImage.ImageI(lsst.geom.ExtentI(w, h))
        image.array[:] = np.floor(np.random.normal(mean, std, (h, w)) + 0.5).astype(int)

        # Note that the mean/median/std may not be quite equal to the requested values
        self.images.append((image, isInt, np.mean(image.array), np.mean(image.array), np.std(image.array)))

        # ImageF
        np.random.seed(666)
        isInt = False
        image = afwImage.ImageF(lsst.geom.ExtentI(w, h))
        image.array[:] = np.random.normal(mean, std, (h, w))

        # Note that the mean/median/std may not be quite equal to the requested values
        self.images.append((image, isInt, np.mean(image.array), np.median(image.array), np.std(image.array)))

    @staticmethod
    def delta(what, isInt):
        # Return a tolerance for a test
        if what == "mean":
            return 4e-6
        elif what == "meanclip":
            return 4e-5
        elif what == "median":
            return 0.00022 if isInt else 0.00000075

    def tearDown(self):
        del self.images

    def testDefaultGet(self):
        """Test that we can get a single statistic without specifying it"""
        for image, isInt, mean, median, std in self.images:
            stats = afwMath.makeStatistics(image, afwMath.MEDIAN)

            self.assertEqual(stats.getValue(), stats.getValue(afwMath.MEDIAN))
            self.assertEqual(stats.getResult()[0], stats.getResult(afwMath.MEDIAN)[0])
            #
            stats = afwMath.makeStatistics(image, afwMath.MEDIAN | afwMath.ERRORS)

            self.assertEqual(stats.getValue(), stats.getValue(afwMath.MEDIAN))
            self.assertEqual(stats.getResult(), stats.getResult(afwMath.MEDIAN))
            self.assertEqual(stats.getError(), stats.getError(afwMath.MEDIAN))

            def tst():
                stats.getValue()
            stats = afwMath.makeStatistics(image, afwMath.MEDIAN | afwMath.MEAN)
            self.assertRaises(lsst.pex.exceptions.InvalidParameterError, tst)

    def testStats1(self):
        for image, isInt, mean, median, std in self.images:
            stats = afwMath.makeStatistics(image, afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM)

            self.assertEqual(stats.getValue(afwMath.NPOINT), image.getWidth()*image.getHeight())
            self.assertEqual(stats.getValue(afwMath.NPOINT)*stats.getValue(afwMath.MEAN),
                             stats.getValue(afwMath.SUM))

            self.assertAlmostEqual(stats.getValue(afwMath.MEAN), mean, delta=self.delta("mean", isInt))
            # didn't ask for error, so it's a NaN
            self.assertTrue(np.isnan(stats.getError(afwMath.MEAN)))
            self.assertAlmostEqual(stats.getValue(afwMath.STDEV), std, delta=0.000008)

    def testStats2(self):
        for image, isInt, mean, median, std in self.images:
            stats = afwMath.makeStatistics(image, afwMath.STDEV | afwMath.MEAN | afwMath.ERRORS)
            meanRes = stats.getResult(afwMath.MEAN)
            sd = stats.getValue(afwMath.STDEV)

            self.assertAlmostEqual(meanRes[0], mean, delta=self.delta("mean", isInt))
            self.assertAlmostEqual(meanRes[1], sd/math.sqrt(image.getWidth()*image.getHeight()))

    def testStats3(self):
        for image, isInt, mean, median, std in self.images:
            stats = afwMath.makeStatistics(image, afwMath.NPOINT)

            def getMean():
                stats.getValue(afwMath.MEAN)

            self.assertRaises(lsst.pex.exceptions.InvalidParameterError, getMean)

    def testStatsZebra(self):
        """Add 1 to every other row"""
        for image, isInt, mean, median, std in self.images:
            image2 = image.clone()
            #
            # Add 1 to every other row, so the variance is increased by 1/4
            #
            self.assertEqual(image2.getHeight() % 2, 0)
            width = image2.getWidth()
            for y in range(1, image2.getHeight(), 2):
                sim = image2[lsst.geom.Box2I(lsst.geom.Point2I(0, y), lsst.geom.Extent2I(width, 1))]
                sim += 1

            if display:
                afwDisplay.Display(frame=0).mtv(image, "Image 1")
                afwDisplay.Display(frame=1).mtv(image2, "Image 2 (var inc by 1/4)")

            stats = afwMath.makeStatistics(image2,
                                           afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.ERRORS)
            meanRes = stats.getResult(afwMath.MEAN)
            n = stats.getValue(afwMath.NPOINT)
            sd = stats.getValue(afwMath.STDEV)

            self.assertAlmostEqual(meanRes[0], mean + 0.5, delta=self.delta("mean", isInt))
            self.assertAlmostEqual(sd, np.hypot(std, 1/math.sqrt(4.0)*math.sqrt(n/(n - 1))),
                                   delta=0.00011)
            self.assertAlmostEqual(meanRes[1], sd/math.sqrt(image2.getWidth()*image2.getHeight()), 10)

            meanSquare = afwMath.makeStatistics(image2, afwMath.MEANSQUARE).getValue()
            self.assertAlmostEqual(meanSquare, 0.5*(mean**2 + (mean + 1)**2) + std**2,
                                   delta=0.00025 if isInt else 0.00006)

    def testStatsStdevclip(self):
        """Test STDEVCLIP; cf. #611"""
        for image, isInt, mean, median, std in self.images:
            image2 = image.clone()

            stats = afwMath.makeStatistics(image2, afwMath.STDEVCLIP | afwMath.NPOINT | afwMath.SUM)
            self.assertAlmostEqual(stats.getValue(afwMath.STDEVCLIP), math.sqrt(self.clippedVariance3)*std,
                                   delta=0.0015)
            #
            # Check we get the correct sum even when clipping
            #
            self.assertEqual(
                stats.getValue(afwMath.NPOINT)*afwMath.makeStatistics(image2, afwMath.MEAN).getValue(),
                stats.getValue(afwMath.SUM))

    def testMedian(self):
        """Test the median code"""
        for image, isInt, mean, median, std in self.images:
            med = afwMath.makeStatistics(image, afwMath.MEDIAN).getValue()
            self.assertAlmostEqual(med, median, delta=self.delta("median", isInt))

            values = [1.0, 2.0, 3.0, 2.0]
            self.assertEqual(afwMath.makeStatistics(values, afwMath.MEDIAN).getValue(), 2.0)

    def testIqrange(self):
        """Test the inter-quartile range"""
        for image, isInt, mean, median, std in self.images:
            iqr = afwMath.makeStatistics(image, afwMath.IQRANGE).getValue()
            # pretty loose constraint for isInt; probably because the distribution
            # isn't very Gaussian with the added rounding to integer values
            self.assertAlmostEqual(iqr, std/0.741301109252802, delta=0.063 if isInt else 0.00011)

    def testMeanClip(self):
        """Test the clipped mean"""

        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(6)

        for image, isInt, mean, median, std in self.images:
            stats = afwMath.makeStatistics(image, afwMath.MEANCLIP | afwMath.NCLIPPED, sctrl)
            self.assertAlmostEqual(stats.getValue(afwMath.MEANCLIP), mean, delta=self.delta("mean", isInt))
            self.assertEqual(stats.getValue(afwMath.NCLIPPED), 0)

    def testVarianceClip(self):
        """Test the 3-sigma clipped standard deviation and variance"""
        for image, isInt, mean, median, std in self.images:
            delta = 0.0006 if isInt else 0.0014
            stdevClip = afwMath.makeStatistics(image, afwMath.STDEVCLIP).getValue()
            self.assertAlmostEqual(stdevClip, math.sqrt(self.clippedVariance3)*std, delta=delta)

            varianceClip = afwMath.makeStatistics(image, afwMath.VARIANCECLIP).getValue()
            self.assertAlmostEqual(varianceClip, self.clippedVariance3*std**2, delta=2*delta)

    def _testBadValue(self, badVal):
        """Test that we can handle an instance of `badVal` in the data correctly

        Note that we only test ImageF here (as ImageI can't contain a NaN)
        """
        mean = self.images[0][1]
        x, y = 10, 10
        for useImage in [True, False]:
            if useImage:
                image = afwImage.ImageF(100, 100)
                image.set(mean)
                image[x, y] = badVal
            else:
                image = afwImage.MaskedImageF(100, 100)
                image.set(mean, 0x0, 1.0)
                image[x, y] = (badVal, 0x0, 1.0)

            self.assertEqual(afwMath.makeStatistics(image, afwMath.MAX).getValue(), mean)
            self.assertEqual(afwMath.makeStatistics(image, afwMath.MEAN).getValue(), mean)

            sctrl = afwMath.StatisticsControl()

            sctrl.setNanSafe(False)
            self.assertFalse(np.isfinite(afwMath.makeStatistics(image, afwMath.MAX, sctrl).getValue()))
            self.assertFalse(np.isfinite(afwMath.makeStatistics(image, afwMath.MEAN, sctrl).getValue()))

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
        img = afwImage.ImageF(lsst.geom.Extent2I(nx, ny))

        z0 = 10.0
        dzdx = 1.0
        mean = z0 + (nx//2)*dzdx
        stdev = 0.0
        for y in range(ny):
            for x in range(nx):
                z = z0 + dzdx*x
                img[x, y] = z
                stdev += (z - mean)*(z - mean)

        stdev = math.sqrt(stdev/(nx*ny - 1))

        stats = afwMath.makeStatistics(
            img, afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN)
        testmean = stats.getValue(afwMath.MEAN)
        teststdev = stats.getValue(afwMath.STDEV)

        self.assertEqual(stats.getValue(afwMath.NPOINT), nx*ny)
        self.assertEqual(testmean, mean)
        self.assertAlmostEqual(teststdev, stdev)

        stats = afwMath.makeStatistics(
            img, afwMath.STDEV | afwMath.MEAN | afwMath.ERRORS)
        mean, meanErr = stats.getResult(afwMath.MEAN)
        sd = stats.getValue(afwMath.STDEV)

        self.assertEqual(mean, img[nx//2, ny//2])
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
        mask = afwImage.Mask(lsst.geom.Extent2I(10, 10))
        mask.set(0x0)

        mask[1, 1] = 0x10
        mask[3, 1] = 0x08
        mask[5, 4] = 0x08
        mask[4, 5] = 0x02

        stats = afwMath.makeStatistics(mask, afwMath.SUM | afwMath.NPOINT)
        self.assertEqual(mask.getWidth()*mask.getHeight(),
                         stats.getValue(afwMath.NPOINT))
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

        mean = self.images[0][1]

        ctrl = afwMath.StatisticsControl()
        ctrl.setAndMask(~0x0)

        mimg = afwImage.MaskedImageF(lsst.geom.Extent2I(10, 10))
        mimg.set([mean, 0x1, mean])

        # test the case with no valid pixels ... both mean and stdev should be
        # nan
        stat = afwMath.makeStatistics(mimg, afwMath.MEAN | afwMath.STDEV, ctrl)
        mean = stat.getValue(afwMath.MEAN)
        stdev = stat.getValue(afwMath.STDEV)
        self.assertNotEqual(mean, mean)   # NaN does not equal itself
        self.assertNotEqual(stdev, stdev)  # NaN does not equal itself

        # test the case with one valid pixel ... mean is ok, but stdev should
        # still be nan
        mimg.getMask()[1, 1] = 0x0
        stat = afwMath.makeStatistics(mimg, afwMath.MEAN | afwMath.STDEV, ctrl)
        mean = stat.getValue(afwMath.MEAN)
        stdev = stat.getValue(afwMath.STDEV)
        self.assertEqual(mean, mean)
        self.assertNotEqual(stdev, stdev)  # NaN does not equal itself

        # test the case with two valid pixels ... both mean and stdev are ok
        mimg.getMask()[1, 2] = 0x0
        stat = afwMath.makeStatistics(mimg, afwMath.MEAN | afwMath.STDEV, ctrl)
        mean = stat.getValue(afwMath.MEAN)
        stdev = stat.getValue(afwMath.STDEV)
        self.assertEqual(mean, mean)
        self.assertEqual(stdev, 0.0)

    def testTicket1125(self):
        """Ticket 1125 reported that the clipped routines were aborting when called with no valid pixels. """

        mean = self.images[0][1]

        mimg = afwImage.MaskedImageF(lsst.geom.Extent2I(10, 10))
        mimg.set([mean, 0x1, mean])

        ctrl = afwMath.StatisticsControl()
        ctrl.setAndMask(~0x0)

        # test the case with no valid pixels ... try MEANCLIP and STDEVCLIP
        stat = afwMath.makeStatistics(
            mimg, afwMath.MEANCLIP | afwMath.STDEVCLIP, ctrl)
        mean = stat.getValue(afwMath.MEANCLIP)
        stdev = stat.getValue(afwMath.STDEVCLIP)
        self.assertNotEqual(mean, mean)   # NaN does not equal itself
        self.assertNotEqual(stdev, stdev)  # NaN does not equal itself

    def testWeightedSum(self):
        ctrl = afwMath.StatisticsControl()
        mi = afwImage.MaskedImageF(lsst.geom.Extent2I(10, 10))
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
        mi = afwImage.MaskedImageF(lsst.geom.Extent2I(10, 10))
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
        mi = afwImage.MaskedImageF(lsst.geom.Extent2I(10, 10))
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
        """Verify that the clipped mean doesn't not return NaN for a single value."""
        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(6)

        for image, isInt, mean, median, std in self.images:
            stats = afwMath.makeStatistics(image, afwMath.MEANCLIP | afwMath.NCLIPPED, sctrl)
            self.assertAlmostEqual(stats.getValue(afwMath.MEANCLIP), mean,
                                   delta=self.delta("meanclip", isInt))
            self.assertEqual(stats.getValue(afwMath.NCLIPPED), 0)

            # this bug was caused by the iterative nature of the MEANCLIP.
            # With only one point, the sample variance returns NaN to avoid a divide by zero error
            # Thus, on the second iteration, the clip width (based on _variance) is NaN and corrupts
            #   all further calculations.
            img = afwImage.ImageF(lsst.geom.Extent2I(1, 1))
            img.set(0)
            stats = afwMath.makeStatistics(img, afwMath.MEANCLIP | afwMath.NCLIPPED)
            self.assertEqual(stats.getValue(afwMath.MEANCLIP), 0)
            self.assertEqual(stats.getValue(afwMath.NCLIPPED), 0)

    def testMismatch(self):
        """Test that we get an exception when there's a size mismatch"""
        scale = 5
        for image, isInt, mean, median, std in self.images:
            dims = image.getDimensions()
            mask = afwImage.Mask(dims*scale)
            mask.set(0xFF)
            ctrl = afwMath.StatisticsControl()
            ctrl.setAndMask(0xFF)
            # If it didn't raise, this would result in a NaN (the image data is
            # completely masked).
            self.assertRaises(lsst.pex.exceptions.InvalidParameterError, afwMath.makeStatistics,
                              image, mask, afwMath.MEDIAN, ctrl)
            subMask = afwImage.Mask(mask, lsst.geom.Box2I(lsst.geom.Point2I(dims*(scale - 1)), dims))
            subMask.set(0)
            # Using subMask is successful.
            self.assertAlmostEqual(afwMath.makeStatistics(image, subMask, afwMath.MEDIAN, ctrl).getValue(),
                                   median, delta=self.delta("median", isInt))

    def testClipping(self):
        """Test that clipping statistics work

        Insert a single bad pixel; it should be clipped.
        """
        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(10)

        for image, isInt, mean, median, std in self.images:
            nval = 1000*mean
            if isInt:
                nval = int(nval)
            image[0, 0] = nval

            stats = afwMath.makeStatistics(image, afwMath.MEANCLIP | afwMath.NCLIPPED | afwMath.NPOINT, sctrl)
            self.assertAlmostEqual(stats.getValue(afwMath.MEANCLIP), mean,
                                   delta=self.delta("meanclip", isInt))
            self.assertEqual(stats.getValue(afwMath.NCLIPPED), 1)
            self.assertEqual(stats.getValue(afwMath.NPOINT), image.getBBox().getArea())

    def testNMasked(self):
        """Test that NMASKED works"""
        maskVal = 0xBE
        ctrl = afwMath.StatisticsControl()
        ctrl.setAndMask(maskVal)
        for image, isInt, mean, median, std in self.images:
            mask = afwImage.Mask(image.getBBox())
            mask.set(0)
            self.assertEqual(afwMath.makeStatistics(image, mask, afwMath.NMASKED, ctrl).getValue(), 0)
            mask[1, 1] = maskVal
            self.assertEqual(afwMath.makeStatistics(image, mask, afwMath.NMASKED, ctrl).getValue(), 1)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

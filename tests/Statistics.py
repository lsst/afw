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

    def testDefaultGet(self):
	"""Test that we can get a single statistic without specifying it"""
        stats = afwMath.StatisticsF(self.image, afwMath.MEDIAN)

        self.assertEqual(stats.getValue(), stats.getValue(afwMath.MEDIAN))
        self.assertEqual(stats.getResult()[0], stats.getResult(afwMath.MEDIAN)[0])
        #
        stats = afwMath.StatisticsF(self.image, afwMath.MEDIAN | afwMath.ERRORS)

        self.assertEqual(stats.getValue(), stats.getValue(afwMath.MEDIAN))
        self.assertEqual(stats.getResult(), stats.getResult(afwMath.MEDIAN))
        self.assertEqual(stats.getError(), stats.getError(afwMath.MEDIAN))

        def tst():
            stats.getValue()
        stats = afwMath.StatisticsF(self.image, afwMath.MEDIAN | afwMath.MEAN)
        utilsTests.assertRaisesLsstCpp(self, lsst.pex.exceptions.InvalidParameterException, tst)

    def testStats1(self):
        stats = afwMath.StatisticsF(self.image, afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM)

        self.assertEqual(stats.getValue(afwMath.NPOINT), self.image.getWidth()*self.image.getHeight())
        self.assertEqual(stats.getValue(afwMath.NPOINT)*stats.getValue(afwMath.MEAN), stats.getValue(afwMath.SUM))
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

        utilsTests.assertRaisesLsstCpp(self, lsst.pex.exceptions.InvalidParameterException, getMean)

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

        stats = afwMath.makeStatistics(image2, afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.ERRORS)
        mean = stats.getResult(afwMath.MEAN)
        n = stats.getValue(afwMath.NPOINT)
        sd = stats.getValue(afwMath.STDEV)

        self.assertEqual(mean[0],  image2.get(0,0) + 0.5)
        self.assertEqual(sd, 1/math.sqrt(4.0)*math.sqrt(n/(n - 1)))
        self.assertAlmostEqual(mean[1], sd/math.sqrt(image2.getWidth()*image2.getHeight()), 10)

    def testStatsStdevclip(self):
	"""Test STDEVCLIP; cf. #611"""
        image2 = self.image.Factory(self.image, True)

        stats = afwMath.makeStatistics(image2, afwMath.STDEVCLIP | afwMath.NPOINT | afwMath.SUM)
        self.assertEqual(stats.getValue(afwMath.STDEVCLIP), 0)
        #
        # Check we get the correct sum even when clipping
        #
        self.assertEqual(stats.getValue(afwMath.NPOINT)*afwMath.makeStatistics(image2, afwMath.MEAN).getValue(),
                         stats.getValue(afwMath.SUM))

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
	stats = afwMath.StatisticsF(self.image, afwMath.STDEVCLIP)

	self.assertEqual(stats.getValue(afwMath.STDEVCLIP), 0)

    def testVarianceClip(self):
	"""Test the 3-sigma clipped variance"""
	stats = afwMath.StatisticsF(self.image, afwMath.VARIANCECLIP)

	self.assertEqual(stats.getValue(afwMath.VARIANCECLIP), 0)

    def testSampleImageStats(self):

	imgfiles = []
        imgfiles.append("v1_i1_g_m400_s20_f.fits")
        imgfiles.append("v1_i1_g_m400_s20_u16.fits")
        imgfiles.append("v1_i2_g_m400_s20_f.fits")
        imgfiles.append("v1_i2_g_m400_s20_u16.fits")
        imgfiles.append("v2_i1_p_m9_f.fits")
        imgfiles.append("v2_i1_p_m9_u16.fits")
        imgfiles.append("v2_i2_p_m9_f.fits")
        imgfiles.append("v2_i2_p_m9_u16.fits")

        afwdata_dir = os.getenv("AFWDATA_DIR")
	for imgfile in imgfiles:
            
            img_path = afwdata_dir + "/Statistics/" + imgfile

            # get the image and header
	    dimg = afwImage.DecoratedImageF(img_path)
	    fitsHdr = dimg.getMetadata()

            # get the true values of the mean and stdev
            true_mean = fitsHdr.getAsDouble("MEANCOMP")
            true_stdev = fitsHdr.getAsDouble("SIGCOMP")

            # measure the mean and stdev with the Statistics class
            img = dimg.getImage()
            statobj = afwMath.StatisticsF(img, afwMath.MEAN | afwMath.STDEV)
            mean = statobj.getValue(afwMath.MEAN)
            stdev = statobj.getValue(afwMath.STDEV)

            # print true_mean, mean, true_stdev, stdev
            self.assertAlmostEqual(mean, true_mean, 8)
            self.assertAlmostEqual(stdev, true_stdev, 8)


    # Now do tests on a 'ramp' (image with constant gradient)
    def testStatisticsRamp(self):

	nx = 101
	ny = 64
	img = afwImage.ImageF(nx,ny)
    
	z0 = 10.0
	dzdx = 1.0
	mean = z0 + (nx/2)*dzdx
	stdev = 0.0
	for y in range(ny):
	    for x in range(nx):
		z = z0 + dzdx*x
		img.set(x, y, z)
		stdev += (z - mean)*(z - mean)
		
	stdev = math.sqrt(stdev/(nx*ny-1))
	    
	stats = afwMath.StatisticsF(img, afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN)
	testmean = stats.getValue(afwMath.MEAN)
	teststdev = stats.getValue(afwMath.STDEV)
	
	self.assertEqual(stats.getValue(afwMath.NPOINT), nx*ny)
	self.assertEqual(testmean, mean)
	self.assertEqual(teststdev, stdev )
	    
	stats = afwMath.StatisticsF(img, afwMath.STDEV | afwMath.MEAN | afwMath.ERRORS)
	mean, mean_err = stats.getResult(afwMath.MEAN)
	sd = stats.getValue(afwMath.STDEV)
	
	self.assertEqual(mean,  img.get(nx/2,ny/2))
	self.assertEqual(mean_err, sd/math.sqrt(img.getWidth()*img.getHeight()))
	
	# ===============================================================================
	# sjb code for percentiles and clipped stats

	stats = afwMath.StatisticsF(img, afwMath.MEDIAN)
	self.assertEqual(z0+dzdx*(nx-1)/2.0, stats.getValue(afwMath.MEDIAN))

	stats = afwMath.StatisticsF(img, afwMath.IQRANGE)
	self.assertEqual(dzdx*(nx-1)/2.0, stats.getValue(afwMath.IQRANGE))

	stats = afwMath.StatisticsF(img, afwMath.MEANCLIP)
	self.assertEqual(z0+dzdx*(nx-1)/2.0, stats.getValue(afwMath.MEANCLIP))
    


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(StatisticsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

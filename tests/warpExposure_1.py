#!/usr/bin/env python
"""
Test warpExposure

Author: Nicole M. Silvestri, University of Washington
Contact: nms@astro.washington.edu
Created on: Thu Sep 20, 2007
"""

import os
import math
import pdb # we may want to say pdb.set_trace()
import unittest

import numpy

import eups
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.image.testUtils as imageTestUtils
import lsst.utils.tests as utilsTests
import lsst.pex.logging as logging

try:
    type(verbose)
except NameError:
    verbose = 0
    logging.Trace_setVerbosity("lsst.afw.math", verbose)

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests")

OriginalExposureName = "med"
SwarpedImageName = "medswarp1lanczos4.fits"

OriginalExposurePath = os.path.join(dataDir, OriginalExposureName)
SwarpedImagePath = os.path.join(dataDir, SwarpedImageName)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class wcsMatchTestCase(unittest.TestCase):
    """
    A test case for warpExposure
    """
    def xtestMatchNull(self):
        """Test that warpExposure maps an image onto itself
        
        This test may be too severe because an exact match is unlikely;
        but the images should be very close. A somewhat gentler test would be to
        only compare good pixels in the warped image.
        """
        originalExposure = afwImage.ExposureD(OriginalExposurePath)
        originalImageArr, originalVarArr, originalMaskArr = \
            imageTestUtils.arraysFromMaskedImage(originalExposure.getMaskedImage())
        
        warpingKernel = afwMath.LanczosWarpingKernel(4)
        print "starting warp"
        numGoodPix = afwMath.warpExposure(afwWarpedExposure, originalExposure, warpingKernel)
        print "ending warp; numGoodPix =", numGoodPix
        afwWarpedExposure.writeFits("afwWarpedNull")
        afwWarpedImageArr, afwWarpedVarArr, afwWarpedMaskArr = \
            imageTestUtils.arraysFromMaskedImage(afwWarpedExposure.getMaskedImage())
        if not numpy.allclose(afwWarpedImageArr, originalImageArr):
            self.fail("afw null-warped image does not match original image")
        if not numpy.allclose(afwWarpedVarArr, originalVarArr):
            self.fail("afw null-warped variance does not match original image")
        if not numpy.allclose(afwWarpedMaskArr, originalMaskArr):
            self.fail("afw null-warped mask does not match original image")
        
    def testMatchSwarp(self):
        """Test that warpExposure matches swarp for a significant change in WCS.
        
        Note that swarp only warps the image plane, so only test that plane.
        """
        originalExposure = afwImage.ExposureD(OriginalExposurePath)
        swarpedDecoratedImage = afwImage.DecoratedImageD(SwarpedImagePath)
        swarpedImage = swarpedDecoratedImage.getImage()
        swarpedImageArr = imageTestUtils.arrayFromImage(swarpedImage)

        swarpedMetadata = swarpedDecoratedImage.getMetadata()
        warpedWcs = afwImage.Wcs(swarpedMetadata)

        afwWarpedMaskedImage = afwImage.MaskedImageD(swarpedImage.getWidth(), swarpedImage.getHeight())
        afwWarpedExposure = afwImage.ExposureD(afwWarpedMaskedImage, warpedWcs)
        
        warpingKernel = afwMath.LanczosWarpingKernel(4)
        print "starting warp"
        numGoodPix = afwMath.warpExposure(afwWarpedExposure, originalExposure, warpingKernel)
        print "ending warp; numGoodPix =", numGoodPix
        afwWarpedExposure.writeFits("afwWarped")
        afwWarpedImageArr, afwWarpedVarArr, afwWarpedMaskArr = imageTestUtils.arraysFromMaskedImage(afwWarpedMaskedImage)
        
        # ignore masked pixels in our warped image
        afwWarpedBoolMask =  (afwWarpedMaskArr != 0)
        afwWarpedImageMaskedArr = numpy.ma.array(afwWarpedImageArr, copy=False, mask = afwWarpedBoolMask)
        afwWarpedImageGoodArr = afwWarpedImageMaskedArr.compressed()
        print "afwWarpedImageGoodArr =", afwWarpedImageGoodArr
        
        swarpedImageMaskedArr = numpy.ma.array(swarpedImageArr, copy=False, mask = afwWarpedBoolMask)
        swarpedImageGoodArr = swarpedImageMaskedArr.compressed()
        print "swarpedImageGoodArr =", swarpedImageGoodArr
        
        if not numpy.allclose(afwWarpedImageGoodArr, swarpedImageGoodArr):
            self.fail("afw warped image does not match swarped image (ignoring bad pixels)")
        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """
    Returns a suite containing all the test cases in this module.
    """
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(wcsMatchTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

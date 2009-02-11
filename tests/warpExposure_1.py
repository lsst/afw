#!/usr/bin/env python
"""Test warpExposure
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

class WarpExposureTestCase(unittest.TestCase):
    """Test case for warpExposure
    """
    def testMatchNull(self):
        """Test that warpExposure maps an image onto itself.
        
        Note: bad pixel areas do get smoothed out so we have to excluded
        all masked bits in the output image, even non-edge pixels.
        """
        originalExposure = afwImage.ExposureF(OriginalExposurePath)
        afwWarpedExposure = afwImage.ExposureF(OriginalExposurePath)
        warpingKernel = afwMath.LanczosWarpingKernel(4)
        numGoodPix = afwMath.warpExposure(afwWarpedExposure, originalExposure, warpingKernel)
        afwWarpedExposure.writeFits("afwWarpedNull")
        afwWarpedMaskedImage = afwWarpedExposure.getMaskedImage()
        afwWarpedMask = afwWarpedMaskedImage.getMask()
        afwWarpedMaskArr = imageTestUtils.arrayFromMask(afwWarpedMask)
        badPlanes = self.compareMaskedImages(afwWarpedMaskedImage,
            originalExposure.getMaskedImage(), skipMaskArr = afwWarpedMaskArr)
        if badPlanes:
            badPlanesStr = str(badPlanes)[1:-1]
            self.fail("afw warped %s do/does not match swarped image (ignoring bad pixels)" % (badPlanesStr,))
        
    def testMatchSwarp(self):
        """Test that warpExposure matches swarp for a significant change in WCS.
        
        Note that swarp only warps the image plane, so only test that plane.
        
        Note: the edge of the good area is slightly different for for swarp and warpExposure.
        I would prefer to grow the EDGE mask by one pixel before comparing the images
        but that is too much hassle with the current afw so instead I ignore edge pixels from swarp and afw.
        """
        originalExposure = afwImage.ExposureF(OriginalExposurePath)
        swarpedDecoratedImage = afwImage.DecoratedImageF(SwarpedImagePath)
        swarpedImage = swarpedDecoratedImage.getImage()
        destWidth = swarpedImage.getWidth()
        destHeight = swarpedImage.getHeight()

        swarpedMetadata = swarpedDecoratedImage.getMetadata()
        warpedWcs = afwImage.Wcs(swarpedMetadata)

        afwWarpedMaskedImage = afwImage.MaskedImageF(destWidth, destHeight)
        afwWarpedExposure = afwImage.ExposureF(afwWarpedMaskedImage, warpedWcs)
        afwWarpedMask = afwWarpedMaskedImage.getMask()
        
        warpingKernel = afwMath.LanczosWarpingKernel(4)
        numGoodPix = afwMath.warpExposure(afwWarpedExposure, originalExposure, warpingKernel)
        afwWarpedExposure.writeFits("afwWarped")
        
        # set 0-value warped image pixels as EDGE in afwWarpedMask
        # and zero pixels from the swarped exposure
        edgeBitMask = afwWarpedMask.getPlaneBitMask("EDGE")
        if edgeBitMask == 0:
            self.fail("warped mask has no EDGE bit")
        skipMaskArr = imageTestUtils.arrayFromMask(afwWarpedMask)
        skipMaskArr &= edgeBitMask
        swarpedImageArr = imageTestUtils.arrayFromMask(swarpedImage)
        swarpedEdgeMaskArr = (swarpedImageArr == 0) * edgeBitMask
        swarpedEdgeMask = imageTestUtils.maskFromArray(swarpedEdgeMaskArr)
        swarpedEdgeMask.writeFits("warpedEdgeMask")
        skipMaskArr |= swarpedEdgeMaskArr
        swarpedMaskedImage = afwImage.MaskedImageF(swarpedImage)
        
        badPlanes = self.compareMaskedImages(afwWarpedMaskedImage, swarpedMaskedImage,
            doImage=True, doMask=False, doVariance=False, skipMaskArr=skipMaskArr, rtol=0.1)
        if badPlanes:
            self.fail("afw warped image does not match swarped image (ignoring bad pixels)")

    def compareMaskedImages(self, maskedImage1, maskedImage2,
        doImage=True, doMask=True, doVariance=True, skipMaskArr=None, rtol=1.0e-05, atol=1e-08):
        """Compare pixels from two masked images
        
        Inputs:
        - maskedImage1: first masked image to compare
        - maskedImage2: second masked image to compare
        - doImage: compare image planes if True
        - doMask: compare mask planes if True
        - doVariance: compare variance planes if True
        - skipMaskArr: pixels to ingore on the image, mask and variance arrays; nonzero values are skipped
        
        Returns a list of names of tested planes that did not match (empty if all match).
        """
        arr1Set = imageTestUtils.arraysFromMaskedImage(maskedImage1)
        arr2Set = imageTestUtils.arraysFromMaskedImage(maskedImage2)

        badPlanes = []
        for ind, (doPlane, planeName) in enumerate(((doImage, "image"), (doMask, "mask"), (doVariance, "variance"))):
            if not doPlane:
                continue
            arr1 = arr1Set[ind]
            arr2 = arr2Set[ind]
            
            if skipMaskArr != None:
                maskedArr1 = numpy.ma.array(arr1, copy=False, mask = skipMaskArr)
                maskedArr2 = numpy.ma.array(arr2, copy=False, mask = skipMaskArr)
                filledArr1 = maskedArr1.filled(0.0)
                filledArr2 = maskedArr2.filled(0.0)
            else:
                filledArr1 = arr1
                filledArr2 = arr2
            
            if not numpy.allclose(filledArr1, filledArr2, rtol=rtol, atol=atol):
                badPlanes.append(planeName)
                errArr = numpy.abs(filledArr1 - filledArr2)
                maxErr = errArr.max()
                maxPosInd = numpy.where(errArr==maxErr)
                maxPosTuple = (maxPosInd[0][0], maxPosInd[1][0])
                print "maxDiff=%s at position %s; value=%s or %s" % (maxErr,maxPosTuple, filledArr1[maxPosInd], filledArr2[maxPosInd])
        return badPlanes
        
        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """
    Returns a suite containing all the test cases in this module.
    """
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(WarpExposureTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

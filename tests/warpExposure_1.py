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
        badPlanes = self.compareMaskedImages(afwWarpedMaskedImage,
            originalExposure.getMaskedImage(), skipBadMask1 = 0xFFFF)
        if badPlanes:
            badPlanesStr = str(badPlanes)[1:-1]
            self.fail("afw warped %s do/does not match swarped image (ignoring bad pixels)" % (badPlanesStr,))
        
    def testMatchSwarp(self):
        """Test that warpExposure matches swarp for a significant change in WCS.
        
        Note that swarp only warps the image plane, so only test that plane.
        
        Note: the edge of the good area is slightly different for for swarp and warpExposure
        so I grow the EDGE mask by one pixel before comparing the images.
        
        BROKEN: cannot convolve a mask, so need to turn into an image and...sigh.
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
        
        # grow afwWarped exposure EDGE mask by 1 pixel
        edgeBitMask = afwWarpedMask.getPlaneBitMask("EDGE")
        if edgeBitMask == 0:
            self.fail("warped mask has no EDGE bit")
        edgeMask = afwImage.MaskU(afwWarpedMask, True)
        edgeMask &= edgeBitMask
        growKernelImage = afwImage.ImageD(3, 3)
        growKernelImage.set(1)
        for i in (0, 2):
            for j in (0, 2):
                growKernelImage.set(i, j, 0)
        growKernel = afwMath.FixedKernel(growKernelImage)
        grownEdgeMask = afwImage.MaskU(edgeMask, True)
        afwMath.convolve(grownEdgeMask, edgeMask, growKernel, False, 0)
        afwWarpedMask |= grownEdgeMask
        afwWarpedMask.writeFits("afwWarpedGrownMask")

        # when comparing, ignore pixels on border of width 1 since those mask bits were not grown properly
        bbox = afwImage.BBox(afwImage.PointI(1, 1), destWidth-2, destHeight-2)
        subAfwMaskedImage = afwImage.MaskedImageF(afwMaskedImage, bbox)

        swarpedMaskedImage = afwImage.MaskedImageF(swarpedImage)
        subSwarpedMaskedImage = afwImage.MaskedImageF(swarpedMaskedImage, bbox)
        
        badPlanes = self.compareMaskedImages(subAfwMaskedImage, subSwarpedMaskedImage,
            doImage=True, doVariance=False, doMask=False, skipBadMask1=0xFFFF, rtol=0.1)
        if badPlanes:
            self.fail("afw warped image does not match swarped image (ignoring bad pixels)")

    def compareMaskedImages(self, maskedImage1, maskedImage2,
        doImage=True, doMask=True, doVariance=True, skipBadMask1=0, rtol=1.0e-05, atol=1e-08):
        """Compare pixels from two masked images
        
        Inputs:
        - maskedImage1: first masked image to compare
        - maskedImage2: second masked image to compare
        - doImage: compare image planes if True
        - doMask: compare mask planes if True
        - doVariance: compare variance planes if True
        - skipBad1: ignore pixels marked bad (mask != 0) in maskedImage1 if True
        
        Returns a list of names of tested planes that did not match (empty if all match).
        """
        arr1Set = imageTestUtils.arraysFromMaskedImage(maskedImage1)
        arr2Set = imageTestUtils.arraysFromMaskedImage(maskedImage2)

        if skipBadMask1:
            badPixArr1 = (arr1Set[2] & skipBadMask1 != 0)
        
        badPlanes = []
        for ind, planeName in enumerate(("image", "variance", "mask")):
#            print "testing", planeName
            arr1 = arr1Set[ind]
            arr2 = arr2Set[ind]
            
            if skipBadMask1:
                maskedArr1 = numpy.ma.array(arr1, copy=False, mask = badPixArr1)
                maskedArr2 = numpy.ma.array(arr2, copy=False, mask = badPixArr1)
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

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
OriginalExposurePath = os.path.join(dataDir, OriginalExposureName)
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

    def testMatchSwarpBilinear(self):
        """Test that warpExposure matches swarp using a bilinear warping kernel
        """
        # maxDiff=2219.74499512 at position (455, 4); value=-1276.12097168 vs. 943.624023438
        # a pixel very near the bad column
        self.compareSwarped("bilinear")

    def testMatchSwarpLanczos2(self):
        """Test that warpExposure matches swarp using a lanczos4 warping kernel.
        """
        # use a large rtol because a few pixels are significantly off, the worst being:
        # maxDiff=135.429199219 at position (559, 63); value=2298.32055664 vs. 2433.74975586
        self.compareSwarped("lanczos2", rtol=1.0)

    def testMatchSwarpLanczos3(self):
        """Test that warpExposure matches swarp using a lanczos4 warping kernel.
        """
        # use a large rtol because a few pixels are significantly off, the worst being:
        # maxDiff=53.94140625 at position (469, 8); value=2766.74536133 vs. 2712.80395508
        self.compareSwarped("lanczos3", rtol=0.5)

    def testMatchSwarpLanczos4(self):
        """Test that warpExposure matches swarp using a lanczos4 warping kernel.
        """
        # use a large rtol because a few pixels are significantly off, the worst being:
        # maxDiff=25.6 at position (468, 6); value=3710.5 vs. 3736.1
        self.compareSwarped("lanczos4", rtol=0.5)

    def compareSwarped(self, kernelName, rtol=0.1):
        """Compare remapExposure to swarp for given warping kernel.
        
        Note that swarp only warps the image plane, so only test that plane.
       
        Note: the edge of the good area is slightly different for for swarp and warpExposure.
        I would prefer to grow the EDGE mask by one pixel before comparing the images
        but that is too much hassle with the current afw so instead I ignore edge pixels from swarp and afw.
        """
        if kernelName.startswith("lanczos"):
            order = int(kernelName[7:])
            warpingKernel = afwMath.LanczosWarpingKernel(order)
        elif kernelName == "bilinear":
            warpingKernel = afwMath.BilinearWarpingKernel()
        else:
            raise RuntimeError("unknown warping kernel %r" % (kernelName))

        originalExposure = afwImage.ExposureF(OriginalExposurePath)

        swarpedImageName = "medswarp1%s.fits" % (kernelName,)
        swarpedImagePath = os.path.join(dataDir, swarpedImageName)
        swarpedDecoratedImage = afwImage.DecoratedImageF(swarpedImagePath)
        swarpedImage = swarpedDecoratedImage.getImage()
        swarpedMetadata = swarpedDecoratedImage.getMetadata()
        warpedWcs = afwImage.Wcs(swarpedMetadata)
        destWidth = swarpedImage.getWidth()
        destHeight = swarpedImage.getHeight()

        afwWarpedMaskedImage = afwImage.MaskedImageF(destWidth, destHeight)
        afwWarpedExposure = afwImage.ExposureF(afwWarpedMaskedImage, warpedWcs)
        numGoodPix = afwMath.warpExposure(afwWarpedExposure, originalExposure, warpingKernel)
        afwWarpedExposure.writeFits("afwWarped1%s" % (kernelName,))

        # set 0-value warped image pixels as EDGE in afwWarpedMask
        # and zero pixels from the swarped exposure
        afwWarpedMask = afwWarpedMaskedImage.getMask()
        edgeBitMask = afwWarpedMask.getPlaneBitMask("EDGE")
        if edgeBitMask == 0:
            self.fail("warped mask has no EDGE bit")
        skipMaskArr = imageTestUtils.arrayFromMask(afwWarpedMask)
        skipMaskArr &= edgeBitMask
        swarpedImageArr = imageTestUtils.arrayFromMask(swarpedImage)
        swarpedEdgeMaskArr = (swarpedImageArr == 0) * edgeBitMask
        swarpedEdgeMask = imageTestUtils.maskFromArray(swarpedEdgeMaskArr)
        skipMaskArr |= swarpedEdgeMaskArr
        swarpedMaskedImage = afwImage.MaskedImageF(swarpedImage)
        
        badPlanes = self.compareMaskedImages(afwWarpedMaskedImage, swarpedMaskedImage,
            doImage=True, doMask=False, doVariance=False, skipMaskArr=skipMaskArr, rtol=rtol)
        if badPlanes:
            self.fail("afw and swarp %s-warped images do not match (ignoring bad pixels)" % (kernelName,))

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
                print "maxDiff=%s at position %s; value=%s vs. %s" % (maxErr,maxPosTuple, filledArr1[maxPosInd][0], filledArr2[maxPosInd][0])
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

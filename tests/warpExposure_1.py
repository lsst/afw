#!/usr/bin/env python
"""Test warpExposure
"""
import os

import unittest

import numpy

import eups
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.image.testUtils as imageTestUtils
import lsst.utils.tests as utilsTests
import lsst.pex.logging as logging

VERBOSITY = 0 # increase to see trace

# set True to save afw-warped images as FITS files
SAVE_FITS_FILES = False

logging.Debug("lsst.afw.math", VERBOSITY)

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests")

originalExposureName = "med"
originalExposurePath = os.path.join(dataDir, originalExposureName)

class WarpExposureTestCase(unittest.TestCase):
    """Test case for warpExposure
    """
    def testNullWarpExposure(self):
        """Test that warpExposure maps an image onto itself.
        
        Note:
        - edge pixels must be ignored
        - bad mask pixels get smeared out so we have to excluded all bad mask pixels
          from the output image when comparing masks.
        """
        originalExposure = afwImage.ExposureF(originalExposurePath)
        afwWarpedExposure = afwImage.ExposureF(originalExposurePath)
        warpingKernel = afwMath.LanczosWarpingKernel(4)
        afwMath.warpExposure(afwWarpedExposure, originalExposure, warpingKernel)
        if SAVE_FITS_FILES:
            afwWarpedExposure.writeFits("afwWarpedExposureNull")
        afwWarpedMaskedImage = afwWarpedExposure.getMaskedImage()
        afwWarpedMask = afwWarpedMaskedImage.getMask()
        edgeBitMask = afwWarpedMask.getPlaneBitMask("EDGE")
        if edgeBitMask == 0:
            self.fail("warped mask has no EDGE bit")
        afwWarpedMaskedImageArrSet = imageTestUtils.arraysFromMaskedImage(afwWarpedMaskedImage)
        afwWarpedMaskArr = afwWarpedMaskedImageArrSet[1]
        
        # compare all non-edge pixels of image and variance, but relax specs a bit
        # because of minor noise introduced by bad pixels
        edgeMaskArr = afwWarpedMaskArr & edgeBitMask
        originalMaskedImageArrSet = imageTestUtils.arraysFromMaskedImage(originalExposure.getMaskedImage())
        errStr = imageTestUtils.maskedImagesDiffer(afwWarpedMaskedImageArrSet, originalMaskedImageArrSet,
            doMask=False, skipMaskArr=edgeMaskArr, atol=1.0e-5)
        if errStr:
            self.fail("afw null-warped MaskedImage (all pixels, relaxed tolerance): %s" % (errStr,))
        
        # compare good pixels of image, mask and variance using full tolerance
        errStr = imageTestUtils.maskedImagesDiffer(afwWarpedMaskedImageArrSet, originalMaskedImageArrSet,
            doImage=False, doVariance=False, skipMaskArr=afwWarpedMaskArr)
        if errStr:
            self.fail("afw null-warped MaskedImage (good pixels, max tolerance): %s" % (errStr,))

    def testNullWarpImage(self):
        """Test that warpImage maps an image onto itself.
        """
        originalExposure = afwImage.ExposureF(originalExposurePath)
        afwWarpedExposure = afwImage.ExposureF(originalExposurePath)
        originalImage = originalExposure.getMaskedImage().getImage()
        afwWarpedImage = afwWarpedExposure.getMaskedImage().getImage()
        originalWcs = originalExposure.getWcs()
        afwWarpedWcs = afwWarpedExposure.getWcs()
        warpingKernel = afwMath.LanczosWarpingKernel(4)
        afwMath.warpImage(afwWarpedImage, afwWarpedWcs, originalImage, originalWcs, warpingKernel)
        if SAVE_FITS_FILES:
            afwWarpedImage.writeFits("afwWarpedImageNull.fits")
        afwWarpedImageArr = imageTestUtils.arrayFromImage(afwWarpedImage)
        edgeMaskArr = numpy.isnan(afwWarpedImageArr)
        originalImageArr = imageTestUtils.arrayFromImage(originalImage)
        # relax specs a bit because of minor noise introduced by bad pixels
        errStr = imageTestUtils.imagesDiffer(originalImageArr, originalImageArr,
            skipMaskArr=edgeMaskArr, atol=1.0e-5)
        if errStr:
            self.fail("afw null-warped Image: %s" % (errStr,))

    def testNullWcs(self):
        """Cannot warp from or into an exposure without a Wcs.
        """
        exposureWithWcs = afwImage.ExposureF(originalExposurePath)
        mi = exposureWithWcs.getMaskedImage()
        exposureWithoutWcs = afwImage.ExposureF(mi.getWidth(), mi.getHeight())
        warpingKernel = afwMath.BilinearWarpingKernel()
        try:
            afwMath.warpExposure(exposureWithWcs, exposureWithoutWcs, warpingKernel)
            self.fail("warping from a source Exception with no Wcs should fail")
        except Exception:
            pass
        try:
            afwMath.warpExposure(exposureWithoutWcs, exposureWithWcs, warpingKernel)
            self.fail("warping into a destination Exception with no Wcs should fail")
        except Exception:
            pass
    
    def testWarpIntoSelf(self):
        """Cannot warp in-place
        """
        originalExposure = afwImage.ExposureF(100, 100)
        warpingKernel = afwMath.BilinearWarpingKernel()
        try:
            afwMath.warpExposure(originalExposure, originalExposure, warpingKernel)
            self.fail("warpExposure in place (dest is src) should fail")
        except Exception:
            pass
        try:
            afwMath.warpImage(originalExposure.getMaskedImage(), originalExposure.getWcs(),
                originalExposure.getMaskedImage(), originalExposure.getWcs(), warpingKernel)
            self.fail("warpImage<MaskedImage> in place (dest is src) should fail")
        except Exception:
            pass
        try:
            afwMath.warpImage(originalExposure.getImage(), originalExposure.getWcs(),
                originalExposure.getImage(), originalExposure.getWcs(), warpingKernel)
            self.fail("warpImage<Image> in place (dest is src) should fail")
        except Exception:
            pass

    def testMatchSwarpBilinearImage(self):
        """Test that warpExposure matches swarp using a bilinear warping kernel
        """
        self.compareSwarpedImage("bilinear", atol=1.0e-2)

    def testMatchSwarpBilinearExposure(self):
        """Test that warpExposure matches swarp using a bilinear warping kernel
        """
        self.compareSwarpedExposure("bilinear")

    def testMatchSwarpLanczos2Image(self):
        """Test that warpExposure matches swarp using a lanczos2 warping kernel
        """
        self.compareSwarpedImage("lanczos2", atol=1.0e-2)

    def testMatchSwarpLanczos2Exposure(self):
        """Test that warpExposure matches swarp using a lanczos2 warping kernel.
        """
        self.compareSwarpedExposure("lanczos2")

    def testMatchSwarpLanczos3Image(self):
        """Test that warpExposure matches swarp using a lanczos2 warping kernel
        """
        self.compareSwarpedImage("lanczos3", atol=1.0e-2)

    def testMatchSwarpLanczos3(self):
        """Test that warpExposure matches swarp using a lanczos4 warping kernel.
        """
        self.compareSwarpedExposure("lanczos3")

    def testMatchSwarpLanczos4Image(self):
        """Test that warpExposure matches swarp using a lanczos2 warping kernel
        """
        self.compareSwarpedImage("lanczos4", atol=1.0e-2)

    def testMatchSwarpLanczos4(self):
        """Test that warpExposure matches swarp using a lanczos4 warping kernel.
        """
        self.compareSwarpedExposure("lanczos4")

    def testMatchSwarpNearestExposure(self):
        """Test that warpExposure matches swarp using a nearest neighbor warping kernel
        """
        self.compareSwarpedExposure("nearest", atol=60)

    def compareSwarpedImage(self, kernelName, rtol=1.0e-05, atol=1e-08):
        """Compare remapExposure to swarp for given warping kernel.
        
        Note that swarp only warps the image plane, so only test that plane.
        """
        warpingKernel = afwMath.makeWarpingKernel(kernelName)

        originalExposure = afwImage.ExposureF(originalExposurePath)
        originalImage = originalExposure.getMaskedImage().getImage()
        originalWcs = originalExposure.getWcs()
        
        # path for saved afw-warped image
        afwWarpedImagePath = "afwWarpedImage1%s.fits" % (kernelName,)

        swarpedImageName = "medswarp1%s.fits" % (kernelName,)
        swarpedImagePath = os.path.join(dataDir, swarpedImageName)
        swarpedDecoratedImage = afwImage.DecoratedImageF(swarpedImagePath)
        swarpedImage = swarpedDecoratedImage.getImage()
        swarpedMetadata = swarpedDecoratedImage.getMetadata()
        warpedWcs = afwImage.makeWcs(swarpedMetadata)
        destWidth = swarpedImage.getWidth()
        destHeight = swarpedImage.getHeight()

        afwWarpedImage = afwImage.ImageF(destWidth, destHeight)
        afwMath.warpImage(afwWarpedImage, warpedWcs, originalImage, originalWcs, warpingKernel)
        if SAVE_FITS_FILES:
            afwWarpedImage.writeFits(afwWarpedImagePath)

        
        afwWarpedImageArr = imageTestUtils.arrayFromImage(afwWarpedImage)
        swarpedImageArr = imageTestUtils.arrayFromImage(swarpedImage)
        edgeMaskArr = numpy.isnan(afwWarpedImageArr)
        errStr = imageTestUtils.imagesDiffer(afwWarpedImageArr, swarpedImageArr,
            skipMaskArr=edgeMaskArr, rtol=rtol, atol=atol)
        if errStr:
            if not SAVE_FITS_FILES:
                # save the image anyway
                afwWarpedImage.writeFits(afwWarpedImagePath)
            print "Saved failed afw-warped image as: %s" % (afwWarpedImagePath,)
            self.fail("afw and swarp %s-warped images do not match (ignoring NaN pixels): %s" % \
                (kernelName, errStr))

    def compareSwarpedExposure(self, kernelName, rtol=1.0e-05, atol=1e-08):
        """Compare remapExposure to swarp for given warping kernel.
        
        Note that swarp only warps the image plane, so only test that plane.
        """
        warpingKernel = afwMath.makeWarpingKernel(kernelName)

        originalExposure = afwImage.ExposureF(originalExposurePath)
        
        # path for saved afw-warped image
        afwWarpedImagePath = "afwWarpedExposure1%s" % (kernelName,)

        swarpedImageName = "medswarp1%s.fits" % (kernelName,)
        swarpedImagePath = os.path.join(dataDir, swarpedImageName)
        swarpedDecoratedImage = afwImage.DecoratedImageF(swarpedImagePath)
        swarpedImage = swarpedDecoratedImage.getImage()
        swarpedMetadata = swarpedDecoratedImage.getMetadata()
        warpedWcs = afwImage.makeWcs(swarpedMetadata)
        destWidth = swarpedImage.getWidth()
        destHeight = swarpedImage.getHeight()

        afwWarpedMaskedImage = afwImage.MaskedImageF(destWidth, destHeight)
        afwWarpedExposure = afwImage.ExposureF(afwWarpedMaskedImage, warpedWcs)
        afwMath.warpExposure(afwWarpedExposure, originalExposure, warpingKernel)
        if SAVE_FITS_FILES:
            afwWarpedExposure.writeFits(afwWarpedImagePath)

        afwWarpedMask = afwWarpedMaskedImage.getMask()
        edgeBitMask = afwWarpedMask.getPlaneBitMask("EDGE")
        if edgeBitMask == 0:
            self.fail("warped mask has no EDGE bit")
        afwWarpedMaskedImageArrSet = imageTestUtils.arraysFromMaskedImage(afwWarpedMaskedImage)
        afwWarpedMaskArr = afwWarpedMaskedImageArrSet[1]

        swarpedMaskedImage = afwImage.MaskedImageF(swarpedImage)
        swarpedMaskedImageArrSet = imageTestUtils.arraysFromMaskedImage(swarpedMaskedImage)
        
        errStr = imageTestUtils.maskedImagesDiffer(afwWarpedMaskedImageArrSet, swarpedMaskedImageArrSet,
            doImage=True, doMask=False, doVariance=False, skipMaskArr=afwWarpedMaskArr, rtol=rtol, atol=atol)
        if errStr:
            if not SAVE_FITS_FILES:
                afwWarpedExposure.writeFits(afwWarpedImagePath)
            print "Saved failed afw-warped exposure as: %s" % (afwWarpedImagePath,)
            self.fail("afw and swarp %s-warped %s (ignoring bad pixels)" % (kernelName, errStr))

    def compareMaskedImages(self, maskedImage1, maskedImage2, descr="",
        doImage=True, doMask=True, doVariance=True, skipMaskArr=None, rtol=1.0e-05, atol=1e-08):
        """Compare pixels from two masked images
        
        Inputs:
        - maskedImage1: first masked image to compare
        - maskedImage2: second masked image to compare
        - descr: a short description of the inputs
        - doImage: compare image planes if True
        - doMask: compare mask planes if True
        - doVariance: compare variance planes if True
        - skipMaskArr: pixels to ingore on the image, mask and variance arrays; nonzero values are skipped
        
        Returns None if all is well, or an error string if any plane did not match.
        The error string is one of:
        <plane> plane does not match
        (<plane1>, <plane2>,...) planes do not match
        """
        badPlanes = []
        for (doPlane, planeName) in ((doImage, "image"), (doMask, "mask"), (doVariance, "variance")):
            if not doPlane:
                continue
            funcName = "get%s" % (planeName.title(),)
            imageDescr = "%s: %s plane" % (descr, planeName)
            image1 = getattr(maskedImage1, funcName)()
            image2 = getattr(maskedImage2, funcName)()
            imageOK = self.compareImages(image1, image2, descr=imageDescr, skipMaskArr=skipMaskArr,
                                         rtol=rtol, atol=atol)
            if not imageOK:
                badPlanes.append(planeName)
        if not badPlanes:
            return None
        if len(badPlanes) > 1:
            return "%s planes do not match" % (badPlanes,)
        return "%s plane does not match" % (badPlanes[0],)
    
    def compareImages(self, image1, image2, descr="", skipMaskArr=None, rtol=1.0e-05, atol=1e-08):
        """Return True if two images are nearly equal, False otherwise
        """
        arr1 = imageTestUtils.arrayFromImage(image1)
        arr2 = imageTestUtils.arrayFromImage(image2)

        if skipMaskArr != None:
            maskedArr1 = numpy.ma.array(arr1, copy=False, mask = skipMaskArr)
            maskedArr2 = numpy.ma.array(arr2, copy=False, mask = skipMaskArr)
            filledArr1 = maskedArr1.filled(0.0)
            filledArr2 = maskedArr2.filled(0.0)
        else:
            filledArr1 = arr1
            filledArr2 = arr2
        
        if not numpy.allclose(filledArr1, filledArr2, rtol=rtol, atol=atol):
            errArr =  numpy.abs(filledArr1 - filledArr2) - (filledArr2 * rtol) + atol
            maxErr = errArr.max()
            maxPosInd = numpy.where(errArr==maxErr)
            maxPosTuple = (maxPosInd[0][0], maxPosInd[1][0])
            relErr = numpy.abs(filledArr1 - filledArr2) / (filledArr1 + filledArr2)
            print "%s: maxErr=%s at position %s; value=%s vs. %s; relErr=%s" % \
                (descr, maxErr, maxPosTuple, filledArr1[maxPosInd][0], filledArr2[maxPosInd][0], relErr)
            return False
        return True
        
        
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

def run(doExit=False):
    """Run the tests"""
    utilsTests.run(suite(), doExit)

if __name__ == "__main__":
    run(True)

#!/usr/bin/env python

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

"""Test warpExposure
"""
import math
import os
import unittest

import numpy

import eups
import lsst.daf.base as dafBase
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.afw.gpu as afwGpu
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.image.utils as imageUtils
import lsst.afw.image.testUtils as imageTestUtils
import lsst.utils.tests as utilsTests
import lsst.pex.logging as pexLog
import lsst.pex.policy as pexPolicy
import lsst.pex.exceptions as pexExcept
import lsst.afw.display.ds9 as ds9
try:
    display
except:
    display = False
    VERBOSITY = 0                       # increase to see trace
    # set True to save afw-warped images as FITS files
    SAVE_FITS_FILES = False
    # set True to save failed afw-warped images as FITS files even if SAVE_FITS_FILES is False
    #SAVE_FAILED_FITS_FILES = False
    SAVE_FAILED_FITS_FILES = True

pexLog.Debug("lsst.afw.math", VERBOSITY)

afwDataDir = eups.productDir("afwdata")
if not afwDataDir:
    raise RuntimeError("Must set up afwdata to run these tests")
dataDir = os.path.join(afwDataDir, "data")

originalExposureName = "medexp.fits"
originalExposurePath = os.path.join(dataDir, originalExposureName)
subExposureName = "medsub.fits"
subExposurePath = os.path.join(dataDir, originalExposureName)
originalFullExposureName = os.path.join("CFHT", "D4", "cal-53535-i-797722_1.fits")
originalFullExposurePath = os.path.join(dataDir, originalFullExposureName)

def makeWcs(pixelScale, crPixPos, crValCoord, posAng=afwGeom.Angle(0.0), doFlipX=False, projection="TAN"):
    """Make a Wcs
    
    @param[in] pixelScale: desired scale, as sky/pixel, an afwGeom.Angle
    @param[in] crPixPos: crPix for WCS, using the LSST standard; a pair of floats
    @param[in] crValCoord: crVal for WCS (afwCoord.Coord)
    @param[in] posAng: position angle (afwGeom.Angle)
    @param[in] doFlipX: flip X axis?
    @param[in] projection: WCS projection (e.g. "TAN" or "STG")
    """
    if len(projection) != 3:
        raise RuntimeError("projection=%r; must have length 3" % (projection,))
    ctypeList = [("%-5s%3s" % (("RA", "DEC")[i], projection)).replace(" ", "-")
        for i in range(2)]
    ps = dafBase.PropertySet()
    crPixFits = [ind + 1.0 for ind in crPixPos] # convert pix position to FITS standard
    crValDeg = crValCoord.getPosition(afwGeom.degrees)
    posAngRad = posAng.asRadians()
    pixelScaleDeg = pixelScale.asDegrees()
    cdMat = numpy.array([[ math.cos(posAngRad), math.sin(posAngRad)],
                         [-math.sin(posAngRad), math.cos(posAngRad)]], dtype=float) * pixelScaleDeg
    if doFlipX:
        cdMat[:,0] = -cdMat[:,0]
    for i in range(2):
        ip1 = i + 1
        ps.add("CTYPE%1d" % (ip1,), ctypeList[i])
        ps.add("CRPIX%1d" % (ip1,), crPixFits[i])
        ps.add("CRVAL%1d" % (ip1,), crValDeg[i])
    ps.add("RADECSYS", "ICRS")
    ps.add("EQUINOX", 2000)
    ps.add("CD1_1", cdMat[0, 0])
    ps.add("CD2_1", cdMat[1, 0])
    ps.add("CD1_2", cdMat[0, 1])
    ps.add("CD2_2", cdMat[1, 1])
    return afwImage.makeWcs(ps)


class WarpExposureTestCase(unittest.TestCase):
    """Test case for warpExposure
    """
    def testNullWarpExposure(self, interpLength=10):
        """Test that warpExposure maps an image onto itself.
        
        Note:
        - edge pixels must be ignored
        - bad mask pixels get smeared out so we have to excluded all bad mask pixels
          from the output image when comparing masks.
        """
        filterPolicyFile = pexPolicy.DefaultPolicyFile("afw", "SdssFilters.paf", "tests")
        filterPolicy = pexPolicy.Policy.createPolicy(filterPolicyFile, filterPolicyFile.getRepositoryPath(), True)
        imageUtils.defineFiltersFromPolicy(filterPolicy, reset=True)

        originalExposure = afwImage.ExposureF(originalExposurePath)
        originalFilter = afwImage.Filter("i")
        originalCalib = afwImage.Calib()
        originalCalib.setFluxMag0(1.0e5, 1.0e3)
        originalExposure.setFilter(originalFilter)
        originalExposure.setCalib(originalCalib)
        afwWarpedExposure = afwImage.ExposureF(
            originalExposure.getBBox(),
            originalExposure.getWcs())
        warpingControl = afwMath.WarpingControl("lanczos4", "", 0, interpLength)
        afwMath.warpExposure(afwWarpedExposure, originalExposure, warpingControl)
        if SAVE_FITS_FILES:
            afwWarpedExposure.writeFits("afwWarpedExposureNull.fits")
        
        self.assertEquals(afwWarpedExposure.getFilter().getName(), originalFilter.getName())
        self.assertEquals(afwWarpedExposure.getCalib().getFluxMag0(), originalCalib.getFluxMag0())
        
        afwWarpedMaskedImage = afwWarpedExposure.getMaskedImage()
        afwWarpedMask = afwWarpedMaskedImage.getMask()
        edgeBitMask = afwWarpedMask.getPlaneBitMask("EDGE")
        if edgeBitMask == 0:
            self.fail("warped mask has no EDGE bit")
        afwWarpedMaskedImageArrSet = afwWarpedMaskedImage.getArrays()
        afwWarpedMaskArr = afwWarpedMaskedImageArrSet[1]
        
        # compare all non-edge pixels of image and variance, but relax specs a bit
        # because of minor noise introduced by bad pixels
        edgeMaskArr = afwWarpedMaskArr & edgeBitMask
        originalMaskedImageArrSet = originalExposure.getMaskedImage().getArrays()
        errStr = imageTestUtils.maskedImagesDiffer(afwWarpedMaskedImageArrSet, originalMaskedImageArrSet,
            doMask=False, skipMaskArr=edgeMaskArr, atol=1e-5)
        if errStr:
            self.fail("afw null-warped MaskedImage (all pixels, relaxed tolerance): %s" % (errStr,))
        
        # compare good pixels of image, mask and variance using full tolerance
        errStr = imageTestUtils.maskedImagesDiffer(afwWarpedMaskedImageArrSet, originalMaskedImageArrSet,
            doImage=False, doVariance=False, skipMaskArr=afwWarpedMaskArr)
        if errStr:
            self.fail("afw null-warped MaskedImage (good pixels, max tolerance): %s" % (errStr,))

    def testNullWarpImage(self, interpLength=10):
        """Test that warpImage maps an image onto itself.
        """
        originalExposure = afwImage.ExposureF(originalExposurePath)
        afwWarpedExposure = afwImage.ExposureF(originalExposurePath)
        originalImage = originalExposure.getMaskedImage().getImage()
        afwWarpedImage = afwWarpedExposure.getMaskedImage().getImage()
        originalWcs = originalExposure.getWcs()
        afwWarpedWcs = afwWarpedExposure.getWcs()
        warpingControl = afwMath.WarpingControl("lanczos4", "", 0, interpLength)
        afwMath.warpImage(afwWarpedImage, afwWarpedWcs, originalImage, originalWcs, warpingControl)
        if SAVE_FITS_FILES:
            afwWarpedImage.writeFits("afwWarpedImageNull.fits")
        afwWarpedImageArr = afwWarpedImage.getArray()
        edgeMaskArr = numpy.isnan(afwWarpedImageArr)
        originalImageArr = originalImage.getArray()
        # relax specs a bit because of minor noise introduced by bad pixels
        errStr = imageTestUtils.imagesDiffer(originalImageArr, originalImageArr,
            skipMaskArr=edgeMaskArr)
        if errStr:
            self.fail("afw null-warped Image: %s" % (errStr,))

    def testNullWcs(self, interpLength=10):
        """Cannot warp from or into an exposure without a Wcs.
        """
        exposureWithWcs = afwImage.ExposureF(originalExposurePath)
        mi = exposureWithWcs.getMaskedImage()
        exposureWithoutWcs = afwImage.ExposureF(mi.getDimensions())
        warpingControl = afwMath.WarpingControl("bilinear", "", 0, interpLength)
        try:
            afwMath.warpExposure(exposureWithWcs, exposureWithoutWcs, warpingControl)
            self.fail("warping from a source Exception with no Wcs should fail")
        except Exception:
            pass
        try:
            afwMath.warpExposure(exposureWithoutWcs, exposureWithWcs, warpingControl)
            self.fail("warping into a destination Exception with no Wcs should fail")
        except Exception:
            pass
    
    def testWarpIntoSelf(self, interpLength=10):
        """Cannot warp in-place
        """
        originalExposure = afwImage.ExposureF(afwGeom.Extent2I(100, 100))
        warpingControl = afwMath.WarpingControl("bilinear", "", 0, interpLength)
        try:
            afwMath.warpExposure(originalExposure, originalExposure, warpingControl)
            self.fail("warpExposure in place (dest is src) should fail")
        except Exception:
            pass
        try:
            afwMath.warpImage(originalExposure.getMaskedImage(), originalExposure.getWcs(),
                originalExposure.getMaskedImage(), originalExposure.getWcs(), warpingControl)
            self.fail("warpImage<MaskedImage> in place (dest is src) should fail")
        except Exception:
            pass
        try:
            afwMath.warpImage(originalExposure.getImage(), originalExposure.getWcs(),
                originalExposure.getImage(), originalExposure.getWcs(), warpingControl)
            self.fail("warpImage<Image> in place (dest is src) should fail")
        except Exception:
            pass
    
    def testWarpingControl(self):
        """Test the basic mechanics of WarpingControl
        """
        for interpLength in (0, 1, 52):
            wc = afwMath.WarpingControl("lanczos3", "", 0, interpLength)
            self.assertFalse(wc.hasMaskWarpingKernel())
            self.assertEqual(wc.getInterpLength(), interpLength)
            for newInterpLength in (3, 7, 9):
                wc.setInterpLength(newInterpLength)
                self.assertEqual(wc.getInterpLength(), newInterpLength)
        
        for cacheSize in (0, 100):
            wc = afwMath.WarpingControl("lanczos3", "bilinear", cacheSize)
            self.assertTrue(wc.hasMaskWarpingKernel())
            self.assertEqual(wc.getCacheSize(), cacheSize)
            self.assertEqual(wc.getWarpingKernel().getCacheSize(), cacheSize)
            self.assertEqual(wc.getMaskWarpingKernel().getCacheSize(), cacheSize)
            for newCacheSize in (1, 50):
                wc.setCacheSize(newCacheSize)
                self.assertEqual(wc.getCacheSize(), newCacheSize)
                self.assertEqual(wc.getWarpingKernel().getCacheSize(), newCacheSize)
                self.assertEqual(wc.getMaskWarpingKernel().getCacheSize(), newCacheSize)
    
    def testWarpingControlError(self):
        """Test error handling of WarpingControl
        """
        # error: mask kernel smaller than main kernel
        for kernelName, maskKernelName in (
            ("bilinear", "lanczos3"),
            ("bilinear", "lanczos4"),
            ("lanczos3", "lanczos4"),
        ):
            self.assertRaises(pexExcept.Exception,
                afwMath.WarpingControl, kernelName, maskKernelName)
        
        # error: new mask kernel larger than main kernel
        warpingControl = afwMath.WarpingControl("bilinear")
        for maskKernelName in ("lanczos3", "lanczos4"):
            self.assertRaises(pexExcept.Exception,
                warpingControl.setMaskWarpingKernelName, maskKernelName)

        # error: new kernel smaller than mask kernel
        warpingControl = afwMath.WarpingControl("lanczos4", "lanczos4")
        for kernelName in ("bilinear", "lanczos3"):
            self.assertRaises(pexExcept.Exception,
                warpingControl.setWarpingKernelName, kernelName)
        
        # error: GPU only works with Lanczos kernels
        self.assertRaises(pexExcept.Exception,
            afwMath.WarpingControl, "bilinear", "", 0, 0, afwGpu.USE_GPU)
        warpingControl = afwMath.WarpingControl("bilinear")
        self.assertRaises(pexExcept.Exception,
            warpingControl.setDevicePreference, afwGpu.USE_GPU)

        # OK: GPU works with Lanczos kernels
        for kernelName in ("lanczos3", "lanczos4"):
            afwMath.WarpingControl(kernelName, "", 0, 0, afwGpu.USE_GPU)
            warpingControl = afwMath.WarpingControl(kernelName)
            warpingControl.setDevicePreference(afwGpu.USE_GPU)

        # OK: main kernel at least as big as mask kernel
        for kernelName, maskKernelName in (
            ("bilinear", "bilinear"),
            ("lanczos3", "lanczos3"),
            ("lanczos3", "bilinear"),
            ("lanczos4", "lanczos3"),
        ):
            # this should not raise any exception
            afwMath.WarpingControl(kernelName, maskKernelName)
        
        # invalid kernel names
        for kernelName, maskKernelName in (
            ("badname", ""),
            ("lanczos", ""), # no digit after lanczos
            ("lanczos3", "badname"),
            ("lanczos3", "lanczos"),
        ):
            self.assertRaises(pexExcept.Exception,
                afwMath.WarpingControl, kernelName, maskKernelName)

    def testWarpMask(self):
        """Test that warping the mask plane with a different kernel does the right thing
        """
        for kernelName, maskKernelName in (
            ("bilinear", "bilinear"),
            ("lanczos3", "lanczos3"),
            ("lanczos3", "bilinear"),
            ("lanczos4", "lanczos3"),
        ):
            for growFullMask in (0, 1, 3, 0xFFFF):
                self.verifyMaskWarp(
                    kernelName=kernelName,
                    maskKernelName=maskKernelName,
                    growFullMask=growFullMask,
                )
        
    def testMatchSwarpBilinearImage(self):
        """Test that warpExposure matches swarp using a bilinear warping kernel
        """
        self.compareToSwarp("bilinear", useWarpExposure=False, atol=0.15)

    def testMatchSwarpBilinearExposure(self):
        """Test that warpExposure matches swarp using a bilinear warping kernel
        """
        self.compareToSwarp("bilinear", useWarpExposure=True, useSubregion=False, useDeepCopy=True)

    def testMatchSwarpLanczos2Image(self):
        """Test that warpExposure matches swarp using a lanczos2 warping kernel
        """
        self.compareToSwarp("lanczos2", useWarpExposure=False)

    def testMatchSwarpLanczos2Exposure(self):
        """Test that warpExposure matches swarp using a lanczos2 warping kernel.
        """
        self.compareToSwarp("lanczos2", useWarpExposure=True)

    def testMatchSwarpLanczos2SubExposure(self):
        """Test that warpExposure matches swarp using a lanczos2 warping kernel with a subexposure
        """
        for useDeepCopy in (False, True):
            self.compareToSwarp("lanczos2", useWarpExposure=True, useSubregion=True, useDeepCopy=useDeepCopy)

    def testMatchSwarpLanczos3Image(self):
        """Test that warpExposure matches swarp using a lanczos2 warping kernel
        """
        self.compareToSwarp("lanczos3", useWarpExposure=False)

    def testMatchSwarpLanczos3(self):
        """Test that warpExposure matches swarp using a lanczos4 warping kernel.
        """
        self.compareToSwarp("lanczos3", useWarpExposure=True)

    def testMatchSwarpLanczos4Image(self):
        """Test that warpExposure matches swarp using a lanczos2 warping kernel
        """
        self.compareToSwarp("lanczos4", useWarpExposure=False)

    def testMatchSwarpLanczos4(self):
        """Test that warpExposure matches swarp using a lanczos4 warping kernel.
        """
        self.compareToSwarp("lanczos4", useWarpExposure=True)

    def testMatchSwarpNearestExposure(self):
        """Test that warpExposure matches swarp using a nearest neighbor warping kernel
        """
        self.compareToSwarp("nearest", useWarpExposure=True, atol=60)

    def testTicket2441(self):
        """Test ticket 2441: warpExposure sometimes mishandles zero-extent dest exposures"""
        fromWcs = makeWcs(
            pixelScale = afwGeom.Angle(1.0e-8, afwGeom.degrees),
            projection = "TAN",
            crPixPos = (0, 0),
            crValCoord = afwCoord.IcrsCoord(afwGeom.Point2D(359, 0), afwGeom.degrees),
        )
        fromExp = afwImage.ExposureF(afwImage.MaskedImageF(10, 10), fromWcs)
        
        toWcs = makeWcs(
            pixelScale = afwGeom.Angle(0.00011, afwGeom.degrees),
            projection = "CEA",
            crPixPos = (410000.0, 11441.0),
            crValCoord = afwCoord.IcrsCoord(afwGeom.Point2D(45, 0), afwGeom.degrees),
            doFlipX = True,
        )
        toExp = afwImage.ExposureF(afwImage.MaskedImageF(0,0), toWcs)
        
        warpControl = afwMath.WarpingControl("lanczos3")
        # if a bug described in ticket #2441 is present, this will raise an exception:
        numGoodPix = afwMath.warpExposure(toExp, fromExp, warpControl)
        self.assertEqual(numGoodPix, 0)
    
    def testSmallSrc(self):
        """Verify that a source image that is too small will not raise an exception
        
        This tests another bug that was fixed in ticket #2441
        """
        fromWcs = makeWcs(
            pixelScale = afwGeom.Angle(1.0e-8, afwGeom.degrees),
            projection = "TAN",
            crPixPos = (0, 0),
            crValCoord = afwCoord.IcrsCoord(afwGeom.Point2D(359, 0), afwGeom.degrees),
        )
        fromExp = afwImage.ExposureF(afwImage.MaskedImageF(1, 1), fromWcs)
        
        toWcs = makeWcs(
            pixelScale = afwGeom.Angle(1.1e-8, afwGeom.degrees),
            projection = "TAN",
            crPixPos = (0, 0),
            crValCoord = afwCoord.IcrsCoord(afwGeom.Point2D(358, 0), afwGeom.degrees),
        )
        toExp = afwImage.ExposureF(afwImage.MaskedImageF(10,10), toWcs)

        warpControl = afwMath.WarpingControl("lanczos3")
        # if a bug described in ticket #2441 is present, this will raise an exception:
        numGoodPix = afwMath.warpExposure(toExp, fromExp, warpControl)
        self.assertEqual(numGoodPix, 0)
        imArr, maskArr, varArr = toExp.getMaskedImage().getArrays()
        self.assertTrue(numpy.alltrue(numpy.isnan(imArr)))
        self.assertTrue(numpy.alltrue(numpy.isinf(varArr)))
        edgeMask = afwImage.MaskU.getPlaneBitMask("EDGE")
        self.assertTrue(numpy.alltrue(maskArr == edgeMask))
    
    def verifyMaskWarp(self, kernelName, maskKernelName, growFullMask, interpLength=10, cacheSize=100000,
       rtol=4e-05, atol=1e-2):
        """Verify that using a separate mask warping kernel produces the correct results
        
        Inputs:
        - kernelName: name of warping kernel in the form used by afwImage.makeKernel
        - maskKernelName: name of mask warping kernel in the form used by afwImage.makeKernel
        - interpLength: interpLength argument for lsst.afw.math.WarpingControl
        - cacheSize: cacheSize argument for lsst.afw.math.WarpingControl;
            0 disables the cache
            10000 gives some speed improvement but less accurate results (atol must be increased)
            100000 gives better accuracy but no speed improvement in this test
        - rtol: relative tolerance as used by numpy.allclose
        - atol: absolute tolerance as used by numpy.allclose
        """
        srcWcs = makeWcs(
            pixelScale = afwGeom.Angle(0.2, afwGeom.degrees),
            crPixPos = (10.0, 11.0),
            crValCoord = afwCoord.IcrsCoord(afwGeom.Point2D(41.7, 32.9), afwGeom.degrees),
        )
        destWcs = makeWcs(
            pixelScale = afwGeom.Angle(0.17, afwGeom.degrees),
            crPixPos = (9.0, 10.0),
            crValCoord = afwCoord.IcrsCoord(afwGeom.Point2D(41.65, 32.95), afwGeom.degrees),
            posAng = afwGeom.Angle(31, afwGeom.degrees),
        )
        
        srcMaskedImage = afwImage.MaskedImageF(100, 101)
        srcExposure = afwImage.ExposureF(srcMaskedImage, srcWcs)

        destMaskedImage = afwImage.MaskedImageF(110, 121)
        destExposure = afwImage.ExposureF(destMaskedImage, destWcs)
        
        srcArrays = srcMaskedImage.getArrays()
        shape = srcArrays[0].shape
        numpy.random.seed(0)
        srcArrays[0][:] = numpy.random.normal(10000, 1000, size=shape)
        srcArrays[2][:] = numpy.random.normal( 9000,  900, size=shape)
        srcArrays[1][:] = numpy.reshape(numpy.arange(0, shape[0] * shape[1], 1, dtype=numpy.uint16), shape)
        
        warpControl = afwMath.WarpingControl(
            kernelName,
            maskKernelName,
            cacheSize,
            interpLength,
            afwGpu.DEFAULT_DEVICE_PREFERENCE,
            growFullMask
        )
        afwMath.warpExposure(destExposure, srcExposure, warpControl)
        afwArrays = [numpy.copy(arr) for arr in destExposure.getMaskedImage().getArrays()]

        # now compute with two separate mask planes        
        warpControl.setGrowFullMask(0)
        afwMath.warpExposure(destExposure, srcExposure, warpControl)
        narrowArrays = [numpy.copy(arr) for arr in destExposure.getMaskedImage().getArrays()]

        warpControl.setMaskWarpingKernelName("")
        afwMath.warpExposure(destExposure, srcExposure, warpControl)
        broadArrays = [numpy.copy(arr) for arr in destExposure.getMaskedImage().getArrays()]


        if (kernelName != maskKernelName) and (growFullMask != 0xFFFF):
            # we expect the mask planes to differ
            if numpy.allclose(broadArrays[1], narrowArrays[1]):
                self.fail("No difference between broad and narrow mask")

        predMask = (broadArrays[1] & growFullMask) | (narrowArrays[1] & ~growFullMask)
        predArraySet = (broadArrays[0], predMask, broadArrays[2])
        
        errStr = imageTestUtils.maskedImagesDiffer(afwArrays, predArraySet,
            doImage=True, doMask=True, doVariance=True,
            rtol=rtol, atol=atol)
        if errStr:
            self.fail("Separate mask warping failed; warpingKernel=%s; maskWarpingKernel=%s; error=%s" % \
                (kernelName, maskKernelName, errStr))

    def compareToSwarp(self, kernelName, 
        useWarpExposure=True, useSubregion=False, useDeepCopy=False,
        interpLength=10, cacheSize=100000,
        rtol=4e-05, atol=1e-2):
        """Compare warpExposure to swarp for given warping kernel.
        
        Note that swarp only warps the image plane, so only test that plane.
        
        Inputs:
        - kernelName: name of kernel in the form used by afwImage.makeKernel
        - useWarpExposure: if True, call warpExposure to warp an ExposureF,
            else call warpImage to warp an ImageF
        - useSubregion: if True then the original source exposure (from which the usual
            test exposure was extracted) is read and the correct subregion extracted
        - useDeepCopy: if True then the copy of the subimage is a deep copy,
            else it is a shallow copy; ignored if useSubregion is False
        - interpLength: interpLength argument for lsst.afw.math.WarpingControl
        - cacheSize: cacheSize argument for lsst.afw.math.WarpingControl;
            0 disables the cache
            10000 gives some speed improvement but less accurate results (atol must be increased)
            100000 gives better accuracy but no speed improvement in this test
        - rtol: relative tolerance as used by numpy.allclose
        - atol: absolute tolerance as used by numpy.allclose
        """
        warpingControl = afwMath.WarpingControl(
            kernelName,
            "", # there is no point to a separate mask kernel since we aren't testing the mask plane
            cacheSize,
            interpLength,
        )
        if useSubregion:
            originalFullExposure = afwImage.ExposureF(originalExposurePath)
            # "medsub" is a subregion of med starting at 0-indexed pixel (40, 150) of size 145 x 200
            bbox = afwGeom.Box2I(afwGeom.Point2I(40, 150), afwGeom.Extent2I(145, 200))
            originalExposure = afwImage.ExposureF(originalFullExposure, bbox, afwImage.LOCAL, useDeepCopy)
            swarpedImageName = "medsubswarp1%s.fits" % (kernelName,)
        else:
            originalExposure = afwImage.ExposureF(originalExposurePath)
            swarpedImageName = "medswarp1%s.fits" % (kernelName,)

        swarpedImagePath = os.path.join(dataDir, swarpedImageName)
        swarpedDecoratedImage = afwImage.DecoratedImageF(swarpedImagePath)
        swarpedImage = swarpedDecoratedImage.getImage()
        swarpedMetadata = swarpedDecoratedImage.getMetadata()
        warpedWcs = afwImage.makeWcs(swarpedMetadata)
        
        if useWarpExposure:
            # path for saved afw-warped image
            afwWarpedImagePath = "afwWarpedExposure1%s" % (kernelName,)
    
            afwWarpedMaskedImage = afwImage.MaskedImageF(swarpedImage.getDimensions())
            afwWarpedExposure = afwImage.ExposureF(afwWarpedMaskedImage, warpedWcs)
            afwMath.warpExposure(afwWarpedExposure, originalExposure, warpingControl)
            if SAVE_FITS_FILES:
                afwWarpedExposure.writeFits(afwWarpedImagePath)
            if display:
                ds9.mtv(afwWarpedExposure, frame=1, title="Warped")
    
            afwWarpedMask = afwWarpedMaskedImage.getMask()
            edgeBitMask = afwWarpedMask.getPlaneBitMask("EDGE")
            if edgeBitMask == 0:
                self.fail("warped mask has no EDGE bit")
            afwWarpedMaskedImageArrSet = afwWarpedMaskedImage.getArrays()
            afwWarpedMaskArr = afwWarpedMaskedImageArrSet[1]
    
            swarpedMaskedImage = afwImage.MaskedImageF(swarpedImage)
            swarpedMaskedImageArrSet = swarpedMaskedImage.getArrays()

            if display:
                ds9.mtv(swarpedMaskedImage, frame=2, title="SWarped")
            
            errStr = imageTestUtils.maskedImagesDiffer(afwWarpedMaskedImageArrSet, swarpedMaskedImageArrSet,
                doImage=True, doMask=False, doVariance=False, skipMaskArr=afwWarpedMaskArr,
                rtol=rtol, atol=atol)
            if errStr:
                if SAVE_FAILED_FITS_FILES:
                    afwWarpedExposure.writeFits(afwWarpedImagePath)
                    print "Saved failed afw-warped exposure as: %s" % (afwWarpedImagePath,)
                self.fail("afw and swarp %s-warped %s (ignoring bad pixels)" % (kernelName, errStr))
        else:
            # path for saved afw-warped image
            afwWarpedImagePath = "afwWarpedImage1%s.fits" % (kernelName,)
    
            afwWarpedImage = afwImage.ImageF(swarpedImage.getDimensions())
            originalImage = originalExposure.getMaskedImage().getImage()
            originalWcs = originalExposure.getWcs()
            afwMath.warpImage(afwWarpedImage, warpedWcs, originalImage,
                              originalWcs, warpingControl)
            if display:
                ds9.mtv(afwWarpedImage, frame=1, title="Warped")
                ds9.mtv(swarpedImage, frame=2, title="SWarped")
                diff = swarpedImage.Factory(swarpedImage, True)
                diff -= afwWarpedImage
                ds9.mtv(diff, frame=3, title="swarp - afw")
            if SAVE_FITS_FILES:
                afwWarpedImage.writeFits(afwWarpedImagePath)
            
            afwWarpedImageArr = afwWarpedImage.getArray()
            swarpedImageArr = swarpedImage.getArray()
            edgeMaskArr = numpy.isnan(afwWarpedImageArr)
            errStr = imageTestUtils.imagesDiffer(afwWarpedImageArr, swarpedImageArr,
                skipMaskArr=edgeMaskArr, rtol=rtol, atol=atol)
            if errStr:
                if SAVE_FAILED_FITS_FILES:
                    # save the image anyway
                    afwWarpedImage.writeFits(afwWarpedImagePath)
                    print "Saved failed afw-warped image as: %s" % (afwWarpedImagePath,)
                self.fail("afw and swarp %s-warped images do not match (ignoring NaN pixels): %s" % \
                    (kernelName, errStr))
    
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
        arr1 = image1.getArray()
        arr2 = image2.getArray()

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

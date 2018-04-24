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
import os
import unittest

import numpy as np

import lsst.utils
import lsst.utils.tests
import lsst.daf.base as dafBase
from lsst.afw.coord import Observatory, Weather
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.image.utils as imageUtils
import lsst.pex.policy as pexPolicy
import lsst.pex.exceptions as pexExcept
import lsst.afw.display.ds9 as ds9
from lsst.log import Log

# Change the level to Log.DEBUG to see debug messages
Log.getLogger("afw.image.Mask").setLevel(Log.INFO)
Log.getLogger("TRACE2.afw.math.warp").setLevel(Log.INFO)
Log.getLogger("TRACE3.afw.math.warp").setLevel(Log.INFO)


display = False
# set True to save afw-warped images as FITS files
SAVE_FITS_FILES = False
# set True to save failed afw-warped images as FITS files even if
# SAVE_FITS_FILES is False
SAVE_FAILED_FITS_FILES = True

try:
    afwdataDir = lsst.utils.getPackageDir("afwdata")
except pexExcept.NotFoundError:
    afwdataDir = None
else:
    dataDir = os.path.join(afwdataDir, "data")

    originalExposureName = "medexp.fits"
    originalExposurePath = os.path.join(dataDir, originalExposureName)
    subExposureName = "medsub.fits"
    subExposurePath = os.path.join(dataDir, originalExposureName)
    originalFullExposureName = os.path.join(
        "CFHT", "D4", "cal-53535-i-797722_1.fits")
    originalFullExposurePath = os.path.join(dataDir, originalFullExposureName)


def makeVisitInfo():
    """Return a non-NaN visitInfo."""
    return afwImage.VisitInfo(exposureId=10313423,
                              exposureTime=10.01,
                              darkTime=11.02,
                              date=dafBase.DateTime(65321.1, dafBase.DateTime.MJD, dafBase.DateTime.TAI),
                              ut1=12345.1,
                              era=45.1*afwGeom.degrees,
                              boresightRaDec=afwGeom.SpherePoint(23.1, 73.2, afwGeom.degrees),
                              boresightAzAlt=afwGeom.SpherePoint(134.5, 33.3, afwGeom.degrees),
                              boresightAirmass=1.73,
                              boresightRotAngle=73.2*afwGeom.degrees,
                              rotType=afwImage.RotType.SKY,
                              observatory=Observatory(11.1*afwGeom.degrees, 22.2*afwGeom.degrees, 0.333),
                              weather=Weather(1.1, 2.2, 34.5),
                              )


class WarpExposureTestCase(lsst.utils.tests.TestCase):
    """Test case for warpExposure
    """

    def setUp(self):
        np.random.seed(0)

    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
    def testNullWarpExposure(self, interpLength=10):
        """Test that warpExposure maps an image onto itself.

        Note:
        - NO_DATA and off-CCD pixels must be ignored
        - bad mask pixels get smeared out so we have to excluded all bad mask pixels
          from the output image when comparing masks.
        """
        filterPolicyFile = pexPolicy.DefaultPolicyFile(
            "afw", "SdssFilters.paf", "tests")
        filterPolicy = pexPolicy.Policy.createPolicy(
            filterPolicyFile, filterPolicyFile.getRepositoryPath(), True)
        imageUtils.defineFiltersFromPolicy(filterPolicy, reset=True)

        originalExposure = afwImage.ExposureF(originalExposurePath)
        originalExposure.getInfo().setVisitInfo(makeVisitInfo())
        originalFilter = afwImage.Filter("i")
        originalCalib = afwImage.Calib()
        originalCalib.setFluxMag0(1.0e5, 1.0e3)
        originalExposure.setFilter(originalFilter)
        originalExposure.setCalib(originalCalib)
        afwWarpedExposure = afwImage.ExposureF(
            originalExposure.getBBox(),
            originalExposure.getWcs())
        warpingControl = afwMath.WarpingControl(
            "lanczos4", "", 0, interpLength)
        afwMath.warpExposure(
            afwWarpedExposure, originalExposure, warpingControl)
        if SAVE_FITS_FILES:
            afwWarpedExposure.writeFits("afwWarpedExposureNull.fits")

        self.assertEqual(afwWarpedExposure.getFilter().getName(),
                         originalFilter.getName())
        self.assertEqual(afwWarpedExposure.getCalib().getFluxMag0(),
                         originalCalib.getFluxMag0())
        self.assertEqual(afwWarpedExposure.getInfo().getVisitInfo(),
                         originalExposure.getInfo().getVisitInfo())

        afwWarpedMaskedImage = afwWarpedExposure.getMaskedImage()
        afwWarpedMask = afwWarpedMaskedImage.getMask()
        noDataBitMask = afwWarpedMask.getPlaneBitMask("NO_DATA")
        afwWarpedMaskedImageArrSet = afwWarpedMaskedImage.getArrays()
        afwWarpedMaskArr = afwWarpedMaskedImageArrSet[1]

        # compare all non-DATA pixels of image and variance, but relax specs a bit
        # because of minor noise introduced by bad pixels
        noDataMaskArr = afwWarpedMaskArr & noDataBitMask
        msg = "afw null-warped MaskedImage (all pixels, relaxed tolerance)"
        self.assertMaskedImagesAlmostEqual(afwWarpedMaskedImage, originalExposure.getMaskedImage(),
                                           doMask=False, skipMask=noDataMaskArr, atol=1e-5, msg=msg)

        # compare good pixels (mask=0) of image, mask and variance using full
        # tolerance
        msg = "afw null-warped MaskedImage (good pixels, max tolerance)"
        self.assertMaskedImagesAlmostEqual(afwWarpedMaskedImage, originalExposure.getMaskedImage(),
                                           skipMask=afwWarpedMask, msg=msg)

    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
    def testNullWarpImage(self, interpLength=10):
        """Test that warpImage maps an image onto itself.
        """
        originalExposure = afwImage.ExposureF(originalExposurePath)
        afwWarpedExposure = afwImage.ExposureF(originalExposurePath)
        originalImage = originalExposure.getMaskedImage().getImage()
        afwWarpedImage = afwWarpedExposure.getMaskedImage().getImage()
        originalWcs = originalExposure.getWcs()
        afwWarpedWcs = afwWarpedExposure.getWcs()
        warpingControl = afwMath.WarpingControl(
            "lanczos4", "", 0, interpLength)
        afwMath.warpImage(afwWarpedImage, afwWarpedWcs,
                          originalImage, originalWcs, warpingControl)
        if SAVE_FITS_FILES:
            afwWarpedImage.writeFits("afwWarpedImageNull.fits")
        afwWarpedImageArr = afwWarpedImage.getArray()
        noDataMaskArr = np.isnan(afwWarpedImageArr)
        # relax specs a bit because of minor noise introduced by bad pixels
        msg = "afw null-warped Image"
        self.assertImagesAlmostEqual(originalImage, afwWarpedImage, skipMask=noDataMaskArr,
                                     atol=1e-5, msg=msg)

    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
    def testNullWcs(self, interpLength=10):
        """Cannot warp from or into an exposure without a Wcs.
        """
        exposureWithWcs = afwImage.ExposureF(originalExposurePath)
        mi = exposureWithWcs.getMaskedImage()
        exposureWithoutWcs = afwImage.ExposureF(mi.getDimensions())
        warpingControl = afwMath.WarpingControl(
            "bilinear", "", 0, interpLength)
        try:
            afwMath.warpExposure(
                exposureWithWcs, exposureWithoutWcs, warpingControl)
            self.fail("warping from a source Exception with no Wcs should fail")
        except Exception:
            pass
        try:
            afwMath.warpExposure(exposureWithoutWcs,
                                 exposureWithWcs, warpingControl)
            self.fail(
                "warping into a destination Exception with no Wcs should fail")
        except Exception:
            pass

    def testWarpIntoSelf(self, interpLength=10):
        """Cannot warp in-place
        """
        originalExposure = afwImage.ExposureF(afwGeom.Extent2I(100, 100))
        warpingControl = afwMath.WarpingControl(
            "bilinear", "", 0, interpLength)
        try:
            afwMath.warpExposure(
                originalExposure, originalExposure, warpingControl)
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
            self.assertEqual(
                wc.getMaskWarpingKernel().getCacheSize(), cacheSize)
            for newCacheSize in (1, 50):
                wc.setCacheSize(newCacheSize)
                self.assertEqual(wc.getCacheSize(), newCacheSize)
                self.assertEqual(
                    wc.getWarpingKernel().getCacheSize(), newCacheSize)
                self.assertEqual(
                    wc.getMaskWarpingKernel().getCacheSize(), newCacheSize)

    def testWarpingControlError(self):
        """Test error handling of WarpingControl
        """
        # error: mask kernel smaller than main kernel
        for kernelName, maskKernelName in (
            ("bilinear", "lanczos3"),
            ("bilinear", "lanczos4"),
            ("lanczos3", "lanczos4"),
        ):
            with self.assertRaises(pexExcept.Exception):
                afwMath.WarpingControl(kernelName, maskKernelName)

        # error: new mask kernel larger than main kernel
        warpingControl = afwMath.WarpingControl("bilinear")
        for maskKernelName in ("lanczos3", "lanczos4"):
            with self.assertRaises(pexExcept.Exception):
                warpingControl.setMaskWarpingKernelName(maskKernelName)

        # error: new kernel smaller than mask kernel
        warpingControl = afwMath.WarpingControl("lanczos4", "lanczos4")
        for kernelName in ("bilinear", "lanczos3"):
            with self.assertRaises(pexExcept.Exception):
                warpingControl.setWarpingKernelName(kernelName)

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
            ("lanczos", ""),  # no digit after lanczos
            ("lanczos3", "badname"),
            ("lanczos3", "lanczos"),
        ):
            with self.assertRaises(pexExcept.Exception):
                afwMath.WarpingControl(kernelName, maskKernelName)

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
        self.compareToSwarp("bilinear", useWarpExposure=True,
                            useSubregion=False, useDeepCopy=True)

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
            self.compareToSwarp("lanczos2", useWarpExposure=True,
                                useSubregion=True, useDeepCopy=useDeepCopy)

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

    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
    def testTransformBasedWarp(self):
        """Test warping using TransformPoint2ToPoint2
        """
        for interpLength in (0, 1, 2, 4):
            kernelName = "lanczos3"
            rtol = 4e-5
            atol = 1e-2
            warpingControl = afwMath.WarpingControl(
                warpingKernelName=kernelName,
                interpLength=interpLength,
            )

            originalExposure = afwImage.ExposureF(originalExposurePath)
            originalMetadata = afwImage.DecoratedImageF(originalExposurePath).getMetadata()
            originalSkyWcs = afwGeom.makeSkyWcs(originalMetadata)

            swarpedImageName = "medswarp1%s.fits" % (kernelName,)
            swarpedImagePath = os.path.join(dataDir, swarpedImageName)
            swarpedDecoratedImage = afwImage.DecoratedImageF(swarpedImagePath)
            swarpedImage = swarpedDecoratedImage.getImage()

            swarpedMetadata = swarpedDecoratedImage.getMetadata()
            warpedSkyWcs = afwGeom.makeSkyWcs(swarpedMetadata)

            # original image is source, warped image is destination
            srcToDest = afwGeom.makeWcsPairTransform(originalSkyWcs, warpedSkyWcs)

            afwWarpedMaskedImage = afwImage.MaskedImageF(swarpedImage.getDimensions())
            originalMaskedImage = originalExposure.getMaskedImage()

            numGoodPix = afwMath.warpImage(afwWarpedMaskedImage, originalMaskedImage,
                                           srcToDest, warpingControl)
            self.assertGreater(numGoodPix, 50)

            afwWarpedImage = afwWarpedMaskedImage.getImage()
            afwWarpedImageArr = afwWarpedImage.getArray()
            noDataMaskArr = np.isnan(afwWarpedImageArr)
            self.assertImagesAlmostEqual(afwWarpedImage, swarpedImage,
                                         skipMask=noDataMaskArr, rtol=rtol, atol=atol)

    def testTicket2441(self):
        """Test ticket 2441: warpExposure sometimes mishandles zero-extent dest exposures"""
        fromWcs = afwGeom.makeSkyWcs(
            crpix=afwGeom.Point2D(0, 0),
            crval=afwGeom.SpherePoint(359, 0, afwGeom.degrees),
            cdMatrix=afwGeom.makeCdMatrix(scale=1.0e-8*afwGeom.degrees),
        )
        fromExp = afwImage.ExposureF(afwImage.MaskedImageF(10, 10), fromWcs)

        toWcs = afwGeom.makeSkyWcs(
            crpix=afwGeom.Point2D(410000, 11441),
            crval=afwGeom.SpherePoint(45, 0, afwGeom.degrees),
            cdMatrix=afwGeom.makeCdMatrix(scale=0.00011*afwGeom.degrees, flipX=True),
            projection="CEA",
        )
        toExp = afwImage.ExposureF(afwImage.MaskedImageF(0, 0), toWcs)

        warpControl = afwMath.WarpingControl("lanczos3")
        # if a bug described in ticket #2441 is present, this will raise an
        # exception:
        numGoodPix = afwMath.warpExposure(toExp, fromExp, warpControl)
        self.assertEqual(numGoodPix, 0)

    def testTicketDM4063(self):
        """Test that a uint16 array can be cast to a bool array, to avoid DM-4063
        """
        a = np.array([0, 1, 0, 23], dtype=np.uint16)
        b = np.array([True, True, False, False], dtype=bool)
        acast = np.array(a != 0, dtype=bool)
        orArr = acast | b
        desOrArr = np.array([True, True, False, True], dtype=bool)
        # Note: assertEqual(bool arr, bool arr) fails with:
        # ValueError: The truth value of an array with more than one element is
        # ambiguous
        try:
            self.assertTrue(np.all(orArr == desOrArr))
        except Exception as e:
            print("Failed: %r != %r: %s" % (orArr, desOrArr, e))
            raise

    def testSmallSrc(self):
        """Verify that a source image that is too small will not raise an exception

        This tests another bug that was fixed in ticket #2441
        """
        fromWcs = afwGeom.makeSkyWcs(
            crpix=afwGeom.Point2D(0, 0),
            crval=afwGeom.SpherePoint(359, 0, afwGeom.degrees),
            cdMatrix=afwGeom.makeCdMatrix(scale=1.0e-8*afwGeom.degrees),
        )
        fromExp = afwImage.ExposureF(afwImage.MaskedImageF(1, 1), fromWcs)

        toWcs = afwGeom.makeSkyWcs(
            crpix=afwGeom.Point2D(0, 0),
            crval=afwGeom.SpherePoint(358, 0, afwGeom.degrees),
            cdMatrix=afwGeom.makeCdMatrix(scale=1.1e-8*afwGeom.degrees),
        )
        toExp = afwImage.ExposureF(afwImage.MaskedImageF(10, 10), toWcs)

        warpControl = afwMath.WarpingControl("lanczos3")
        # if a bug described in ticket #2441 is present, this will raise an
        # exception:
        numGoodPix = afwMath.warpExposure(toExp, fromExp, warpControl)
        self.assertEqual(numGoodPix, 0)
        imArr, maskArr, varArr = toExp.getMaskedImage().getArrays()
        self.assertTrue(np.all(np.isnan(imArr)))
        self.assertTrue(np.all(np.isinf(varArr)))
        noDataBitMask = afwImage.Mask.getPlaneBitMask("NO_DATA")
        self.assertTrue(np.all(maskArr == noDataBitMask))

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
        - rtol: relative tolerance as used by np.allclose
        - atol: absolute tolerance as used by np.allclose
        """
        srcWcs = afwGeom.makeSkyWcs(
            crpix=afwGeom.Point2D(10, 11),
            crval=afwGeom.SpherePoint(41.7, 32.9, afwGeom.degrees),
            cdMatrix=afwGeom.makeCdMatrix(scale=0.2*afwGeom.degrees),
        )
        destWcs = afwGeom.makeSkyWcs(
            crpix=afwGeom.Point2D(9, 10),
            crval=afwGeom.SpherePoint(41.65, 32.95, afwGeom.degrees),
            cdMatrix=afwGeom.makeCdMatrix(scale=0.17*afwGeom.degrees),
        )

        srcMaskedImage = afwImage.MaskedImageF(100, 101)
        srcExposure = afwImage.ExposureF(srcMaskedImage, srcWcs)

        srcArrays = srcMaskedImage.getArrays()
        shape = srcArrays[0].shape
        srcArrays[0][:] = np.random.normal(10000, 1000, size=shape)
        srcArrays[2][:] = np.random.normal(9000, 900, size=shape)
        srcArrays[1][:] = np.reshape(
            np.arange(0, shape[0] * shape[1], 1, dtype=np.uint16), shape)

        warpControl = afwMath.WarpingControl(
            kernelName,
            maskKernelName,
            cacheSize,
            interpLength,
            growFullMask
        )
        destMaskedImage = afwImage.MaskedImageF(110, 121)
        destExposure = afwImage.ExposureF(destMaskedImage, destWcs)
        afwMath.warpExposure(destExposure, srcExposure, warpControl)

        # now compute with two separate mask planes
        warpControl.setGrowFullMask(0)
        narrowMaskedImage = afwImage.MaskedImageF(110, 121)
        narrowExposure = afwImage.ExposureF(narrowMaskedImage, destWcs)
        afwMath.warpExposure(narrowExposure, srcExposure, warpControl)
        narrowArrays = narrowExposure.getMaskedImage().getArrays()

        warpControl.setMaskWarpingKernelName("")
        broadMaskedImage = afwImage.MaskedImageF(110, 121)
        broadExposure = afwImage.ExposureF(broadMaskedImage, destWcs)
        afwMath.warpExposure(broadExposure, srcExposure, warpControl)
        broadArrays = broadExposure.getMaskedImage().getArrays()

        if (kernelName != maskKernelName) and (growFullMask != 0xFFFF):
            # we expect the mask planes to differ
            if np.all(narrowArrays[1] == broadArrays[1]):
                self.fail("No difference between broad and narrow mask")

        predMask = (broadArrays[1] & growFullMask) | (
            narrowArrays[1] & ~growFullMask).astype(np.uint16)
        predArraySet = (broadArrays[0], predMask, broadArrays[2])
        predExposure = afwImage.makeMaskedImageFromArrays(*predArraySet)

        msg = "Separate mask warping failed; warpingKernel=%s; maskWarpingKernel=%s" % \
            (kernelName, maskKernelName)
        self.assertMaskedImagesAlmostEqual(destExposure.getMaskedImage(), predExposure,
                                           doImage=True, doMask=True, doVariance=True,
                                           rtol=rtol, atol=atol, msg=msg)

    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
    def compareToSwarp(self, kernelName,
                       useWarpExposure=True, useSubregion=False, useDeepCopy=False,
                       interpLength=10, cacheSize=100000,
                       rtol=4e-05, atol=1e-2):
        """Compare warpExposure to swarp for given warping kernel.

        Note that swarp only warps the image plane, so only test that plane.

        Inputs:
        - kernelName: name of kernel in the form used by afwImage.makeKernel
        - useWarpExposure: if True, call warpExposure to warp an ExposureF,
            else call warpImage to warp an ImageF and also call the Transform version
        - useSubregion: if True then the original source exposure (from which the usual
            test exposure was extracted) is read and the correct subregion extracted
        - useDeepCopy: if True then the copy of the subimage is a deep copy,
            else it is a shallow copy; ignored if useSubregion is False
        - interpLength: interpLength argument for lsst.afw.math.WarpingControl
        - cacheSize: cacheSize argument for lsst.afw.math.WarpingControl;
            0 disables the cache
            10000 gives some speed improvement but less accurate results (atol must be increased)
            100000 gives better accuracy but no speed improvement in this test
        - rtol: relative tolerance as used by np.allclose
        - atol: absolute tolerance as used by np.allclose
        """
        warpingControl = afwMath.WarpingControl(
            kernelName,
            "",  # there is no point to a separate mask kernel since we aren't testing the mask plane
            cacheSize,
            interpLength,
        )
        if useSubregion:
            originalFullExposure = afwImage.ExposureF(originalExposurePath)
            # "medsub" is a subregion of med starting at 0-indexed pixel (40, 150) of size 145 x 200
            bbox = afwGeom.Box2I(afwGeom.Point2I(40, 150),
                                 afwGeom.Extent2I(145, 200))
            originalExposure = afwImage.ExposureF(
                originalFullExposure, bbox, afwImage.LOCAL, useDeepCopy)
            swarpedImageName = "medsubswarp1%s.fits" % (kernelName,)
        else:
            originalExposure = afwImage.ExposureF(originalExposurePath)
            swarpedImageName = "medswarp1%s.fits" % (kernelName,)

        swarpedImagePath = os.path.join(dataDir, swarpedImageName)
        swarpedDecoratedImage = afwImage.DecoratedImageF(swarpedImagePath)
        swarpedImage = swarpedDecoratedImage.getImage()
        swarpedMetadata = swarpedDecoratedImage.getMetadata()
        warpedWcs = afwGeom.makeSkyWcs(swarpedMetadata)

        if useWarpExposure:
            # path for saved afw-warped image
            afwWarpedImagePath = "afwWarpedExposure1%s.fits" % (kernelName,)

            afwWarpedMaskedImage = afwImage.MaskedImageF(
                swarpedImage.getDimensions())
            afwWarpedExposure = afwImage.ExposureF(
                afwWarpedMaskedImage, warpedWcs)
            afwMath.warpExposure(
                afwWarpedExposure, originalExposure, warpingControl)
            afwWarpedMask = afwWarpedMaskedImage.getMask()
            if SAVE_FITS_FILES:
                afwWarpedExposure.writeFits(afwWarpedImagePath)
            if display:
                ds9.mtv(afwWarpedExposure, frame=1, title="Warped")

            swarpedMaskedImage = afwImage.MaskedImageF(swarpedImage)

            if display:
                ds9.mtv(swarpedMaskedImage, frame=2, title="SWarped")

            msg = "afw and swarp %s-warped differ (ignoring bad pixels)" % (
                kernelName,)
            try:
                self.assertMaskedImagesAlmostEqual(afwWarpedMaskedImage, swarpedMaskedImage,
                                                   doImage=True, doMask=False, doVariance=False,
                                                   skipMask=afwWarpedMask, rtol=rtol, atol=atol, msg=msg)
            except Exception:
                if SAVE_FAILED_FITS_FILES:
                    afwWarpedExposure.writeFits(afwWarpedImagePath)
                    print("Saved failed afw-warped exposure as: %s" %
                          (afwWarpedImagePath,))
                raise
        else:
            # path for saved afw-warped image
            afwWarpedImagePath = "afwWarpedImage1%s.fits" % (kernelName,)
            afwWarpedImage2Path = "afwWarpedImage1%s_xyTransform.fits" % (
                kernelName,)

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
            noDataMaskArr = np.isnan(afwWarpedImageArr)
            msg = "afw and swarp %s-warped images do not match (ignoring NaN pixels)" % \
                (kernelName,)
            try:
                self.assertImagesAlmostEqual(afwWarpedImage, swarpedImage,
                                             skipMask=noDataMaskArr, rtol=rtol, atol=atol, msg=msg)
            except Exception:
                if SAVE_FAILED_FITS_FILES:
                    # save the image anyway
                    afwWarpedImage.writeFits(afwWarpedImagePath)
                    print("Saved failed afw-warped image as: %s" %
                          (afwWarpedImagePath,))
                raise

            afwWarpedImage2 = afwImage.ImageF(swarpedImage.getDimensions())
            srcToDest = afwGeom.makeWcsPairTransform(originalWcs, warpedWcs)
            afwMath.warpImage(afwWarpedImage2, originalImage,
                              srcToDest, warpingControl)
            msg = "afw transform-based and WCS-based %s-warped images do not match" % (
                kernelName,)
            try:
                self.assertImagesAlmostEqual(afwWarpedImage2, afwWarpedImage,
                                             rtol=rtol, atol=atol, msg=msg)
            except Exception:
                if SAVE_FAILED_FITS_FILES:
                    # save the image anyway
                    afwWarpedImage.writeFits(afwWarpedImagePath)
                    print("Saved failed afw-warped image as: %s" %
                          (afwWarpedImage2Path,))
                raise


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

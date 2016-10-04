#!/usr/bin/env python
from __future__ import absolute_import, division
from __future__ import print_function
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

"""Test warpExposure
"""
import math
import os
import unittest

import numpy

import lsst.utils
import lsst.utils.tests
import lsst.daf.base as dafBase
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.afw.gpu as afwGpu
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
# set True to save failed afw-warped images as FITS files even if SAVE_FITS_FILES is False
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
    originalFullExposureName = os.path.join("CFHT", "D4", "cal-53535-i-797722_1.fits")
    originalFullExposurePath = os.path.join(dataDir, originalFullExposureName)


def makeWcs(pixelScale, crPixPos, crValCoord, posAng=afwGeom.Angle(0.0), doFlipX=False, projection="TAN",
            radDecCSys="ICRS", equinox=2000):
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
    crPixFits = [ind + 1.0 for ind in crPixPos]  # convert pix position to FITS standard
    crValDeg = crValCoord.getPosition(afwGeom.degrees)
    posAngRad = posAng.asRadians()
    pixelScaleDeg = pixelScale.asDegrees()
    cdMat = numpy.array([[math.cos(posAngRad), math.sin(posAngRad)],
                         [-math.sin(posAngRad), math.cos(posAngRad)]], dtype=float) * pixelScaleDeg
    if doFlipX:
        cdMat[:, 0] = -cdMat[:, 0]
    for i in range(2):
        ip1 = i + 1
        ps.add("CTYPE%1d" % (ip1,), ctypeList[i])
        ps.add("CRPIX%1d" % (ip1,), crPixFits[i])
        ps.add("CRVAL%1d" % (ip1,), crValDeg[i])
    ps.add("RADECSYS", radDecCSys)
    ps.add("EQUINOX", equinox)
    ps.add("CD1_1", cdMat[0, 0])
    ps.add("CD2_1", cdMat[1, 0])
    ps.add("CD1_2", cdMat[0, 1])
    ps.add("CD2_2", cdMat[1, 1])
    return afwImage.makeWcs(ps)


class WarpExposureTestCase(lsst.utils.tests.TestCase):
    """Test case for warpExposure
    """

    def setUp(self):
        numpy.random.seed(0)

    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
    def testNullWarpExposure(self, interpLength=10):
        """Test that warpExposure maps an image onto itself.

        Note:
        - NO_DATA and off-CCD pixels must be ignored
        - bad mask pixels get smeared out so we have to excluded all bad mask pixels
          from the output image when comparing masks.
        """
        filterPolicyFile = pexPolicy.DefaultPolicyFile("afw", "SdssFilters.paf", "tests")
        filterPolicy = pexPolicy.Policy.createPolicy(
            filterPolicyFile, filterPolicyFile.getRepositoryPath(), True)
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

        self.assertEqual(afwWarpedExposure.getFilter().getName(), originalFilter.getName())
        self.assertEqual(afwWarpedExposure.getCalib().getFluxMag0(), originalCalib.getFluxMag0())

        afwWarpedMaskedImage = afwWarpedExposure.getMaskedImage()
        afwWarpedMask = afwWarpedMaskedImage.getMask()
        noDataBitMask = afwWarpedMask.getPlaneBitMask("NO_DATA")
        afwWarpedMaskedImageArrSet = afwWarpedMaskedImage.getArrays()
        afwWarpedMaskArr = afwWarpedMaskedImageArrSet[1]

        # compare all non-DATA pixels of image and variance, but relax specs a bit
        # because of minor noise introduced by bad pixels
        noDataMaskArr = afwWarpedMaskArr & noDataBitMask
        msg = "afw null-warped MaskedImage (all pixels, relaxed tolerance)"
        self.assertMaskedImagesNearlyEqual(afwWarpedMaskedImage, originalExposure.getMaskedImage(),
                                           doMask=False, skipMask=noDataMaskArr, atol=1e-5, msg=msg)

        # compare good pixels (mask=0) of image, mask and variance using full tolerance
        msg = "afw null-warped MaskedImage (good pixels, max tolerance)"
        self.assertMaskedImagesNearlyEqual(afwWarpedMaskedImage, originalExposure.getMaskedImage(),
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
        warpingControl = afwMath.WarpingControl("lanczos4", "", 0, interpLength)
        afwMath.warpImage(afwWarpedImage, afwWarpedWcs, originalImage, originalWcs, warpingControl)
        if SAVE_FITS_FILES:
            afwWarpedImage.writeFits("afwWarpedImageNull.fits")
        afwWarpedImageArr = afwWarpedImage.getArray()
        noDataMaskArr = numpy.isnan(afwWarpedImageArr)
        # relax specs a bit because of minor noise introduced by bad pixels
        msg = "afw null-warped Image"
        self.assertImagesNearlyEqual(originalImage, afwWarpedImage, skipMask=noDataMaskArr,
                                     atol=1e-5, msg=msg)

    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
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

        # error: GPU only works with Lanczos kernels
        with self.assertRaises(pexExcept.Exception):
            afwMath.WarpingControl("bilinear", "", 0, 0, afwGpu.USE_GPU)
        warpingControl = afwMath.WarpingControl("bilinear")
        with self.assertRaises(pexExcept.Exception):
            warpingControl.setDevicePreference(afwGpu.USE_GPU)

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

    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
    def testNonIcrs(self):
        """Test that warping to a non-ICRS-like coordinate system produces different results

        It would be better to also test that the results are as expected,
        but I have not been able to get swarp to perform this operation,
        so have not found an independent means of generating the expected results.
        """
        kernelName = "lanczos3"
        rtol = 4e-5
        atol = 1e-2
        warpingControl = afwMath.WarpingControl(
            kernelName,
        )

        originalExposure = afwImage.ExposureF(originalExposurePath)
        originalImage = originalExposure.getMaskedImage().getImage()
        originalWcs = originalExposure.getWcs()

        swarpedImageName = "medswarp1%s.fits" % (kernelName,)
        swarpedImagePath = os.path.join(dataDir, swarpedImageName)
        swarpedDecoratedImage = afwImage.DecoratedImageF(swarpedImagePath)
        swarpedImage = swarpedDecoratedImage.getImage()

        for changeEquinox in (False, True):
            swarpedMetadata = swarpedDecoratedImage.getMetadata()
            if changeEquinox:
                swarpedMetadata.set("RADECSYS", "FK5")
                swarpedMetadata.set("EQUINOX", swarpedMetadata.get("EQUINOX") + 1)
            warpedWcs = afwImage.makeWcs(swarpedMetadata)

            afwWarpedImage = afwImage.ImageF(swarpedImage.getDimensions())
            originalImage = originalExposure.getMaskedImage().getImage()
            originalWcs = originalExposure.getWcs()
            numGoodPix = afwMath.warpImage(afwWarpedImage, warpedWcs, originalImage,
                                           originalWcs, warpingControl)
            self.assertGreater(numGoodPix, 50)

            afwWarpedImageArr = afwWarpedImage.getArray()
            noDataMaskArr = numpy.isnan(afwWarpedImageArr)
            if changeEquinox:
                with self.assertRaises(AssertionError):
                    self.assertImagesNearlyEqual(afwWarpedImage, swarpedImage,
                                                 skipMask=noDataMaskArr, rtol=rtol, atol=atol)
            else:
                self.assertImagesNearlyEqual(afwWarpedImage, swarpedImage,
                                             skipMask=noDataMaskArr, rtol=rtol, atol=atol)

    def testTicket2441(self):
        """Test ticket 2441: warpExposure sometimes mishandles zero-extent dest exposures"""
        fromWcs = makeWcs(
            pixelScale=afwGeom.Angle(1.0e-8, afwGeom.degrees),
            projection="TAN",
            crPixPos=(0, 0),
            crValCoord=afwCoord.IcrsCoord(afwGeom.Point2D(359, 0), afwGeom.degrees),
        )
        fromExp = afwImage.ExposureF(afwImage.MaskedImageF(10, 10), fromWcs)

        toWcs = makeWcs(
            pixelScale=afwGeom.Angle(0.00011, afwGeom.degrees),
            projection="CEA",
            crPixPos=(410000.0, 11441.0),
            crValCoord=afwCoord.IcrsCoord(afwGeom.Point2D(45, 0), afwGeom.degrees),
            doFlipX=True,
        )
        toExp = afwImage.ExposureF(afwImage.MaskedImageF(0, 0), toWcs)

        warpControl = afwMath.WarpingControl("lanczos3")
        # if a bug described in ticket #2441 is present, this will raise an exception:
        numGoodPix = afwMath.warpExposure(toExp, fromExp, warpControl)
        self.assertEqual(numGoodPix, 0)

    def testTicketDM4063(self):
        """Test that a uint16 array can be cast to a bool array, to avoid DM-4063
        """
        a = numpy.array([0, 1, 0, 23], dtype=numpy.uint16)
        b = numpy.array([True, True, False, False], dtype=bool)
        acast = numpy.array(a != 0, dtype=bool)
        orArr = acast | b
        desOrArr = numpy.array([True, True, False, True], dtype=bool)
        # Note: assertEqual(bool arr, bool arr) fails with:
        # ValueError: The truth value of an array with more than one element is ambiguous
        try:
            self.assertTrue(numpy.all(orArr == desOrArr))
        except Exception as e:
            print("Failed: %r != %r: %s" % (orArr, desOrArr, e))
            raise

    def testSmallSrc(self):
        """Verify that a source image that is too small will not raise an exception

        This tests another bug that was fixed in ticket #2441
        """
        fromWcs = makeWcs(
            pixelScale=afwGeom.Angle(1.0e-8, afwGeom.degrees),
            projection="TAN",
            crPixPos=(0, 0),
            crValCoord=afwCoord.IcrsCoord(afwGeom.Point2D(359, 0), afwGeom.degrees),
        )
        fromExp = afwImage.ExposureF(afwImage.MaskedImageF(1, 1), fromWcs)

        toWcs = makeWcs(
            pixelScale=afwGeom.Angle(1.1e-8, afwGeom.degrees),
            projection="TAN",
            crPixPos=(0, 0),
            crValCoord=afwCoord.IcrsCoord(afwGeom.Point2D(358, 0), afwGeom.degrees),
        )
        toExp = afwImage.ExposureF(afwImage.MaskedImageF(10, 10), toWcs)

        warpControl = afwMath.WarpingControl("lanczos3")
        # if a bug described in ticket #2441 is present, this will raise an exception:
        numGoodPix = afwMath.warpExposure(toExp, fromExp, warpControl)
        self.assertEqual(numGoodPix, 0)
        imArr, maskArr, varArr = toExp.getMaskedImage().getArrays()
        self.assertTrue(numpy.all(numpy.isnan(imArr)))
        self.assertTrue(numpy.all(numpy.isinf(varArr)))
        noDataBitMask = afwImage.MaskU.getPlaneBitMask("NO_DATA")
        self.assertTrue(numpy.all(maskArr == noDataBitMask))

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
            pixelScale=afwGeom.Angle(0.2, afwGeom.degrees),
            crPixPos=(10.0, 11.0),
            crValCoord=afwCoord.IcrsCoord(afwGeom.Point2D(41.7, 32.9), afwGeom.degrees),
        )
        destWcs = makeWcs(
            pixelScale=afwGeom.Angle(0.17, afwGeom.degrees),
            crPixPos=(9.0, 10.0),
            crValCoord=afwCoord.IcrsCoord(afwGeom.Point2D(41.65, 32.95), afwGeom.degrees),
            posAng=afwGeom.Angle(31, afwGeom.degrees),
        )

        srcMaskedImage = afwImage.MaskedImageF(100, 101)
        srcExposure = afwImage.ExposureF(srcMaskedImage, srcWcs)

        srcArrays = srcMaskedImage.getArrays()
        shape = srcArrays[0].shape
        srcArrays[0][:] = numpy.random.normal(10000, 1000, size=shape)
        srcArrays[2][:] = numpy.random.normal(9000, 900, size=shape)
        srcArrays[1][:] = numpy.reshape(numpy.arange(0, shape[0] * shape[1], 1, dtype=numpy.uint16), shape)

        warpControl = afwMath.WarpingControl(
            kernelName,
            maskKernelName,
            cacheSize,
            interpLength,
            afwGpu.DEFAULT_DEVICE_PREFERENCE,
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
            if numpy.all(narrowArrays[1] == broadArrays[1]):
                self.fail("No difference between broad and narrow mask")

        predMask = (broadArrays[1] & growFullMask) | (narrowArrays[1] & ~growFullMask).astype(numpy.uint16)
        predArraySet = (broadArrays[0], predMask, broadArrays[2])
        predExposure = afwImage.makeMaskedImageFromArrays(*predArraySet)

        msg = "Separate mask warping failed; warpingKernel=%s; maskWarpingKernel=%s" % \
            (kernelName, maskKernelName)
        self.assertMaskedImagesNearlyEqual(destExposure.getMaskedImage(), predExposure,
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
            else call warpImage to warp an ImageF and also call the XYTransform version
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
            "",  # there is no point to a separate mask kernel since we aren't testing the mask plane
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
            afwWarpedImagePath = "afwWarpedExposure1%s.fits" % (kernelName,)

            afwWarpedMaskedImage = afwImage.MaskedImageF(swarpedImage.getDimensions())
            afwWarpedExposure = afwImage.ExposureF(afwWarpedMaskedImage, warpedWcs)
            afwMath.warpExposure(afwWarpedExposure, originalExposure, warpingControl)
            afwWarpedMask = afwWarpedMaskedImage.getMask()
            if SAVE_FITS_FILES:
                afwWarpedExposure.writeFits(afwWarpedImagePath)
            if display:
                ds9.mtv(afwWarpedExposure, frame=1, title="Warped")

            swarpedMaskedImage = afwImage.MaskedImageF(swarpedImage)

            if display:
                ds9.mtv(swarpedMaskedImage, frame=2, title="SWarped")

            msg = "afw and swarp %s-warped differ (ignoring bad pixels)" % (kernelName,)
            try:
                self.assertMaskedImagesNearlyEqual(afwWarpedMaskedImage, swarpedMaskedImage,
                                                   doImage=True, doMask=False, doVariance=False,
                                                   skipMask=afwWarpedMask, rtol=rtol, atol=atol, msg=msg)
            except Exception:
                if SAVE_FAILED_FITS_FILES:
                    afwWarpedExposure.writeFits(afwWarpedImagePath)
                    print("Saved failed afw-warped exposure as: %s" % (afwWarpedImagePath,))
                raise
        else:
            # path for saved afw-warped image
            afwWarpedImagePath = "afwWarpedImage1%s.fits" % (kernelName,)
            afwWarpedImage2Path = "afwWarpedImage1%s_xyTransform.fits" % (kernelName,)

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
            noDataMaskArr = numpy.isnan(afwWarpedImageArr)
            msg = "afw and swarp %s-warped images do not match (ignoring NaN pixels)" % \
                (kernelName,)
            try:
                self.assertImagesNearlyEqual(afwWarpedImage, swarpedImage,
                                             skipMask=noDataMaskArr, rtol=rtol, atol=atol, msg=msg)
            except Exception:
                if SAVE_FAILED_FITS_FILES:
                    # save the image anyway
                    afwWarpedImage.writeFits(afwWarpedImagePath)
                    print("Saved failed afw-warped image as: %s" % (afwWarpedImagePath,))
                raise

            afwWarpedImage2 = afwImage.ImageF(swarpedImage.getDimensions())
            xyTransform = afwImage.XYTransformFromWcsPair(warpedWcs, originalWcs)
            afwMath.warpImage(afwWarpedImage2, originalImage, xyTransform, warpingControl)
            msg = "afw xyTransform-based and WCS-based %s-warped images do not match" % (kernelName,)
            try:
                self.assertImagesNearlyEqual(afwWarpedImage2, afwWarpedImage,
                                             rtol=rtol, atol=atol, msg=msg)
            except Exception:
                if SAVE_FAILED_FITS_FILES:
                    # save the image anyway
                    afwWarpedImage.writeFits(afwWarpedImagePath)
                    print("Saved failed afw-warped image as: %s" % (afwWarpedImage2Path,))
                raise


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

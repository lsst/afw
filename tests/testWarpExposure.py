#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11#"""Test warpExposure
#pybind11#"""
#pybind11#import math
#pybind11#import os
#pybind11#import unittest
#pybind11#
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.utils.tests
#pybind11#import lsst.daf.base as dafBase
#pybind11#import lsst.afw.coord as afwCoord
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.gpu as afwGpu
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.image.utils as imageUtils
#pybind11#import lsst.pex.policy as pexPolicy
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#from lsst.log import Log
#pybind11#
#pybind11## Change the level to Log.DEBUG to see debug messages
#pybind11#Log.getLogger("afw.image.Mask").setLevel(Log.INFO)
#pybind11#Log.getLogger("TRACE2.afw.math.warp").setLevel(Log.INFO)
#pybind11#Log.getLogger("TRACE3.afw.math.warp").setLevel(Log.INFO)
#pybind11#
#pybind11#
#pybind11#try:
#pybind11#    display
#pybind11#except:
#pybind11#    display = False
#pybind11#    # set True to save afw-warped images as FITS files
#pybind11#    SAVE_FITS_FILES = False
#pybind11#    # set True to save failed afw-warped images as FITS files even if SAVE_FITS_FILES is False
#pybind11#    #SAVE_FAILED_FITS_FILES = False
#pybind11#    SAVE_FAILED_FITS_FILES = True
#pybind11#
#pybind11#try:
#pybind11#    afwdataDir = lsst.utils.getPackageDir("afwdata")
#pybind11#except pexExcept.NotFoundError:
#pybind11#    afwdataDir = None
#pybind11#else:
#pybind11#    dataDir = os.path.join(afwdataDir, "data")
#pybind11#
#pybind11#    originalExposureName = "medexp.fits"
#pybind11#    originalExposurePath = os.path.join(dataDir, originalExposureName)
#pybind11#    subExposureName = "medsub.fits"
#pybind11#    subExposurePath = os.path.join(dataDir, originalExposureName)
#pybind11#    originalFullExposureName = os.path.join("CFHT", "D4", "cal-53535-i-797722_1.fits")
#pybind11#    originalFullExposurePath = os.path.join(dataDir, originalFullExposureName)
#pybind11#
#pybind11#
#pybind11#def makeWcs(pixelScale, crPixPos, crValCoord, posAng=afwGeom.Angle(0.0), doFlipX=False, projection="TAN",
#pybind11#            radDecCSys="ICRS", equinox=2000):
#pybind11#    """Make a Wcs
#pybind11#
#pybind11#    @param[in] pixelScale: desired scale, as sky/pixel, an afwGeom.Angle
#pybind11#    @param[in] crPixPos: crPix for WCS, using the LSST standard; a pair of floats
#pybind11#    @param[in] crValCoord: crVal for WCS (afwCoord.Coord)
#pybind11#    @param[in] posAng: position angle (afwGeom.Angle)
#pybind11#    @param[in] doFlipX: flip X axis?
#pybind11#    @param[in] projection: WCS projection (e.g. "TAN" or "STG")
#pybind11#    """
#pybind11#    if len(projection) != 3:
#pybind11#        raise RuntimeError("projection=%r; must have length 3" % (projection,))
#pybind11#    ctypeList = [("%-5s%3s" % (("RA", "DEC")[i], projection)).replace(" ", "-")
#pybind11#                 for i in range(2)]
#pybind11#    ps = dafBase.PropertySet()
#pybind11#    crPixFits = [ind + 1.0 for ind in crPixPos]  # convert pix position to FITS standard
#pybind11#    crValDeg = crValCoord.getPosition(afwGeom.degrees)
#pybind11#    posAngRad = posAng.asRadians()
#pybind11#    pixelScaleDeg = pixelScale.asDegrees()
#pybind11#    cdMat = numpy.array([[math.cos(posAngRad), math.sin(posAngRad)],
#pybind11#                         [-math.sin(posAngRad), math.cos(posAngRad)]], dtype=float) * pixelScaleDeg
#pybind11#    if doFlipX:
#pybind11#        cdMat[:, 0] = -cdMat[:, 0]
#pybind11#    for i in range(2):
#pybind11#        ip1 = i + 1
#pybind11#        ps.add("CTYPE%1d" % (ip1,), ctypeList[i])
#pybind11#        ps.add("CRPIX%1d" % (ip1,), crPixFits[i])
#pybind11#        ps.add("CRVAL%1d" % (ip1,), crValDeg[i])
#pybind11#    ps.add("RADECSYS", radDecCSys)
#pybind11#    ps.add("EQUINOX", equinox)
#pybind11#    ps.add("CD1_1", cdMat[0, 0])
#pybind11#    ps.add("CD2_1", cdMat[1, 0])
#pybind11#    ps.add("CD1_2", cdMat[0, 1])
#pybind11#    ps.add("CD2_2", cdMat[1, 1])
#pybind11#    return afwImage.makeWcs(ps)
#pybind11#
#pybind11#
#pybind11#class WarpExposureTestCase(lsst.utils.tests.TestCase):
#pybind11#    """Test case for warpExposure
#pybind11#    """
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        numpy.random.seed(0)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testNullWarpExposure(self, interpLength=10):
#pybind11#        """Test that warpExposure maps an image onto itself.
#pybind11#
#pybind11#        Note:
#pybind11#        - NO_DATA and off-CCD pixels must be ignored
#pybind11#        - bad mask pixels get smeared out so we have to excluded all bad mask pixels
#pybind11#          from the output image when comparing masks.
#pybind11#        """
#pybind11#        filterPolicyFile = pexPolicy.DefaultPolicyFile("afw", "SdssFilters.paf", "tests")
#pybind11#        filterPolicy = pexPolicy.Policy.createPolicy(
#pybind11#            filterPolicyFile, filterPolicyFile.getRepositoryPath(), True)
#pybind11#        imageUtils.defineFiltersFromPolicy(filterPolicy, reset=True)
#pybind11#
#pybind11#        originalExposure = afwImage.ExposureF(originalExposurePath)
#pybind11#        originalFilter = afwImage.Filter("i")
#pybind11#        originalCalib = afwImage.Calib()
#pybind11#        originalCalib.setFluxMag0(1.0e5, 1.0e3)
#pybind11#        originalExposure.setFilter(originalFilter)
#pybind11#        originalExposure.setCalib(originalCalib)
#pybind11#        afwWarpedExposure = afwImage.ExposureF(
#pybind11#            originalExposure.getBBox(),
#pybind11#            originalExposure.getWcs())
#pybind11#        warpingControl = afwMath.WarpingControl("lanczos4", "", 0, interpLength)
#pybind11#        afwMath.warpExposure(afwWarpedExposure, originalExposure, warpingControl)
#pybind11#        if SAVE_FITS_FILES:
#pybind11#            afwWarpedExposure.writeFits("afwWarpedExposureNull.fits")
#pybind11#
#pybind11#        self.assertEqual(afwWarpedExposure.getFilter().getName(), originalFilter.getName())
#pybind11#        self.assertEqual(afwWarpedExposure.getCalib().getFluxMag0(), originalCalib.getFluxMag0())
#pybind11#
#pybind11#        afwWarpedMaskedImage = afwWarpedExposure.getMaskedImage()
#pybind11#        afwWarpedMask = afwWarpedMaskedImage.getMask()
#pybind11#        noDataBitMask = afwWarpedMask.getPlaneBitMask("NO_DATA")
#pybind11#        afwWarpedMaskedImageArrSet = afwWarpedMaskedImage.getArrays()
#pybind11#        afwWarpedMaskArr = afwWarpedMaskedImageArrSet[1]
#pybind11#
#pybind11#        # compare all non-DATA pixels of image and variance, but relax specs a bit
#pybind11#        # because of minor noise introduced by bad pixels
#pybind11#        noDataMaskArr = afwWarpedMaskArr & noDataBitMask
#pybind11#        msg = "afw null-warped MaskedImage (all pixels, relaxed tolerance)"
#pybind11#        self.assertMaskedImagesNearlyEqual(afwWarpedMaskedImage, originalExposure.getMaskedImage(),
#pybind11#                                           doMask=False, skipMask=noDataMaskArr, atol=1e-5, msg=msg)
#pybind11#
#pybind11#        # compare good pixels (mask=0) of image, mask and variance using full tolerance
#pybind11#        msg = "afw null-warped MaskedImage (good pixels, max tolerance)"
#pybind11#        self.assertMaskedImagesNearlyEqual(afwWarpedMaskedImage, originalExposure.getMaskedImage(),
#pybind11#                                           skipMask=afwWarpedMask, msg=msg)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testNullWarpImage(self, interpLength=10):
#pybind11#        """Test that warpImage maps an image onto itself.
#pybind11#        """
#pybind11#        originalExposure = afwImage.ExposureF(originalExposurePath)
#pybind11#        afwWarpedExposure = afwImage.ExposureF(originalExposurePath)
#pybind11#        originalImage = originalExposure.getMaskedImage().getImage()
#pybind11#        afwWarpedImage = afwWarpedExposure.getMaskedImage().getImage()
#pybind11#        originalWcs = originalExposure.getWcs()
#pybind11#        afwWarpedWcs = afwWarpedExposure.getWcs()
#pybind11#        warpingControl = afwMath.WarpingControl("lanczos4", "", 0, interpLength)
#pybind11#        afwMath.warpImage(afwWarpedImage, afwWarpedWcs, originalImage, originalWcs, warpingControl)
#pybind11#        if SAVE_FITS_FILES:
#pybind11#            afwWarpedImage.writeFits("afwWarpedImageNull.fits")
#pybind11#        afwWarpedImageArr = afwWarpedImage.getArray()
#pybind11#        noDataMaskArr = numpy.isnan(afwWarpedImageArr)
#pybind11#        # relax specs a bit because of minor noise introduced by bad pixels
#pybind11#        msg = "afw null-warped Image"
#pybind11#        self.assertImagesNearlyEqual(originalImage, afwWarpedImage, skipMask=noDataMaskArr,
#pybind11#                                     atol=1e-5, msg=msg)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testNullWcs(self, interpLength=10):
#pybind11#        """Cannot warp from or into an exposure without a Wcs.
#pybind11#        """
#pybind11#        exposureWithWcs = afwImage.ExposureF(originalExposurePath)
#pybind11#        mi = exposureWithWcs.getMaskedImage()
#pybind11#        exposureWithoutWcs = afwImage.ExposureF(mi.getDimensions())
#pybind11#        warpingControl = afwMath.WarpingControl("bilinear", "", 0, interpLength)
#pybind11#        try:
#pybind11#            afwMath.warpExposure(exposureWithWcs, exposureWithoutWcs, warpingControl)
#pybind11#            self.fail("warping from a source Exception with no Wcs should fail")
#pybind11#        except Exception:
#pybind11#            pass
#pybind11#        try:
#pybind11#            afwMath.warpExposure(exposureWithoutWcs, exposureWithWcs, warpingControl)
#pybind11#            self.fail("warping into a destination Exception with no Wcs should fail")
#pybind11#        except Exception:
#pybind11#            pass
#pybind11#
#pybind11#    def testWarpIntoSelf(self, interpLength=10):
#pybind11#        """Cannot warp in-place
#pybind11#        """
#pybind11#        originalExposure = afwImage.ExposureF(afwGeom.Extent2I(100, 100))
#pybind11#        warpingControl = afwMath.WarpingControl("bilinear", "", 0, interpLength)
#pybind11#        try:
#pybind11#            afwMath.warpExposure(originalExposure, originalExposure, warpingControl)
#pybind11#            self.fail("warpExposure in place (dest is src) should fail")
#pybind11#        except Exception:
#pybind11#            pass
#pybind11#        try:
#pybind11#            afwMath.warpImage(originalExposure.getMaskedImage(), originalExposure.getWcs(),
#pybind11#                              originalExposure.getMaskedImage(), originalExposure.getWcs(), warpingControl)
#pybind11#            self.fail("warpImage<MaskedImage> in place (dest is src) should fail")
#pybind11#        except Exception:
#pybind11#            pass
#pybind11#        try:
#pybind11#            afwMath.warpImage(originalExposure.getImage(), originalExposure.getWcs(),
#pybind11#                              originalExposure.getImage(), originalExposure.getWcs(), warpingControl)
#pybind11#            self.fail("warpImage<Image> in place (dest is src) should fail")
#pybind11#        except Exception:
#pybind11#            pass
#pybind11#
#pybind11#    def testWarpingControl(self):
#pybind11#        """Test the basic mechanics of WarpingControl
#pybind11#        """
#pybind11#        for interpLength in (0, 1, 52):
#pybind11#            wc = afwMath.WarpingControl("lanczos3", "", 0, interpLength)
#pybind11#            self.assertFalse(wc.hasMaskWarpingKernel())
#pybind11#            self.assertEqual(wc.getInterpLength(), interpLength)
#pybind11#            for newInterpLength in (3, 7, 9):
#pybind11#                wc.setInterpLength(newInterpLength)
#pybind11#                self.assertEqual(wc.getInterpLength(), newInterpLength)
#pybind11#
#pybind11#        for cacheSize in (0, 100):
#pybind11#            wc = afwMath.WarpingControl("lanczos3", "bilinear", cacheSize)
#pybind11#            self.assertTrue(wc.hasMaskWarpingKernel())
#pybind11#            self.assertEqual(wc.getCacheSize(), cacheSize)
#pybind11#            self.assertEqual(wc.getWarpingKernel().getCacheSize(), cacheSize)
#pybind11#            self.assertEqual(wc.getMaskWarpingKernel().getCacheSize(), cacheSize)
#pybind11#            for newCacheSize in (1, 50):
#pybind11#                wc.setCacheSize(newCacheSize)
#pybind11#                self.assertEqual(wc.getCacheSize(), newCacheSize)
#pybind11#                self.assertEqual(wc.getWarpingKernel().getCacheSize(), newCacheSize)
#pybind11#                self.assertEqual(wc.getMaskWarpingKernel().getCacheSize(), newCacheSize)
#pybind11#
#pybind11#    def testWarpingControlError(self):
#pybind11#        """Test error handling of WarpingControl
#pybind11#        """
#pybind11#        # error: mask kernel smaller than main kernel
#pybind11#        for kernelName, maskKernelName in (
#pybind11#            ("bilinear", "lanczos3"),
#pybind11#            ("bilinear", "lanczos4"),
#pybind11#            ("lanczos3", "lanczos4"),
#pybind11#        ):
#pybind11#            with self.assertRaises(pexExcept.Exception):
#pybind11#                afwMath.WarpingControl(kernelName, maskKernelName)
#pybind11#
#pybind11#        # error: new mask kernel larger than main kernel
#pybind11#        warpingControl = afwMath.WarpingControl("bilinear")
#pybind11#        for maskKernelName in ("lanczos3", "lanczos4"):
#pybind11#            with self.assertRaises(pexExcept.Exception):
#pybind11#                warpingControl.setMaskWarpingKernelName(maskKernelName)
#pybind11#
#pybind11#        # error: new kernel smaller than mask kernel
#pybind11#        warpingControl = afwMath.WarpingControl("lanczos4", "lanczos4")
#pybind11#        for kernelName in ("bilinear", "lanczos3"):
#pybind11#            with self.assertRaises(pexExcept.Exception):
#pybind11#                warpingControl.setWarpingKernelName(kernelName)
#pybind11#
#pybind11#        # error: GPU only works with Lanczos kernels
#pybind11#        with self.assertRaises(pexExcept.Exception):
#pybind11#            afwMath.WarpingControl("bilinear", "", 0, 0, afwGpu.USE_GPU)
#pybind11#        warpingControl = afwMath.WarpingControl("bilinear")
#pybind11#        with self.assertRaises(pexExcept.Exception):
#pybind11#            warpingControl.setDevicePreference(afwGpu.USE_GPU)
#pybind11#
#pybind11#        # OK: GPU works with Lanczos kernels
#pybind11#        for kernelName in ("lanczos3", "lanczos4"):
#pybind11#            afwMath.WarpingControl(kernelName, "", 0, 0, afwGpu.USE_GPU)
#pybind11#            warpingControl = afwMath.WarpingControl(kernelName)
#pybind11#            warpingControl.setDevicePreference(afwGpu.USE_GPU)
#pybind11#
#pybind11#        # OK: main kernel at least as big as mask kernel
#pybind11#        for kernelName, maskKernelName in (
#pybind11#            ("bilinear", "bilinear"),
#pybind11#            ("lanczos3", "lanczos3"),
#pybind11#            ("lanczos3", "bilinear"),
#pybind11#            ("lanczos4", "lanczos3"),
#pybind11#        ):
#pybind11#            # this should not raise any exception
#pybind11#            afwMath.WarpingControl(kernelName, maskKernelName)
#pybind11#
#pybind11#        # invalid kernel names
#pybind11#        for kernelName, maskKernelName in (
#pybind11#            ("badname", ""),
#pybind11#            ("lanczos", ""),  # no digit after lanczos
#pybind11#            ("lanczos3", "badname"),
#pybind11#            ("lanczos3", "lanczos"),
#pybind11#        ):
#pybind11#            with self.assertRaises(pexExcept.Exception):
#pybind11#                afwMath.WarpingControl(kernelName, maskKernelName)
#pybind11#
#pybind11#    def testWarpMask(self):
#pybind11#        """Test that warping the mask plane with a different kernel does the right thing
#pybind11#        """
#pybind11#        for kernelName, maskKernelName in (
#pybind11#            ("bilinear", "bilinear"),
#pybind11#            ("lanczos3", "lanczos3"),
#pybind11#            ("lanczos3", "bilinear"),
#pybind11#            ("lanczos4", "lanczos3"),
#pybind11#        ):
#pybind11#            for growFullMask in (0, 1, 3, 0xFFFF):
#pybind11#                self.verifyMaskWarp(
#pybind11#                    kernelName=kernelName,
#pybind11#                    maskKernelName=maskKernelName,
#pybind11#                    growFullMask=growFullMask,
#pybind11#                )
#pybind11#
#pybind11#    def testMatchSwarpBilinearImage(self):
#pybind11#        """Test that warpExposure matches swarp using a bilinear warping kernel
#pybind11#        """
#pybind11#        self.compareToSwarp("bilinear", useWarpExposure=False, atol=0.15)
#pybind11#
#pybind11#    def testMatchSwarpBilinearExposure(self):
#pybind11#        """Test that warpExposure matches swarp using a bilinear warping kernel
#pybind11#        """
#pybind11#        self.compareToSwarp("bilinear", useWarpExposure=True, useSubregion=False, useDeepCopy=True)
#pybind11#
#pybind11#    def testMatchSwarpLanczos2Image(self):
#pybind11#        """Test that warpExposure matches swarp using a lanczos2 warping kernel
#pybind11#        """
#pybind11#        self.compareToSwarp("lanczos2", useWarpExposure=False)
#pybind11#
#pybind11#    def testMatchSwarpLanczos2Exposure(self):
#pybind11#        """Test that warpExposure matches swarp using a lanczos2 warping kernel.
#pybind11#        """
#pybind11#        self.compareToSwarp("lanczos2", useWarpExposure=True)
#pybind11#
#pybind11#    def testMatchSwarpLanczos2SubExposure(self):
#pybind11#        """Test that warpExposure matches swarp using a lanczos2 warping kernel with a subexposure
#pybind11#        """
#pybind11#        for useDeepCopy in (False, True):
#pybind11#            self.compareToSwarp("lanczos2", useWarpExposure=True, useSubregion=True, useDeepCopy=useDeepCopy)
#pybind11#
#pybind11#    def testMatchSwarpLanczos3Image(self):
#pybind11#        """Test that warpExposure matches swarp using a lanczos2 warping kernel
#pybind11#        """
#pybind11#        self.compareToSwarp("lanczos3", useWarpExposure=False)
#pybind11#
#pybind11#    def testMatchSwarpLanczos3(self):
#pybind11#        """Test that warpExposure matches swarp using a lanczos4 warping kernel.
#pybind11#        """
#pybind11#        self.compareToSwarp("lanczos3", useWarpExposure=True)
#pybind11#
#pybind11#    def testMatchSwarpLanczos4Image(self):
#pybind11#        """Test that warpExposure matches swarp using a lanczos2 warping kernel
#pybind11#        """
#pybind11#        self.compareToSwarp("lanczos4", useWarpExposure=False)
#pybind11#
#pybind11#    def testMatchSwarpLanczos4(self):
#pybind11#        """Test that warpExposure matches swarp using a lanczos4 warping kernel.
#pybind11#        """
#pybind11#        self.compareToSwarp("lanczos4", useWarpExposure=True)
#pybind11#
#pybind11#    def testMatchSwarpNearestExposure(self):
#pybind11#        """Test that warpExposure matches swarp using a nearest neighbor warping kernel
#pybind11#        """
#pybind11#        self.compareToSwarp("nearest", useWarpExposure=True, atol=60)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testNonIcrs(self):
#pybind11#        """Test that warping to a non-ICRS-like coordinate system produces different results
#pybind11#
#pybind11#        It would be better to also test that the results are as expected,
#pybind11#        but I have not been able to get swarp to perform this operation,
#pybind11#        so have not found an independent means of generating the expected results.
#pybind11#        """
#pybind11#        kernelName = "lanczos3"
#pybind11#        rtol = 4e-5
#pybind11#        atol = 1e-2
#pybind11#        warpingControl = afwMath.WarpingControl(
#pybind11#            kernelName,
#pybind11#        )
#pybind11#
#pybind11#        originalExposure = afwImage.ExposureF(originalExposurePath)
#pybind11#        originalImage = originalExposure.getMaskedImage().getImage()
#pybind11#        originalWcs = originalExposure.getWcs()
#pybind11#
#pybind11#        swarpedImageName = "medswarp1%s.fits" % (kernelName,)
#pybind11#        swarpedImagePath = os.path.join(dataDir, swarpedImageName)
#pybind11#        swarpedDecoratedImage = afwImage.DecoratedImageF(swarpedImagePath)
#pybind11#        swarpedImage = swarpedDecoratedImage.getImage()
#pybind11#
#pybind11#        for changeEquinox in (False, True):
#pybind11#            swarpedMetadata = swarpedDecoratedImage.getMetadata()
#pybind11#            if changeEquinox:
#pybind11#                swarpedMetadata.set("RADECSYS", "FK5")
#pybind11#                swarpedMetadata.set("EQUINOX", swarpedMetadata.get("EQUINOX") + 1)
#pybind11#            warpedWcs = afwImage.makeWcs(swarpedMetadata)
#pybind11#
#pybind11#            afwWarpedImage = afwImage.ImageF(swarpedImage.getDimensions())
#pybind11#            originalImage = originalExposure.getMaskedImage().getImage()
#pybind11#            originalWcs = originalExposure.getWcs()
#pybind11#            numGoodPix = afwMath.warpImage(afwWarpedImage, warpedWcs, originalImage,
#pybind11#                                           originalWcs, warpingControl)
#pybind11#            self.assertGreater(numGoodPix, 50)
#pybind11#
#pybind11#            afwWarpedImageArr = afwWarpedImage.getArray()
#pybind11#            noDataMaskArr = numpy.isnan(afwWarpedImageArr)
#pybind11#            if changeEquinox:
#pybind11#                with self.assertRaises(AssertionError):
#pybind11#                    self.assertImagesNearlyEqual(afwWarpedImage, swarpedImage,
#pybind11#                                                 skipMask=noDataMaskArr, rtol=rtol, atol=atol)
#pybind11#            else:
#pybind11#                self.assertImagesNearlyEqual(afwWarpedImage, swarpedImage,
#pybind11#                                             skipMask=noDataMaskArr, rtol=rtol, atol=atol)
#pybind11#
#pybind11#    def testTicket2441(self):
#pybind11#        """Test ticket 2441: warpExposure sometimes mishandles zero-extent dest exposures"""
#pybind11#        fromWcs = makeWcs(
#pybind11#            pixelScale=afwGeom.Angle(1.0e-8, afwGeom.degrees),
#pybind11#            projection="TAN",
#pybind11#            crPixPos=(0, 0),
#pybind11#            crValCoord=afwCoord.IcrsCoord(afwGeom.Point2D(359, 0), afwGeom.degrees),
#pybind11#        )
#pybind11#        fromExp = afwImage.ExposureF(afwImage.MaskedImageF(10, 10), fromWcs)
#pybind11#
#pybind11#        toWcs = makeWcs(
#pybind11#            pixelScale=afwGeom.Angle(0.00011, afwGeom.degrees),
#pybind11#            projection="CEA",
#pybind11#            crPixPos=(410000.0, 11441.0),
#pybind11#            crValCoord=afwCoord.IcrsCoord(afwGeom.Point2D(45, 0), afwGeom.degrees),
#pybind11#            doFlipX=True,
#pybind11#        )
#pybind11#        toExp = afwImage.ExposureF(afwImage.MaskedImageF(0, 0), toWcs)
#pybind11#
#pybind11#        warpControl = afwMath.WarpingControl("lanczos3")
#pybind11#        # if a bug described in ticket #2441 is present, this will raise an exception:
#pybind11#        numGoodPix = afwMath.warpExposure(toExp, fromExp, warpControl)
#pybind11#        self.assertEqual(numGoodPix, 0)
#pybind11#
#pybind11#    def testTicketDM4063(self):
#pybind11#        """Test that a uint16 array can be cast to a bool array, to avoid DM-4063
#pybind11#        """
#pybind11#        a = numpy.array([0, 1, 0, 23], dtype=numpy.uint16)
#pybind11#        b = numpy.array([True, True, False, False], dtype=bool)
#pybind11#        acast = numpy.array(a != 0, dtype=bool)
#pybind11#        orArr = acast | b
#pybind11#        desOrArr = numpy.array([True, True, False, True], dtype=bool)
#pybind11#        # Note: assertEqual(bool arr, bool arr) fails with:
#pybind11#        # ValueError: The truth value of an array with more than one element is ambiguous
#pybind11#        try:
#pybind11#            self.assertTrue(numpy.all(orArr == desOrArr))
#pybind11#        except Exception as e:
#pybind11#            print("Failed: %r != %r: %s" % (orArr, desOrArr, e))
#pybind11#            raise
#pybind11#
#pybind11#    def testSmallSrc(self):
#pybind11#        """Verify that a source image that is too small will not raise an exception
#pybind11#
#pybind11#        This tests another bug that was fixed in ticket #2441
#pybind11#        """
#pybind11#        fromWcs = makeWcs(
#pybind11#            pixelScale=afwGeom.Angle(1.0e-8, afwGeom.degrees),
#pybind11#            projection="TAN",
#pybind11#            crPixPos=(0, 0),
#pybind11#            crValCoord=afwCoord.IcrsCoord(afwGeom.Point2D(359, 0), afwGeom.degrees),
#pybind11#        )
#pybind11#        fromExp = afwImage.ExposureF(afwImage.MaskedImageF(1, 1), fromWcs)
#pybind11#
#pybind11#        toWcs = makeWcs(
#pybind11#            pixelScale=afwGeom.Angle(1.1e-8, afwGeom.degrees),
#pybind11#            projection="TAN",
#pybind11#            crPixPos=(0, 0),
#pybind11#            crValCoord=afwCoord.IcrsCoord(afwGeom.Point2D(358, 0), afwGeom.degrees),
#pybind11#        )
#pybind11#        toExp = afwImage.ExposureF(afwImage.MaskedImageF(10, 10), toWcs)
#pybind11#
#pybind11#        warpControl = afwMath.WarpingControl("lanczos3")
#pybind11#        # if a bug described in ticket #2441 is present, this will raise an exception:
#pybind11#        numGoodPix = afwMath.warpExposure(toExp, fromExp, warpControl)
#pybind11#        self.assertEqual(numGoodPix, 0)
#pybind11#        imArr, maskArr, varArr = toExp.getMaskedImage().getArrays()
#pybind11#        self.assertTrue(numpy.all(numpy.isnan(imArr)))
#pybind11#        self.assertTrue(numpy.all(numpy.isinf(varArr)))
#pybind11#        noDataBitMask = afwImage.MaskU.getPlaneBitMask("NO_DATA")
#pybind11#        self.assertTrue(numpy.all(maskArr == noDataBitMask))
#pybind11#
#pybind11#    def verifyMaskWarp(self, kernelName, maskKernelName, growFullMask, interpLength=10, cacheSize=100000,
#pybind11#                       rtol=4e-05, atol=1e-2):
#pybind11#        """Verify that using a separate mask warping kernel produces the correct results
#pybind11#
#pybind11#        Inputs:
#pybind11#        - kernelName: name of warping kernel in the form used by afwImage.makeKernel
#pybind11#        - maskKernelName: name of mask warping kernel in the form used by afwImage.makeKernel
#pybind11#        - interpLength: interpLength argument for lsst.afw.math.WarpingControl
#pybind11#        - cacheSize: cacheSize argument for lsst.afw.math.WarpingControl;
#pybind11#            0 disables the cache
#pybind11#            10000 gives some speed improvement but less accurate results (atol must be increased)
#pybind11#            100000 gives better accuracy but no speed improvement in this test
#pybind11#        - rtol: relative tolerance as used by numpy.allclose
#pybind11#        - atol: absolute tolerance as used by numpy.allclose
#pybind11#        """
#pybind11#        srcWcs = makeWcs(
#pybind11#            pixelScale=afwGeom.Angle(0.2, afwGeom.degrees),
#pybind11#            crPixPos=(10.0, 11.0),
#pybind11#            crValCoord=afwCoord.IcrsCoord(afwGeom.Point2D(41.7, 32.9), afwGeom.degrees),
#pybind11#        )
#pybind11#        destWcs = makeWcs(
#pybind11#            pixelScale=afwGeom.Angle(0.17, afwGeom.degrees),
#pybind11#            crPixPos=(9.0, 10.0),
#pybind11#            crValCoord=afwCoord.IcrsCoord(afwGeom.Point2D(41.65, 32.95), afwGeom.degrees),
#pybind11#            posAng=afwGeom.Angle(31, afwGeom.degrees),
#pybind11#        )
#pybind11#
#pybind11#        srcMaskedImage = afwImage.MaskedImageF(100, 101)
#pybind11#        srcExposure = afwImage.ExposureF(srcMaskedImage, srcWcs)
#pybind11#
#pybind11#        srcArrays = srcMaskedImage.getArrays()
#pybind11#        shape = srcArrays[0].shape
#pybind11#        srcArrays[0][:] = numpy.random.normal(10000, 1000, size=shape)
#pybind11#        srcArrays[2][:] = numpy.random.normal(9000, 900, size=shape)
#pybind11#        srcArrays[1][:] = numpy.reshape(numpy.arange(0, shape[0] * shape[1], 1, dtype=numpy.uint16), shape)
#pybind11#
#pybind11#        warpControl = afwMath.WarpingControl(
#pybind11#            kernelName,
#pybind11#            maskKernelName,
#pybind11#            cacheSize,
#pybind11#            interpLength,
#pybind11#            afwGpu.DEFAULT_DEVICE_PREFERENCE,
#pybind11#            growFullMask
#pybind11#        )
#pybind11#        destMaskedImage = afwImage.MaskedImageF(110, 121)
#pybind11#        destExposure = afwImage.ExposureF(destMaskedImage, destWcs)
#pybind11#        afwMath.warpExposure(destExposure, srcExposure, warpControl)
#pybind11#
#pybind11#        # now compute with two separate mask planes
#pybind11#        warpControl.setGrowFullMask(0)
#pybind11#        narrowMaskedImage = afwImage.MaskedImageF(110, 121)
#pybind11#        narrowExposure = afwImage.ExposureF(narrowMaskedImage, destWcs)
#pybind11#        afwMath.warpExposure(narrowExposure, srcExposure, warpControl)
#pybind11#        narrowArrays = narrowExposure.getMaskedImage().getArrays()
#pybind11#
#pybind11#        warpControl.setMaskWarpingKernelName("")
#pybind11#        broadMaskedImage = afwImage.MaskedImageF(110, 121)
#pybind11#        broadExposure = afwImage.ExposureF(broadMaskedImage, destWcs)
#pybind11#        afwMath.warpExposure(broadExposure, srcExposure, warpControl)
#pybind11#        broadArrays = broadExposure.getMaskedImage().getArrays()
#pybind11#
#pybind11#        if (kernelName != maskKernelName) and (growFullMask != 0xFFFF):
#pybind11#            # we expect the mask planes to differ
#pybind11#            if numpy.all(narrowArrays[1] == broadArrays[1]):
#pybind11#                self.fail("No difference between broad and narrow mask")
#pybind11#
#pybind11#        predMask = (broadArrays[1] & growFullMask) | (narrowArrays[1] & ~growFullMask).astype(numpy.uint16)
#pybind11#        predArraySet = (broadArrays[0], predMask, broadArrays[2])
#pybind11#        predExposure = afwImage.makeMaskedImageFromArrays(*predArraySet)
#pybind11#
#pybind11#        msg = "Separate mask warping failed; warpingKernel=%s; maskWarpingKernel=%s" % \
#pybind11#            (kernelName, maskKernelName)
#pybind11#        self.assertMaskedImagesNearlyEqual(destExposure.getMaskedImage(), predExposure,
#pybind11#                                           doImage=True, doMask=True, doVariance=True,
#pybind11#                                           rtol=rtol, atol=atol, msg=msg)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def compareToSwarp(self, kernelName,
#pybind11#                       useWarpExposure=True, useSubregion=False, useDeepCopy=False,
#pybind11#                       interpLength=10, cacheSize=100000,
#pybind11#                       rtol=4e-05, atol=1e-2):
#pybind11#        """Compare warpExposure to swarp for given warping kernel.
#pybind11#
#pybind11#        Note that swarp only warps the image plane, so only test that plane.
#pybind11#
#pybind11#        Inputs:
#pybind11#        - kernelName: name of kernel in the form used by afwImage.makeKernel
#pybind11#        - useWarpExposure: if True, call warpExposure to warp an ExposureF,
#pybind11#            else call warpImage to warp an ImageF and also call the XYTransform version
#pybind11#        - useSubregion: if True then the original source exposure (from which the usual
#pybind11#            test exposure was extracted) is read and the correct subregion extracted
#pybind11#        - useDeepCopy: if True then the copy of the subimage is a deep copy,
#pybind11#            else it is a shallow copy; ignored if useSubregion is False
#pybind11#        - interpLength: interpLength argument for lsst.afw.math.WarpingControl
#pybind11#        - cacheSize: cacheSize argument for lsst.afw.math.WarpingControl;
#pybind11#            0 disables the cache
#pybind11#            10000 gives some speed improvement but less accurate results (atol must be increased)
#pybind11#            100000 gives better accuracy but no speed improvement in this test
#pybind11#        - rtol: relative tolerance as used by numpy.allclose
#pybind11#        - atol: absolute tolerance as used by numpy.allclose
#pybind11#        """
#pybind11#        warpingControl = afwMath.WarpingControl(
#pybind11#            kernelName,
#pybind11#            "",  # there is no point to a separate mask kernel since we aren't testing the mask plane
#pybind11#            cacheSize,
#pybind11#            interpLength,
#pybind11#        )
#pybind11#        if useSubregion:
#pybind11#            originalFullExposure = afwImage.ExposureF(originalExposurePath)
#pybind11#            # "medsub" is a subregion of med starting at 0-indexed pixel (40, 150) of size 145 x 200
#pybind11#            bbox = afwGeom.Box2I(afwGeom.Point2I(40, 150), afwGeom.Extent2I(145, 200))
#pybind11#            originalExposure = afwImage.ExposureF(originalFullExposure, bbox, afwImage.LOCAL, useDeepCopy)
#pybind11#            swarpedImageName = "medsubswarp1%s.fits" % (kernelName,)
#pybind11#        else:
#pybind11#            originalExposure = afwImage.ExposureF(originalExposurePath)
#pybind11#            swarpedImageName = "medswarp1%s.fits" % (kernelName,)
#pybind11#
#pybind11#        swarpedImagePath = os.path.join(dataDir, swarpedImageName)
#pybind11#        swarpedDecoratedImage = afwImage.DecoratedImageF(swarpedImagePath)
#pybind11#        swarpedImage = swarpedDecoratedImage.getImage()
#pybind11#        swarpedMetadata = swarpedDecoratedImage.getMetadata()
#pybind11#        warpedWcs = afwImage.makeWcs(swarpedMetadata)
#pybind11#
#pybind11#        if useWarpExposure:
#pybind11#            # path for saved afw-warped image
#pybind11#            afwWarpedImagePath = "afwWarpedExposure1%s.fits" % (kernelName,)
#pybind11#
#pybind11#            afwWarpedMaskedImage = afwImage.MaskedImageF(swarpedImage.getDimensions())
#pybind11#            afwWarpedExposure = afwImage.ExposureF(afwWarpedMaskedImage, warpedWcs)
#pybind11#            afwMath.warpExposure(afwWarpedExposure, originalExposure, warpingControl)
#pybind11#            afwWarpedMask = afwWarpedMaskedImage.getMask()
#pybind11#            if SAVE_FITS_FILES:
#pybind11#                afwWarpedExposure.writeFits(afwWarpedImagePath)
#pybind11#            if display:
#pybind11#                ds9.mtv(afwWarpedExposure, frame=1, title="Warped")
#pybind11#
#pybind11#            swarpedMaskedImage = afwImage.MaskedImageF(swarpedImage)
#pybind11#
#pybind11#            if display:
#pybind11#                ds9.mtv(swarpedMaskedImage, frame=2, title="SWarped")
#pybind11#
#pybind11#            msg = "afw and swarp %s-warped differ (ignoring bad pixels)" % (kernelName,)
#pybind11#            try:
#pybind11#                self.assertMaskedImagesNearlyEqual(afwWarpedMaskedImage, swarpedMaskedImage,
#pybind11#                                                   doImage=True, doMask=False, doVariance=False, skipMask=afwWarpedMask,
#pybind11#                                                   rtol=rtol, atol=atol, msg=msg)
#pybind11#            except Exception:
#pybind11#                if SAVE_FAILED_FITS_FILES:
#pybind11#                    afwWarpedExposure.writeFits(afwWarpedImagePath)
#pybind11#                    print("Saved failed afw-warped exposure as: %s" % (afwWarpedImagePath,))
#pybind11#                raise
#pybind11#        else:
#pybind11#            # path for saved afw-warped image
#pybind11#            afwWarpedImagePath = "afwWarpedImage1%s.fits" % (kernelName,)
#pybind11#            afwWarpedImage2Path = "afwWarpedImage1%s_xyTransform.fits" % (kernelName,)
#pybind11#
#pybind11#            afwWarpedImage = afwImage.ImageF(swarpedImage.getDimensions())
#pybind11#            originalImage = originalExposure.getMaskedImage().getImage()
#pybind11#            originalWcs = originalExposure.getWcs()
#pybind11#            afwMath.warpImage(afwWarpedImage, warpedWcs, originalImage,
#pybind11#                              originalWcs, warpingControl)
#pybind11#            if display:
#pybind11#                ds9.mtv(afwWarpedImage, frame=1, title="Warped")
#pybind11#                ds9.mtv(swarpedImage, frame=2, title="SWarped")
#pybind11#                diff = swarpedImage.Factory(swarpedImage, True)
#pybind11#                diff -= afwWarpedImage
#pybind11#                ds9.mtv(diff, frame=3, title="swarp - afw")
#pybind11#            if SAVE_FITS_FILES:
#pybind11#                afwWarpedImage.writeFits(afwWarpedImagePath)
#pybind11#
#pybind11#            afwWarpedImageArr = afwWarpedImage.getArray()
#pybind11#            noDataMaskArr = numpy.isnan(afwWarpedImageArr)
#pybind11#            msg = "afw and swarp %s-warped images do not match (ignoring NaN pixels)" % \
#pybind11#                (kernelName,)
#pybind11#            try:
#pybind11#                self.assertImagesNearlyEqual(afwWarpedImage, swarpedImage,
#pybind11#                                             skipMask=noDataMaskArr, rtol=rtol, atol=atol, msg=msg)
#pybind11#            except Exception:
#pybind11#                if SAVE_FAILED_FITS_FILES:
#pybind11#                    # save the image anyway
#pybind11#                    afwWarpedImage.writeFits(afwWarpedImagePath)
#pybind11#                    print("Saved failed afw-warped image as: %s" % (afwWarpedImagePath,))
#pybind11#                raise
#pybind11#
#pybind11#            afwWarpedImage2 = afwImage.ImageF(swarpedImage.getDimensions())
#pybind11#            xyTransform = afwImage.XYTransformFromWcsPair(warpedWcs, originalWcs)
#pybind11#            afwMath.warpImage(afwWarpedImage2, originalImage, xyTransform, warpingControl)
#pybind11#            msg = "afw xyTransform-based and WCS-based %s-warped images do not match" % (kernelName,)
#pybind11#            try:
#pybind11#                self.assertImagesNearlyEqual(afwWarpedImage2, afwWarpedImage,
#pybind11#                                             rtol=rtol, atol=atol, msg=msg)
#pybind11#            except Exception:
#pybind11#                if SAVE_FAILED_FITS_FILES:
#pybind11#                    # save the image anyway
#pybind11#                    afwWarpedImage.writeFits(afwWarpedImagePath)
#pybind11#                    print("Saved failed afw-warped image as: %s" % (afwWarpedImage2Path,))
#pybind11#                raise
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

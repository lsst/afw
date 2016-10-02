#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
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
#pybind11#"""Basic test of Warp (the warping algorithm is thoroughly tested in lsst.afw.math)
#pybind11#"""
#pybind11#import os
#pybind11#import unittest
#pybind11#import warnings
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.image.utils as imageUtils
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.pex.policy as pexPolicy
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#from lsst.log import Log
#pybind11#
#pybind11## Change the level to Log.DEBUG to see debug messages
#pybind11#Log.getLogger("afw.image.Mask").setLevel(Log.INFO)
#pybind11#Log.getLogger("TRACE3.afw.math.warp").setLevel(Log.INFO)
#pybind11#Log.getLogger("TRACE4.afw.math.warp").setLevel(Log.INFO)
#pybind11#
#pybind11#try:
#pybind11#    afwdataDir = lsst.utils.getPackageDir("afwdata")
#pybind11#except pexExcept.NotFoundError:
#pybind11#    afwdataDir = None
#pybind11#    dataDir = None
#pybind11#else:
#pybind11#    dataDir = os.path.join(afwdataDir, "data")
#pybind11#    originalExposureName = "medexp.fits"
#pybind11#    originalExposurePath = os.path.join(dataDir, originalExposureName)
#pybind11#    subExposureName = "medsub.fits"
#pybind11#    subExposurePath = os.path.join(dataDir, originalExposureName)
#pybind11#    originalFullExposureName = os.path.join("CFHT", "D4", "cal-53535-i-797722_1.fits")
#pybind11#    originalFullExposurePath = os.path.join(dataDir, originalFullExposureName)
#pybind11#
#pybind11#
#pybind11#class WarpExposureTestCase(lsst.utils.tests.TestCase):
#pybind11#    """Test case for Warp
#pybind11#    """
#pybind11#
#pybind11#    def testMatchSwarpLanczos2Exposure(self):
#pybind11#        """Test that warpExposure matches swarp using a lanczos2 warping kernel.
#pybind11#        """
#pybind11#        self.compareToSwarp("lanczos2")
#pybind11#
#pybind11#    def testMatchSwarpLanczos2SubExposure(self):
#pybind11#        """Test that warpExposure matches swarp using a lanczos2 warping kernel with a subexposure
#pybind11#        """
#pybind11#        for useDeepCopy in (False, True):
#pybind11#            self.compareToSwarp("lanczos2", useSubregion=True, useDeepCopy=useDeepCopy)
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testBBox(self):
#pybind11#        """Test that the default bounding box includes all warped pixels
#pybind11#        """
#pybind11#        kernelName = "lanczos2"
#pybind11#        warper = afwMath.Warper(kernelName)
#pybind11#        originalExposure, swarpedImage, swarpedWcs = self.getSwarpedImage(
#pybind11#            kernelName=kernelName, useSubregion=True, useDeepCopy=False)
#pybind11#
#pybind11#        filterPolicyFile = pexPolicy.DefaultPolicyFile("afw", "SdssFilters.paf", "tests")
#pybind11#        filterPolicy = pexPolicy.Policy.createPolicy(
#pybind11#            filterPolicyFile, filterPolicyFile.getRepositoryPath(), True)
#pybind11#        imageUtils.defineFiltersFromPolicy(filterPolicy, reset=True)
#pybind11#
#pybind11#        originalFilter = afwImage.Filter("i")
#pybind11#        originalCalib = afwImage.Calib()
#pybind11#        originalCalib.setFluxMag0(1.0e5, 1.0e3)
#pybind11#        originalExposure.setFilter(originalFilter)
#pybind11#        originalExposure.setCalib(originalCalib)
#pybind11#
#pybind11#        warpedExposure1 = warper.warpExposure(destWcs=swarpedWcs, srcExposure=originalExposure)
#pybind11#        # the default size must include all good pixels, so growing the bbox should not add any
#pybind11#        warpedExposure2 = warper.warpExposure(destWcs=swarpedWcs, srcExposure=originalExposure, border=1)
#pybind11#        # a bit of excess border is allowed, but surely not as much as 10 (in fact it is approx. 5)
#pybind11#        warpedExposure3 = warper.warpExposure(destWcs=swarpedWcs, srcExposure=originalExposure, border=-10)
#pybind11#        # assert that warpedExposure and warpedExposure2 have the same number of non-no_data pixels
#pybind11#        # and that warpedExposure3 has fewer
#pybind11#        noDataBitMask = afwImage.MaskU.getPlaneBitMask("NO_DATA")
#pybind11#        mask1Arr = warpedExposure1.getMaskedImage().getMask().getArray()
#pybind11#        mask2Arr = warpedExposure2.getMaskedImage().getMask().getArray()
#pybind11#        mask3Arr = warpedExposure3.getMaskedImage().getMask().getArray()
#pybind11#        nGood1 = (mask1Arr & noDataBitMask == 0).sum()
#pybind11#        nGood2 = (mask2Arr & noDataBitMask == 0).sum()
#pybind11#        nGood3 = (mask3Arr & noDataBitMask == 0).sum()
#pybind11#        self.assertEqual(nGood1, nGood2)
#pybind11#        self.assertLess(nGood3, nGood1)
#pybind11#
#pybind11#        self.assertEqual(warpedExposure1.getFilter().getName(), originalFilter.getName())
#pybind11#        self.assertEqual(warpedExposure1.getCalib().getFluxMag0(), originalCalib.getFluxMag0())
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testDestBBox(self):
#pybind11#        """Test that the destBBox argument works
#pybind11#        """
#pybind11#        kernelName = "lanczos2"
#pybind11#        warper = afwMath.Warper(kernelName)
#pybind11#        originalExposure, swarpedImage, swarpedWcs = self.getSwarpedImage(
#pybind11#            kernelName=kernelName, useSubregion=True, useDeepCopy=False)
#pybind11#
#pybind11#        bbox = afwGeom.Box2I(afwGeom.Point2I(100, 25), afwGeom.Extent2I(3, 7))
#pybind11#        warpedExposure = warper.warpExposure(
#pybind11#            destWcs=swarpedWcs,
#pybind11#            srcExposure=originalExposure,
#pybind11#            destBBox=bbox,
#pybind11#            border=-2,  # should be ignored
#pybind11#            maxBBox=afwGeom.Box2I(afwGeom.Point2I(1, 2), afwGeom.Extent2I(8, 9)),  # should be ignored
#pybind11#        )
#pybind11#        self.assertEqual(bbox, warpedExposure.getBBox(afwImage.PARENT))
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def getSwarpedImage(self, kernelName, useSubregion=False, useDeepCopy=False):
#pybind11#        """
#pybind11#        Inputs:
#pybind11#        - kernelName: name of kernel in the form used by afwImage.makeKernel
#pybind11#        - useSubregion: if True then the original source exposure (from which the usual
#pybind11#            test exposure was extracted) is read and the correct subregion extracted
#pybind11#        - useDeepCopy: if True then the copy of the subimage is a deep copy,
#pybind11#            else it is a shallow copy; ignored if useSubregion is False
#pybind11#
#pybind11#        Returns:
#pybind11#        - originalExposure
#pybind11#        - swarpedImage
#pybind11#        - swarpedWcs
#pybind11#        """
#pybind11#        if useSubregion:
#pybind11#            originalFullExposure = afwImage.ExposureF(originalExposurePath)
#pybind11#            # "medsub" is a subregion of med starting at 0-indexed pixel (40, 150) of size 145 x 200
#pybind11#            bbox = afwGeom.Box2I(afwGeom.Point2I(40, 150), afwGeom.Extent2I(145, 200))
#pybind11#            originalExposure = afwImage.ExposureF(originalFullExposure, bbox, useDeepCopy)
#pybind11#            swarpedImageName = "medsubswarp1%s.fits" % (kernelName,)
#pybind11#        else:
#pybind11#            originalExposure = afwImage.ExposureF(originalExposurePath)
#pybind11#            swarpedImageName = "medswarp1%s.fits" % (kernelName,)
#pybind11#
#pybind11#        swarpedImagePath = os.path.join(dataDir, swarpedImageName)
#pybind11#        swarpedDecoratedImage = afwImage.DecoratedImageF(swarpedImagePath)
#pybind11#        swarpedImage = swarpedDecoratedImage.getImage()
#pybind11#        swarpedMetadata = swarpedDecoratedImage.getMetadata()
#pybind11#        swarpedWcs = afwImage.makeWcs(swarpedMetadata)
#pybind11#        return (originalExposure, swarpedImage, swarpedWcs)
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def compareToSwarp(self, kernelName,
#pybind11#                       useSubregion=False, useDeepCopy=False,
#pybind11#                       interpLength=10, cacheSize=100000,
#pybind11#                       rtol=4e-05, atol=1e-2):
#pybind11#        """Compare warpExposure to swarp for given warping kernel.
#pybind11#
#pybind11#        Note that swarp only warps the image plane, so only test that plane.
#pybind11#
#pybind11#        Inputs:
#pybind11#        - kernelName: name of kernel in the form used by afwImage.makeKernel
#pybind11#        - useSubregion: if True then the original source exposure (from which the usual
#pybind11#            test exposure was extracted) is read and the correct subregion extracted
#pybind11#        - useDeepCopy: if True then the copy of the subimage is a deep copy,
#pybind11#            else it is a shallow copy; ignored if useSubregion is False
#pybind11#        - interpLength: interpLength argument for lsst.afw.math.warpExposure
#pybind11#        - cacheSize: cacheSize argument for lsst.afw.math.SeparableKernel.computeCache;
#pybind11#            0 disables the cache
#pybind11#            10000 gives some speed improvement but less accurate results (atol must be increased)
#pybind11#            100000 gives better accuracy but no speed improvement in this test
#pybind11#        - rtol: relative tolerance as used by numpy.allclose
#pybind11#        - atol: absolute tolerance as used by numpy.allclose
#pybind11#        """
#pybind11#        warper = afwMath.Warper(kernelName)
#pybind11#
#pybind11#        originalExposure, swarpedImage, swarpedWcs = self.getSwarpedImage(
#pybind11#            kernelName=kernelName, useSubregion=useSubregion, useDeepCopy=useDeepCopy)
#pybind11#        maxBBox = afwGeom.Box2I(
#pybind11#            afwGeom.Point2I(swarpedImage.getX0(), swarpedImage.getY0()),
#pybind11#            afwGeom.Extent2I(swarpedImage.getWidth(), swarpedImage.getHeight()))
#pybind11#
#pybind11#        # warning: this test assumes that the swarped image is smaller than it needs to be
#pybind11#        # to hold all of the warped pixels
#pybind11#        afwWarpedExposure = warper.warpExposure(
#pybind11#            destWcs=swarpedWcs,
#pybind11#            srcExposure=originalExposure,
#pybind11#            maxBBox=maxBBox,
#pybind11#        )
#pybind11#        afwWarpedMaskedImage = afwWarpedExposure.getMaskedImage()
#pybind11#
#pybind11#        afwWarpedMask = afwWarpedMaskedImage.getMask()
#pybind11#        noDataBitMask = afwImage.MaskU.getPlaneBitMask("NO_DATA")
#pybind11#        noDataMask = afwWarpedMask.getArray() & noDataBitMask
#pybind11#
#pybind11#        msg = "afw and swarp %s-warped %s (ignoring bad pixels)"
#pybind11#        self.assertImagesNearlyEqual(afwWarpedMaskedImage.getImage(), swarpedImage,
#pybind11#                                     skipMask=noDataMask, rtol=rtol, atol=atol, msg=msg)
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

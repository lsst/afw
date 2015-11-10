#!/usr/bin/env python2
from __future__ import absolute_import, division
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
"""Basic test of Warp (the warping algorithm is thoroughly tested in lsst.afw.math)
"""
import os
import unittest
import warnings

import lsst.utils
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.image.utils as imageUtils
import lsst.afw.math as afwMath
import lsst.utils.tests as utilsTests
import lsst.pex.logging as pexLog
import lsst.pex.policy as pexPolicy

VERBOSITY = 0                       # increase to see trace

pexLog.Debug("lsst.afw.math", VERBOSITY)

try:
    afwDataDir = lsst.utils.getPackageDir("afwdata")
except Exception:
    warnings.warn("skipping all tests because afwdata is not setup")
    dataDir = None
else:
    dataDir = os.path.join(afwDataDir, "data")
    originalExposureName = "medexp.fits"
    originalExposurePath = os.path.join(dataDir, originalExposureName)
    subExposureName = "medsub.fits"
    subExposurePath = os.path.join(dataDir, originalExposureName)
    originalFullExposureName = os.path.join("CFHT", "D4", "cal-53535-i-797722_1.fits")
    originalFullExposurePath = os.path.join(dataDir, originalFullExposureName)

class WarpExposureTestCase(utilsTests.TestCase):
    """Test case for Warp
    """
    def testMatchSwarpLanczos2Exposure(self):
        """Test that warpExposure matches swarp using a lanczos2 warping kernel.
        """
        self.compareToSwarp("lanczos2")

    def testMatchSwarpLanczos2SubExposure(self):
        """Test that warpExposure matches swarp using a lanczos2 warping kernel with a subexposure
        """
        for useDeepCopy in (False, True):
            self.compareToSwarp("lanczos2", useSubregion=True, useDeepCopy=useDeepCopy)
    
    def testBBox(self):
        """Test that the default bounding box includes all warped pixels
        """
        kernelName = "lanczos2"
        warper = afwMath.Warper(kernelName)
        originalExposure, swarpedImage, swarpedWcs = self.getSwarpedImage(
            kernelName=kernelName, useSubregion=True, useDeepCopy=False)

        filterPolicyFile = pexPolicy.DefaultPolicyFile("afw", "SdssFilters.paf", "tests")
        filterPolicy = pexPolicy.Policy.createPolicy(filterPolicyFile, filterPolicyFile.getRepositoryPath(), True)
        imageUtils.defineFiltersFromPolicy(filterPolicy, reset=True)

        originalFilter = afwImage.Filter("i")
        originalCalib = afwImage.Calib()
        originalCalib.setFluxMag0(1.0e5, 1.0e3)
        originalExposure.setFilter(originalFilter)
        originalExposure.setCalib(originalCalib)

        warpedExposure1 = warper.warpExposure(destWcs=swarpedWcs, srcExposure=originalExposure)
        # the default size must include all good pixels, so growing the bbox should not add any
        warpedExposure2 = warper.warpExposure(destWcs=swarpedWcs, srcExposure=originalExposure, border=1)
        # a bit of excess border is allowed, but surely not as much as 10 (in fact it is approx. 5)
        warpedExposure3 = warper.warpExposure(destWcs=swarpedWcs, srcExposure=originalExposure, border=-10)
        # assert that warpedExposure and warpedExposure2 have the same number of non-no_data pixels
        # and that warpedExposure3 has fewer
        noDataBitMask = afwImage.MaskU.getPlaneBitMask("NO_DATA")
        mask1Arr = warpedExposure1.getMaskedImage().getMask().getArray()
        mask2Arr = warpedExposure2.getMaskedImage().getMask().getArray()
        mask3Arr = warpedExposure3.getMaskedImage().getMask().getArray()
        nGood1 = (mask1Arr & noDataBitMask == 0).sum()
        nGood2 = (mask2Arr & noDataBitMask == 0).sum()
        nGood3 = (mask3Arr & noDataBitMask == 0).sum()
        self.assertEqual(nGood1, nGood2)
        self.assertTrue(nGood3 < nGood1)

        self.assertEquals(warpedExposure1.getFilter().getName(), originalFilter.getName())
        self.assertEquals(warpedExposure1.getCalib().getFluxMag0(), originalCalib.getFluxMag0())
        
    def testDestBBox(self):
        """Test that the destBBox argument works
        """
        kernelName = "lanczos2"
        warper = afwMath.Warper(kernelName)
        originalExposure, swarpedImage, swarpedWcs = self.getSwarpedImage(
            kernelName=kernelName, useSubregion=True, useDeepCopy=False)
        
        bbox = afwGeom.Box2I(afwGeom.Point2I(100, 25), afwGeom.Extent2I(3, 7))
        warpedExposure = warper.warpExposure(
            destWcs = swarpedWcs,
            srcExposure = originalExposure,
            destBBox = bbox,
            border = -2, # should be ignored
            maxBBox = afwGeom.Box2I(afwGeom.Point2I(1, 2), afwGeom.Extent2I(8, 9)), # should be ignored
        )
        self.assertTrue(bbox == warpedExposure.getBBox(afwImage.PARENT))
    
    def getSwarpedImage(self, kernelName, useSubregion=False, useDeepCopy=False):
        """
        Inputs:
        - kernelName: name of kernel in the form used by afwImage.makeKernel
        - useSubregion: if True then the original source exposure (from which the usual
            test exposure was extracted) is read and the correct subregion extracted
        - useDeepCopy: if True then the copy of the subimage is a deep copy,
            else it is a shallow copy; ignored if useSubregion is False
        
        Returns:
        - originalExposure
        - swarpedImage
        - swarpedWcs
        """
        if useSubregion:
            originalFullExposure = afwImage.ExposureF(originalExposurePath)
            # "medsub" is a subregion of med starting at 0-indexed pixel (40, 150) of size 145 x 200
            bbox = afwGeom.Box2I(afwGeom.Point2I(40, 150), afwGeom.Extent2I(145, 200))
            originalExposure = afwImage.ExposureF(originalFullExposure, bbox, useDeepCopy)
            swarpedImageName = "medsubswarp1%s.fits" % (kernelName,)
        else:
            originalExposure = afwImage.ExposureF(originalExposurePath)
            swarpedImageName = "medswarp1%s.fits" % (kernelName,)

        swarpedImagePath = os.path.join(dataDir, swarpedImageName)
        swarpedDecoratedImage = afwImage.DecoratedImageF(swarpedImagePath)
        swarpedImage = swarpedDecoratedImage.getImage()
        swarpedMetadata = swarpedDecoratedImage.getMetadata()
        swarpedWcs = afwImage.makeWcs(swarpedMetadata)
        return (originalExposure, swarpedImage, swarpedWcs)

    def compareToSwarp(self, kernelName, 
                       useSubregion=False, useDeepCopy=False,
                       interpLength=10, cacheSize=100000,
                       rtol=4e-05, atol=1e-2):
        """Compare warpExposure to swarp for given warping kernel.
        
        Note that swarp only warps the image plane, so only test that plane.
        
        Inputs:
        - kernelName: name of kernel in the form used by afwImage.makeKernel
        - useSubregion: if True then the original source exposure (from which the usual
            test exposure was extracted) is read and the correct subregion extracted
        - useDeepCopy: if True then the copy of the subimage is a deep copy,
            else it is a shallow copy; ignored if useSubregion is False
        - interpLength: interpLength argument for lsst.afw.math.warpExposure
        - cacheSize: cacheSize argument for lsst.afw.math.SeparableKernel.computeCache;
            0 disables the cache
            10000 gives some speed improvement but less accurate results (atol must be increased)
            100000 gives better accuracy but no speed improvement in this test
        - rtol: relative tolerance as used by numpy.allclose
        - atol: absolute tolerance as used by numpy.allclose
        """
        warper = afwMath.Warper(kernelName)

        originalExposure, swarpedImage, swarpedWcs = self.getSwarpedImage(
            kernelName=kernelName, useSubregion=useSubregion, useDeepCopy=useDeepCopy)
        maxBBox = afwGeom.Box2I(
            afwGeom.Point2I(swarpedImage.getX0(), swarpedImage.getY0()),
            afwGeom.Extent2I(swarpedImage.getWidth(), swarpedImage.getHeight()))

        # warning: this test assumes that the swarped image is smaller than it needs to be
        # to hold all of the warped pixels
        afwWarpedExposure = warper.warpExposure(
            destWcs = swarpedWcs,
            srcExposure = originalExposure,
            maxBBox = maxBBox,
        )
        afwWarpedMaskedImage = afwWarpedExposure.getMaskedImage()

        afwWarpedMask = afwWarpedMaskedImage.getMask()
        noDataBitMask = afwImage.MaskU.getPlaneBitMask("NO_DATA")
        noDataMask = afwWarpedMask.getArray() & noDataBitMask

        msg = "afw and swarp %s-warped %s (ignoring bad pixels)"
        self.assertImagesNearlyEqual(afwWarpedMaskedImage.getImage(), swarpedImage,
            skipMask=noDataMask, rtol=rtol, atol=atol, msg=msg)
        
        
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
    if dataDir == None:
        return
    utilsTests.run(suite(), doExit)

if __name__ == "__main__":
    run(True)

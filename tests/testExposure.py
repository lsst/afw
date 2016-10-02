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
#pybind11#"""
#pybind11#Test lsst.afw.image.Exposure
#pybind11#
#pybind11#Author: Nicole M. Silvestri, University of Washington
#pybind11#Contact: nms@astro.washington.edu
#pybind11#Created on: Mon Sep 10, 2007
#pybind11#"""
#pybind11#
#pybind11#import os.path
#pybind11#import unittest
#pybind11#
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.utils.tests
#pybind11#import lsst.daf.base as dafBase
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.table as afwTable
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#import lsst.pex.policy as pexPolicy
#pybind11#import lsst.afw.fits
#pybind11#from lsst.afw.cameraGeom.testUtils import DetectorWrapper
#pybind11#from lsst.log import Log
#pybind11#from testTableArchivesLib import DummyPsf
#pybind11#
#pybind11#Log.getLogger("afw.image.Mask").setLevel(Log.INFO)
#pybind11#
#pybind11#try:
#pybind11#    dataDir = os.path.join(lsst.utils.getPackageDir("afwdata"), "data")
#pybind11#except pexExcept.NotFoundError:
#pybind11#    dataDir = None
#pybind11#else:
#pybind11#    InputMaskedImageName = "871034p_1_MI.fits"
#pybind11#    InputMaskedImageNameSmall = "small_MI.fits"
#pybind11#    InputImageNameSmall = "small"
#pybind11#    OutputMaskedImageName = "871034p_1_MInew.fits"
#pybind11#
#pybind11#    currDir = os.path.abspath(os.path.dirname(__file__))
#pybind11#    inFilePath = os.path.join(dataDir, InputMaskedImageName)
#pybind11#    inFilePathSmall = os.path.join(dataDir, InputMaskedImageNameSmall)
#pybind11#    inFilePathSmallImage = os.path.join(dataDir, InputImageNameSmall)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#@unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#class ExposureTestCase(lsst.utils.tests.TestCase):
#pybind11#    """
#pybind11#    A test case for the Exposure Class
#pybind11#    """
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        maskedImage = afwImage.MaskedImageF(inFilePathSmall)
#pybind11#        maskedImageMD = afwImage.readMetadata(inFilePathSmall)
#pybind11#
#pybind11#        self.smallExposure = afwImage.ExposureF(inFilePathSmall)
#pybind11#        self.width = maskedImage.getWidth()
#pybind11#        self.height = maskedImage.getHeight()
#pybind11#        self.wcs = afwImage.makeWcs(maskedImageMD)
#pybind11#        self.psf = DummyPsf(2.0)
#pybind11#        self.detector = DetectorWrapper().detector
#pybind11#
#pybind11#        self.exposureBlank = afwImage.ExposureF()
#pybind11#        self.exposureMiOnly = afwImage.makeExposure(maskedImage)
#pybind11#        self.exposureMiWcs = afwImage.makeExposure(maskedImage, self.wcs)
#pybind11#        self.exposureCrWcs = afwImage.ExposureF(100, 100, self.wcs)         # n.b. the (100, 100, ...) form
#pybind11#        self.exposureCrOnly = afwImage.ExposureF(afwGeom.ExtentI(100, 100))  # test with ExtentI(100, 100) too
#pybind11#
#pybind11#        afwImage.Filter.reset()
#pybind11#        afwImage.FilterProperty.reset()
#pybind11#
#pybind11#        filterPolicy = pexPolicy.Policy()
#pybind11#        filterPolicy.add("lambdaEff", 470.0)
#pybind11#        afwImage.Filter.define(afwImage.FilterProperty("g", filterPolicy))
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.smallExposure
#pybind11#        del self.wcs
#pybind11#        del self.psf
#pybind11#        del self.detector
#pybind11#
#pybind11#        del self.exposureBlank
#pybind11#        del self.exposureMiOnly
#pybind11#        del self.exposureMiWcs
#pybind11#        del self.exposureCrWcs
#pybind11#        del self.exposureCrOnly
#pybind11#
#pybind11#    def testGetMaskedImage(self):
#pybind11#        """
#pybind11#        Test to ensure a MaskedImage can be obtained from each
#pybind11#        Exposure. An Exposure is required to have a MaskedImage,
#pybind11#        therefore each of the Exposures should return a MaskedImage.
#pybind11#
#pybind11#        MaskedImage class should throw appropriate
#pybind11#        lsst::pex::exceptions::NotFound if the MaskedImage can not be
#pybind11#        obtained.
#pybind11#        """
#pybind11#        maskedImageBlank = self.exposureBlank.getMaskedImage()
#pybind11#        blankWidth = maskedImageBlank.getWidth()
#pybind11#        blankHeight = maskedImageBlank.getHeight()
#pybind11#        if blankWidth != blankHeight != 0:
#pybind11#            self.fail("%s = %s != 0" % (blankWidth, blankHeight))
#pybind11#
#pybind11#        maskedImageMiOnly = self.exposureMiOnly.getMaskedImage()
#pybind11#        miOnlyWidth = maskedImageMiOnly.getWidth()
#pybind11#        miOnlyHeight = maskedImageMiOnly.getHeight()
#pybind11#        self.assertAlmostEqual(miOnlyWidth, self.width)
#pybind11#        self.assertAlmostEqual(miOnlyHeight, self.height)
#pybind11#
#pybind11#        # NOTE: Unittests for Exposures created from a MaskedImage and
#pybind11#        # a WCS object are incomplete.  No way to test the validity of
#pybind11#        # the WCS being copied/created.
#pybind11#
#pybind11#        maskedImageMiWcs = self.exposureMiWcs.getMaskedImage()
#pybind11#        miWcsWidth = maskedImageMiWcs.getWidth()
#pybind11#        miWcsHeight = maskedImageMiWcs.getHeight()
#pybind11#        self.assertAlmostEqual(miWcsWidth, self.width)
#pybind11#        self.assertAlmostEqual(miWcsHeight, self.height)
#pybind11#
#pybind11#        maskedImageCrWcs = self.exposureCrWcs.getMaskedImage()
#pybind11#        crWcsWidth = maskedImageCrWcs.getWidth()
#pybind11#        crWcsHeight = maskedImageCrWcs.getHeight()
#pybind11#        if crWcsWidth != crWcsHeight != 0:
#pybind11#            self.fail("%s != %s != 0" % (crWcsWidth, crWcsHeight))
#pybind11#
#pybind11#        maskedImageCrOnly = self.exposureCrOnly.getMaskedImage()
#pybind11#        crOnlyWidth = maskedImageCrOnly.getWidth()
#pybind11#        crOnlyHeight = maskedImageCrOnly.getHeight()
#pybind11#        if crOnlyWidth != crOnlyHeight != 0:
#pybind11#            self.fail("%s != %s != 0" % (crOnlyWidth, crOnlyHeight))
#pybind11#
#pybind11#        # Check Exposure.getWidth() returns the MaskedImage's width
#pybind11#        self.assertEqual(crOnlyWidth, self.exposureCrOnly.getWidth())
#pybind11#        self.assertEqual(crOnlyHeight, self.exposureCrOnly.getHeight())
#pybind11#
#pybind11#    def testGetWcs(self):
#pybind11#        """
#pybind11#        Test if a WCS can be obtained from each Exposure created with
#pybind11#        a WCS.
#pybind11#
#pybind11#        Test that appropriate exceptions are thrown if a WCS is
#pybind11#        requested from an Exposure that was not created with a WCS.
#pybind11#        Python turns the pex::exceptions in the Exposure and
#pybind11#        MaskedImage classes into IndexErrors.
#pybind11#
#pybind11#        The exposureBlank, exposureMiOnly, and exposureCrOnly
#pybind11#        Exposures should throw a lsst::pex::exceptions::NotFound.
#pybind11#        """
#pybind11#
#pybind11#        self.assertFalse(self.exposureBlank.getWcs())
#pybind11#        self.assertFalse(self.exposureMiOnly.getWcs())
#pybind11#
#pybind11#        # These two should pass
#pybind11#        self.exposureMiWcs.getWcs()
#pybind11#        self.exposureCrWcs.getWcs()
#pybind11#
#pybind11#        self.assertFalse(self.exposureCrOnly.getWcs())
#pybind11#
#pybind11#    def testSetMembers(self):
#pybind11#        """
#pybind11#        Test that the MaskedImage and the WCS of an Exposure can be set.
#pybind11#        """
#pybind11#        exposure = afwImage.ExposureF()
#pybind11#
#pybind11#        maskedImage = afwImage.MaskedImageF(inFilePathSmall)
#pybind11#        exposure.setMaskedImage(maskedImage)
#pybind11#        exposure.setWcs(self.wcs)
#pybind11#        exposure.setDetector(self.detector)
#pybind11#        exposure.setFilter(afwImage.Filter("g"))
#pybind11#
#pybind11#        self.assertEquals(exposure.getDetector().getName(), self.detector.getName())
#pybind11#        self.assertEquals(exposure.getDetector().getSerial(), self.detector.getSerial())
#pybind11#        self.assertEquals(exposure.getFilter().getName(), "g")
#pybind11#
#pybind11#        try:
#pybind11#            exposure.getWcs()
#pybind11#        except pexExcept.Exception as e:
#pybind11#            print("caught expected exception (getWcs): %s" % e)
#pybind11#            pass
#pybind11#        #
#pybind11#        # Test the Calib member.  The Calib tests are in color.py, here we just check that it's in Exposure
#pybind11#        #
#pybind11#        calib = exposure.getCalib()
#pybind11#        dt = 10
#pybind11#        calib.setExptime(dt)
#pybind11#        self.assertEqual(exposure.getCalib().getExptime(), dt)
#pybind11#        #
#pybind11#        # now check that we can set Calib
#pybind11#        #
#pybind11#        calib = afwImage.Calib()
#pybind11#        dt = 666
#pybind11#        calib.setExptime(dt)
#pybind11#
#pybind11#        exposure.setCalib(calib)
#pybind11#
#pybind11#        self.assertEqual(exposure.getCalib().getExptime(), dt)
#pybind11#        #
#pybind11#        # Psfs next
#pybind11#        #
#pybind11#        self.assertFalse(exposure.hasPsf())
#pybind11#        exposure.setPsf(self.psf)
#pybind11#        self.assertTrue(exposure.hasPsf())
#pybind11#
#pybind11#        exposure.setPsf(DummyPsf(1.0))  # we can reset the Psf
#pybind11#
#pybind11#        # Test that we can set the MaskedImage and WCS of an Exposure
#pybind11#        # that already has both
#pybind11#        self.exposureMiWcs.setMaskedImage(maskedImage)
#pybind11#        exposure.setWcs(self.wcs)
#pybind11#
#pybind11#    def testHasWcs(self):
#pybind11#        """
#pybind11#        Test if an Exposure has a WCS or not.
#pybind11#        """
#pybind11#        self.assertFalse(self.exposureBlank.hasWcs())
#pybind11#
#pybind11#        self.assertFalse(self.exposureMiOnly.hasWcs())
#pybind11#        self.assertTrue(self.exposureMiWcs.hasWcs())
#pybind11#        self.assertTrue(self.exposureCrWcs.hasWcs())
#pybind11#        self.assertFalse(self.exposureCrOnly.hasWcs())
#pybind11#
#pybind11#    def testGetSubExposure(self):
#pybind11#        """
#pybind11#        Test that a subExposure of the original Exposure can be obtained.
#pybind11#
#pybind11#        The MaskedImage class should throw a
#pybind11#        lsst::pex::exceptions::InvalidParameter if the requested
#pybind11#        subRegion is not fully contained within the original
#pybind11#        MaskedImage.
#pybind11#
#pybind11#        """
#pybind11#        #
#pybind11#        # This subExposure is valid
#pybind11#        #
#pybind11#        subBBox = afwGeom.Box2I(afwGeom.Point2I(40, 50), afwGeom.Extent2I(10, 10))
#pybind11#        subExposure = self.exposureCrWcs.Factory(self.exposureCrWcs, subBBox, afwImage.LOCAL)
#pybind11#
#pybind11#        self.checkWcs(self.exposureCrWcs, subExposure)
#pybind11#
#pybind11#        # this subRegion is not valid and should trigger an exception
#pybind11#        # from the MaskedImage class and should trigger an exception
#pybind11#        # from the WCS class for the MaskedImage 871034p_1_MI.
#pybind11#
#pybind11#        subRegion3 = afwGeom.Box2I(afwGeom.Point2I(100, 100), afwGeom.Extent2I(10, 10))
#pybind11#
#pybind11#        def getSubRegion():
#pybind11#            self.exposureCrWcs.Factory(self.exposureCrWcs, subRegion3, afwImage.LOCAL)
#pybind11#
#pybind11#        self.assertRaises(pexExcept.LengthError, getSubRegion)
#pybind11#
#pybind11#        # this subRegion is not valid and should trigger an exception
#pybind11#        # from the MaskedImage class only for the MaskedImage small_MI.
#pybind11#        # small_MI (cols, rows) = (256, 256)
#pybind11#
#pybind11#        subRegion4 = afwGeom.Box2I(afwGeom.Point2I(250, 250), afwGeom.Extent2I(10, 10))
#pybind11#
#pybind11#        def getSubRegion():
#pybind11#            self.exposureCrWcs.Factory(self.exposureCrWcs, subRegion4, afwImage.LOCAL)
#pybind11#
#pybind11#        self.assertRaises(pexExcept.LengthError, getSubRegion)
#pybind11#
#pybind11#        # check the sub- and parent- exposures are using the same Wcs transformation
#pybind11#        subBBox = afwGeom.Box2I(afwGeom.Point2I(40, 50), afwGeom.Extent2I(10, 10))
#pybind11#        subExposure = self.exposureCrWcs.Factory(self.exposureCrWcs, subBBox, afwImage.LOCAL)
#pybind11#        parentPos = self.exposureCrWcs.getWcs().pixelToSky(0, 0)
#pybind11#
#pybind11#        parentPos = parentPos.getPosition()
#pybind11#
#pybind11#        subExpPos = subExposure.getWcs().pixelToSky(0, 0).getPosition()
#pybind11#
#pybind11#        for i in range(2):
#pybind11#            self.assertAlmostEqual(parentPos[i], subExpPos[i], 9, "Wcs in sub image has changed")
#pybind11#
#pybind11#    def testReadWriteFits(self):
#pybind11#        """Test readFits and writeFits.
#pybind11#        """
#pybind11#        # This should pass without an exception
#pybind11#        mainExposure = afwImage.ExposureF(inFilePathSmall)
#pybind11#        mainExposure.setDetector(self.detector)
#pybind11#
#pybind11#        subBBox = afwGeom.Box2I(afwGeom.Point2I(10, 10), afwGeom.Extent2I(40, 50))
#pybind11#        subExposure = mainExposure.Factory(mainExposure, subBBox, afwImage.LOCAL)
#pybind11#        self.checkWcs(mainExposure, subExposure)
#pybind11#        det = subExposure.getDetector()
#pybind11#        self.assertTrue(det)
#pybind11#
#pybind11#        subExposure = afwImage.ExposureF(inFilePathSmall, subBBox, afwImage.LOCAL)
#pybind11#
#pybind11#        self.checkWcs(mainExposure, subExposure)
#pybind11#
#pybind11#        # This should throw an exception
#pybind11#        def getExposure():
#pybind11#            afwImage.ExposureF(inFilePathSmallImage)
#pybind11#
#pybind11#        self.assertRaises(lsst.afw.fits.FitsError, getExposure)
#pybind11#
#pybind11#        mainExposure.setPsf(self.psf)
#pybind11#
#pybind11#        # Make sure we can write without an exception
#pybind11#        mainExposure.getCalib().setExptime(10)
#pybind11#        mainExposure.getCalib().setMidTime(dafBase.DateTime())
#pybind11#        midMjd = mainExposure.getCalib().getMidTime().get()
#pybind11#        fluxMag0, fluxMag0Err = 1e12, 1e10
#pybind11#        mainExposure.getCalib().setFluxMag0(fluxMag0, fluxMag0Err)
#pybind11#
#pybind11#        # Check scaling of Calib
#pybind11#        scale = 2.0
#pybind11#        calib = mainExposure.getCalib()
#pybind11#        calib *= scale
#pybind11#        self.assertEqual((fluxMag0*scale, fluxMag0Err*scale), calib.getFluxMag0())
#pybind11#        calib /= scale
#pybind11#        self.assertEqual((fluxMag0, fluxMag0Err), calib.getFluxMag0())
#pybind11#
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            mainExposure.writeFits(tmpFile)
#pybind11#
#pybind11#            readExposure = type(mainExposure)(tmpFile)
#pybind11#
#pybind11#            #
#pybind11#            # Check the round-tripping
#pybind11#            #
#pybind11#            self.assertEqual(mainExposure.getFilter().getName(), readExposure.getFilter().getName())
#pybind11#
#pybind11#            self.assertEqual(mainExposure.getCalib().getExptime(), readExposure.getCalib().getExptime())
#pybind11#            self.assertEqual(midMjd, readExposure.getCalib().getMidTime().get())
#pybind11#            self.assertEqual((fluxMag0, fluxMag0Err), readExposure.getCalib().getFluxMag0())
#pybind11#
#pybind11#            psf = readExposure.getPsf()
#pybind11#            self.assertIsNotNone(psf)
#pybind11#            dummyPsf = DummyPsf.swigConvert(psf)
#pybind11#            self.assertIsNotNone(dummyPsf)
#pybind11#            self.assertEqual(dummyPsf.getValue(), self.psf.getValue())
#pybind11#
#pybind11#    def checkWcs(self, parentExposure, subExposure):
#pybind11#        """Compare WCS at corner points of a sub-exposure and its parent exposure
#pybind11#           By using the function indexToPosition, we should be able to convert the indices
#pybind11#           (of the four corners (of the sub-exposure)) to positions and use the wcs
#pybind11#           to get the same sky coordinates for each.
#pybind11#        """
#pybind11#        subMI = subExposure.getMaskedImage()
#pybind11#        subDim = subMI.getDimensions()
#pybind11#
#pybind11#        # Note: pixel positions must be computed relative to XY0 when working with WCS
#pybind11#        mainWcs = parentExposure.getWcs()
#pybind11#        subWcs = subExposure.getWcs()
#pybind11#
#pybind11#        for xSubInd in (0, subDim.getX()-1):
#pybind11#            for ySubInd in (0, subDim.getY()-1):
#pybind11#                mainWcs.pixelToSky(
#pybind11#                    afwImage.indexToPosition(xSubInd),
#pybind11#                    afwImage.indexToPosition(ySubInd),
#pybind11#                )
#pybind11#                subWcs.pixelToSky(
#pybind11#                    afwImage.indexToPosition(xSubInd),
#pybind11#                    afwImage.indexToPosition(ySubInd),
#pybind11#                )
#pybind11#
#pybind11#    def cmpExposure(self, e1, e2):
#pybind11#        self.assertEqual(e1.getDetector().getName(), e2.getDetector().getName())
#pybind11#        self.assertEqual(e1.getDetector().getSerial(), e2.getDetector().getSerial())
#pybind11#        self.assertEqual(e1.getFilter().getName(), e2.getFilter().getName())
#pybind11#        xy = afwGeom.Point2D(0, 0)
#pybind11#        self.assertEqual(e1.getWcs().pixelToSky(xy)[0], e2.getWcs().pixelToSky(xy)[0])
#pybind11#        self.assertEqual(e1.getCalib().getExptime(), e2.getCalib().getExptime())
#pybind11#        # check PSF identity
#pybind11#        if not e1.getPsf():
#pybind11#            self.assertFalse(e2.getPsf())
#pybind11#        else:
#pybind11#            psf1 = DummyPsf.swigConvert(e1.getPsf())
#pybind11#            psf2 = DummyPsf.swigConvert(e2.getPsf())
#pybind11#            self.assertEqual(psf1.getValue(), psf2.getValue())
#pybind11#
#pybind11#    def testCopyExposure(self):
#pybind11#        """Copy an Exposure (maybe changing type)"""
#pybind11#
#pybind11#        exposureU = afwImage.ExposureU(inFilePathSmall)
#pybind11#        exposureU.setWcs(self.wcs)
#pybind11#        exposureU.setDetector(self.detector)
#pybind11#        exposureU.setFilter(afwImage.Filter("g"))
#pybind11#        exposureU.getCalib().setExptime(666)
#pybind11#        exposureU.setPsf(DummyPsf(4.0))
#pybind11#
#pybind11#        exposureF = exposureU.convertF()
#pybind11#        self.cmpExposure(exposureF, exposureU)
#pybind11#
#pybind11#        nexp = exposureF.Factory(exposureF, False)
#pybind11#        self.cmpExposure(exposureF, nexp)
#pybind11#
#pybind11#        # Ensure that the copy was deep.
#pybind11#        # (actually this test is invalid since getDetector() returns a CONST_PTR)
#pybind11#        # cen0 = exposureU.getDetector().getCenterPixel()
#pybind11#        # x0,y0 = cen0
#pybind11#        # det = exposureF.getDetector()
#pybind11#        # det.setCenterPixel(afwGeom.Point2D(999.0, 437.8))
#pybind11#        # self.assertEqual(exposureU.getDetector().getCenterPixel()[0], x0)
#pybind11#        # self.assertEqual(exposureU.getDetector().getCenterPixel()[1], y0)
#pybind11#
#pybind11#    def testDeepCopyData(self):
#pybind11#        """Make sure a deep copy of an Exposure has its own data (ticket #2625)
#pybind11#        """
#pybind11#        exp = afwImage.ExposureF(6, 7)
#pybind11#        mi = exp.getMaskedImage()
#pybind11#        mi.getImage().set(100)
#pybind11#        mi.getMask().set(5)
#pybind11#        mi.getVariance().set(200)
#pybind11#
#pybind11#        expCopy = exp.clone()
#pybind11#        miCopy = expCopy.getMaskedImage()
#pybind11#        miCopy.getImage().set(-50)
#pybind11#        miCopy.getMask().set(2)
#pybind11#        miCopy.getVariance().set(175)
#pybind11#
#pybind11#        self.assertFloatsAlmostEqual(miCopy.getImage().getArray(), -50)
#pybind11#        self.assertTrue(numpy.all(miCopy.getMask().getArray() == 2))
#pybind11#        self.assertFloatsAlmostEqual(miCopy.getVariance().getArray(), 175)
#pybind11#
#pybind11#        self.assertFloatsAlmostEqual(mi.getImage().getArray(), 100)
#pybind11#        self.assertTrue(numpy.all(mi.getMask().getArray() == 5))
#pybind11#        self.assertFloatsAlmostEqual(mi.getVariance().getArray(), 200)
#pybind11#
#pybind11#    def testDeepCopySubData(self):
#pybind11#        """Make sure a deep copy of a subregion of an Exposure has its own data (ticket #2625)
#pybind11#        """
#pybind11#        exp = afwImage.ExposureF(6, 7)
#pybind11#        mi = exp.getMaskedImage()
#pybind11#        mi.getImage().set(100)
#pybind11#        mi.getMask().set(5)
#pybind11#        mi.getVariance().set(200)
#pybind11#
#pybind11#        bbox = afwGeom.Box2I(afwGeom.Point2I(1, 0), afwGeom.Extent2I(5, 4))
#pybind11#        expCopy = exp.Factory(exp, bbox, afwImage.PARENT, True)
#pybind11#        miCopy = expCopy.getMaskedImage()
#pybind11#        miCopy.getImage().set(-50)
#pybind11#        miCopy.getMask().set(2)
#pybind11#        miCopy.getVariance().set(175)
#pybind11#
#pybind11#        self.assertFloatsAlmostEqual(miCopy.getImage().getArray(), -50)
#pybind11#        self.assertTrue(numpy.all(miCopy.getMask().getArray() == 2))
#pybind11#        self.assertFloatsAlmostEqual(miCopy.getVariance().getArray(), 175)
#pybind11#
#pybind11#        self.assertFloatsAlmostEqual(mi.getImage().getArray(), 100)
#pybind11#        self.assertTrue(numpy.all(mi.getMask().getArray() == 5))
#pybind11#        self.assertFloatsAlmostEqual(mi.getVariance().getArray(), 200)
#pybind11#
#pybind11#    def testDeepCopyMetadata(self):
#pybind11#        """Make sure a deep copy of an Exposure has a deep copy of metadata (ticket #2568)
#pybind11#        """
#pybind11#        exp = afwImage.ExposureF(10, 10)
#pybind11#        expMeta = exp.getMetadata()
#pybind11#        expMeta.set("foo", 5)
#pybind11#        expCopy = exp.clone()
#pybind11#        expCopyMeta = expCopy.getMetadata()
#pybind11#        expCopyMeta.set("foo", 6)
#pybind11#        self.assertEqual(expCopyMeta.get("foo"), 6)
#pybind11#        self.assertEqual(expMeta.get("foo"), 5)  # this will fail if the bug is present
#pybind11#
#pybind11#    def testDeepCopySubMetadata(self):
#pybind11#        """Make sure a deep copy of a subregion of an Exposure has a deep copy of metadata (ticket #2568)
#pybind11#        """
#pybind11#        exp = afwImage.ExposureF(10, 10)
#pybind11#        expMeta = exp.getMetadata()
#pybind11#        expMeta.set("foo", 5)
#pybind11#        bbox = afwGeom.Box2I(afwGeom.Point2I(1, 0), afwGeom.Extent2I(5, 5))
#pybind11#        expCopy = exp.Factory(exp, bbox, afwImage.PARENT, True)
#pybind11#        expCopyMeta = expCopy.getMetadata()
#pybind11#        expCopyMeta.set("foo", 6)
#pybind11#        self.assertEqual(expCopyMeta.get("foo"), 6)
#pybind11#        self.assertEqual(expMeta.get("foo"), 5)  # this will fail if the bug is present
#pybind11#
#pybind11#    def testMakeExposureLeaks(self):
#pybind11#        """Test for memory leaks in makeExposure (the test is in lsst.utils.tests.MemoryTestCase)"""
#pybind11#        afwImage.makeMaskedImage(afwImage.ImageU(afwGeom.Extent2I(10, 20)))
#pybind11#        afwImage.makeExposure(afwImage.makeMaskedImage(afwImage.ImageU(afwGeom.Extent2I(10, 20))))
#pybind11#
#pybind11#    def testImageSlices(self):
#pybind11#        """Test image slicing, which generate sub-images using Box2I under the covers"""
#pybind11#        exp = afwImage.ExposureF(10, 20)
#pybind11#        mi = exp.getMaskedImage()
#pybind11#        mi[9, 19] = 10
#pybind11#        # N.b. Exposures don't support setting/getting the pixels so can't replicate e.g. Image's slice tests
#pybind11#        sexp = exp[1:4, 6:10]
#pybind11#        self.assertEqual(sexp.getDimensions(), afwGeom.ExtentI(3, 4))
#pybind11#        sexp = exp[..., -3:]
#pybind11#        self.assertEqual(sexp.getDimensions(), afwGeom.ExtentI(exp.getWidth(), 3))
#pybind11#        self.assertEqual(sexp.getMaskedImage().get(sexp.getWidth() - 1, sexp.getHeight() - 1),
#pybind11#                         exp.getMaskedImage().get(exp.getWidth() - 1, exp.getHeight() - 1))
#pybind11#
#pybind11#    def testConversionToScalar(self):
#pybind11#        """Test that even 1-pixel Exposures can't be converted to scalars"""
#pybind11#        im = afwImage.ExposureF(10, 20)
#pybind11#
#pybind11#        self.assertRaises(TypeError, float, im)  # only single pixel images may be converted
#pybind11#        self.assertRaises(TypeError, float, im[0, 0])  # actually, can't convert (img, msk, var) to scalar
#pybind11#
#pybind11#    def testReadMetadata(self):
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            self.exposureCrWcs.getMetadata().set("FRAZZLE", True)
#pybind11#            # This will write the main metadata (inc. FRAZZLE) to the primary HDU, and the
#pybind11#            # WCS to subsequent HDUs, along with INHERIT=T.
#pybind11#            self.exposureCrWcs.writeFits(tmpFile)
#pybind11#            # This should read the first non-empty HDU (i.e. it skips the primary), but
#pybind11#            # goes back and reads it if it finds INHERIT=T.  That should let us read
#pybind11#            # frazzle and the Wcs from the PropertySet returned by readMetadata.
#pybind11#            md = afwImage.readMetadata(tmpFile)
#pybind11#            wcs = afwImage.makeWcs(md, True)
#pybind11#            self.assertEqual(wcs.getPixelOrigin(), self.wcs.getPixelOrigin())
#pybind11#            self.assertEqual(wcs.getSkyOrigin(), self.wcs.getSkyOrigin())
#pybind11#            self.assertTrue(numpy.all(wcs.getCDMatrix() == self.wcs.getCDMatrix()))
#pybind11#            frazzle = md.get("FRAZZLE")
#pybind11#            self.assertTrue(frazzle)
#pybind11#
#pybind11#    def testArchiveKeys(self):
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            exposure1 = afwImage.ExposureF(100, 100, self.wcs)
#pybind11#            exposure1.setPsf(self.psf)
#pybind11#            exposure1.writeFits(tmpFile)
#pybind11#            exposure2 = afwImage.ExposureF(tmpFile)
#pybind11#            self.assertFalse(exposure2.getMetadata().exists("AR_ID"))
#pybind11#            self.assertFalse(exposure2.getMetadata().exists("PSF_ID"))
#pybind11#            self.assertFalse(exposure2.getMetadata().exists("WCS_ID"))
#pybind11#
#pybind11#    def testTicket2861(self):
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            exposure1 = afwImage.ExposureF(100, 100, self.wcs)
#pybind11#            exposure1.setPsf(self.psf)
#pybind11#            schema = afwTable.ExposureTable.makeMinimalSchema()
#pybind11#            coaddInputs = afwImage.CoaddInputs(schema, schema)
#pybind11#            exposure1.getInfo().setCoaddInputs(coaddInputs)
#pybind11#            exposure2 = afwImage.ExposureF(exposure1, True)
#pybind11#            self.assertIsNotNone(exposure2.getInfo().getCoaddInputs())
#pybind11#            exposure2.writeFits(tmpFile)
#pybind11#            exposure3 = afwImage.ExposureF(tmpFile)
#pybind11#            self.assertIsNotNone(exposure3.getInfo().getCoaddInputs())
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

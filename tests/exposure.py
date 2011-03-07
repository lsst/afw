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

"""
Test lsst.afw.image.Exposure

Author: Nicole M. Silvestri, University of Washington
Contact: nms@astro.washington.edu
Created on: Mon Sep 10, 2007
"""

import os

import unittest

import eups
import lsst.daf.base as dafBase
import lsst.afw.detection as afwDetection
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.afw.cameraGeom as cameraGeom
import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.pex.logging as pexLog
import lsst.pex.policy as pexPolicy

try:
    type(VERBOSITY)
except:
    VERBOSITY = 0                       # increase to see trace

pexLog.Debug("lsst.afw.image", VERBOSITY)

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests") 

InputMaskedImageName = "871034p_1_MI"
InputMaskedImageNameSmall = "small_MI"
InputImageNameSmall = "small"
OutputMaskedImageName = "871034p_1_MInew.fits"

currDir = os.path.abspath(os.path.dirname(__file__))
inFilePath = os.path.join(dataDir, InputMaskedImageName)
inFilePathSmall = os.path.join(dataDir, InputMaskedImageNameSmall)
inFilePathSmallImage = os.path.join(dataDir, InputImageNameSmall)
outFile = OutputMaskedImageName

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ExposureTestCase(unittest.TestCase):
    """
    A test case for the Exposure Class
    """

    def setUp(self):
        maskedImage = afwImage.MaskedImageF(inFilePathSmall)
        maskedImageMD = afwImage.readMetadata(inFilePathSmall + "_img.fits")

        self.smallExposure = afwImage.ExposureF(inFilePathSmall)
        self.width =  maskedImage.getWidth()
        self.height = maskedImage.getHeight()
        self.wcs = afwImage.makeWcs(maskedImageMD)

        self.exposureBlank = afwImage.ExposureF()
        self.exposureMiOnly = afwImage.makeExposure(maskedImage)
        self.exposureMiWcs = afwImage.makeExposure(maskedImage, self.wcs)
        self.exposureCrWcs = afwImage.ExposureF(afwGeom.Extent2I(100, 100), self.wcs)
        self.exposureCrOnly = afwImage.ExposureF(afwGeom.Extent2I(100, 100))

        afwImage.Filter.reset()
        afwImage.FilterProperty.reset()

        filterPolicy = pexPolicy.Policy()
        filterPolicy.add("lambdaEff", 470.0)
        afwImage.Filter.define(afwImage.FilterProperty("g", filterPolicy))
            
    def tearDown(self):
        del self.smallExposure
        del self.wcs

        del self.exposureBlank 
        del self.exposureMiOnly
        del self.exposureMiWcs
        del self.exposureCrWcs
        del self.exposureCrOnly

    def testGetMaskedImage(self):
        """
        Test to ensure a MaskedImage can be obtained from each
        Exposure. An Exposure is required to have a MaskedImage,
        therefore each of the Exposures should return a MaskedImage.

        MaskedImage class should throw appropriate
        lsst::pex::exceptions::NotFound if the MaskedImage can not be
        obtained.
        """
        maskedImageBlank = self.exposureBlank.getMaskedImage()        
        blankWidth = maskedImageBlank.getWidth()
        blankHeight = maskedImageBlank.getHeight()
        if blankWidth != blankHeight != 0:
            self.fail("%s = %s != 0" % (blankWidth, blankHeight))           
        
        maskedImageMiOnly = self.exposureMiOnly.getMaskedImage()
        miOnlyWidth = maskedImageMiOnly.getWidth()
        miOnlyHeight = maskedImageMiOnly.getHeight()
        self.assertAlmostEqual(miOnlyWidth, self.width)
        self.assertAlmostEqual(miOnlyHeight, self.height)
        
        # NOTE: Unittests for Exposures created from a MaskedImage and
        # a WCS object are incomplete.  No way to test the validity of
        # the WCS being copied/created.
        
        maskedImageMiWcs = self.exposureMiWcs.getMaskedImage()
        miWcsWidth = maskedImageMiWcs.getWidth()
        miWcsHeight = maskedImageMiWcs.getHeight()
        self.assertAlmostEqual(miWcsWidth, self.width)
        self.assertAlmostEqual(miWcsHeight, self.height)
       
        maskedImageCrWcs = self.exposureCrWcs.getMaskedImage()       
        crWcsWidth = maskedImageCrWcs.getWidth()
        crWcsHeight = maskedImageCrWcs.getHeight()
        if crWcsWidth != crWcsHeight != 0:
            self.fail("%s != %s != 0" % (crWcsWidth, crWcsHeight))   
        
        maskedImageCrOnly = self.exposureCrOnly.getMaskedImage()
        crOnlyWidth = maskedImageCrOnly.getWidth()
        crOnlyHeight = maskedImageCrOnly.getHeight()
        if crOnlyWidth != crOnlyHeight != 0:
            self.fail("%s != %s != 0" % (crOnlyWidth, crOnlyHeight)) 

        # Check Exposure.getWidth() returns the MaskedImage's width
        self.assertEqual(crOnlyWidth, self.exposureCrOnly.getWidth())
        self.assertEqual(crOnlyHeight, self.exposureCrOnly.getHeight())

    def testGetWcs(self):
        """
        Test if a WCS can be obtained from each Exposure created with
        a WCS.
    
        Test that appropriate exceptions are thrown if a WCS is
        requested from an Exposure that was not created with a WCS.
        Python turns the pex::exceptions in the Exposure and
        MaskedImage classes into IndexErrors.

        The exposureBlank, exposureMiOnly, and exposureCrOnly
        Exposures should throw a lsst::pex::exceptions::NotFound.
        """

        self.assertTrue(not self.exposureBlank.getWcs())
        self.assertTrue(not self.exposureMiOnly.getWcs())

        # These two should pass
        self.exposureMiWcs.getWcs()
        self.exposureCrWcs.getWcs()
       
        self.assertTrue(not self.exposureCrOnly.getWcs())
            
    def testSetMembers(self):
        """
        Test that the MaskedImage and the WCS of an Exposure can be set.
        """
        exposure = afwImage.ExposureF()       

        maskedImage = afwImage.MaskedImageF(inFilePathSmall)
        exposure.setMaskedImage(maskedImage)
        exposure.setWcs(self.wcs)
        exposure.setDetector(cameraGeom.Detector(cameraGeom.Id(666)))
        exposure.setFilter(afwImage.Filter("g"))

        self.assertEquals(exposure.getDetector().getId().getSerial(), 666)
        self.assertEquals(exposure.getFilter().getName(), "g")
        
        try:
            exposure.getWcs()
        except pexExcept.LsstCppException, e:
            print "caught expected exception (getWcs): %s" % e   
            pass
        #
        # Test the Calib member.  The Calib tests are in color.py, here we just check that it's in Exposure
        #
        calib = exposure.getCalib()
        dt = 10
        calib.setExptime(dt)
        self.assertEqual(exposure.getCalib().getExptime(), dt)
        #
        # Psfs next
        #
        w, h = 11, 11
        self.assertFalse(exposure.hasPsf())
        psf = afwDetection.createPsf("DoubleGaussian", w, h, 3)
        exposure.setPsf(psf)
        self.assertTrue(exposure.hasPsf())
        self.assertEqual(exposure.getPsf().getKernel().getDimensions(), afwGeom.Extent2I(w, h))

        exposure.setPsf(afwDetection.createPsf("DoubleGaussian", w, h, 1)) # we can reset the Psf
         
        # Test that we can set the MaskedImage and WCS of an Exposure
        # that already has both
        self.exposureMiWcs.setMaskedImage(maskedImage)
        exposure.setWcs(self.wcs)
       
    def testHasWcs(self):
        """
        Test if an Exposure has a WCS or not.
        """
        self.assertFalse(self.exposureBlank.hasWcs())       

        self.assertFalse(self.exposureMiOnly.hasWcs())        
        self.assertTrue(self.exposureMiWcs.hasWcs())        
        self.assertTrue(self.exposureCrWcs.hasWcs())       
        self.assertFalse(self.exposureCrOnly.hasWcs())
       
    def testGetSubExposure(self):
        """
        Test that a subExposure of the original Exposure can be obtained.

        The MaskedImage class should throw a
        lsst::pex::exceptions::InvalidParameter if the requested
        subRegion is not fully contained within the original
        MaskedImage.
        
        """
        #
        # This subExposure is valid
        #
        subBBox = afwGeom.BoxI(afwGeom.PointI(40, 50), afwGeom.ExtentI(10, 10))
        subExposure = self.exposureCrWcs.Factory(self.exposureCrWcs, subBBox, afwImage.LOCAL)
        
        self.checkWcs(self.exposureCrWcs, subExposure)

        # this subRegion is not valid and should trigger an exception
        # from the MaskedImage class and should trigger an exception
        # from the WCS class for the MaskedImage 871034p_1_MI.
        
        subRegion3 = afwGeom.BoxI(afwGeom.PointI(100, 100), afwGeom.ExtentI(10, 10))
        def getSubRegion():
            self.exposureCrWcs.Factory(self.exposureCrWcs, subRegion3, afwImage.LOCAL)

        utilsTests.assertRaisesLsstCpp(self, pexExcept.LengthErrorException, getSubRegion)

        # this subRegion is not valid and should trigger an exception
        # from the MaskedImage class only for the MaskedImage small_MI.
        # small_MI (cols, rows) = (256, 256) 

        subRegion4 = afwGeom.BoxI(afwGeom.PointI(250, 250), afwGeom.ExtentI(10, 10))
        def getSubRegion():
            self.exposureCrWcs.Factory(self.exposureCrWcs, subRegion4, afwImage.LOCAL)

        utilsTests.assertRaisesLsstCpp(self, pexExcept.LengthErrorException, getSubRegion)

        #check the sub- and parent- exposures are using the same Wcs transformation
        subBBox = afwGeom.BoxI(afwGeom.PointI(40, 50), afwGeom.ExtentI(10, 10))
        subExposure = self.exposureCrWcs.Factory(self.exposureCrWcs, subBBox, afwImage.LOCAL)
        parentPos = self.exposureCrWcs.getWcs().pixelToSky(0,0)
        
        parentPos = parentPos.getPosition()
        
        subExpPos = subExposure.getWcs().pixelToSky(0,0).getPosition()
        
        for i in range(2):
            self.assertAlmostEqual(parentPos[i], subExpPos[i], 9, "Wcs in sub image has changed")

    def testReadWriteFits(self):
        """Test readFits and writeFits.
        """
        # This should pass without an exception
        mainExposure = afwImage.ExposureF(inFilePathSmall)
        mainExposure.setDetector(cameraGeom.Detector(cameraGeom.Id(666)))
        
        subBBox = afwGeom.BoxI(afwGeom.PointI(10, 10), afwGeom.ExtentI(40, 50))
        subExposure = mainExposure.Factory(mainExposure, subBBox, afwImage.LOCAL)
        self.checkWcs(mainExposure, subExposure)
        det = subExposure.getDetector()
        self.assertTrue(det)
        
        hdu = 0
        subExposure = afwImage.ExposureF(inFilePathSmall, hdu, subBBox)
        
        self.checkWcs(mainExposure, subExposure)
        
        # This should throw an exception
        def getExposure():
            afwImage.ExposureF(inFilePathSmallImage)
        
        utilsTests.assertRaisesLsstCpp(self, pexExcept.NotFoundException, getExposure)
        
        # Make sure we can write without an exception
        mainExposure.getCalib().setExptime(10)
        mainExposure.getCalib().setMidTime(dafBase.DateTime())
        midMjd = mainExposure.getCalib().getMidTime().get()
        fluxMag0, fluxMag0Err = 1e12, 1e10
        mainExposure.getCalib().setFluxMag0(fluxMag0, fluxMag0Err)

        mainExposure.writeFits(outFile)

        readExposure = type(mainExposure)(outFile)

        os.remove(outFile)
        #
        # Check the round-tripping
        #
        self.assertEqual(mainExposure.getFilter().getName(), readExposure.getFilter().getName())

        self.assertEqual(mainExposure.getCalib().getExptime(), readExposure.getCalib().getExptime())
        self.assertEqual(midMjd, readExposure.getCalib().getMidTime().get())
        self.assertEqual((fluxMag0, fluxMag0Err), readExposure.getCalib().getFluxMag0())

    def checkWcs(self, parentExposure, subExposure):
        """Compare WCS at corner points of a sub-exposure and its parent exposure
           By using the function indexToPosition, we should be able to convert the indices
           (of the four corners (of the sub-exposure)) to positions and use the wcs
           to get the same sky coordinates for each.
        """
        subMI = subExposure.getMaskedImage()
        subDim = subMI.getDimensions()
        subXY0 = subMI.getXY0()

        # Note: pixel positions must be computed relative to XY0 when working with WCS
        mainWcs = parentExposure.getWcs()
        subWcs = subExposure.getWcs()

        for xSubInd in (0, subDim.getX()-1):
            for ySubInd in (0, subDim.getY()-1):
                p0 = mainWcs.pixelToSky(
                    afwImage.indexToPosition(xSubInd),
                    afwImage.indexToPosition(ySubInd),
                )
                p1 = subWcs.pixelToSky(
                    afwImage.indexToPosition(xSubInd),
                    afwImage.indexToPosition(ySubInd),
                )

    def testCopyExposure(self):
        """Convert an Exposure from one type to another"""

        exposureU = afwImage.ExposureU(inFilePathSmall)
        exposureU.setWcs(self.wcs)
        exposureU.setDetector(cameraGeom.Detector(cameraGeom.Id(666)))
        exposureU.setFilter(afwImage.Filter("g"))
        exposureU.getCalib().setExptime(666)

        exposureF = exposureU.convertF()

        self.assertEqual(exposureU.getDetector(), exposureF.getDetector())
        self.assertEqual(exposureU.getFilter().getName(), exposureF.getFilter().getName())
        xy = afwGeom.PointD(0, 0)
        self.assertEqual(exposureU.getWcs().pixelToSky(xy)[0], exposureF.getWcs().pixelToSky(xy)[0])
        self.assertEqual(exposureU.getCalib().getExptime(), exposureF.getCalib().getExptime())

    def testMakeExposureLeaks(self):
        """Test for memory leaks in makeExposure (the test is in utilsTests.MemoryTestCase)"""
        m = afwImage.makeMaskedImage(afwImage.ImageU(afwGeom.Extent2I(10, 20)))
        e = afwImage.makeExposure(afwImage.makeMaskedImage(afwImage.ImageU(afwGeom.Extent2I(10, 20))))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """
    Returns a suite containing all the test cases in this module.
    """
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ExposureTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

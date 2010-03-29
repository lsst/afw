#!/usr/bin/env python
"""
Test lsst.afw.image.Exposure

Author: Nicole M. Silvestri, University of Washington
Contact: nms@astro.washington.edu
Created on: Mon Sep 10, 2007
"""

import os

import unittest

import eups
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.utils.tests as utilsTests
import lsst.pex.logging as pexLog
import lsst.pex.exceptions as pexExcept

VERBOSITY = 0 # increase to see trace

pexLog.Debug("lsst.afw.image", VERBOSITY)

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests") 

InputMaskedImageName = "871034p_1_MI"
InputMaskedImageNameSmall = "small_MI"
InputImageNameSmall = "small"
OutputMaskedImageName = "871034p_1_MInew"

currDir = os.path.abspath(os.path.dirname(__file__))
inFilePath = os.path.join(dataDir, InputMaskedImageName)
inFilePathSmall = os.path.join(dataDir, InputMaskedImageNameSmall)
inFilePathSmallImage = os.path.join(dataDir, InputImageNameSmall)
outFilePath = OutputMaskedImageName
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ExposureTestCase(unittest.TestCase):
    """
    A test case for the Exposure Class
    """

    def setUp(self):
        maskedImage = afwImage.MaskedImageF(inFilePathSmall)

        self.smallExposure = afwImage.ExposureF(inFilePathSmall)
        self.width =  maskedImage.getWidth()
        self.height = maskedImage.getHeight()
        self.wcs = afwImage.makeWcs(self.smallExposure.getMetadata())

        self.exposureBlank = afwImage.ExposureF()
        self.exposureMiOnly = afwImage.makeExposure(maskedImage)
        self.exposureMiWcs = afwImage.makeExposure(maskedImage, self.wcs)
        self.exposureCrWcs = afwImage.ExposureF(100, 100, self.wcs)
        self.exposureCrOnly = afwImage.ExposureF(100, 100)
            
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
        
        try:
            exposure.getWcs()
        except pexExcept.LsstCppException, e:
            print "caught expected exception (getWcs): %s" % e   
            pass
               
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
        subBBox = afwImage.BBox(afwImage.PointI(40, 50), 10, 10)
        subExposure = self.exposureCrWcs.Factory(self.exposureCrWcs, subBBox)
        
        self.checkWcs(self.exposureCrWcs, subExposure)

        # this subRegion is not valid and should trigger an exception
        # from the MaskedImage class and should trigger an exception
        # from the WCS class for the MaskedImage 871034p_1_MI.
        
        subRegion3 = afwImage.BBox(afwImage.PointI(100, 100), 10, 10)
        def getSubRegion():
            self.exposureCrWcs.Factory(self.exposureCrWcs, subRegion3)

        utilsTests.assertRaisesLsstCpp(self, pexExcept.LengthErrorException, getSubRegion)

        # this subRegion is not valid and should trigger an exception
        # from the MaskedImage class only for the MaskedImage small_MI.
        # small_MI (cols, rows) = (256, 256) 

        subRegion4 = afwImage.BBox(afwImage.PointI(250, 250), 10, 10)        
        def getSubRegion():
            self.exposureCrWcs.Factory(self.exposureCrWcs, subRegion4)

        utilsTests.assertRaisesLsstCpp(self, pexExcept.LengthErrorException, getSubRegion)

        #check the sub- and parent- exposures are using the same Wcs transformation
        subBBox = afwImage.BBox(afwImage.PointI(40, 50), 10, 10)
        subExposure = self.exposureCrWcs.Factory(self.exposureCrWcs, subBBox)
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
        
        subBBox = afwImage.BBox(afwImage.PointI(10, 10), 40, 50)
        subExposure = mainExposure.Factory(mainExposure, subBBox)
        self.checkWcs(mainExposure, subExposure)
        
        hdu = 0
        subExposure = afwImage.ExposureF(inFilePathSmall, hdu, subBBox)
        
        self.checkWcs(mainExposure, subExposure)
        
        # This should throw an exception
        def getExposure():
            afwImage.ExposureF(inFilePathSmallImage)
        
        utilsTests.assertRaisesLsstCpp(self, pexExcept.NotFoundException, getExposure)
        
        # Make sure we can write without an exception
        mainExposure.writeFits(outFilePath)

        os.remove(afwImage.MaskedImageF.imageFileName(outFilePath))
        os.remove(afwImage.MaskedImageF.maskFileName(outFilePath))
        os.remove(afwImage.MaskedImageF.varianceFileName(outFilePath))

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

        for xSubInd in (0, subDim[0]-1):
            for ySubInd in (0, subDim[1]-1):
                p0 = mainWcs.pixelToSky(
                    afwImage.indexToPosition(xSubInd),
                    afwImage.indexToPosition(ySubInd),
                )
                p1 = subWcs.pixelToSky(
                    afwImage.indexToPosition(xSubInd),
                    afwImage.indexToPosition(ySubInd),
                )


         
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

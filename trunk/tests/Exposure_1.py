#!/usr/bin/env python
"""
Test lsst.afw.image.Exposure

Author: Nicole M. Silvestri, University of Washington
Contact: nms@astro.washington.edu
Created on: Mon Sep 10, 2007
"""

import os
import math
import pdb # we may want to say pdb.set_trace()
import unittest

import numpy

import eups
import lsst.afw.image as afwImage
import lsst.utils.tests as utilsTests
import lsst.pex.logging as pexLog
import lsst.pex.exceptions as pexEx

Verbosity = 0 # increase to see trace
pexLog.Trace_setVerbosity("lsst.afw.image", Verbosity)

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
outFilePath = os.path.join(dataDir, OutputMaskedImageName)
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
        self.wcs = afwImage.Wcs(self.smallExposure.getMetadata())

        self.exposureBlank = afwImage.ExposureF()
        self.exposureMiOnly = afwImage.ExposureF(maskedImage)
        self.exposureMiWcs = afwImage.ExposureF(maskedImage, self.wcs)
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

    def testTearDown(self):
        """Test that tearDown tears everything down"""
        pass

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
            self.fail("%s = %s != 0" (blankWidth, blankHeight))           
        
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
            self.fail("%s != %s != 0" (crWcsWidth, crWcsHeight))   
        
        maskedImageCrOnly = self.exposureCrOnly.getMaskedImage()
        crOnlyWidth = maskedImageCrOnly.getWidth()
        crOnlyHeight = maskedImageCrOnly.getHeight()
        if crOnlyWidth != crOnlyHeight != 0:
            self.fail("%s != %s != 0" (crOnlyWidth, crOnlyRows)) 
       
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
        wcsMiWcs = self.exposureMiWcs.getWcs()
        wcsCrWcs = self.exposureCrWcs.getWcs()
       
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
            theWcs = exposure.getWcs();
        except pexEx.LsstExceptionStack, e:
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

        Member has not been fully implemented yet (as of Sep 19 2007)
        so this test should throw a
        lsst::pex::exceptions::InvalidParameter when a subExposure is
        requested.
 
        The MaskedImage class should throw a
        lsst::pex::exceptions::InvalidParameter if the requested
        subRegion is not fully contained within the original
        MaskedImage.
        
        """
        
        # the following subRegion is valid and should not trigger an
        # exception from the MaskedImage class, however the WCS class
        # may throw an exception if the WCS FITS cards for the image
        # are not found (this fails for 871034p_1_MI because the Fits
        # header cards are not found).
        
        subRegion1 = afwImage.BBox(afwImage.PointI(50, 50), 10, 10)
        try:
            subExposure = self.exposureCrWcs.getSubExposure(subRegion1)
           
        except Exception, e:
           
            pass
        
        subRegion2 = afwImage.BBox(afwImage.PointI(0, 0), 5, 5)
        try:
            subExposure = afwImage.ExposureF(self.smallExposure, subRegion2)
        except IndexError, e:
            pass
        
        # this subRegion is not valid and should trigger an exception
        # from the MaskedImage class and should trigger an exception
        # from the WCS class for the MaskedImage 871034p_1_MI.
        
        def getSubRegion():
            subExposure = afwImage.ExposureF(self.exposureCrWcs, subRegion3)

        subRegion3 = afwImage.BBox(afwImage.PointI(100, 100), 10, 10)
        self.assertRaises(pexEx.LsstLengthError, getSubRegion)

        # this subRegion is not valid and should trigger an exception
        # from the MaskedImage class only for the MaskedImage small_MI.
        # small_MI (cols, rows) = (256, 256) 

        def getSubRegion():
            subExposure = afwImage.ExposureF(self.exposureCrWcs, subRegion3)

        subRegion4 = afwImage.BBox(afwImage.PointI(250, 250), 10, 10)        
        self.assertRaises(pexEx.LsstLengthError, getSubRegion)
        
    def testReadWriteFits(self):
         """

         Test that the readFits member can read an Exposure given the
         name of the Exposure.

         The constructor taking a basename should read the Exposure's MaskedImage
         using the MaskedImage class' constructor and read the WCS
         metadata into a WCS object.  Currently the WCS class lacks
         the capability to return the metadata to the user so a
         readFits request should simply reset the _wcsPtr with the
         metadata obtained frm the MaskedImage.  Exposure's readFits
         only take a MaskedImage for now.  The MaskedImage class will
         throw an exception if the MaskedImage can't be found.

         The writeFits member is not yet fully implemented (as of Sep
         19 2007) therefore this member should throw a
         lsst::pex::exceptions::InvalidParameter.
         """
         # This should pass without an exception
         exposure = afwImage.ExposureF(inFilePathSmall)

         # This should throw an exception
         def getExposure():
             exposure = afwImage.ExposureF(inFilePathSmallImage)
             
         self.assertRaises(pexEx.LsstNotFound, getExposure)

         # This should not throw an exception 
         exposure.writeFits(outFilePath)
         
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

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

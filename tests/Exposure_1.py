"""
Test lsst.afwImage.Exposure

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
import lsst.daf.tests as dafTests
import lsst.daf.utils as dafUtils
import lsst.pex.exceptions as pexEx

verbosity = 0 # increase to see trace
dafUtils.Trace_setVerbosity("lsst.afw", verbosity)

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests") 

InputMaskedImageName = "871034p_1_MI"
InputMaskedImageNameSmall = "small_MI"
InputImageNameSmall = "small_"
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
        self.maskedImage = afwImage.MaskedImageF()
        self.maskedImage.readFits(inFilePathSmall)
        self.wcs = afw.WCS(self.maskedImage.getImage().getMetaData())
        self.exposureBlank = afw.ExposureF()
        self.exposureMiOnly = afw.ExposureF(self.maskedImage)
        self.exposureMiWcs = afw.ExposureF(self.maskedImage, self.wcs)
        self.exposureCrWcs = afw.ExposureF(100, 100, self.wcs)
        self.exposureCrOnly = afw.ExposureF(100, 100)

    def tearDown(self):
        del self.exposureBlank 
        del self.exposureMiOnly
        del self.exposureMiWcs
        del self.exposureCrWcs
        del self.exposureCrOnly
        del self.maskedImage
        del self.wcs

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
        blankCols = maskedImageBlank.getCols()
        blankRows = maskedImageBlank.getRows()
        if blankCols != blankRows != 0:
            self.fail("%s = %s != 0" (blankCols, blankRows))           
        
        maskedImageMiOnly = self.exposureMiOnly.getMaskedImage()
        miOnlyCols = maskedImageMiOnly.getCols()
        miOnlyRows = maskedImageMiOnly.getRows()
        miCols = self.maskedImage.getCols()
        miRows = self.maskedImage.getRows()
        self.assertAlmostEqual(miOnlyCols, miCols)
        self.assertAlmostEqual(miOnlyRows, miRows)
        
        # NOTE: Unittests for Exposures created from a MaskedImage and
        # a WCS object are incomplete.  No way to test the validity of
        # the WCS being copied/created.
        
        maskedImageMiWcs = self.exposureMiWcs.getMaskedImage()
        miWcsCols = maskedImageMiWcs.getCols()
        miWcsRows = maskedImageMiWcs.getRows()
        self.assertAlmostEqual(miWcsCols, miCols)
        self.assertAlmostEqual(miWcsRows, miRows)
       
        maskedImageCrWcs = self.exposureCrWcs.getMaskedImage()       
        crWcsCols = maskedImageCrWcs.getCols()
        crWcsRows = maskedImageCrWcs.getRows()
        if crWcsCols != crWcsRows != 0:
            self.fail("%s != %s != 0" (crWcsCols, crWcsRows))   
        
        maskedImageCrOnly = self.exposureCrOnly.getMaskedImage()
        crOnlyCols = maskedImageCrOnly.getCols()
        crOnlyRows = maskedImageCrOnly.getRows()
        if crOnlyCols != crOnlyRows != 0:
            self.fail("%s != %s != 0" (crOnlyCols, crOnlyRows)) 
       
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
        try:
            wcsBlank = self.exposureBlank.getWcs()
            self.fail("No exception raised for wcsBlank")
        except pexEx.LsstExceptionStack, e:
            print "caught expected exception (wcsBlank): %s" % e
            pass
            
        try:    
            wcsMiOnly = self.exposureMiOnly.getWcs()
            self.fail("No exception raised for wcsMiOnly")
        except pexEx.LsstExceptionStack, e:
            print "caught expected exception (wcsMiOnly): %s" % e
            pass

        # These two should pass
        wcsMiWcs = self.exposureMiWcs.getWcs()
        wcsCrWcs = self.exposureCrWcs.getWcs()
       
        try:
            wcsCrOnly = self.exposureCrOnly.getWcs()
            self.fail("No exception raised for wcsCrOnly")
        except pexEx.LsstExceptionStack, e:
            print "caught expected exception (wcsCrOnly): %s" % e
            pass

            
    def testSetMembers(self):
        """
        Test that the MaskedImage and the WCS of an Exposure can be set.
        """
        maskedImage = afwImage.MaskedImageF()           
        maskedImage.readFits(inFilePathSmall)
        
        exposure = afw.ExposureF()       
        exposure.setMaskedImage(maskedImage)        
        wcs = afw.WCS(maskedImage.getImage().getMetaData())
        exposure.setWcs(wcs)
        
        try:
            theWcs = exposure.getWcs();
        except pexEx.LsstExceptionStack, e:
            print "caught expected exception (getWcs): %s" % e   
            pass
               
        # Test that we can set the MaskedImage and WCS of an Exposure
        # that already has both
        
        self.exposureMiWcs.setMaskedImage(maskedImage)
        
        bigMiCols = self.maskedImage.getCols()
        bigMiRows = self.maskedImage.getRows()
        smallMiCols = maskedImage.getCols()
        smallMiRows = maskedImage.getRows()
       
      #  if bigMiCols == smallMiCols |  bigMiRows == smallMiRows:
      #      self.fail("%s = %s or %s = %s; MaskedImage was not set properly" (bigMiCols, smallMiCols, bigMiRows, smallMiRows)) 
        
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
        
        subRegion1 = afwImage.BBox2i(50, 50, 10, 10)
        try:
            subExposure = self.exposureCrWcs.getSubExposure(subRegion1)
           
        except Exception, e:
           
            pass
        
        smallMaskedImage = afwImage.MaskedImageF()
        smallMaskedImage.readFits(inFilePathSmall)
        wcs = afw.WCS(smallMaskedImage.getImage().getMetaData())
        smallExposure = afw.ExposureF(smallMaskedImage, wcs)
    
        subRegion2 = afwImage.BBox2i(0, 0, 5, 5)
        try:
            subExposure = smallExposure.getSubExposure(subRegion2)
        except IndexError, e:
       
            pass
        
        # this subRegion is not valid and should trigger an exception
        # from the MaskedImage class and should trigger an exception
        # from the WCS class for the MaskedImage 871034p_1_MI.
        
        subRegion3 = afwImage.BBox2i(100, 100, 10, 10)
        try:
            subExposure = self.exposureCrWcs.getSubExposure(subRegion3)
            self.fail("No exception raised for getSubExposureLargeMI")
        except pexEx.LsstExceptionStack, e:
            print "caught expected exception: %s" % e
            pass

        # this subRegion is not valid and should trigger an exception
        # from the MaskedImage class only for the MaskedImage small_MI.
        # small_MI (cols, rows) = (256, 256) 

        subRegion4 = afwImage.BBox2i(250, 250, 10, 10)        
        try:
            subExposure = smallExposure.getSubExposure(subRegion4)
            self.fail("No exception raised for getSubExposureSmallMI")
        except pexEx.LsstExceptionStack, e:
            print "caught expected exception: %s" % e
            pass
        
    def testReadWriteFits(self):
         """

         Test that the readFits member can read an Exposure given the
         name of the Exposure.

         The readFits member should read the Exposure's MaskedImage
         using the MaskedImage class' readFits member and read the WCS
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
         exposure = afw.ExposureF()

         # This should pass without an exception
         exposure.readFits(inFilePathSmall)

         # This should throw an exception
         try:
             exposure.readFits(inFilePathSmallImage)
             self.fail("No exception raised for readFits")
         except pexEx.LsstNotFound, e:
             print "caught expected exception: %s" % e
             pass

         # This should not throw an exception 
         #try:
         exposure.writeFits(outFilePath)
         #exposure.writeFits(OutputMaskedImageName)         
         #    self.fail("No exception raised for writeFits")
         #except pexEx.LsstInvalidParameter, e:
         #    print "caught expected exception: %s" % e
         #    pass

         
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """
    Returns a suite containing all the test cases in this module.
    """
    dafTests.init()

    suites = []
    suites += unittest.makeSuite(ExposureTestCase)
    suites += unittest.makeSuite(dafTests.MemoryTestCase)

    return unittest.TestSuite(suites)

if __name__ == "__main__":
    dafTests.run(suite())

#!/usr/bin/env python

##test1079
##\brief Test that the wcs of sub-images are written and read from disk correctly
##$Id$
##\author Fergal Mullally


import os
import pdb                          # we may want to say pdb.set_trace()
import unittest

import eups
import lsst.afw.image as afwImg
import lsst.afw.geom as afwGeom
import lsst.utils.tests as utilsTests
import lsst.pex.exceptions.exceptionsLib as exceptions


try:
    type(verbose)
except NameError:
    verbose = 0



class SavingSubImagesTest(unittest.TestCase):
    """
    Tests for changes made for ticket #1079. In the LSST wcs transformations are done in terms
    of pixel position, which is measured from the lower left hand corner of the parent image from
    which this sub-image is drawn. However, when saving a sub-image to disk, the fits standards 
    has no concept of parent- and sub- images, and specifies that the wcs is measured relative to 
    the pixel index (i.e the lower left hand corner of the sub-image). This test makes sure 
    we're saving and reading wcs headers from sub-images correctly.
    """
    
    def setUp(self):
        path = eups.productDir("afw")
        self.parentFile = os.path.join(path, "tests", "data", "parent.fits")
        
        self.parent = afwImg.ExposureF(self.parentFile)
        self.llcParent = self.parent.getMaskedImage().getXY0()
        self.oParent = self.parent.getWcs().getPixelOrigin()
        
        #A list of pixel positions to test
        self.testPositions = []
        self.testPositions.append(afwGeom.makePointD(128, 128))
        self.testPositions.append(afwGeom.makePointD(0,0))        
        self.testPositions.append(afwGeom.makePointD(20,30))        
        self.testPositions.append(afwGeom.makePointD(60,50))        
        self.testPositions.append(afwGeom.makePointD(80, 80))        
        self.testPositions.append(afwGeom.makePointD(256,256))        

    def tearDown(self):
        del self.parent
        del self.oParent
        del self.testPositions

    def testInvarianceOfCrpix1(self):
        """Test that crpix is the same for parent and sub-image. Also tests that llc of sub-image
        saved correctly"""
        
        llc = afwImg.PointI(20, 30)
        bbox = afwImg.BBox(llc, 60, 50)
        subImg = afwImg.ExposureF(self.parent, bbox)

        subImgLlc = subImg.getMaskedImage().getXY0()
        oSubImage = subImg.getWcs().getPixelOrigin()
        
        #Useful for debugging
        if False:
            print self.parent.getMaskedImage().getXY0()
            print subImg.getMaskedImage().getXY0()
            print self.parent.getWcs().getFitsMetadata().toString()
            print subImg.getWcs().getFitsMetadata().toString()
            print self.oParent, oSubImage
        
        for i in range(2):
            self.assertEqual(llc[i], subImgLlc[i], "Corner of sub-image not correct")
            self.assertAlmostEqual(self.oParent[i], oSubImage[i], 6, "Crpix of sub-image not correct")


    def testInvarianceOfCrpix2(self):
        """For sub-images loaded from disk, test that crpix is the same for parent and sub-image. 
        Also tests that llc of sub-image saved correctly"""
        
        #Load sub-image directly off of disk
        llc = afwImg.PointI(20, 30)
        bbox = afwImg.BBox(llc, 60, 50)
        hdu=0
        subImg = afwImg.ExposureF(self.parentFile, hdu, bbox)
        oSubImage = subImg.getWcs().getPixelOrigin()
        subImgLlc = subImg.getMaskedImage().getXY0()
       
        #Useful for debugging
        if False:
            print self.parent.getMaskedImage().getXY0()
            print subImg.getMaskedImage().getXY0()
            print self.parent.getWcs().getFitsMetadata().toString()
            print subImg.getWcs().getFitsMetadata().toString()
            print self.oParent, oSubImage
        
        for i in range(2):
            self.assertEqual(llc[i], subImgLlc[i], "Corner of sub-image not correct")
            self.assertAlmostEqual(self.oParent[i], oSubImage[i], 6, "Crpix of sub-image not correct")
    
    
    def testInvarianceOfPixelToSky(self):

        llc = afwImg.PointI(20, 30)
        bbox = afwImg.BBox(llc, 60, 50)
        subImg = afwImg.ExposureF(self.parent, bbox)

        for p in self.testPositions:
            adParent = self.parent.getWcs().pixelToSky(p)
            adSub = subImg.getWcs().pixelToSky(p)
            
            for i in range(2):
                msg = "Subimage radec is wrong. Expected %7f %.7f got %.7f %.7f" % \
                    (adParent[0], adParent[1], adSub[0], adSub[1])


    def testSubSubImage(self):
        """Check that a sub-image of a sub-image is equivalent to a sub image, i.e
        that the parent is an invarient"""                
        
        llc1 = afwImg.PointI(20, 30)
        bbox = afwImg.BBox(llc1, 60, 50)
        hdu=0
        subImg = afwImg.ExposureF(self.parentFile, hdu, bbox)


        llc2 = afwImg.PointI(22, 23)

        #This subsub image should fail. Although it's big enough to fit in the parent image
        #it's too small for the sub-image
        bbox = afwImg.BBox(llc2, 100, 110)
        self.assertRaises(exceptions.LsstCppException, afwImg.ExposureF, subImg, bbox)
        
        bbox = afwImg.BBox(llc2, 10, 11)
        subSubImg = afwImg.ExposureF(subImg, bbox)
        
        sub0 = subImg.getMaskedImage().getXY0()
        subsub0= subSubImg.getMaskedImage().getXY0()
        
        
        if False:
            print sub0
            print subsub0
            
            
        for i in range(2):
            self.assertEqual(llc1[i], sub0[i], "XY0 don't match (1)")
            self.assertEqual(llc1[i] + llc2[i], subsub0[i], "XY0 don't match (2)")
        
        subCrpix = subImg.getWcs().getPixelOrigin()
        subsubCrpix = subSubImg.getWcs().getPixelOrigin()

        for i in range(2):
            self.assertAlmostEqual(subCrpix[i], subsubCrpix[i], 6, "crpix don't match")
        
     
    def testRoundTrip(self):
        """Test that saving and retrieving an image doesn't alter the metadata"""
        llc = afwImg.PointI(20, 30)
        bbox = afwImg.BBox(llc, 60, 50)
        subImg = afwImg.ExposureF(self.parent, bbox)

        subImg.writeFits("tmp.fits")
        
        newImg = afwImg.ExposureF("tmp.fits")
        
        subXY0 = subImg.getMaskedImage().getXY0()
        newXY0 = newImg.getMaskedImage().getXY0()
        
        parentCrpix = self.parent.getWcs().getPixelOrigin()
        subCrpix = subImg.getWcs().getPixelOrigin()
        newCrpix = newImg.getWcs().getPixelOrigin()
        
        if False:
            print self.parent.getWcs().getFitsMetadata().toString()
            print subImg.getWcs().getFitsMetadata().toString()
            print newImg.getWcs().getFitsMetadata().toString()
            
        for i in range(2):
            #Sanity check. subImg's crpix is the same as the parent
            self.assertAlmostEqual(parentCrpix[i], subCrpix[i], 6,"parent/sub crpix disagree")
            
            self.assertEqual(subXY0[i], newXY0[i], "Origin has changed")
            self.assertAlmostEqual(subCrpix[i], newCrpix[i], 6,"crpix has changed")


    def testFitsHeader(self):
        """Test that XY0 and crpix are written to the header as expected"""
        
        parentCrpix = self.parent.getWcs().getPixelOrigin()
        
        #Make a sub-image
        x0, y0 = 20, 30
        llc = afwImg.PointI(x0, y0)
        bbox = afwImg.BBox(llc, 60, 50)
        subImg = afwImg.ExposureF(self.parent, bbox)
        
        subImg.writeFits("tmp.fits")
        hdr = afwImg.readMetadata("tmp.fits")
        
        self.assertTrue( hdr.exists("LTV1"), "LTV1 not saved to fits header")
        self.assertTrue( hdr.exists("LTV2"), "LTV2 not saved to fits header")
        self.assertEqual(hdr.get("LTV1"), -1*x0, "LTV1 has wrong value")
        self.assertEqual(hdr.get("LTV2"), -1*y0, "LTV1 has wrong value")


        self.assertTrue( hdr.exists("CRPIX1"), "CRPIX1 not saved to fits header")
        self.assertTrue( hdr.exists("CRPIX2"), "CRPIX2 not saved to fits header")
        
        fitsCrpix = [hdr.get("CRPIX1"), hdr.get("CRPIX2")]
        self.assertAlmostEqual(fitsCrpix[0] - hdr.get("LTV1"), parentCrpix[0], 6, "CRPIX1 saved wrong")
        self.assertAlmostEqual(fitsCrpix[1] - hdr.get("LTV2"), parentCrpix[1], 6, "CRPIX2 saved wrong")
        
#####

def suite():
    """Returns a suite containing all the test cases in this mod        hdr = afwImg.getMetadata("tmp.fits")
ule."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SavingSubImagesTest)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

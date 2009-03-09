#!/usr/bin/env python
"""
Tests for MaskedImages

Run with:
   python MaskedImage.py
or
   python
   >>> import MaskedImage; MaskedImage.run()
"""

import os
import pdb  # we may want to say pdb.set_trace()
import sys
import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.daf.base
import lsst.afw.image.imageLib as afwImage
import eups
import lsst.afw.display.ds9 as ds9

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class MaskedImageTestCase(unittest.TestCase):
    """A test case for MaskedImage"""
    def setUp(self):
        self.imgVal1, self.varVal1 = 100.0, 10.0
        self.imgVal2, self.varVal2 = 200.0, 15.0
        self.mimage = afwImage.MaskedImageF(100, 200)

        self.mimage.getImage().set(self.imgVal1)
        #
        # Set center of mask to 0, with 2 pixel border set to EDGE
        #
        self.BAD = afwImage.MaskU_getPlaneBitMask("BAD")
        self.EDGE = afwImage.MaskU_getPlaneBitMask("EDGE")
        
        self.mimage.getMask().set(self.EDGE)
        centre = afwImage.MaskU(self.mimage.getMask(),
                                afwImage.BBox(afwImage.PointI(2,2), self.mimage.getWidth() - 4, self.mimage.getHeight() - 4))
        centre.set(0x0)
        #
        self.mimage.getVariance().set(self.varVal1)
        #
        # Second MaskedImage
        #
        self.mimage2 = afwImage.MaskedImageF(self.mimage.getDimensions())
        self.mimage2.getImage().set(self.imgVal2)
        self.mimage2.getVariance().set(self.varVal2)

    def tearDown(self):
        del self.mimage
        del self.mimage2

    def testSetGetValues(self):
        self.assertEqual(self.mimage.getImage().get(0,0), self.imgVal1)
        
        self.assertEqual(self.mimage.getMask().get(0,0), self.EDGE)
        self.assertEqual(self.mimage.getMask().get(1,1), self.EDGE)
        self.assertEqual(self.mimage.getMask().get(2,2), 0x0)
        
        self.assertEqual(self.mimage.getVariance().get(0,0), self.varVal1)
    
    def testAddImages(self):
        "Test addition"
        # add an image
        self.mimage2 += self.mimage

        self.assertEqual(self.mimage2.getImage().get(0,0), self.imgVal1 + self.imgVal2)
        self.assertEqual(self.mimage2.getMask().get(0,0), self.EDGE)
        self.assertEqual(self.mimage2.getVariance().get(0,0), self.varVal1 + self.varVal2)
        # add a scalar
        self.mimage += self.imgVal1
        
        self.assertEqual(self.mimage.getImage().get(0,0), 2*self.imgVal1)

        self.assertEqual(self.mimage.getMask().get(0,0), self.EDGE)
        self.assertEqual(self.mimage.getMask().get(1,1), self.EDGE)
        self.assertEqual(self.mimage.getMask().get(2,2), 0x0)

        self.assertEqual(self.mimage.getVariance().get(0,0), self.varVal1)
    
    def testAddScaledImages(self):
        "Test addition by a scaled MaskedImage"
        # add an image
        c = 10.0
        mimage2_copy = self.mimage2.Factory(self.mimage2, True) # make a copy
        self.mimage2.scaledPlus(c, self.mimage)
        #
        # Now repeat calculation using a temporary
        #
        tmp = self.mimage.Factory(self.mimage, True)
        tmp *= c
        mimage2_copy += tmp

        self.assertEqual(self.mimage2.getImage().get(0,0), mimage2_copy.getImage().get(0,0))
        self.assertEqual(self.mimage2.getMask().get(0,0), mimage2_copy.getMask().get(0,0))
        self.assertEqual(self.mimage2.getVariance().get(0,0), mimage2_copy.getVariance().get(0,0))

    def testSubtractImages(self):
        "Test subtraction by a scaled MaskedImage"
        # subtract a scaled image
        c = 10.0
        mimage2_copy = self.mimage2.Factory(self.mimage2, True) # make a copy
        self.mimage2.scaledMinus(c, self.mimage)
        #
        # Now repeat calculation using a temporary
        #
        tmp = self.mimage.Factory(self.mimage, True)
        tmp *= c
        mimage2_copy -= tmp

        self.assertEqual(self.mimage2.getImage().get(0,0), mimage2_copy.getImage().get(0,0))
        self.assertEqual(self.mimage2.getMask().get(0,0), mimage2_copy.getMask().get(0,0))
        self.assertEqual(self.mimage2.getVariance().get(0,0), mimage2_copy.getVariance().get(0,0))
    
    def testMultiplyImages(self):
        """Test multiplication"""
        # Multiply by an image
        self.mimage2 *= self.mimage

        self.assertEqual(self.mimage2.getImage().get(0,0), self.imgVal2*self.imgVal1)
        self.assertEqual(self.mimage2.getMask().get(0,0), self.EDGE)
        self.assertEqual(self.mimage2.getVariance().get(0,0),
                         self.varVal2*pow(self.imgVal1,2) + self.varVal1*pow(self.imgVal2, 2))
        # multiply by a scalar
        self.mimage *= self.imgVal1
        
        self.assertEqual(self.mimage.getImage().get(0,0), self.imgVal1*self.imgVal1)

        self.assertEqual(self.mimage.getMask().get(0,0), self.EDGE)
        self.assertEqual(self.mimage.getMask().get(1,1), self.EDGE)
        self.assertEqual(self.mimage.getMask().get(2,2), 0x0)

        self.assertEqual(self.mimage.getVariance().get(0,0), self.varVal1*pow(self.imgVal1, 2))
    
    def testScaledMultiplyImages(self):
        """Test multiplication by a scaled image"""
        # Multiply by an image
        c = 10.0
        mimage2_copy = self.mimage2.Factory(self.mimage2, True) # make a copy
        self.mimage2.scaledMultiplies(c, self.mimage)
        #
        # Now repeat calculation using a temporary
        #
        tmp = self.mimage.Factory(self.mimage, True)
        tmp *= c
        mimage2_copy *= tmp

        self.assertEqual(self.mimage2.getImage().get(0,0), mimage2_copy.getImage().get(0,0))
        self.assertEqual(self.mimage2.getMask().get(0,0), mimage2_copy.getMask().get(0,0))
        self.assertEqual(self.mimage2.getVariance().get(0,0), mimage2_copy.getVariance().get(0,0))

    def testDivideImages(self):
        """Test division"""
        # Divide by an image
        self.mimage2 /= self.mimage

        self.assertEqual(self.mimage2.getImage().get(0,0), self.imgVal2/self.imgVal1)
        self.assertEqual(self.mimage2.getMask().get(0,0), self.EDGE)
        self.assertAlmostEqual(self.mimage2.getVariance().get(0,0),
                               (self.varVal2*pow(self.imgVal1,2) + self.varVal1*pow(self.imgVal2, 2))/pow(self.imgVal1, 4), 10)
        # divide by a scalar
        self.mimage /= self.imgVal1
        
        self.assertEqual(self.mimage.getImage().get(0,0), self.imgVal1/self.imgVal1)

        self.assertEqual(self.mimage.getMask().get(0,0), self.EDGE)
        self.assertEqual(self.mimage.getMask().get(1,1), self.EDGE)
        self.assertEqual(self.mimage.getMask().get(2,2), 0x0)

        self.assertAlmostEqual(self.mimage.getVariance().get(0,0), self.varVal1/pow(self.imgVal1, 2), 9)
        
    def testScaledDivideImages(self):
        """Test division by a scaled image"""
        # Divide by an image
        c = 10.0
        mimage2_copy = self.mimage2.Factory(self.mimage2, True) # make a copy
        self.mimage2.scaledDivides(c, self.mimage)
        #
        # Now repeat calculation using a temporary
        #
        tmp = self.mimage.Factory(self.mimage, True)
        tmp *= c
        mimage2_copy /= tmp

        self.assertEqual(self.mimage2.getImage().get(0,0), mimage2_copy.getImage().get(0,0))
        self.assertEqual(self.mimage2.getMask().get(0,0), mimage2_copy.getMask().get(0,0))
        self.assertEqual(self.mimage2.getVariance().get(0,0), mimage2_copy.getVariance().get(0,0))

    def testCopyConstructors(self):
        dimage = afwImage.MaskedImageF(self.mimage, True) # deep copy
        simage = afwImage.MaskedImageF(self.mimage) # shallow copy
        
        self.mimage += 2                # should only change dimage
        self.assertEqual(dimage.getImage().get(0,0), self.imgVal1)
        self.assertEqual(simage.getImage().get(0,0), self.imgVal1 + 2)

    def checkImgPatch12(self, img, x0, y0):
        """Check that a patch of an image is correct; origin of patch is at (x0, y0) in full image
        N.b. This isn't a general routine!  Works only for testSubimages[12]"""
        
        self.assertEqual(img.get(x0 - 1, y0 - 1), (self.imgVal1, self.EDGE, self.varVal1))
        self.assertEqual(img.get(x0,     y0),     (666,          self.BAD,  0))
        self.assertEqual(img.get(x0 + 3, y0),     (self.imgVal1, 0x0,       self.varVal1))
        self.assertEqual(img.get(x0,     y0 + 1), (666,          self.BAD,  0))
        self.assertEqual(img.get(x0 + 3, y0 + 1), (self.imgVal1, 0x0,       self.varVal1))
        self.assertEqual(img.get(x0,     y0 + 2), (self.imgVal1, 0x0,       self.varVal1))

    def testSubimages1(self):
        smimage = afwImage.MaskedImageF(self.mimage, afwImage.BBox(afwImage.PointI(1, 1), 10, 5))
        
        simage = afwImage.MaskedImageF(smimage, afwImage.BBox(afwImage.PointI(1, 1), 3, 2))
        self.assertEqual(simage.getX0(), 2); self.assertEqual(simage.getY0(), 2) # i.e. wrt self.mimage

        mimage2 = afwImage.MaskedImageF(simage.getDimensions())
        mimage2.getImage().set(666)
        mimage2.getMask().set(self.BAD)
        simage <<= mimage2

        del simage; del mimage2

        self.checkImgPatch12(self.mimage, 2, 2)
        self.checkImgPatch12(smimage, 1, 1)

    def testSubimages2(self):
        """Test subimages when we've played with the (x0, y0) value"""

        self.mimage.set(9, 4, (888, 0x0, 0))
        #printImg(afwImage.ImageF(self.mimage, afwImage.BBox(afwImage.PointI(0, 0), 10, 5))); print

        smimage = afwImage.MaskedImageF(self.mimage, afwImage.BBox(afwImage.PointI(1, 1), 10, 5))
        smimage.setXY0(afwImage.PointI(0, 0)) # reset origin; doesn't affect pixel coordinate systems

        simage = afwImage.MaskedImageF(smimage, afwImage.BBox(afwImage.PointI(1, 1), 3, 2))
        self.assertEqual(simage.getX0(), 1); self.assertEqual(simage.getY0(), 1)

        mimage2 = afwImage.MaskedImageF(simage.getDimensions())
        mimage2.set(666, self.BAD, 0.0)
        simage <<= mimage2
        del simage; del mimage2
        
        self.checkImgPatch12(self.mimage, 2, 2)
        self.checkImgPatch12(smimage, 1, 1)

    def checkImgPatch3(self, img, deep):
        """Check that a patch of an image is correct; origin of patch is at (x0, y0) in full image
        N.b. This isn't a general routine!  Works only for testSubimages3"""

        # Include deep in comparison so we can see which test fails
        self.assertEqual(img.get(0,   0) + (deep, ), (100, 0x0, self.varVal1, deep))
        self.assertEqual(img.get(10, 10) + (deep, ), (200, 0xf, self.varVal1, deep))

    def testSubimages3(self):
        """Test subimages when we've played with the (x0, y0) value"""

        self.mimage.getImage().set(20, 20, 200); self.mimage.getMask().set(20, 20, 0xf)

        for deep in (True, False):
            mimage = self.mimage.Factory(self.mimage, afwImage.BBox(afwImage.PointI(10, 10), 64, 64), deep)
            mimage.setXY0(afwImage.PointI(0, 0))
            mimage2 = mimage.Factory(mimage)
            
            if display:
                ds9.mtv(mimage2)
                
            self.checkImgPatch3(mimage2, deep)

    def testSetCopiedMask(self):
        """Check that we can set the Mask with a copied Mask"""
        
        crMask = self.mimage.getMask().Factory(self.mimage.getMask(), True)
        msk = self.mimage.getMask(); msk |= crMask; del msk

    def testVariance(self):
        """Check that we can set the variance from the gain"""
        gain = 2

        var = self.mimage.getVariance()
        var <<= self.mimage.getImage();
	var /= gain

    def testTicket653(self):
        """How-to-repeat for #653"""
        # The original ticket read this file, but it doesn't reproduce for me,
        # As I don't see how reading an exposure from disk could make a difference
        # it's easier to just build an Image
        if False:
            im = afwImage.ImageF(os.path.join(eups.productDir("afwdata"), "med_img.fits"))
        else:
            im = afwImage.ImageF(10, 10)
        mi = afwImage.MaskedImageF(im)
        exp = afwImage.ExposureF(mi)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def printImg(img):
    print "%4s " % "",
    for c in range(img.getWidth()):
        print "%7d" % c,
    print

    for r in range(img.getHeight() - 1, -1, -1):
        print "%4d " % r,
        for c in range(img.getWidth()):
            print "%7.1f" % float(img.get(c, r)),
        print

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(MaskedImageTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

#!/usr/bin/env python
"""
Tests for Images

Run with:
   ./Image.py
or
   python
   >>> import Image; Image.run()
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

class ImageTestCase(unittest.TestCase):
    """A test case for Image"""
    def setUp(self):
        self.val1, self.val2 = 0.01, 1.0
        self.image1 = afwImage.ImageF(100, 200); self.image1.set(self.val1)
        self.image2 = afwImage.ImageF(self.image1.getDimensions()); self.image2.set(self.val2)

    def tearDown(self):
        del self.image1
        del self.image2

    def testSetGetImages(self):
        self.assertEqual(self.image1.get(0,0), self.val1)
        self.assertEqual(self.image2.get(0,0), self.val2)
    
    def testAddImages(self):
        self.image2 += self.image1
        self.image1 += self.val1
        
        self.assertEqual(self.image1.get(0,0), 2*self.val1)
        self.assertEqual(self.image2.get(0,0), self.val1 + self.val2)
    
    def testAddScaledImages(self):
        c = 10.0
        self.image1.scaledPlus(c, self.image2)
        
        self.assertEqual(self.image1.get(0,0), self.val1 + c*self.val2)
    
    def testSubtractImages(self):
        self.image2 -= self.image1
        self.image1 -= self.val1
        
        self.assertEqual(self.image1.get(0,0), 0)
        self.assertEqual(self.image2.get(0,0), self.val2 - self.val1)
    
    def testSubtractScaledImages(self):
        c = 10.0
        self.image1.scaledMinus(c, self.image2)
        
        self.assertEqual(self.image1.get(0,0), self.val1 - c*self.val2)
    
    def testMultiplyImages(self):
        self.image2 *= self.image1
        self.image1 *= self.val1
        
        self.assertEqual(self.image1.get(0,0), self.val1*self.val1)
        self.assertEqual(self.image2.get(0,0), self.val2*self.val1)
    
    def testMultiplesScaledImages(self):
        c = 10.0
        self.image1.scaledMultiplies(c, self.image2)
        
        self.assertEqual(self.image1.get(0,0), self.val1 * c*self.val2)
    
    def testDivideImages(self):
        self.image2 /= self.image1
        self.image1 /= self.val1
        
        self.assertEqual(self.image1.get(0,0), 1)
        self.assertEqual(self.image2.get(0,0), self.val2/self.val1)
    
    def testDividesScaledImages(self):
        c = 10.0
        self.image1.scaledDivides(c, self.image2)
        
        self.assertAlmostEqual(self.image1.get(0,0), self.val1/(c*self.val2))
    
    def testCopyConstructors(self):
        dimage = afwImage.ImageF(self.image1, True) # deep copy
        simage = afwImage.ImageF(self.image1) # shallow copy
        
        self.image1 += 2                # should only change dimage
        self.assertEqual(dimage.get(0,0), self.val1)
        self.assertEqual(simage.get(0,0), self.val1 + 2)

    def testGeneralisedCopyConstructors(self):
        imageU = self.image1.convertU16() # these are generalised (templated) copy constructors in C++
        imageF = imageU.convertFloat()

        self.assertEqual(imageU.get(0,0), self.val1)
        self.assertEqual(imageF.get(0,0), self.val1)
            

    def testSubimages(self):
        simage1 = afwImage.ImageF(self.image1, afwImage.BBox(afwImage.PointI(1, 1), 10, 5))
        
        simage = afwImage.ImageF(simage1, afwImage.BBox(afwImage.PointI(1, 1), 3, 2))
        self.assertEqual(simage.getX0(), 2); self.assertEqual(simage.getY0(), 2) # i.e. wrt self.image1

        image2 = afwImage.ImageF(simage.getDimensions())
        image2.set(666)
        simage <<= image2
        del simage; del image2

        self.checkImgPatch(self.image1, 2, 2)
        self.checkImgPatch(simage1, 1, 1)

    def testSubimages2(self):
        """Test subimages when we've played with the (x0, y0) value"""

        self.image1.set(9, 4, 888)
        #printImg(afwImage.ImageF(self.image1, afwImage.BBox(afwImage.PointI(0, 0), 10, 5))); print

        simage1 = afwImage.ImageF(self.image1, afwImage.BBox(afwImage.PointI(1, 1), 10, 5))
        simage1.setXY0(afwImage.PointI(0, 0)) # reset origin; doesn't affect pixel coordinate systems

        simage = afwImage.ImageF(simage1, afwImage.BBox(afwImage.PointI(1, 1), 3, 2))
        self.assertEqual(simage.getX0(), 1); self.assertEqual(simage.getY0(), 1)

        image2 = afwImage.ImageF(simage.getDimensions())
        image2.set(666)
        simage <<= image2
        del simage; del image2
        
        self.checkImgPatch(self.image1, 2, 2)
        self.checkImgPatch(simage1, 1, 1)

    def testBadSubimages(self):
        def tst():
            simage1 = afwImage.ImageF(self.image1, afwImage.BBox(afwImage.PointI(1, -1), 10, 5))

        utilsTests.assertRaisesLsstCpp(self, lsst.pex.exceptions.LengthErrorException, tst)
        

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ImageTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

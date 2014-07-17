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
Tests for MaskedImages

Run with:
   python MaskedImage.py
or
   python
   >>> import MaskedImage; MaskedImage.run()
"""

import os

import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.daf.base
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
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
        centre = afwImage.MaskU(
            self.mimage.getMask(),
            afwGeom.Box2I(afwGeom.Point2I(2, 2), self.mimage.getDimensions() - afwGeom.Extent2I(4)),
            afwImage.LOCAL)
        centre.set(0x0)
        #
        self.mimage.getVariance().set(self.varVal1)
        #
        # Second MaskedImage
        #
        self.mimage2 = afwImage.MaskedImageF(self.mimage.getDimensions())
        self.mimage2.getImage().set(self.imgVal2)
        self.mimage2.getVariance().set(self.varVal2)
        #
        # a Function2
        #
        self.function = afwMath.PolynomialFunction2D(2)
        self.function.setParameters(range(self.function.getNParameters()))

    def tearDown(self):
        del self.mimage
        del self.mimage2
        del self.function

    def testArrays(self):
        image, mask, variance = self.mimage.getArrays()
        self.assert_((self.mimage.getImage().getArray() == image).all())
        self.assert_((self.mimage.getMask().getArray() == mask).all())
        self.assert_((self.mimage.getVariance().getArray() == variance).all())
        mimage2 = afwImage.makeMaskedImageFromArrays(image, mask, variance)
        self.assertEqual(type(mimage2), type(self.mimage))

    def testSetGetValues(self):
        self.assertEqual(self.mimage.get(0, 0), (self.imgVal1, self.EDGE, self.varVal1))

        self.assertEqual(self.mimage.getMask().get(1, 1), self.EDGE)
        self.assertEqual(self.mimage.getMask().get(2, 2), 0x0)
    
    def testMaskedImageFromImage(self):
        w, h = 10, 20
        dims = afwGeom.Extent2I(w, h)
        im, mask, var = afwImage.ImageF(dims), afwImage.MaskU(dims), afwImage.ImageF(dims)
        im.set(666)

        maskedImage = afwImage.MaskedImageF(im, mask, var)

        maskedImage = afwImage.makeMaskedImage(im, mask, var)

        maskedImage = afwImage.MaskedImageF(im)
        self.assertEqual(im.getDimensions(), maskedImage.getImage().getDimensions())
        self.assertEqual(im.getDimensions(), maskedImage.getMask().getDimensions())
        self.assertEqual(im.getDimensions(), maskedImage.getVariance().getDimensions())

        self.assertEqual(maskedImage.get(0, 0), (im.get(0, 0), 0x0, 0.0))

    def testMakeMaskedImageXY0(self):
        """Test that makeMaskedImage sets XY0 correctly"""
        im = afwImage.ImageF(200, 300)
        xy0 = afwGeom.PointI(10, 20)
        im.setXY0(*xy0)
        mi = afwImage.makeMaskedImage(im)

        self.assertEqual(mi.getImage().getXY0(),    xy0)
        self.assertEqual(mi.getMask().getXY0(),     xy0)
        self.assertEqual(mi.getVariance().getXY0(), xy0)

    def testCopyMaskedImage(self):
        """Test copy constructor"""
        #
        # shallow copy
        #
        mi = self.mimage.Factory(self.mimage, False)

        val00 = self.mimage.get(0, 0)
        nval00 = (100, 0xff, -1)        # the new value we'll set
        self.assertNotEqual(val00, nval00)

        self.assertEqual(mi.get(0, 0), val00)
        mi.set(0, 0, nval00)

        self.assertEqual(self.mimage.get(0, 0), nval00)
        self.assertEqual(mi.get(0, 0), nval00)
        mi.set(0, 0, val00)             # reinstate initial value
        #
        # deep copy
        #
        mi = self.mimage.Factory(self.mimage, True)

        self.assertEqual(mi.get(0, 0), val00)
        mi.set(0, 0, nval00)

        self.assertEqual(self.mimage.get(0, 0), val00)
        self.assertEqual(mi.get(0, 0), nval00)
        #
        # Copy with change of Image type
        #
        mi = self.mimage.convertD()

        self.assertEqual(mi.get(0, 0), val00)
        mi.set(0, 0, nval00)

        self.assertEqual(self.mimage.get(0, 0), val00)
        self.assertEqual(mi.get(0, 0), nval00)
        #
        # Convert from U to F
        #
        mi = afwImage.MaskedImageU(afwGeom.Extent2I(10, 20))
        val00 = (10, 0x10, 1)
        mi.set(val00)
        self.assertEqual(mi.get(0, 0), val00)

        fmi = mi.convertF()
        self.assertEqual(fmi.get(0, 0), val00)

    def testAddImages(self):
        "Test addition"
        # add an image
        self.mimage2 += self.mimage

        self.assertEqual(self.mimage2.get(0, 0), (self.imgVal1 + self.imgVal2, self.EDGE, 
                                                 self.varVal1 + self.varVal2))

        # Add an Image<int> to a MaskedImage<int>
        mimage_i = afwImage.MaskedImageI(self.mimage2.getDimensions())
        mimage_i.set(900, 0x0, 1000.0)
        image_i = afwImage.ImageI(mimage_i.getDimensions(), 2)

        mimage_i += image_i

        self.assertEqual(mimage_i.get(0, 0), (902, 0x0, 1000.0))

        # add a scalar
        self.mimage += self.imgVal1
        
        self.assertEqual(self.mimage.get(0, 0), (2*self.imgVal1, self.EDGE, self.varVal1))

        self.assertEqual(self.mimage.getMask().get(1, 1), self.EDGE)
        self.assertEqual(self.mimage.getMask().get(2, 2), 0x0)

        # add a function
        self.mimage.set(self.imgVal1, 0x0, 0.0)
        self.mimage += self.function

        for i, j in [(2, 3)]:
            self.assertEqual(self.mimage.getImage().get(i, j), self.imgVal1 + self.function(i, j))
    
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

        self.assertEqual(self.mimage2.get(0, 0), mimage2_copy.get(0, 0))

    def testSubtractImages(self):
        "Test subtraction"
        # subtract an image
        self.mimage2 -= self.mimage
        self.assertEqual(self.mimage2.get(0, 0),
                         (self.imgVal2 - self.imgVal1, self.EDGE, self.varVal2 + self.varVal1))

        # Subtract an Image<int> from a MaskedImage<int>
        mimage_i = afwImage.MaskedImageI(self.mimage2.getDimensions())
        mimage_i.set(900, 0x0, 1000.0)
        image_i = afwImage.ImageI(mimage_i.getDimensions(), 2)

        mimage_i -= image_i

        self.assertEqual(mimage_i.get(0, 0), (898, 0x0, 1000.0))

        # subtract a scalar
        self.mimage -= self.imgVal1
        self.assertEqual(self.mimage.get(0, 0), (0.0, self.EDGE, self.varVal1))

    def testSubtractScaledImages(self):
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

        self.assertEqual(self.mimage2.get(0, 0), mimage2_copy.get(0, 0))

    def testArithmeticImagesMismatch(self):
        "Test arithmetic operations on MaskedImages of different sizes"
        i1 = afwImage.MaskedImageF(afwGeom.Extent2I(100, 100))
        i1.set(100)
        i2 = afwImage.MaskedImageF(afwGeom.Extent2I(10, 10))
        i2.set(10)
        
        def tst1(i1, i2):
            i1 -= i2
        def tst2(i1, i2):
            i1.scaledMinus(1.0, i2)
        def tst3(i1, i2):
            i1 += i2
        def tst4(i1, i2):
            i1.scaledPlus(1.0, i2)
        def tst5(i1, i2):
            i1 *= i2
        def tst6(i1, i2):
            i1.scaledMultiplies(1.0, i2)
        def tst7(i1, i2):
            i1 /= i2
        def tst8(i1, i2):
            i1.scaledDivides(1.0, i2)

        tsts12 = [tst1, tst3, tst5, tst7]
        for tst in tsts12:
            self.assertRaises(lsst.pex.exceptions.LengthError, tst, i1, i2)

        tsts21 = [tst2, tst4, tst6, tst8]
        for tst in tsts21:
            self.assertRaises(lsst.pex.exceptions.LengthError, tst, i2, i1)

            
    def testMultiplyImages(self):
        """Test multiplication"""
        # Multiply by a MaskedImage
        self.mimage2 *= self.mimage

        self.assertEqual(self.mimage2.get(0, 0),
                         (self.imgVal2*self.imgVal1, self.EDGE,
                          self.varVal2*pow(self.imgVal1,2) + self.varVal1*pow(self.imgVal2, 2)))

        # Divide a MaskedImage<int> by an Image<int>; this divides the variance Image<float>
        # by an Image<int> in C++
        mimage_i = afwImage.MaskedImageI(self.mimage2.getDimensions())
        mimage_i.set(900, 0x0, 1000.0)
        image_i = afwImage.ImageI(mimage_i.getDimensions(), 2)

        mimage_i *= image_i

        self.assertEqual(mimage_i.get(0, 0), (1800, 0x0, 4000.0))

        # multiply by a scalar
        self.mimage *= self.imgVal1
        
        self.assertEqual(self.mimage.get(0, 0),
                         (self.imgVal1*self.imgVal1, self.EDGE, self.varVal1*pow(self.imgVal1, 2)))

        self.assertEqual(self.mimage.getMask().get(1, 1), self.EDGE)
        self.assertEqual(self.mimage.getMask().get(2, 2), 0x0)

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

        self.assertEqual(self.mimage2.get(0, 0), mimage2_copy.get(0, 0))

    def testDivideImages(self):
        """Test division"""
        # Divide by a MaskedImage
        mimage2_copy = self.mimage2.Factory(self.mimage2, True) # make a copy
        mimage2_copy /= self.mimage

        self.assertEqual(mimage2_copy.getImage().get(0, 0), self.imgVal2/self.imgVal1)
        self.assertEqual(mimage2_copy.getMask().get(0, 0), self.EDGE)
        self.assertAlmostEqual(mimage2_copy.getVariance().get(0, 0),
                               (self.varVal2*pow(self.imgVal1,2) +
                                self.varVal1*pow(self.imgVal2, 2))/pow(self.imgVal1, 4), 10)
        # Divide by an Image (of the same type as MaskedImage.getImage())
        mimage = self.mimage2.Factory(self.mimage2, True)
        mimage /= mimage.getImage()

        self.assertEqual(mimage.get(0, 0), (self.imgVal2/self.imgVal2, 0x0, self.varVal2))

        # Divide by an Image (of a different type from MaskedImage.getImage())
        if False:                       # this isn't supported from python (it's OK in C++)
            mimage = self.mimage2.Factory(self.mimage2, True)
            image = afwImage.ImageI(mimage.getDimensions(), 1)
            mimage /= image
            
            self.assertEqual(mimage.get(0, 0), (self.imgVal2, 0x0, self.varVal2))

        # Divide a MaskedImage<int> by an Image<int>; this divides the variance Image<float>
        # by an Image<int> in C++
        mimage_i = afwImage.MaskedImageI(self.mimage2.getDimensions())
        mimage_i.set(900, 0x0, 1000.0)
        image_i = afwImage.ImageI(mimage_i.getDimensions(), 2)

        mimage_i /= image_i

        self.assertEqual(mimage_i.get(0, 0), (450, 0x0, 250.0))

        # divide by a scalar
        self.mimage /= self.imgVal1
        
        self.assertEqual(self.mimage.getImage().get(0, 0), self.imgVal1/self.imgVal1)
        self.assertEqual(self.mimage.getMask().get(0, 0), self.EDGE)
        self.assertAlmostEqual(self.mimage.getVariance().get(0, 0), self.varVal1/pow(self.imgVal1, 2), 9)

        self.assertEqual(self.mimage.getMask().get(1, 1), self.EDGE)
        self.assertEqual(self.mimage.getMask().get(2, 2), 0x0)
        
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

        self.assertEqual(self.mimage2.get(0, 0), mimage2_copy.get(0, 0))

    def testCopyConstructors(self):
        dimage = afwImage.MaskedImageF(self.mimage, True) # deep copy
        simage = afwImage.MaskedImageF(self.mimage) # shallow copy
        
        self.mimage += 2                # should only change dimage
        self.assertEqual(dimage.getImage().get(0, 0), self.imgVal1)
        self.assertEqual(simage.getImage().get(0, 0), self.imgVal1 + 2)

    def checkImgPatch12(self, img, x0, y0):
        """Check that a patch of an image is correct; origin of patch is at (x0, y0) in full image
        N.b. This isn't a general routine!  Works only for testSubimages[12]"""
        
        self.assertEqual(img.get(x0 - 1, y0 - 1), (self.imgVal1, self.EDGE, self.varVal1))
        self.assertEqual(img.get(x0,     y0),     (666,          self.BAD,  0))
        self.assertEqual(img.get(x0 + 3, y0),     (self.imgVal1, 0x0,       self.varVal1))
        self.assertEqual(img.get(x0,     y0 + 1), (666,          self.BAD,  0))
        self.assertEqual(img.get(x0 + 3, y0 + 1), (self.imgVal1, 0x0,       self.varVal1))
        self.assertEqual(img.get(x0,     y0 + 2), (self.imgVal1, 0x0,       self.varVal1))

    def testOrigin(self):
        """Check that we can set and read the origin"""

        im = afwImage.MaskedImageF(afwGeom.ExtentI(10, 20))
        x0 = y0 = 0
        
        self.assertEqual(im.getX0(), x0)
        self.assertEqual(im.getY0(), y0)
        self.assertEqual(im.getXY0(), afwGeom.PointI(x0, y0))

        x0, y0 = 3, 5
        im.setXY0(x0, y0)
        self.assertEqual(im.getX0(), x0)
        self.assertEqual(im.getY0(), y0)
        self.assertEqual(im.getXY0(), afwGeom.PointI(x0, y0))

        x0, y0 = 30, 50
        im.setXY0(afwGeom.Point2I(x0, y0))
        self.assertEqual(im.getX0(), x0)
        self.assertEqual(im.getY0(), y0)
        self.assertEqual(im.getXY0(), afwGeom.Point2I(x0, y0))

    def testSubimages1(self):
        smimage = afwImage.MaskedImageF(
            self.mimage,
            afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.Extent2I(10, 5)),
            afwImage.LOCAL
            )
        
        simage = afwImage.MaskedImageF(
            smimage,
            afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.Extent2I(3, 2)),
            afwImage.LOCAL
            )
        self.assertEqual(simage.getX0(), 2)
        self.assertEqual(simage.getY0(), 2) # i.e. wrt self.mimage

        mimage2 = afwImage.MaskedImageF(simage.getDimensions())
        mimage2.getImage().set(666)
        mimage2.getMask().set(self.BAD)
        simage <<= mimage2

        del simage
        del mimage2

        self.checkImgPatch12(self.mimage, 2, 2)
        self.checkImgPatch12(smimage, 1, 1)

    def testSubimages2(self):
        """Test subimages when we've played with the (x0, y0) value"""

        self.mimage.set(9, 4, (888, 0x0, 0))
        #printImg(afwImage.ImageF(self.mimage, afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(10, 5)))); print

        smimage = afwImage.MaskedImageF(
            self.mimage, 
            afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.Extent2I(10, 5)),
            afwImage.LOCAL
            )
        smimage.setXY0(afwGeom.Point2I(0, 0)) # reset origin; doesn't affect pixel coordinate systems

        simage = afwImage.MaskedImageF(
            smimage, afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.Extent2I(3, 2)),
            afwImage.LOCAL
            )
        self.assertEqual(simage.getX0(), 1)
        self.assertEqual(simage.getY0(), 1)

        mimage2 = afwImage.MaskedImageF(simage.getDimensions())
        mimage2.set(666, self.BAD, 0.0)
        simage <<= mimage2
        del simage
        del mimage2
        
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

        self.mimage.getImage().set(20, 20, 200)
        self.mimage.getMask().set(20, 20, 0xf)

        for deep in (True, False):
            mimage = self.mimage.Factory(
                self.mimage,
                afwGeom.Box2I(afwGeom.Point2I(10, 10), afwGeom.Extent2I(64, 64)), 
                afwImage.LOCAL,
                deep)
            mimage.setXY0(afwGeom.Point2I(0, 0))
            mimage2 = mimage.Factory(mimage)
            
            if display:
                ds9.mtv(mimage2)
                
            self.checkImgPatch3(mimage2, deep)

    def testSetCopiedMask(self):
        """Check that we can set the Mask with a copied Mask"""
        
        crMask = self.mimage.getMask().Factory(self.mimage.getMask(), True)
        msk = self.mimage.getMask()
        msk |= crMask
        del msk

    def testVariance(self):
        """Check that we can set the variance from the gain"""
        gain = 2

        var = self.mimage.getVariance()
        var <<= self.mimage.getImage()
        var /= gain

    def testTicket653(self):
        """How-to-repeat for #653"""
        # The original ticket read this file, but it doesn't reproduce for me,
        # As I don't see how reading an exposure from disk could make a difference
        # it's easier to just build an Image
        if False:
            im = afwImage.ImageF(os.path.join(eups.productDir("afwdata"), "med_img.fits"))
        else:
            im = afwImage.ImageF(afwGeom.Extent2I(10, 10))
        mi = afwImage.MaskedImageF(im)
        exp = afwImage.ExposureF(mi)

    def testMaskedImageInitialisation(self):
        dims = self.mimage.getDimensions()
        factory = self.mimage.Factory

        self.mimage.set(666)

        del self.mimage                 # tempt C++ to reuse the memory
        self.mimage = factory(dims)
        self.assertEqual(self.mimage.get(10, 10), (0, 0x0, 0))

        del self.mimage
        self.mimage = factory(afwGeom.Extent2I(20, 20))
        self.assertEqual(self.mimage.get(10, 10), (0, 0x0, 0))

    def testImageSlices(self):
        """Test image slicing, which generate sub-images using Box2I under the covers"""
        im = afwImage.MaskedImageF(10, 20)
        im[4,10] = (10, 0x2, 100)
        im[-3:, -2:] = 100
        sim = im[1:4, 6:10]
        nan = -666                      #  a real NaN != NaN so tests fail
        sim[:] = (-1, 0x8, nan)
        im[0:4, 0:4] = im[2:6, 8:12]

        if display:
            ds9.mtv(im)

        self.assertEqual(im.get(0,  6), ( 0, 0x0,   0))
        self.assertEqual(im.get(6, 17), ( 0, 0x0,   0))
        self.assertEqual(im.get(7, 18), (100,0x0,   0))
        self.assertEqual(im.get(9, 19), (100,0x0,   0))
        self.assertEqual(im.get(1,  6), (-1, 0x8, nan))
        self.assertEqual(im.get(3,  9), (-1, 0x8, nan))
        self.assertEqual(im.get(4, 10), (10, 0x2, 100))
        self.assertEqual(im.get(4,  9), ( 0, 0x0,   0))
        self.assertEqual(im.get(2,  2), (10, 0x2, 100))
        self.assertEqual(im.get(0,  0), (-1, 0x8, nan))

    def testConversionToScalar(self):
        """Test that even 1-pixel MaskedImages can't be converted to scalars"""
        im = afwImage.MaskedImageF(10, 20)

        self.assertRaises(TypeError, float, im) # only single pixel images may be converted
        self.assertRaises(TypeError, float, im[0,0]) # actually, can't convert (img, msk, var) to scalar

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

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

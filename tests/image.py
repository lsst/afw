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
Tests for Images

Run with:
   ./Image.py
or
   python
   >>> import Image; Image.run()
"""

import os

import sys
import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.daf.base
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
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
        self.val1, self.val2 = 10, 100
        self.image1 = afwImage.ImageF(100, 200)
        self.image1.set(self.val1)
        self.image2 = afwImage.ImageF(self.image1.getDimensions())
        self.image2.set(self.val2)
        self.function = afwMath.PolynomialFunction2D(2)
        self.function.setParameters(range(self.function.getNParameters()))

    def tearDown(self):
        del self.image1
        del self.image2
        del self.function

    def testInitializeImages(self):
        val = 666
        for ctor in (afwImage.ImageU, afwImage.ImageI, afwImage.ImageF, afwImage.ImageD):
            im = ctor(10, 10, val)
            self.assertEqual(im.get(0, 0), val)

            im2 = ctor(afwImage.pairIntInt(10, 10), val)
            self.assertEqual(im2.get(0, 0), val)

    def testSetGetImages(self):
        self.assertEqual(self.image1.get(0, 0), self.val1)
    
    def testAddImages(self):
        self.image2 += self.image1
        self.image1 += self.val1
        
        self.assertEqual(self.image1.get(0, 0), 2*self.val1)
        self.assertEqual(self.image2.get(0, 0), self.val1 + self.val2)

        self.image1.set(self.val1)
        self.image1 += self.function

        for j in range(self.image1.getHeight()):
            for i in range(self.image1.getWidth()):
                self.assertEqual(self.image1.get(i, j), self.val1 + self.function(i, j))

    def testBoundsChecking(self):
        """Check that pixel indexes are checked in python"""
        tsts = []
        def tst():
            self.image1.get(-1, 0)
        tsts.append(tst)

        def tst():
            self.image1.get(0, -1)
        tsts.append(tst)

        def tst():
            self.image1.get(self.image1.getWidth(), 0)
        tsts.append(tst)

        def tst():
            self.image1.get(0, self.image1.getHeight())
        tsts.append(tst)

        for tst in tsts:
            utilsTests.assertRaisesLsstCpp(self, lsst.pex.exceptions.LengthErrorException, tst)

    def testAddScaledImages(self):
        c = 10.0
        self.image1.scaledPlus(c, self.image2)
        
        self.assertEqual(self.image1.get(0, 0), self.val1 + c*self.val2)
    
    def testSubtractImages(self):
        self.image2 -= self.image1
        self.image1 -= self.val1
        
        self.assertEqual(self.image1.get(0, 0), 0)
        self.assertEqual(self.image2.get(0, 0), self.val2 - self.val1)

        self.image1.set(self.val1)
        self.image1 -= self.function

        for j in range(self.image1.getHeight()):
            for i in range(self.image1.getWidth()):
                self.assertEqual(self.image1.get(i, j), self.val1 - self.function(i, j))
    
    def testArithmeticImagesMismatch(self):
        "Test arithmetic operations on Images of different sizes"
        i1 = afwImage.ImageF(100, 100)
        i1.set(100)
        i2 = afwImage.ImageF(10, 10)
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
            utilsTests.assertRaisesLsstCpp(self, lsst.pex.exceptions.LengthErrorException, tst, i1, i2)

        tsts21 = [tst2, tst4, tst6, tst8]
        for tst in tsts21:
            utilsTests.assertRaisesLsstCpp(self, lsst.pex.exceptions.LengthErrorException, tst, i2, i1)

        
    def testSubtractScaledImages(self):
        c = 10.0
        self.image1.scaledMinus(c, self.image2)
        
        self.assertEqual(self.image1.get(0, 0), self.val1 - c*self.val2)
    
    def testMultiplyImages(self):
        self.image2 *= self.image1
        self.image1 *= self.val1
        
        self.assertEqual(self.image1.get(0, 0), self.val1*self.val1)
        self.assertEqual(self.image2.get(0, 0), self.val2*self.val1)
    
    def testMultiplesScaledImages(self):
        c = 10.0
        self.image1.scaledMultiplies(c, self.image2)
        
        self.assertEqual(self.image1.get(0, 0), self.val1 * c*self.val2)
    
    def testDivideImages(self):
        self.image2 /= self.image1
        self.image1 /= self.val1
        
        self.assertEqual(self.image1.get(0, 0), 1)
        self.assertEqual(self.image2.get(0, 0), self.val2/self.val1)
    
    def testDividesScaledImages(self):
        c = 10.0
        self.image1.scaledDivides(c, self.image2)
        
        self.assertAlmostEqual(self.image1.get(0, 0), self.val1/(c*self.val2))
    
    def testCopyConstructors(self):
        dimage = afwImage.ImageF(self.image1, True) # deep copy
        simage = afwImage.ImageF(self.image1) # shallow copy
        
        self.image1 += 2                # should only change dimage
        self.assertEqual(dimage.get(0, 0), self.val1)
        self.assertEqual(simage.get(0, 0), self.val1 + 2)

    def testGeneralisedCopyConstructors(self):
        imageU = self.image1.convertU() # these are generalised (templated) copy constructors in C++
        imageF = imageU.convertF()

        self.assertEqual(imageU.get(0, 0), self.val1)
        self.assertEqual(imageF.get(0, 0), self.val1)
            
    def testPoint(self):
        """Test PointD and PointI"""
        x, y = 10, 20
        for point in (afwImage.PointD, afwImage.PointI):
            p = point(x, y)
            
            self.assertEqual(x, p.getX())
            self.assertEqual(x, p[0])

            self.assertEqual(y, p.getY())
            self.assertEqual(y, p[1])

            def tst1(): p[-1]
            self.assertRaises(IndexError, tst1)

            def tst2(): p[2]
            self.assertRaises(IndexError, tst2)
        
    def testBBox(self):
        x0, y0, width, height = 1, 2, 10, 20
        x1, y1 = x0 + width - 1, y0 + height - 1
        llc = afwImage.PointI(x0, y0)
        
        bbox = afwImage.BBox()
        self.assertEqual(bbox.getWidth(), 0)
        self.assertEqual(bbox.getHeight(), 0)

        bbox = afwImage.BBox(llc)
        self.assertEqual(bbox.getX0(), x0)
        self.assertEqual(bbox.getY0(), y0)
        self.assertEqual(bbox.getX1(), bbox.getX0())
        self.assertEqual(bbox.getY1(), bbox.getY0())
        self.assertEqual(bbox.getWidth(), 1)
        self.assertEqual(bbox.getHeight(), 1)

        bbox = afwImage.BBox(llc, width, height)
        self.assertEqual(bbox.getX0(), x0)
        self.assertEqual(bbox.getY0(), y0)
        self.assertEqual(bbox.getX1(), x1)
        self.assertEqual(bbox.getY1(), y1)
        self.assertEqual(bbox.getWidth(), width)
        self.assertEqual(bbox.getHeight(), height)

        urc = afwImage.PointI(x1, y1)
        bbox2 = afwImage.BBox(llc, urc)
        self.assertEqual(bbox, bbox2)
        
        bbox2 = afwImage.BBox(llc, width, height+1)
        self.assertNotEqual(bbox, bbox2)

        bbox = afwImage.BBox()
        point = afwImage.PointI(1, 1)
        
        bbox.grow(point)

        self.assert_(bbox.contains(point))
        #
        # Test changing the corners
        #
        bbox = afwImage.BBox(llc, width, height)

        bbox.setX0(x0 - 1) 
        self.assertEqual(bbox.getX0(), x0 - 1)
        self.assertEqual(bbox.getX1(), x1)
        bbox.setX1(x1 + 1) 
        self.assertEqual(bbox.getX1(), x1 + 1)

        bbox.setY0(y0 - 1) 
        self.assertEqual(bbox.getY0(), y0 - 1)
        self.assertEqual(bbox.getY1(), y1)
        bbox.setY1(y1 + 1) 
        self.assertEqual(bbox.getY1(), y1 + 1)
        #
        # Test clipping a BBox
        #
        bbox = afwImage.BBox(llc, width, height)
        cbox = afwImage.BBox(llc, width - 1, height - 1)

        bbox2 = bbox.clip(cbox)

        self.assertEqual(bbox.getX0(), x0)
        self.assertEqual(bbox.getY0(), y0)
        self.assertEqual(bbox.getX1(), x1 - 1)
        self.assertEqual(bbox.getY1(), y1 - 1)
        self.assertEqual(bbox.getWidth(), width - 1)
        self.assertEqual(bbox.getHeight(), height - 1)

        bbox = afwImage.BBox(llc, width, height)
        cbox = afwImage.BBox(afwImage.PointI(x0 + 1, y0 + 2), width + 10, height + 10)
        bbox2 = bbox.clip(cbox)

        self.assertEqual(bbox.getX0(), x0 + 1)
        self.assertEqual(bbox.getY0(), y0 + 2)
        self.assertEqual(bbox.getX1(), x1)
        self.assertEqual(bbox.getY1(), y1)
        self.assertEqual(bbox.getWidth(), width - 1)
        self.assertEqual(bbox.getHeight(), height - 2)

        bbox = afwImage.BBox(llc, width, height)
        cbox = afwImage.BBox(afwImage.PointI(x0 - 1, y0 - 2), width + 10, height + 20)
        bbox2 = bbox.clip(cbox)

        self.assertEqual(bbox.getX0(), x0)
        self.assertEqual(bbox.getY0(), y0)
        self.assertEqual(bbox.getX1(), x1)
        self.assertEqual(bbox.getY1(), y1)
        self.assertEqual(bbox.getWidth(), width)
        self.assertEqual(bbox.getHeight(), height)

    def testBCircle(self):
        """Check BCircle"""

        x0, y0, r = 10, 20, 3.5
        bc = afwImage.BCircle(afwImage.PointI(x0, y0), r)
        self.assertEqual(bc.getCenter().getX(), x0)
        self.assertEqual(bc.getCenter().getY(), y0)
        self.assertEqual(bc.getRadius(), r)

        ir = int(r + 0.5)
        self.assertEqual(bc.getBBox().getLLC(), afwImage.PointI(x0 - ir, y0 - ir))
        self.assertEqual(bc.getBBox().getDimensions()[0], 2*ir + 1)
        self.assertEqual(bc.getBBox().getDimensions()[1], 2*ir + 1)

    def checkImgPatch(self, img, x0=0, y0=0):
        """Check that a patch of an image is correct; origin of patch is at (x0, y0)"""
        
        self.assertEqual(img.get(x0 - 1, y0 - 1), self.val1)
        self.assertEqual(img.get(x0,     y0),     666)
        self.assertEqual(img.get(x0 + 3, y0),     self.val1)
        self.assertEqual(img.get(x0,     y0 + 1), 666)
        self.assertEqual(img.get(x0 + 3, y0 + 1), self.val1)
        self.assertEqual(img.get(x0,     y0 + 2), self.val1)

    def testOrigin(self):
        """Check that we can set and read the origin"""

        im = afwImage.ImageF(10, 20)
        x0 = y0 = 0
        
        self.assertEqual(im.getX0(), x0)
        self.assertEqual(im.getY0(), y0)
        self.assertEqual(im.getXY0(), afwImage.PointI(x0, y0))

        x0, y0 = 3, 5
        im.setXY0(x0, y0)
        self.assertEqual(im.getX0(), x0)
        self.assertEqual(im.getY0(), y0)
        self.assertEqual(im.getXY0(), afwImage.PointI(x0, y0))

        x0, y0 = 30, 50
        im.setXY0(afwImage.PointI(x0, y0))
        self.assertEqual(im.getX0(), x0)
        self.assertEqual(im.getY0(), y0)
        self.assertEqual(im.getXY0(), afwImage.PointI(x0, y0))

    def testSubimages(self):
        simage1 = afwImage.ImageF(self.image1, afwImage.BBox(afwImage.PointI(1, 1), 10, 5))
        
        simage = afwImage.ImageF(simage1, afwImage.BBox(afwImage.PointI(1, 1), 3, 2))
        self.assertEqual(simage.getX0(), 2)
        self.assertEqual(simage.getY0(), 2) # i.e. wrt self.image1

        image2 = afwImage.ImageF(simage.getDimensions())
        image2.set(666)
        simage <<= image2
        del simage
        del image2

        self.checkImgPatch(self.image1, 2, 2)
        self.checkImgPatch(simage1, 1, 1)

    def testSubimages2(self):
        """Test subimages when we've played with the (x0, y0) value"""

        self.image1.set(9, 4, 888)
        #printImg(afwImage.ImageF(self.image1, afwImage.BBox(afwImage.PointI(0, 0), 10, 5))); print

        simage1 = afwImage.ImageF(self.image1, afwImage.BBox(afwImage.PointI(1, 1), 10, 5))
        simage1.setXY0(afwImage.PointI(0, 0)) # reset origin; doesn't affect pixel coordinate systems

        simage = afwImage.ImageF(simage1, afwImage.BBox(afwImage.PointI(1, 1), 3, 2))
        self.assertEqual(simage.getX0(), 1)
        self.assertEqual(simage.getY0(), 1)

        image2 = afwImage.ImageF(simage.getDimensions())
        image2.set(666)
        simage <<= image2
        del simage
        del image2
        
        self.checkImgPatch(self.image1, 2, 2)
        self.checkImgPatch(simage1, 1, 1)

    def testBadSubimages(self):
        def tst():
            simage1 = afwImage.ImageF(self.image1, afwImage.BBox(afwImage.PointI(1, -1), 10, 5))

        utilsTests.assertRaisesLsstCpp(self, lsst.pex.exceptions.LengthErrorException, tst)

    def testImageInitialisation(self):
        dims = self.image1.getDimensions()
        factory = self.image1.Factory

        self.image1.set(666)

        del self.image1                 # tempt C++ to reuse the memory
        self.image1 = factory(dims)
        self.assertEqual(self.image1.get(10, 10), 0)

        del self.image1
        self.image1 = factory(20, 20)
        self.assertEqual(self.image1.get(10, 10), 0)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class DecoratedImageTestCase(unittest.TestCase):
    """A test case for DecoratedImage"""
    def setUp(self):
        self.val1, self.val2 = 10, 100
        self.width, self.height = 200, 100
        self.dimage1 = afwImage.DecoratedImageF(self.width, self.height)
        self.dimage1.getImage().set(self.val1)

        dataDir = eups.productDir("afwdata")
        if dataDir:
            self.fileForMetadata = os.path.join(dataDir, "small_MI_img.fits")
            self.trueMetadata = {"RELHUMID" : 10.69}
        else:
            self.fileForMetadata = None

    def tearDown(self):
        del self.dimage1

    def testCreateDecoratedImage(self):
        self.assertEqual(self.dimage1.getWidth(), self.width)
        self.assertEqual(self.dimage1.getHeight(), self.height)
        self.assertEqual(self.dimage1.getImage().get(0, 0), self.val1)

    def testCreateDecoratedImageFromImage(self):
        image = afwImage.ImageF(self.width, self.height)
        image <<= self.dimage1.getImage()

        dimage = afwImage.DecoratedImageF(image)
        self.assertEqual(dimage.getWidth(), self.width)
        self.assertEqual(dimage.getHeight(), self.height)
        self.assertEqual(dimage.getImage().get(0, 0), self.val1)
    
    def testCopyConstructors(self):
        dimage = afwImage.DecoratedImageF(self.dimage1, True) # deep copy
        self.dimage1.getImage().set(0, 0, 1 + 2*self.val1)
        self.assertEqual(dimage.getImage().get(0, 0), self.val1)

        dimage = afwImage.DecoratedImageF(self.dimage1) # shallow copy
        self.dimage1.getImage().set(0, 0, 1 + 2*self.val1)
        self.assertNotEqual(dimage.getImage().get(0, 0), self.val1)

    def testReadFits(self):
        """Test reading FITS files"""
        
        dataDir = eups.productDir("afwdata")
        if not dataDir:
            print >> sys.stderr, "Warning: afwdata is not set up; not running the FITS I/O tests"
            return
        
        files = {}
        files["img"] = os.path.join(dataDir, "small_MI_img.fits") # an S16 fits file
        files["msk"] = os.path.join(dataDir, "small_MI_msk.fits") # an U8 fits file
        files["var"] = os.path.join(dataDir, "small_MI_var.fits") # an F32 fits file

        imgU = afwImage.DecoratedImageF(files["img"]) # read as unsigned short
        imgF = afwImage.DecoratedImageF(files["img"]) # read as float

        self.assertEqual(imgU.getHeight(), 256)
        self.assertEqual(imgF.getImage().getWidth(), 256)
        self.assertEqual(imgU.getImage().get(0, 0), imgF.getImage().get(0, 0))
        #
        # Check the metadata
        #
        meta = self.trueMetadata
        for k in meta.keys():
            self.assertEqual(imgU.getMetadata().getAsDouble(k), meta[k])
            self.assertEqual(imgF.getMetadata().getAsDouble(k), meta[k])
        #
        # Read an F32 image
        #
        varU = afwImage.DecoratedImageF(files["var"]) # read as unsigned short
        varF = afwImage.DecoratedImageF(files["var"]) # read as float

        self.assertEqual(varU.getHeight(), 256)
        self.assertEqual(varF.getImage().getWidth(), 256)
        self.assertEqual(varU.getImage().get(0, 0), varF.getImage().get(0, 0))
        #
        # Read a char image
        #
        maskImg = afwImage.DecoratedImageU(files["msk"]).getImage() # read a char file

        self.assertEqual(maskImg.getHeight(), 256)
        self.assertEqual(maskImg.getWidth(), 256)
        self.assertEqual(maskImg.get(0, 0), 1)
        #
        # Read a U16 image
        #
        tmpFile = "foo.fits"

        imgU.writeFits(tmpFile)

        try:
            imgU16 = afwImage.DecoratedImageF(tmpFile) # read as unsigned short
        except:
            os.remove(tmpFile)
            raise

        os.remove(tmpFile)

    def testWriteFits(self):
        """Test writing FITS files"""

        tmpFile = "foo.fits"

        if self.fileForMetadata:
            imgU = afwImage.DecoratedImageF(self.fileForMetadata)
        else:
            print >> sys.stderr, "Warning: afwdata is not set up; not running the FITS metadata I/O tests"
            imgU = afwImage.DecoratedImageF()

        self.dimage1.writeFits(tmpFile, imgU.getMetadata())
        #
        # Read it back
        #
        rimage = afwImage.DecoratedImageF(tmpFile)
        os.remove(tmpFile)

        self.assertEqual(self.dimage1.getImage().get(0, 0), rimage.getImage().get(0, 0))
        #
        # Check that we wrote (and read) the metadata successfully
        if self.fileForMetadata:
            meta = self.trueMetadata
            for k in meta.keys():
                self.assertEqual(rimage.getMetadata().getAsDouble(k), meta[k])

    def testReadWriteXY0(self):
        """Test that we read and write (X0, Y0) correctly"""
        im = afwImage.ImageF(10, 20)

        x0, y0 = 1, 2
        im.setXY0(x0, y0)
        tmpFile = "foo.fits"
        im.writeFits(tmpFile)

        im2 = im.Factory(tmpFile)
        os.remove(tmpFile)

        self.assertEqual(im2.getX0(), x0)
        self.assertEqual(im2.getY0(), y0)

    def testReadMetadata(self):
        if self.fileForMetadata:
            im = afwImage.DecoratedImageF(self.fileForMetadata)
        else:
            print >> sys.stderr, "Warning: afwdata is not set up; not running the FITS metadata I/O tests"
            return

        meta = afwImage.readMetadata(self.fileForMetadata)
        self.assertTrue("NAXIS1" in meta.names())
        self.assertEqual(im.getWidth(), meta.get("NAXIS1"))
        self.assertEqual(im.getHeight(), meta.get("NAXIS2"))

    def testTicket1040(self):
        """ How to repeat from #1040"""
        image        = afwImage.ImageD(6, 6)
        image.set(2, 2, 100)

        bbox    = afwImage.BBox(afwImage.PointI(1, 1), 5, 5)
        subImage = image.Factory(image, bbox)
        subImageF = subImage.convertFloat()
        
        if display:
            ds9.mtv(subImage, frame=0, title="subImage")
            ds9.mtv(subImageF, frame=1, title="converted subImage")

        self.assertEqual(subImage.get(1, 1), subImageF.get(1, 1))
        
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
    suites += unittest.makeSuite(ImageTestCase)
    suites += unittest.makeSuite(DecoratedImageTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

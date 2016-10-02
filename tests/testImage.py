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
#pybind11#Tests for Images
#pybind11#
#pybind11#Run with:
#pybind11#   ./Image.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import Image; Image.run()
#pybind11#"""
#pybind11#
#pybind11#import os.path
#pybind11#import shutil
#pybind11#import sys
#pybind11#import tempfile
#pybind11#import unittest
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.daf.base
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#
#pybind11#try:
#pybind11#    afwdataDir = lsst.utils.getPackageDir("afwdata")
#pybind11#except pexExcept.NotFoundError:
#pybind11#    afwdataDir = None
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#def makeRampImage(width, height, imgClass=afwImage.ImageF):
#pybind11#    """Make a ramp image of the specified size and image class
#pybind11#
#pybind11#    Values start from 0 at the lower left corner and increase by 1 along rows
#pybind11#    """
#pybind11#    im = imgClass(width, height)
#pybind11#    val = 0
#pybind11#    for yInd in range(height):
#pybind11#        for xInd in range(width):
#pybind11#            im.set(xInd, yInd, val)
#pybind11#            val += 1
#pybind11#    return im
#pybind11#
#pybind11#
#pybind11#class ImageTestCase(lsst.utils.tests.TestCase):
#pybind11#    """A test case for Image"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        numpy.random.seed(1)
#pybind11#        self.val1, self.val2 = 10, 100
#pybind11#        self.image1 = afwImage.ImageF(afwGeom.ExtentI(100, 200))
#pybind11#        self.image1.set(self.val1)
#pybind11#        self.image2 = afwImage.ImageF(self.image1.getDimensions())
#pybind11#        self.image2.set(self.val2)
#pybind11#        self.function = afwMath.PolynomialFunction2D(2)
#pybind11#        self.function.setParameters(list(range(self.function.getNParameters())))
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.image1
#pybind11#        del self.image2
#pybind11#        del self.function
#pybind11#
#pybind11#    def testArrays(self):
#pybind11#        for cls in (afwImage.ImageU, afwImage.ImageI, afwImage.ImageF, afwImage.ImageD):
#pybind11#            image1 = cls(afwGeom.Extent2I(5, 6))
#pybind11#            array1 = image1.getArray()
#pybind11#            self.assertEqual(array1.shape[0], image1.getHeight())
#pybind11#            self.assertEqual(array1.shape[1], image1.getWidth())
#pybind11#            image2 = cls(array1, False)
#pybind11#            self.assertEqual(array1.shape[0], image2.getHeight())
#pybind11#            self.assertEqual(array1.shape[1], image2.getWidth())
#pybind11#            image3 = afwImage.makeImageFromArray(array1)
#pybind11#            self.assertEqual(array1.shape[0], image2.getHeight())
#pybind11#            self.assertEqual(array1.shape[1], image2.getWidth())
#pybind11#            self.assertEqual(type(image3), cls)
#pybind11#            array1[:, :] = numpy.random.uniform(low=0, high=10, size=array1.shape)
#pybind11#            for j in range(image1.getHeight()):
#pybind11#                for i in range(image1.getWidth()):
#pybind11#                    self.assertEqual(image1.get(i, j), array1[j, i])
#pybind11#                    self.assertEqual(image2.get(i, j), array1[j, i])
#pybind11#
#pybind11#    def testInitializeImages(self):
#pybind11#        val = 666
#pybind11#        for ctor in (afwImage.ImageU, afwImage.ImageI, afwImage.ImageF, afwImage.ImageD):
#pybind11#            im = ctor(10, 10, val)
#pybind11#            self.assertEqual(im.get(0, 0), val)
#pybind11#
#pybind11#            im2 = ctor(afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(10, 10)), val)
#pybind11#            self.assertEqual(im2.get(0, 0), val)
#pybind11#
#pybind11#    def testSetGetImages(self):
#pybind11#        self.assertEqual(self.image1.get(0, 0), self.val1)
#pybind11#
#pybind11#    def testGetSet0Images(self):
#pybind11#        self.assertEqual(self.image1.get0(0, 0), self.val1)
#pybind11#        self.image1.setXY0(3, 4)
#pybind11#        self.assertEqual(self.image1.get0(3, 4), self.val1)
#pybind11#
#pybind11#        def f1():
#pybind11#            return self.image1.get0(0, 0)
#pybind11#        self.assertRaises(lsst.pex.exceptions.LengthError, f1)
#pybind11#        self.image1.set(0, 0, 42.)
#pybind11#        self.assertEqual(self.image1.get0(3, 4), 42.)
#pybind11#        self.image1.set0(3, 4, self.val1)
#pybind11#        self.assertEqual(self.image1.get0(3, 4), self.val1)
#pybind11#        self.assertEqual(self.image1.get(0, 0), self.val1)
#pybind11#
#pybind11#    def testAllocateLargeImages(self):
#pybind11#        """Try to allocate a Very large image"""
#pybind11#        bbox = afwGeom.BoxI(afwGeom.PointI(-1 << 30, -1 << 30), afwGeom.PointI(1 << 30, 1 << 30))
#pybind11#
#pybind11#        def tst():
#pybind11#            afwImage.ImageF(bbox)
#pybind11#
#pybind11#        self.assertRaises(lsst.pex.exceptions.LengthError, tst)
#pybind11#
#pybind11#    def testAddImages(self):
#pybind11#        self.image2 += self.image1
#pybind11#        self.image1 += self.val1
#pybind11#
#pybind11#        self.assertEqual(self.image1.get(0, 0), 2*self.val1)
#pybind11#        self.assertEqual(self.image2.get(0, 0), self.val1 + self.val2)
#pybind11#
#pybind11#        self.image1.set(self.val1)
#pybind11#        self.image1 += self.function
#pybind11#
#pybind11#        for j in range(self.image1.getHeight()):
#pybind11#            for i in range(self.image1.getWidth()):
#pybind11#                self.assertEqual(self.image1.get(i, j), self.val1 + self.function(i, j))
#pybind11#
#pybind11#    def testAssignWithBBox(self):
#pybind11#        """Test assign(rhs, bbox) with non-empty bbox
#pybind11#        """
#pybind11#        for xy0 in (afwGeom.Point2I(*val) for val in (
#pybind11#            (0, 0),
#pybind11#            (-100, 120),  # an arbitrary value that is off the image
#pybind11#        )):
#pybind11#            destImDim = afwGeom.Extent2I(5, 4)
#pybind11#            srcImDim = afwGeom.Extent2I(3, 2)
#pybind11#            destIm = afwImage.ImageF(destImDim)
#pybind11#            destIm.setXY0(xy0)
#pybind11#            srcIm = makeRampImage(*srcImDim)
#pybind11#            srcIm.setXY0(55, -33)  # an arbitrary value that should be ignored
#pybind11#            self.assertRaises(Exception, destIm.set, srcIm)  # size mismatch
#pybind11#
#pybind11#            for validMin in (afwGeom.Point2I(*val) for val in (
#pybind11#                (0, 0),
#pybind11#                (2, 0),
#pybind11#                (0, 1),
#pybind11#                (1, 2),
#pybind11#            )):
#pybind11#                for origin in (None, afwImage.PARENT, afwImage.LOCAL):  # None to omit the argument
#pybind11#                    destIm[:] = -1.0
#pybind11#                    bbox = afwGeom.Box2I(validMin, srcIm.getDimensions())
#pybind11#                    if origin != afwImage.LOCAL:
#pybind11#                        bbox.shift(afwGeom.Extent2I(xy0))
#pybind11#                    if origin is None:
#pybind11#                        destIm.assign(srcIm, bbox)
#pybind11#                        destImView = afwImage.ImageF(destIm, bbox)
#pybind11#                    else:
#pybind11#                        destIm.assign(srcIm, bbox, origin)
#pybind11#                        destImView = afwImage.ImageF(destIm, bbox, origin)
#pybind11#                    self.assertFloatsEqual(destImView.getArray(), srcIm.getArray())
#pybind11#                    numPixNotAssigned = (destImDim[0] * destImDim[1]) - (srcImDim[0] * srcImDim[1])
#pybind11#                    self.assertEqual(numpy.sum(destIm.getArray() < -0.5), numPixNotAssigned)
#pybind11#
#pybind11#            for badMin in (afwGeom.Point2I(*val) + afwGeom.Extent2I(xy0) for val in (
#pybind11#                (-1, 0),
#pybind11#                (3, 0),
#pybind11#                (0, -1),
#pybind11#                (1, 3),
#pybind11#            )):
#pybind11#                for origin in (None, afwImage.PARENT, afwImage.LOCAL):  # None to omit the argument
#pybind11#                    bbox = afwGeom.Box2I(badMin, srcIm.getDimensions())
#pybind11#                    if origin != afwImage.LOCAL:
#pybind11#                        bbox.shift(afwGeom.Extent2I(xy0))
#pybind11#                    if origin is None:
#pybind11#                        self.assertRaises(Exception, destIm.set, srcIm, bbox)
#pybind11#                    else:
#pybind11#                        self.assertRaises(Exception, destIm.set, srcIm, bbox, origin)
#pybind11#
#pybind11#    def testAssignWithoutBBox(self):
#pybind11#        """Test assign(rhs, [bbox]) with an empty bbox and with no bbox specified; both set all pixels
#pybind11#        """
#pybind11#        for xy0 in (afwGeom.Point2I(*val) for val in (
#pybind11#            (0, 0),
#pybind11#            (-100, 120),  # an arbitrary value that is off the image
#pybind11#        )):
#pybind11#            destImDim = afwGeom.Extent2I(5, 4)
#pybind11#            destIm = afwImage.ImageF(destImDim)
#pybind11#            destIm.setXY0(xy0)
#pybind11#            srcIm = makeRampImage(*destImDim)
#pybind11#            srcIm.setXY0(55, -33)  # an arbitrary value that should be ignored
#pybind11#
#pybind11#            destIm[:] = -1.0
#pybind11#            destIm.assign(srcIm)
#pybind11#            self.assertFloatsEqual(destIm.getArray(), srcIm.getArray())
#pybind11#
#pybind11#            destIm[:] = -1.0
#pybind11#            destIm.assign(srcIm, afwGeom.Box2I())
#pybind11#            self.assertFloatsEqual(destIm.getArray(), srcIm.getArray())
#pybind11#
#pybind11#    def testBoundsChecking(self):
#pybind11#        """Check that pixel indexes are checked in python"""
#pybind11#        tsts = []
#pybind11#
#pybind11#        def tst():
#pybind11#            self.image1.get(-1, 0)
#pybind11#        tsts.append(tst)
#pybind11#
#pybind11#        def tst():
#pybind11#            self.image1.get(0, -1)
#pybind11#        tsts.append(tst)
#pybind11#
#pybind11#        def tst():
#pybind11#            self.image1.get(self.image1.getWidth(), 0)
#pybind11#        tsts.append(tst)
#pybind11#
#pybind11#        def tst():
#pybind11#            self.image1.get(0, self.image1.getHeight())
#pybind11#        tsts.append(tst)
#pybind11#
#pybind11#        for tst in tsts:
#pybind11#            self.assertRaises(lsst.pex.exceptions.LengthError, tst)
#pybind11#
#pybind11#    def testAddScaledImages(self):
#pybind11#        c = 10.0
#pybind11#        self.image1.scaledPlus(c, self.image2)
#pybind11#
#pybind11#        self.assertEqual(self.image1.get(0, 0), self.val1 + c*self.val2)
#pybind11#
#pybind11#    def testSubtractImages(self):
#pybind11#        self.image2 -= self.image1
#pybind11#        self.image1 -= self.val1
#pybind11#
#pybind11#        self.assertEqual(self.image1.get(0, 0), 0)
#pybind11#        self.assertEqual(self.image2.get(0, 0), self.val2 - self.val1)
#pybind11#
#pybind11#        self.image1.set(self.val1)
#pybind11#        self.image1 -= self.function
#pybind11#
#pybind11#        for j in range(self.image1.getHeight()):
#pybind11#            for i in range(self.image1.getWidth()):
#pybind11#                self.assertEqual(self.image1.get(i, j), self.val1 - self.function(i, j))
#pybind11#
#pybind11#    def testArithmeticImagesMismatch(self):
#pybind11#        "Test arithmetic operations on Images of different sizes"
#pybind11#        i1 = afwImage.ImageF(100, 100)
#pybind11#        i1.set(100)
#pybind11#        i2 = afwImage.ImageF(10, 10)
#pybind11#        i2.set(10)
#pybind11#
#pybind11#        def tst1(i1, i2):
#pybind11#            i1 -= i2
#pybind11#
#pybind11#        def tst2(i1, i2):
#pybind11#            i1.scaledMinus(1.0, i2)
#pybind11#
#pybind11#        def tst3(i1, i2):
#pybind11#            i1 += i2
#pybind11#
#pybind11#        def tst4(i1, i2):
#pybind11#            i1.scaledPlus(1.0, i2)
#pybind11#
#pybind11#        def tst5(i1, i2):
#pybind11#            i1 *= i2
#pybind11#
#pybind11#        def tst6(i1, i2):
#pybind11#            i1.scaledMultiplies(1.0, i2)
#pybind11#
#pybind11#        def tst7(i1, i2):
#pybind11#            i1 /= i2
#pybind11#
#pybind11#        def tst8(i1, i2):
#pybind11#            i1.scaledDivides(1.0, i2)
#pybind11#
#pybind11#        tsts12 = [tst1, tst3, tst5, tst7]
#pybind11#        for tst in tsts12:
#pybind11#            self.assertRaises(lsst.pex.exceptions.LengthError, tst, i1, i2)
#pybind11#
#pybind11#        tsts21 = [tst2, tst4, tst6, tst8]
#pybind11#        for tst in tsts21:
#pybind11#            self.assertRaises(lsst.pex.exceptions.LengthError, tst, i2, i1)
#pybind11#
#pybind11#    def testSubtractScaledImages(self):
#pybind11#        c = 10.0
#pybind11#        self.image1.scaledMinus(c, self.image2)
#pybind11#
#pybind11#        self.assertEqual(self.image1.get(0, 0), self.val1 - c*self.val2)
#pybind11#
#pybind11#    def testMultiplyImages(self):
#pybind11#        self.image2 *= self.image1
#pybind11#        self.image1 *= self.val1
#pybind11#
#pybind11#        self.assertEqual(self.image1.get(0, 0), self.val1*self.val1)
#pybind11#        self.assertEqual(self.image2.get(0, 0), self.val2*self.val1)
#pybind11#
#pybind11#    def testMultiplesScaledImages(self):
#pybind11#        c = 10.0
#pybind11#        self.image1.scaledMultiplies(c, self.image2)
#pybind11#
#pybind11#        self.assertEqual(self.image1.get(0, 0), self.val1 * c*self.val2)
#pybind11#
#pybind11#    def testDivideImages(self):
#pybind11#        self.image2 /= self.image1
#pybind11#        self.image1 /= self.val1
#pybind11#
#pybind11#        self.assertEqual(self.image1.get(0, 0), 1)
#pybind11#        self.assertEqual(self.image2.get(0, 0), self.val2/self.val1)
#pybind11#
#pybind11#    def testDividesScaledImages(self):
#pybind11#        c = 10.0
#pybind11#        self.image1.scaledDivides(c, self.image2)
#pybind11#
#pybind11#        self.assertAlmostEqual(self.image1.get(0, 0), self.val1/(c*self.val2))
#pybind11#
#pybind11#    def testCopyConstructors(self):
#pybind11#        dimage = afwImage.ImageF(self.image1, True)  # deep copy
#pybind11#        simage = afwImage.ImageF(self.image1)  # shallow copy
#pybind11#
#pybind11#        self.image1 += 2                # should only change dimage
#pybind11#        self.assertEqual(dimage.get(0, 0), self.val1)
#pybind11#        self.assertEqual(simage.get(0, 0), self.val1 + 2)
#pybind11#
#pybind11#    def testGeneralisedCopyConstructors(self):
#pybind11#        imageU = self.image1.convertU()  # these are generalised (templated) copy constructors in C++
#pybind11#        imageF = imageU.convertF()
#pybind11#        imageD = imageF.convertD()
#pybind11#
#pybind11#        self.assertEqual(imageU.get(0, 0), self.val1)
#pybind11#        self.assertEqual(imageF.get(0, 0), self.val1)
#pybind11#        self.assertEqual(imageD.get(0, 0), self.val1)
#pybind11#
#pybind11#    def checkImgPatch(self, img, x0=0, y0=0):
#pybind11#        """Check that a patch of an image is correct; origin of patch is at (x0, y0)"""
#pybind11#
#pybind11#        self.assertEqual(img.get(x0 - 1, y0 - 1), self.val1)
#pybind11#        self.assertEqual(img.get(x0, y0), 666)
#pybind11#        self.assertEqual(img.get(x0 + 3, y0), self.val1)
#pybind11#        self.assertEqual(img.get(x0, y0 + 1), 666)
#pybind11#        self.assertEqual(img.get(x0 + 3, y0 + 1), self.val1)
#pybind11#        self.assertEqual(img.get(x0, y0 + 2), self.val1)
#pybind11#
#pybind11#    def testOrigin(self):
#pybind11#        """Check that we can set and read the origin"""
#pybind11#
#pybind11#        im = afwImage.ImageF(10, 20)
#pybind11#        x0 = y0 = 0
#pybind11#
#pybind11#        self.assertEqual(im.getX0(), x0)
#pybind11#        self.assertEqual(im.getY0(), y0)
#pybind11#        self.assertEqual(im.getXY0(), afwGeom.Point2I(x0, y0))
#pybind11#
#pybind11#        x0, y0 = 3, 5
#pybind11#        im.setXY0(x0, y0)
#pybind11#        self.assertEqual(im.getX0(), x0)
#pybind11#        self.assertEqual(im.getY0(), y0)
#pybind11#        self.assertEqual(im.getXY0(), afwGeom.Point2I(x0, y0))
#pybind11#
#pybind11#        x0, y0 = 30, 50
#pybind11#        im.setXY0(afwGeom.Point2I(x0, y0))
#pybind11#        self.assertEqual(im.getX0(), x0)
#pybind11#        self.assertEqual(im.getY0(), y0)
#pybind11#        self.assertEqual(im.getXY0(), afwGeom.Point2I(x0, y0))
#pybind11#
#pybind11#    def testSubimages(self):
#pybind11#        simage1 = afwImage.ImageF(
#pybind11#            self.image1,
#pybind11#            afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.Extent2I(10, 5)),
#pybind11#            afwImage.LOCAL)
#pybind11#
#pybind11#        simage = afwImage.ImageF(
#pybind11#            simage1,
#pybind11#            afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.Extent2I(3, 2)),
#pybind11#            afwImage.LOCAL
#pybind11#        )
#pybind11#        self.assertEqual(simage.getX0(), 2)
#pybind11#        self.assertEqual(simage.getY0(), 2)  # i.e. wrt self.image1
#pybind11#
#pybind11#        image2 = afwImage.ImageF(simage.getDimensions())
#pybind11#        image2.set(666)
#pybind11#        simage[:] = image2
#pybind11#        del simage
#pybind11#        del image2
#pybind11#
#pybind11#        self.checkImgPatch(self.image1, 2, 2)
#pybind11#        self.checkImgPatch(simage1, 1, 1)
#pybind11#
#pybind11#    def testSubimages2(self):
#pybind11#        """Test subimages when we've played with the (x0, y0) value"""
#pybind11#
#pybind11#        self.image1.set(9, 4, 888)
#pybind11#        #printImg(afwImage.ImageF(self.image1, afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(10, 5)))); print
#pybind11#
#pybind11#        simage1 = afwImage.ImageF(
#pybind11#            self.image1,
#pybind11#            afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.Extent2I(10, 5)),
#pybind11#            afwImage.LOCAL
#pybind11#        )
#pybind11#        simage1.setXY0(afwGeom.Point2I(0, 0))  # reset origin; doesn't affect pixel coordinate systems
#pybind11#
#pybind11#        simage = afwImage.ImageF(
#pybind11#            simage1,
#pybind11#            afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.Extent2I(3, 2)),
#pybind11#            afwImage.LOCAL
#pybind11#        )
#pybind11#        self.assertEqual(simage.getX0(), 1)
#pybind11#        self.assertEqual(simage.getY0(), 1)
#pybind11#
#pybind11#        image2 = afwImage.ImageF(simage.getDimensions())
#pybind11#        image2.set(666)
#pybind11#        simage[:] = image2
#pybind11#        del simage
#pybind11#        del image2
#pybind11#
#pybind11#        self.checkImgPatch(self.image1, 2, 2)
#pybind11#        self.checkImgPatch(simage1, 1, 1)
#pybind11#
#pybind11#    def testBadSubimages(self):
#pybind11#        def tst():
#pybind11#            afwImage.ImageF(
#pybind11#                self.image1,
#pybind11#                afwGeom.Box2I(afwGeom.Point2I(1, -1), afwGeom.Extent2I(10, 5)),
#pybind11#                afwImage.LOCAL
#pybind11#            )
#pybind11#
#pybind11#        self.assertRaises(lsst.pex.exceptions.LengthError, tst)
#pybind11#
#pybind11#    def testImageInitialisation(self):
#pybind11#        dims = self.image1.getDimensions()
#pybind11#        factory = self.image1.Factory
#pybind11#
#pybind11#        self.image1.set(666)
#pybind11#
#pybind11#        del self.image1                 # tempt C++ to reuse the memory
#pybind11#        self.image1 = factory(dims)
#pybind11#        self.assertEqual(self.image1.get(10, 10), 0)
#pybind11#
#pybind11#    def testImageSlices(self):
#pybind11#        """Test image slicing, which generate sub-images using Box2I under the covers"""
#pybind11#        im = afwImage.ImageF(10, 20)
#pybind11#        im[-1, :] = -5
#pybind11#        im[..., 18] = -5              # equivalent to im[:, 18]
#pybind11#        im[4, 10] = 10
#pybind11#        im[-3:, -2:] = 100
#pybind11#        im[-2, -2] = -10
#pybind11#        sim = im[1:4, 6:10]
#pybind11#        sim[:] = -1
#pybind11#        im[0:4, 0:4] = im[2:6, 8:12]
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(im)
#pybind11#
#pybind11#        self.assertEqual(im.get(0, 6), 0)
#pybind11#        self.assertEqual(im.get(9, 15), -5)
#pybind11#        self.assertEqual(im.get(5, 18), -5)
#pybind11#        self.assertEqual(im.get(6, 17), 0)
#pybind11#        self.assertEqual(im.get(7, 18), 100)
#pybind11#        self.assertEqual(im.get(9, 19), 100)
#pybind11#        self.assertEqual(im.get(8, 18), -10)
#pybind11#        self.assertEqual(im.get(1, 6), -1)
#pybind11#        self.assertEqual(im.get(3, 9), -1)
#pybind11#        self.assertEqual(im.get(4, 10), 10)
#pybind11#        self.assertEqual(im.get(4, 9), 0)
#pybind11#        self.assertEqual(im.get(2, 2), 10)
#pybind11#        self.assertEqual(im.get(0, 0), -1)
#pybind11#
#pybind11#    def testImageSliceFromBox(self):
#pybind11#        """Test using a Box2I to index an Image"""
#pybind11#        im = afwImage.ImageF(10, 20)
#pybind11#        bbox = afwGeom.BoxI(afwGeom.PointI(1, 3), afwGeom.PointI(6, 9))
#pybind11#        im[bbox] = -1
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(im)
#pybind11#
#pybind11#        self.assertEqual(im.get(0, 6), 0)
#pybind11#        self.assertEqual(im.get(1, 6), -1)
#pybind11#        self.assertEqual(im.get(3, 9), -1)
#pybind11#
#pybind11#    def testConversionToScalar(self):
#pybind11#        """Test that 1-pixel images can be converted to scalars"""
#pybind11#        self.assertEqual(int(afwImage.ImageI(1, 1)), 0.0)
#pybind11#        self.assertEqual(float(afwImage.ImageI(1, 1)), 0.0)
#pybind11#
#pybind11#        im = afwImage.ImageF(10, 20)
#pybind11#        im.set(666)
#pybind11#
#pybind11#        self.assertEqual(float(im[0, 0]), 666)
#pybind11#        self.assertEqual(int(im[0, 0]), 666)
#pybind11#
#pybind11#        self.assertRaises(TypeError, int, im)  # only single pixel images may be converted
#pybind11#        self.assertRaises(TypeError, float, im)  # only single pixel images may be converted
#pybind11#
#pybind11#    def testClone(self):
#pybind11#        """Test that clone works properly"""
#pybind11#        im = afwImage.ImageF(10, 20)
#pybind11#        im[0, 0] = 100
#pybind11#
#pybind11#        im2 = im.clone()                # check that clone with no arguments makes a deep copy
#pybind11#        self.assertEqual(im.getDimensions(), im2.getDimensions())
#pybind11#        self.assertEqual(im.get(0, 0), im2.get(0, 0))
#pybind11#        im2[0, 0] += 100
#pybind11#        self.assertNotEqual(im.get(0, 0), im2.get(0, 0))  # so it's a deep copy
#pybind11#
#pybind11#        im2 = im[0:3, 0:5].clone()  # check that we can slice-then-clone
#pybind11#        self.assertEqual(im2.getDimensions(), afwGeom.ExtentI(3, 5))
#pybind11#        self.assertEqual(im.get(0, 0), im2.get(0, 0))
#pybind11#        im2[0, 0] += 10
#pybind11#        self.assertNotEqual(float(im[0, 0]), float(im2[0, 0]))  # equivalent to im.get(0, 0) etc.
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class DecoratedImageTestCase(lsst.utils.tests.TestCase):
#pybind11#    """A test case for DecoratedImage"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        numpy.random.seed(1)
#pybind11#        self.val1, self.val2 = 10, 100
#pybind11#        self.width, self.height = 200, 100
#pybind11#        self.dimage1 = afwImage.DecoratedImageF(
#pybind11#            afwGeom.Extent2I(self.width, self.height)
#pybind11#        )
#pybind11#        self.dimage1.getImage().set(self.val1)
#pybind11#
#pybind11#        if afwdataDir is not None:
#pybind11#            self.fileForMetadata = os.path.join(afwdataDir, "data", "small_MI.fits")
#pybind11#            self.trueMetadata = {"RELHUMID": 10.69}
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.dimage1
#pybind11#
#pybind11#    def testCreateDecoratedImage(self):
#pybind11#        self.assertEqual(self.dimage1.getWidth(), self.width)
#pybind11#        self.assertEqual(self.dimage1.getHeight(), self.height)
#pybind11#        self.assertEqual(self.dimage1.getImage().get(0, 0), self.val1)
#pybind11#
#pybind11#    def testCreateDecoratedImageFromImage(self):
#pybind11#        image = afwImage.ImageF(afwGeom.Extent2I(self.width, self.height))
#pybind11#        image[:] = self.dimage1.getImage()
#pybind11#
#pybind11#        dimage = afwImage.DecoratedImageF(image)
#pybind11#        self.assertEqual(dimage.getWidth(), self.width)
#pybind11#        self.assertEqual(dimage.getHeight(), self.height)
#pybind11#        self.assertEqual(dimage.getImage().get(0, 0), self.val1)
#pybind11#
#pybind11#    def testCopyConstructors(self):
#pybind11#        dimage = afwImage.DecoratedImageF(self.dimage1, True)  # deep copy
#pybind11#        self.dimage1.getImage().set(0, 0, 1 + 2*self.val1)
#pybind11#        self.assertEqual(dimage.getImage().get(0, 0), self.val1)
#pybind11#
#pybind11#        dimage = afwImage.DecoratedImageF(self.dimage1)  # shallow copy
#pybind11#        self.dimage1.getImage().set(0, 0, 1 + 2*self.val1)
#pybind11#        self.assertNotEqual(dimage.getImage().get(0, 0), self.val1)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testReadFits(self):
#pybind11#        """Test reading FITS files"""
#pybind11#
#pybind11#        hdus = {}
#pybind11#        hdus["img"] = 2  # an S16 fits HDU
#pybind11#        hdus["msk"] = 3  # an U8 fits HDU
#pybind11#        hdus["var"] = 4  # an F32 fits HDU
#pybind11#
#pybind11#        imgU = afwImage.DecoratedImageU(self.fileForMetadata, hdus["img"])  # read as unsigned short
#pybind11#        imgF = afwImage.DecoratedImageF(self.fileForMetadata, hdus["img"])  # read as float
#pybind11#
#pybind11#        self.assertEqual(imgU.getHeight(), 256)
#pybind11#        self.assertEqual(imgF.getImage().getWidth(), 256)
#pybind11#        self.assertEqual(imgU.getImage().get(0, 0), imgF.getImage().get(0, 0))
#pybind11#        #
#pybind11#        # Check the metadata
#pybind11#        #
#pybind11#        meta = self.trueMetadata
#pybind11#        for k in meta.keys():
#pybind11#            self.assertEqual(imgU.getMetadata().getAsDouble(k), meta[k])
#pybind11#            self.assertEqual(imgF.getMetadata().getAsDouble(k), meta[k])
#pybind11#        #
#pybind11#        # Read an F32 image
#pybind11#        #
#pybind11#        varU = afwImage.DecoratedImageF(self.fileForMetadata, hdus["var"])  # read as unsigned short
#pybind11#        varF = afwImage.DecoratedImageF(self.fileForMetadata, hdus["var"])  # read as float
#pybind11#
#pybind11#        self.assertEqual(varU.getHeight(), 256)
#pybind11#        self.assertEqual(varF.getImage().getWidth(), 256)
#pybind11#        self.assertEqual(varU.getImage().get(0, 0), varF.getImage().get(0, 0))
#pybind11#        #
#pybind11#        # Read a char image
#pybind11#        #
#pybind11#        maskImg = afwImage.DecoratedImageU(self.fileForMetadata, hdus["msk"]).getImage()  # read a char file
#pybind11#
#pybind11#        self.assertEqual(maskImg.getHeight(), 256)
#pybind11#        self.assertEqual(maskImg.getWidth(), 256)
#pybind11#        self.assertEqual(maskImg.get(0, 0), 1)
#pybind11#        #
#pybind11#        # Read a U16 image
#pybind11#        #
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            imgU.writeFits(tmpFile)
#pybind11#
#pybind11#            afwImage.DecoratedImageF(tmpFile)  # read as unsigned short
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testWriteFits(self):
#pybind11#        """Test writing FITS files"""
#pybind11#
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            if self.fileForMetadata:
#pybind11#                imgU = afwImage.DecoratedImageF(self.fileForMetadata)
#pybind11#            else:
#pybind11#                imgU = afwImage.DecoratedImageF()
#pybind11#
#pybind11#            self.dimage1.writeFits(tmpFile, imgU.getMetadata())
#pybind11#            #
#pybind11#            # Read it back
#pybind11#            #
#pybind11#            rimage = afwImage.DecoratedImageF(tmpFile)
#pybind11#
#pybind11#            self.assertEqual(self.dimage1.getImage().get(0, 0), rimage.getImage().get(0, 0))
#pybind11#            #
#pybind11#            # Check that we wrote (and read) the metadata successfully
#pybind11#            if self.fileForMetadata:
#pybind11#                meta = self.trueMetadata
#pybind11#                for k in meta.keys():
#pybind11#                    self.assertEqual(rimage.getMetadata().getAsDouble(k), meta[k])
#pybind11#
#pybind11#    def testReadWriteXY0(self):
#pybind11#        """Test that we read and write (X0, Y0) correctly"""
#pybind11#        im = afwImage.ImageF(afwGeom.Extent2I(10, 20))
#pybind11#
#pybind11#        x0, y0 = 1, 2
#pybind11#        im.setXY0(x0, y0)
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            im.writeFits(tmpFile)
#pybind11#
#pybind11#            im2 = im.Factory(tmpFile)
#pybind11#
#pybind11#            self.assertEqual(im2.getX0(), x0)
#pybind11#            self.assertEqual(im2.getY0(), y0)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testReadMetadata(self):
#pybind11#        im = afwImage.DecoratedImageF(self.fileForMetadata)
#pybind11#
#pybind11#        meta = afwImage.readMetadata(self.fileForMetadata)
#pybind11#        self.assertIn("NAXIS1", meta.names())
#pybind11#        self.assertEqual(im.getWidth(), meta.get("NAXIS1"))
#pybind11#        self.assertEqual(im.getHeight(), meta.get("NAXIS2"))
#pybind11#
#pybind11#    def testTicket1040(self):
#pybind11#        """ How to repeat from #1040"""
#pybind11#        image = afwImage.ImageD(afwGeom.Extent2I(6, 6))
#pybind11#        image.set(2, 2, 100)
#pybind11#
#pybind11#        bbox = afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.Extent2I(5, 5))
#pybind11#        subImage = image.Factory(image, bbox)
#pybind11#        subImageF = subImage.convertFloat()
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(subImage, frame=0, title="subImage")
#pybind11#            ds9.mtv(subImageF, frame=1, title="converted subImage")
#pybind11#
#pybind11#        self.assertEqual(subImage.get(1, 1), subImageF.get(1, 1))
#pybind11#
#pybind11#    def testDM882(self):
#pybind11#        """Test that we can write a dotted header unit to a FITS file. See DM-882."""
#pybind11#        self.dimage1.getMetadata().add("A.B.C.D", 12345)
#pybind11#        tempdir = tempfile.mkdtemp()
#pybind11#        testfile = os.path.join(tempdir, "test.fits")
#pybind11#        try:
#pybind11#            self.dimage1.writeFits(testfile)
#pybind11#            meta = afwImage.readMetadata(testfile)
#pybind11#            self.assertEqual(meta.get("A.B.C.D"), 12345)
#pybind11#        finally:
#pybind11#            shutil.rmtree(tempdir)
#pybind11#
#pybind11#    def testLargeImage(self):
#pybind11#        """Test that creating an extremely large image raises, rather than segfaulting. DM-89, -527."""
#pybind11#        for imtype in (afwImage.ImageD, afwImage.ImageF, afwImage.ImageI, afwImage.ImageU):
#pybind11#            self.assertRaises(lsst.pex.exceptions.LengthError, imtype, 60000, 60000)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#def printImg(img):
#pybind11#    print("%4s " % "", end=' ')
#pybind11#    for c in range(img.getWidth()):
#pybind11#        print("%7d" % c, end=' ')
#pybind11#    print()
#pybind11#
#pybind11#    for r in range(img.getHeight() - 1, -1, -1):
#pybind11#        print("%4d " % r, end=' ')
#pybind11#        for c in range(img.getWidth()):
#pybind11#            print("%7.1f" % float(img.get(c, r)), end=' ')
#pybind11#        print()
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

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
#pybind11#Tests for MaskedImages
#pybind11#
#pybind11#Run with:
#pybind11#   python MaskedImage.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import MaskedImage; MaskedImage.run()
#pybind11#"""
#pybind11#
#pybind11#import os
#pybind11#import unittest
#pybind11#
#pybind11#import numpy as np
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.daf.base
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11#
#pybind11#def makeRampImage(width, height, imgClass=afwImage.MaskedImageF):
#pybind11#    """Make a ramp image of the specified size and image class
#pybind11#
#pybind11#    Image values start from 0 at the lower left corner and increase by 1 along rows
#pybind11#    Variance values equal image values + 100
#pybind11#    Mask values equal image values modulo 8 bits (leaving plenty of unused values)
#pybind11#    """
#pybind11#    mi = imgClass(width, height)
#pybind11#    image = mi.getImage()
#pybind11#    mask = mi.getMask()
#pybind11#    variance = mi.getVariance()
#pybind11#    val = 0
#pybind11#    for yInd in range(height):
#pybind11#        for xInd in range(width):
#pybind11#            image.set(xInd, yInd, val)
#pybind11#            variance.set(xInd, yInd, val + 100)
#pybind11#            mask.set(xInd, yInd, val % 0x100)
#pybind11#            val += 1
#pybind11#    return mi
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class MaskedImageTestCase(lsst.utils.tests.TestCase):
#pybind11#    """A test case for MaskedImage"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.imgVal1, self.varVal1 = 100.0, 10.0
#pybind11#        self.imgVal2, self.varVal2 = 200.0, 15.0
#pybind11#        self.mimage = afwImage.MaskedImageF(100, 200)
#pybind11#
#pybind11#        self.mimage.getImage().set(self.imgVal1)
#pybind11#        #
#pybind11#        # Set center of mask to 0, with 2 pixel border set to EDGE
#pybind11#        #
#pybind11#        self.BAD = afwImage.MaskU_getPlaneBitMask("BAD")
#pybind11#        self.EDGE = afwImage.MaskU_getPlaneBitMask("EDGE")
#pybind11#
#pybind11#        self.mimage.getMask().set(self.EDGE)
#pybind11#        centre = afwImage.MaskU(
#pybind11#            self.mimage.getMask(),
#pybind11#            afwGeom.Box2I(afwGeom.Point2I(2, 2), self.mimage.getDimensions() - afwGeom.Extent2I(4)),
#pybind11#            afwImage.LOCAL)
#pybind11#        centre.set(0x0)
#pybind11#        #
#pybind11#        self.mimage.getVariance().set(self.varVal1)
#pybind11#        #
#pybind11#        # Second MaskedImage
#pybind11#        #
#pybind11#        self.mimage2 = afwImage.MaskedImageF(self.mimage.getDimensions())
#pybind11#        self.mimage2.getImage().set(self.imgVal2)
#pybind11#        self.mimage2.getVariance().set(self.varVal2)
#pybind11#        #
#pybind11#        # a Function2
#pybind11#        #
#pybind11#        self.function = afwMath.PolynomialFunction2D(2)
#pybind11#        self.function.setParameters(list(range(self.function.getNParameters())))
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.mimage
#pybind11#        del self.mimage2
#pybind11#        del self.function
#pybind11#
#pybind11#    def testArrays(self):
#pybind11#        """
#pybind11#        This method is testing that ``lsst.afw.image.MaskedImageF.getArrays()``
#pybind11#        returns the proper image, mask, and variance.
#pybind11#        """
#pybind11#        image, mask, variance = self.mimage.getArrays()
#pybind11#        self.assertFloatsEqual(self.mimage.getImage().getArray(), image)
#pybind11#        self.assertFloatsEqual(self.mimage.getMask().getArray(), mask)
#pybind11#        self.assertFloatsEqual(self.mimage.getVariance().getArray(), variance)
#pybind11#        mimage2 = afwImage.makeMaskedImageFromArrays(image, mask, variance)
#pybind11#        self.assertEqual(type(mimage2), type(self.mimage))
#pybind11#
#pybind11#    def testSetGetValues(self):
#pybind11#        self.assertEqual(self.mimage.get(0, 0), (self.imgVal1, self.EDGE, self.varVal1))
#pybind11#
#pybind11#        self.assertEqual(self.mimage.getMask().get(1, 1), self.EDGE)
#pybind11#        self.assertEqual(self.mimage.getMask().get(2, 2), 0x0)
#pybind11#
#pybind11#    def testMaskedImageFromImage(self):
#pybind11#        w, h = 10, 20
#pybind11#        dims = afwGeom.Extent2I(w, h)
#pybind11#        im, mask, var = afwImage.ImageF(dims), afwImage.MaskU(dims), afwImage.ImageF(dims)
#pybind11#        im.set(666)
#pybind11#
#pybind11#        maskedImage = afwImage.MaskedImageF(im, mask, var)
#pybind11#
#pybind11#        maskedImage = afwImage.makeMaskedImage(im, mask, var)
#pybind11#
#pybind11#        maskedImage = afwImage.MaskedImageF(im)
#pybind11#        self.assertEqual(im.getDimensions(), maskedImage.getImage().getDimensions())
#pybind11#        self.assertEqual(im.getDimensions(), maskedImage.getMask().getDimensions())
#pybind11#        self.assertEqual(im.getDimensions(), maskedImage.getVariance().getDimensions())
#pybind11#
#pybind11#        self.assertEqual(maskedImage.get(0, 0), (im.get(0, 0), 0x0, 0.0))
#pybind11#
#pybind11#    def testMakeMaskedImageXY0(self):
#pybind11#        """Test that makeMaskedImage sets XY0 correctly"""
#pybind11#        im = afwImage.ImageF(200, 300)
#pybind11#        xy0 = afwGeom.PointI(10, 20)
#pybind11#        im.setXY0(*xy0)
#pybind11#        mi = afwImage.makeMaskedImage(im)
#pybind11#
#pybind11#        self.assertEqual(mi.getImage().getXY0(), xy0)
#pybind11#        self.assertEqual(mi.getMask().getXY0(), xy0)
#pybind11#        self.assertEqual(mi.getVariance().getXY0(), xy0)
#pybind11#
#pybind11#    def testCopyMaskedImage(self):
#pybind11#        """Test copy constructor"""
#pybind11#        #
#pybind11#        # shallow copy
#pybind11#        #
#pybind11#        mi = self.mimage.Factory(self.mimage, False)
#pybind11#
#pybind11#        val00 = self.mimage.get(0, 0)
#pybind11#        nval00 = (100, 0xff, -1)        # the new value we'll set
#pybind11#        self.assertNotEqual(val00, nval00)
#pybind11#
#pybind11#        self.assertEqual(mi.get(0, 0), val00)
#pybind11#        mi.set(0, 0, nval00)
#pybind11#
#pybind11#        self.assertEqual(self.mimage.get(0, 0), nval00)
#pybind11#        self.assertEqual(mi.get(0, 0), nval00)
#pybind11#        mi.set(0, 0, val00)             # reinstate initial value
#pybind11#        #
#pybind11#        # deep copy
#pybind11#        #
#pybind11#        mi = self.mimage.Factory(self.mimage, True)
#pybind11#
#pybind11#        self.assertEqual(mi.get(0, 0), val00)
#pybind11#        mi.set(0, 0, nval00)
#pybind11#
#pybind11#        self.assertEqual(self.mimage.get(0, 0), val00)
#pybind11#        self.assertEqual(mi.get(0, 0), nval00)
#pybind11#        #
#pybind11#        # Copy with change of Image type
#pybind11#        #
#pybind11#        mi = self.mimage.convertD()
#pybind11#
#pybind11#        self.assertEqual(mi.get(0, 0), val00)
#pybind11#        mi.set(0, 0, nval00)
#pybind11#
#pybind11#        self.assertEqual(self.mimage.get(0, 0), val00)
#pybind11#        self.assertEqual(mi.get(0, 0), nval00)
#pybind11#        #
#pybind11#        # Convert from U to F
#pybind11#        #
#pybind11#        mi = afwImage.MaskedImageU(afwGeom.Extent2I(10, 20))
#pybind11#        val00 = (10, 0x10, 1)
#pybind11#        mi.set(val00)
#pybind11#        self.assertEqual(mi.get(0, 0), val00)
#pybind11#
#pybind11#        fmi = mi.convertF()
#pybind11#        self.assertEqual(fmi.get(0, 0), val00)
#pybind11#
#pybind11#    def testAddImages(self):
#pybind11#        "Test addition"
#pybind11#        # add an image
#pybind11#        self.mimage2 += self.mimage
#pybind11#
#pybind11#        self.assertEqual(self.mimage2.get(0, 0), (self.imgVal1 + self.imgVal2, self.EDGE,
#pybind11#                                                  self.varVal1 + self.varVal2))
#pybind11#
#pybind11#        # Add an Image<int> to a MaskedImage<int>
#pybind11#        mimage_i = afwImage.MaskedImageI(self.mimage2.getDimensions())
#pybind11#        mimage_i.set(900, 0x0, 1000.0)
#pybind11#        image_i = afwImage.ImageI(mimage_i.getDimensions(), 2)
#pybind11#
#pybind11#        mimage_i += image_i
#pybind11#
#pybind11#        self.assertEqual(mimage_i.get(0, 0), (902, 0x0, 1000.0))
#pybind11#
#pybind11#        # add a scalar
#pybind11#        self.mimage += self.imgVal1
#pybind11#
#pybind11#        self.assertEqual(self.mimage.get(0, 0), (2*self.imgVal1, self.EDGE, self.varVal1))
#pybind11#
#pybind11#        self.assertEqual(self.mimage.getMask().get(1, 1), self.EDGE)
#pybind11#        self.assertEqual(self.mimage.getMask().get(2, 2), 0x0)
#pybind11#
#pybind11#        # add a function
#pybind11#        self.mimage.set(self.imgVal1, 0x0, 0.0)
#pybind11#        self.mimage += self.function
#pybind11#
#pybind11#        for i, j in [(2, 3)]:
#pybind11#            self.assertEqual(self.mimage.getImage().get(i, j), self.imgVal1 + self.function(i, j))
#pybind11#
#pybind11#    def testAddScaledImages(self):
#pybind11#        "Test addition by a scaled MaskedImage"
#pybind11#        # add an image
#pybind11#        c = 10.0
#pybind11#        mimage2_copy = self.mimage2.Factory(self.mimage2, True)  # make a copy
#pybind11#        self.mimage2.scaledPlus(c, self.mimage)
#pybind11#        #
#pybind11#        # Now repeat calculation using a temporary
#pybind11#        #
#pybind11#        tmp = self.mimage.Factory(self.mimage, True)
#pybind11#        tmp *= c
#pybind11#        mimage2_copy += tmp
#pybind11#
#pybind11#        self.assertEqual(self.mimage2.get(0, 0), mimage2_copy.get(0, 0))
#pybind11#
#pybind11#    def testAssignWithBBox(self):
#pybind11#        """Test assign(rhs, bbox) with non-empty bbox
#pybind11#        """
#pybind11#        for xy0 in (afwGeom.Point2I(*val) for val in (
#pybind11#            (0, 0),
#pybind11#            (-100, 120),  # an arbitrary value that is off the image
#pybind11#        )):
#pybind11#            destMIDim = afwGeom.Extent2I(5, 4)
#pybind11#            srcMIDim = afwGeom.Extent2I(3, 2)
#pybind11#            destMI = afwImage.MaskedImageF(destMIDim)
#pybind11#            destImage = destMI.getImage()
#pybind11#            destVariance = destMI.getVariance()
#pybind11#            destMask = destMI.getMask()
#pybind11#            destMI.setXY0(xy0)
#pybind11#            srcMI = makeRampImage(*srcMIDim)
#pybind11#            srcMI.setXY0(55, -33)  # an arbitrary value that should be ignored
#pybind11#            self.assertRaises(Exception, destMI.set, srcMI)  # size mismatch
#pybind11#
#pybind11#            for validMin in (afwGeom.Point2I(*val) for val in (
#pybind11#                (0, 0),
#pybind11#                (2, 0),
#pybind11#                (0, 1),
#pybind11#                (1, 2),
#pybind11#            )):
#pybind11#                for origin in (None, afwImage.PARENT, afwImage.LOCAL):  # None to omit the argument
#pybind11#                    destImage[:] = -1.0
#pybind11#                    destVariance[:] = -1.0
#pybind11#                    destMask[:] = 0xFFFF
#pybind11#                    bbox = afwGeom.Box2I(validMin, srcMI.getDimensions())
#pybind11#                    if origin != afwImage.LOCAL:
#pybind11#                        bbox.shift(afwGeom.Extent2I(xy0))
#pybind11#                    if origin is None:
#pybind11#                        destMI.assign(srcMI, bbox)
#pybind11#                        destMIView = afwImage.MaskedImageF(destMI, bbox)
#pybind11#                    else:
#pybind11#                        destMI.assign(srcMI, bbox, origin)
#pybind11#                        destMIView = afwImage.MaskedImageF(destMI, bbox, origin)
#pybind11#                    for i in range(3):
#pybind11#                        self.assertListEqual(destMIView.getArrays()[i].flatten().tolist(),
#pybind11#                                             srcMI.getArrays()[i].flatten().tolist())
#pybind11#                    numPixNotAssigned = (destMIDim[0] * destMIDim[1]) - (srcMIDim[0] * srcMIDim[1])
#pybind11#                    self.assertEqual(np.sum(destImage.getArray() < -0.5), numPixNotAssigned)
#pybind11#                    self.assertEqual(np.sum(destVariance.getArray() < -0.5), numPixNotAssigned)
#pybind11#                    self.assertEqual(np.sum(destMask.getArray() == 0xFFFF), numPixNotAssigned)
#pybind11#
#pybind11#            for badMin in (afwGeom.Point2I(*val) + afwGeom.Extent2I(xy0) for val in (
#pybind11#                (-1, 0),
#pybind11#                (3, 0),
#pybind11#                (0, -1),
#pybind11#                (1, 3),
#pybind11#            )):
#pybind11#                for origin in (None, afwImage.PARENT, afwImage.LOCAL):  # None to omit the argument
#pybind11#                    bbox = afwGeom.Box2I(validMin, srcMI.getDimensions())
#pybind11#                    if origin != afwImage.LOCAL:
#pybind11#                        bbox.shift(afwGeom.Extent2I(xy0))
#pybind11#                    if origin is None:
#pybind11#                        self.assertRaises(Exception, destMI.set, srcMI, bbox)
#pybind11#                    else:
#pybind11#                        self.assertRaises(Exception, destMI.set, srcMI, bbox, origin)
#pybind11#
#pybind11#    def testAssignWithoutBBox(self):
#pybind11#        """Test assign(rhs, [bbox]) with an empty bbox and with no bbox specified; both set all pixels
#pybind11#        """
#pybind11#        for xy0 in (afwGeom.Point2I(*val) for val in (
#pybind11#            (0, 0),
#pybind11#            (-100, 120),  # an arbitrary value that is off the image
#pybind11#        )):
#pybind11#            destMIDim = afwGeom.Extent2I(5, 4)
#pybind11#            destMI = afwImage.MaskedImageF(destMIDim)
#pybind11#            destMI.setXY0(xy0)
#pybind11#            destImage = destMI.getImage()
#pybind11#            destVariance = destMI.getVariance()
#pybind11#            destMask = destMI.getMask()
#pybind11#            srcMI = makeRampImage(*destMIDim)
#pybind11#            srcMI.setXY0(55, -33)  # an arbitrary value that should be ignored
#pybind11#
#pybind11#            destImage[:] = -1.0
#pybind11#            destVariance[:] = -1.0
#pybind11#            destMask[:] = 0xFFFF
#pybind11#            destMI.assign(srcMI)
#pybind11#            for i in range(3):
#pybind11#                self.assertListEqual(destMI.getArrays()[i].flatten().tolist(),
#pybind11#                                     srcMI.getArrays()[i].flatten().tolist())
#pybind11#
#pybind11#            destImage[:] = -1.0
#pybind11#            destVariance[:] = -1.0
#pybind11#            destMask[:] = 0xFFFF
#pybind11#            destMI.assign(srcMI, afwGeom.Box2I())
#pybind11#            for i in range(3):
#pybind11#                self.assertListEqual(destMI.getArrays()[i].flatten().tolist(),
#pybind11#                                     srcMI.getArrays()[i].flatten().tolist())
#pybind11#
#pybind11#    def testSubtractImages(self):
#pybind11#        "Test subtraction"
#pybind11#        # subtract an image
#pybind11#        self.mimage2 -= self.mimage
#pybind11#        self.assertEqual(self.mimage2.get(0, 0),
#pybind11#                         (self.imgVal2 - self.imgVal1, self.EDGE, self.varVal2 + self.varVal1))
#pybind11#
#pybind11#        # Subtract an Image<int> from a MaskedImage<int>
#pybind11#        mimage_i = afwImage.MaskedImageI(self.mimage2.getDimensions())
#pybind11#        mimage_i.set(900, 0x0, 1000.0)
#pybind11#        image_i = afwImage.ImageI(mimage_i.getDimensions(), 2)
#pybind11#
#pybind11#        mimage_i -= image_i
#pybind11#
#pybind11#        self.assertEqual(mimage_i.get(0, 0), (898, 0x0, 1000.0))
#pybind11#
#pybind11#        # subtract a scalar
#pybind11#        self.mimage -= self.imgVal1
#pybind11#        self.assertEqual(self.mimage.get(0, 0), (0.0, self.EDGE, self.varVal1))
#pybind11#
#pybind11#    def testSubtractScaledImages(self):
#pybind11#        "Test subtraction by a scaled MaskedImage"
#pybind11#        # subtract a scaled image
#pybind11#        c = 10.0
#pybind11#        mimage2_copy = self.mimage2.Factory(self.mimage2, True)  # make a copy
#pybind11#        self.mimage2.scaledMinus(c, self.mimage)
#pybind11#        #
#pybind11#        # Now repeat calculation using a temporary
#pybind11#        #
#pybind11#        tmp = self.mimage.Factory(self.mimage, True)
#pybind11#        tmp *= c
#pybind11#        mimage2_copy -= tmp
#pybind11#
#pybind11#        self.assertEqual(self.mimage2.get(0, 0), mimage2_copy.get(0, 0))
#pybind11#
#pybind11#    def testArithmeticImagesMismatch(self):
#pybind11#        "Test arithmetic operations on MaskedImages of different sizes"
#pybind11#        i1 = afwImage.MaskedImageF(afwGeom.Extent2I(100, 100))
#pybind11#        i1.set(100)
#pybind11#        i2 = afwImage.MaskedImageF(afwGeom.Extent2I(10, 10))
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
#pybind11#    def testMultiplyImages(self):
#pybind11#        """Test multiplication"""
#pybind11#        # Multiply by a MaskedImage
#pybind11#        self.mimage2 *= self.mimage
#pybind11#
#pybind11#        self.assertEqual(self.mimage2.get(0, 0),
#pybind11#                         (self.imgVal2*self.imgVal1, self.EDGE,
#pybind11#                          self.varVal2*pow(self.imgVal1, 2) + self.varVal1*pow(self.imgVal2, 2)))
#pybind11#
#pybind11#        # Divide a MaskedImage<int> by an Image<int>; this divides the variance Image<float>
#pybind11#        # by an Image<int> in C++
#pybind11#        mimage_i = afwImage.MaskedImageI(self.mimage2.getDimensions())
#pybind11#        mimage_i.set(900, 0x0, 1000.0)
#pybind11#        image_i = afwImage.ImageI(mimage_i.getDimensions(), 2)
#pybind11#
#pybind11#        mimage_i *= image_i
#pybind11#
#pybind11#        self.assertEqual(mimage_i.get(0, 0), (1800, 0x0, 4000.0))
#pybind11#
#pybind11#        # multiply by a scalar
#pybind11#        self.mimage *= self.imgVal1
#pybind11#
#pybind11#        self.assertEqual(self.mimage.get(0, 0),
#pybind11#                         (self.imgVal1*self.imgVal1, self.EDGE, self.varVal1*pow(self.imgVal1, 2)))
#pybind11#
#pybind11#        self.assertEqual(self.mimage.getMask().get(1, 1), self.EDGE)
#pybind11#        self.assertEqual(self.mimage.getMask().get(2, 2), 0x0)
#pybind11#
#pybind11#    def testScaledMultiplyImages(self):
#pybind11#        """Test multiplication by a scaled image"""
#pybind11#        # Multiply by an image
#pybind11#        c = 10.0
#pybind11#        mimage2_copy = self.mimage2.Factory(self.mimage2, True)  # make a copy
#pybind11#        self.mimage2.scaledMultiplies(c, self.mimage)
#pybind11#        #
#pybind11#        # Now repeat calculation using a temporary
#pybind11#        #
#pybind11#        tmp = self.mimage.Factory(self.mimage, True)
#pybind11#        tmp *= c
#pybind11#        mimage2_copy *= tmp
#pybind11#
#pybind11#        self.assertEqual(self.mimage2.get(0, 0), mimage2_copy.get(0, 0))
#pybind11#
#pybind11#    def testDivideImages(self):
#pybind11#        """Test division"""
#pybind11#        # Divide by a MaskedImage
#pybind11#        mimage2_copy = self.mimage2.Factory(self.mimage2, True)  # make a copy
#pybind11#        mimage2_copy /= self.mimage
#pybind11#
#pybind11#        self.assertEqual(mimage2_copy.getImage().get(0, 0), self.imgVal2/self.imgVal1)
#pybind11#        self.assertEqual(mimage2_copy.getMask().get(0, 0), self.EDGE)
#pybind11#        self.assertAlmostEqual(mimage2_copy.getVariance().get(0, 0),
#pybind11#                               (self.varVal2*pow(self.imgVal1, 2) +
#pybind11#                                self.varVal1*pow(self.imgVal2, 2))/pow(self.imgVal1, 4), 10)
#pybind11#        # Divide by an Image (of the same type as MaskedImage.getImage())
#pybind11#        mimage = self.mimage2.Factory(self.mimage2, True)
#pybind11#        mimage /= mimage.getImage()
#pybind11#
#pybind11#        self.assertEqual(mimage.get(0, 0), (self.imgVal2/self.imgVal2, 0x0, self.varVal2))
#pybind11#
#pybind11#        # Divide by an Image (of a different type from MaskedImage.getImage())
#pybind11#        if False:                       # this isn't supported from python (it's OK in C++)
#pybind11#            mimage = self.mimage2.Factory(self.mimage2, True)
#pybind11#            image = afwImage.ImageI(mimage.getDimensions(), 1)
#pybind11#            mimage /= image
#pybind11#
#pybind11#            self.assertEqual(mimage.get(0, 0), (self.imgVal2, 0x0, self.varVal2))
#pybind11#
#pybind11#        # Divide a MaskedImage<int> by an Image<int>; this divides the variance Image<float>
#pybind11#        # by an Image<int> in C++
#pybind11#        mimage_i = afwImage.MaskedImageI(self.mimage2.getDimensions())
#pybind11#        mimage_i.set(900, 0x0, 1000.0)
#pybind11#        image_i = afwImage.ImageI(mimage_i.getDimensions(), 2)
#pybind11#
#pybind11#        mimage_i /= image_i
#pybind11#
#pybind11#        self.assertEqual(mimage_i.get(0, 0), (450, 0x0, 250.0))
#pybind11#
#pybind11#        # divide by a scalar
#pybind11#        self.mimage /= self.imgVal1
#pybind11#
#pybind11#        self.assertEqual(self.mimage.getImage().get(0, 0), self.imgVal1/self.imgVal1)
#pybind11#        self.assertEqual(self.mimage.getMask().get(0, 0), self.EDGE)
#pybind11#        self.assertAlmostEqual(self.mimage.getVariance().get(0, 0), self.varVal1/pow(self.imgVal1, 2), 9)
#pybind11#
#pybind11#        self.assertEqual(self.mimage.getMask().get(1, 1), self.EDGE)
#pybind11#        self.assertEqual(self.mimage.getMask().get(2, 2), 0x0)
#pybind11#
#pybind11#    def testScaledDivideImages(self):
#pybind11#        """Test division by a scaled image"""
#pybind11#        # Divide by an image
#pybind11#        c = 10.0
#pybind11#        mimage2_copy = self.mimage2.Factory(self.mimage2, True)  # make a copy
#pybind11#        self.mimage2.scaledDivides(c, self.mimage)
#pybind11#        #
#pybind11#        # Now repeat calculation using a temporary
#pybind11#        #
#pybind11#        tmp = self.mimage.Factory(self.mimage, True)
#pybind11#        tmp *= c
#pybind11#        mimage2_copy /= tmp
#pybind11#
#pybind11#        self.assertEqual(self.mimage2.get(0, 0), mimage2_copy.get(0, 0))
#pybind11#
#pybind11#    def testCopyConstructors(self):
#pybind11#        dimage = afwImage.MaskedImageF(self.mimage, True)  # deep copy
#pybind11#        simage = afwImage.MaskedImageF(self.mimage)  # shallow copy
#pybind11#
#pybind11#        self.mimage += 2                # should only change dimage
#pybind11#        self.assertEqual(dimage.getImage().get(0, 0), self.imgVal1)
#pybind11#        self.assertEqual(simage.getImage().get(0, 0), self.imgVal1 + 2)
#pybind11#
#pybind11#    def checkImgPatch12(self, img, x0, y0):
#pybind11#        """Check that a patch of an image is correct; origin of patch is at (x0, y0) in full image
#pybind11#        N.b. This isn't a general routine!  Works only for testSubimages[12]"""
#pybind11#
#pybind11#        self.assertEqual(img.get(x0 - 1, y0 - 1), (self.imgVal1, self.EDGE, self.varVal1))
#pybind11#        self.assertEqual(img.get(x0, y0), (666, self.BAD, 0))
#pybind11#        self.assertEqual(img.get(x0 + 3, y0), (self.imgVal1, 0x0, self.varVal1))
#pybind11#        self.assertEqual(img.get(x0, y0 + 1), (666, self.BAD, 0))
#pybind11#        self.assertEqual(img.get(x0 + 3, y0 + 1), (self.imgVal1, 0x0, self.varVal1))
#pybind11#        self.assertEqual(img.get(x0, y0 + 2), (self.imgVal1, 0x0, self.varVal1))
#pybind11#
#pybind11#    def testOrigin(self):
#pybind11#        """Check that we can set and read the origin"""
#pybind11#
#pybind11#        im = afwImage.MaskedImageF(afwGeom.ExtentI(10, 20))
#pybind11#        x0 = y0 = 0
#pybind11#
#pybind11#        self.assertEqual(im.getX0(), x0)
#pybind11#        self.assertEqual(im.getY0(), y0)
#pybind11#        self.assertEqual(im.getXY0(), afwGeom.PointI(x0, y0))
#pybind11#
#pybind11#        x0, y0 = 3, 5
#pybind11#        im.setXY0(x0, y0)
#pybind11#        self.assertEqual(im.getX0(), x0)
#pybind11#        self.assertEqual(im.getY0(), y0)
#pybind11#        self.assertEqual(im.getXY0(), afwGeom.PointI(x0, y0))
#pybind11#
#pybind11#        x0, y0 = 30, 50
#pybind11#        im.setXY0(afwGeom.Point2I(x0, y0))
#pybind11#        self.assertEqual(im.getX0(), x0)
#pybind11#        self.assertEqual(im.getY0(), y0)
#pybind11#        self.assertEqual(im.getXY0(), afwGeom.Point2I(x0, y0))
#pybind11#
#pybind11#    def testSubimages1(self):
#pybind11#        smimage = afwImage.MaskedImageF(
#pybind11#            self.mimage,
#pybind11#            afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.Extent2I(10, 5)),
#pybind11#            afwImage.LOCAL
#pybind11#        )
#pybind11#
#pybind11#        simage = afwImage.MaskedImageF(
#pybind11#            smimage,
#pybind11#            afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.Extent2I(3, 2)),
#pybind11#            afwImage.LOCAL
#pybind11#        )
#pybind11#        self.assertEqual(simage.getX0(), 2)
#pybind11#        self.assertEqual(simage.getY0(), 2)  # i.e. wrt self.mimage
#pybind11#
#pybind11#        mimage2 = afwImage.MaskedImageF(simage.getDimensions())
#pybind11#        mimage2.getImage().set(666)
#pybind11#        mimage2.getMask().set(self.BAD)
#pybind11#        simage[:] = mimage2
#pybind11#
#pybind11#        del simage
#pybind11#        del mimage2
#pybind11#
#pybind11#        self.checkImgPatch12(self.mimage, 2, 2)
#pybind11#        self.checkImgPatch12(smimage, 1, 1)
#pybind11#
#pybind11#    def testSubimages2(self):
#pybind11#        """Test subimages when we've played with the (x0, y0) value"""
#pybind11#
#pybind11#        self.mimage.set(9, 4, (888, 0x0, 0))
#pybind11#        #printImg(afwImage.ImageF(self.mimage, afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(10, 5)))); print
#pybind11#
#pybind11#        smimage = afwImage.MaskedImageF(
#pybind11#            self.mimage,
#pybind11#            afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.Extent2I(10, 5)),
#pybind11#            afwImage.LOCAL
#pybind11#        )
#pybind11#        smimage.setXY0(afwGeom.Point2I(0, 0))  # reset origin; doesn't affect pixel coordinate systems
#pybind11#
#pybind11#        simage = afwImage.MaskedImageF(
#pybind11#            smimage, afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.Extent2I(3, 2)),
#pybind11#            afwImage.LOCAL
#pybind11#        )
#pybind11#        self.assertEqual(simage.getX0(), 1)
#pybind11#        self.assertEqual(simage.getY0(), 1)
#pybind11#
#pybind11#        mimage2 = afwImage.MaskedImageF(simage.getDimensions())
#pybind11#        mimage2.set(666, self.BAD, 0.0)
#pybind11#        simage[:] = mimage2
#pybind11#        del simage
#pybind11#        del mimage2
#pybind11#
#pybind11#        self.checkImgPatch12(self.mimage, 2, 2)
#pybind11#        self.checkImgPatch12(smimage, 1, 1)
#pybind11#
#pybind11#    def checkImgPatch3(self, img, deep):
#pybind11#        """Check that a patch of an image is correct; origin of patch is at (x0, y0) in full image
#pybind11#        N.b. This isn't a general routine!  Works only for testSubimages3"""
#pybind11#
#pybind11#        # Include deep in comparison so we can see which test fails
#pybind11#        self.assertEqual(img.get(0, 0) + (deep, ), (100, 0x0, self.varVal1, deep))
#pybind11#        self.assertEqual(img.get(10, 10) + (deep, ), (200, 0xf, self.varVal1, deep))
#pybind11#
#pybind11#    def testSubimages3(self):
#pybind11#        """Test subimages when we've played with the (x0, y0) value"""
#pybind11#
#pybind11#        self.mimage.getImage().set(20, 20, 200)
#pybind11#        self.mimage.getMask().set(20, 20, 0xf)
#pybind11#
#pybind11#        for deep in (True, False):
#pybind11#            mimage = self.mimage.Factory(
#pybind11#                self.mimage,
#pybind11#                afwGeom.Box2I(afwGeom.Point2I(10, 10), afwGeom.Extent2I(64, 64)),
#pybind11#                afwImage.LOCAL,
#pybind11#                deep)
#pybind11#            mimage.setXY0(afwGeom.Point2I(0, 0))
#pybind11#            mimage2 = mimage.Factory(mimage)
#pybind11#
#pybind11#            if display:
#pybind11#                ds9.mtv(mimage2)
#pybind11#
#pybind11#            self.checkImgPatch3(mimage2, deep)
#pybind11#
#pybind11#    def testSetCopiedMask(self):
#pybind11#        """Check that we can set the Mask with a copied Mask"""
#pybind11#
#pybind11#        crMask = self.mimage.getMask().Factory(self.mimage.getMask(), True)
#pybind11#        msk = self.mimage.getMask()
#pybind11#        msk |= crMask
#pybind11#        del msk
#pybind11#
#pybind11#    def testVariance(self):
#pybind11#        """Check that we can set the variance from the gain"""
#pybind11#        gain = 2
#pybind11#
#pybind11#        var = self.mimage.getVariance()
#pybind11#        var[:] = self.mimage.getImage()
#pybind11#        var /= gain
#pybind11#
#pybind11#    def testTicket653(self):
#pybind11#        """How-to-repeat for #653"""
#pybind11#        # The original ticket read this file, but it doesn't reproduce for me,
#pybind11#        # As I don't see how reading an exposure from disk could make a difference
#pybind11#        # it's easier to just build an Image
#pybind11#        if False:
#pybind11#            im = afwImage.ImageF(os.path.join(lsst.utils.getPackageDir("afwdata"), "med_img.fits"))
#pybind11#        else:
#pybind11#            im = afwImage.ImageF(afwGeom.Extent2I(10, 10))
#pybind11#        mi = afwImage.MaskedImageF(im)
#pybind11#        afwImage.ExposureF(mi)
#pybind11#
#pybind11#    def testMaskedImageInitialisation(self):
#pybind11#        dims = self.mimage.getDimensions()
#pybind11#        factory = self.mimage.Factory
#pybind11#
#pybind11#        self.mimage.set(666)
#pybind11#
#pybind11#        del self.mimage                 # tempt C++ to reuse the memory
#pybind11#        self.mimage = factory(dims)
#pybind11#        self.assertEqual(self.mimage.get(10, 10), (0, 0x0, 0))
#pybind11#
#pybind11#        del self.mimage
#pybind11#        self.mimage = factory(afwGeom.Extent2I(20, 20))
#pybind11#        self.assertEqual(self.mimage.get(10, 10), (0, 0x0, 0))
#pybind11#
#pybind11#    def testImageSlices(self):
#pybind11#        """Test image slicing, which generate sub-images using Box2I under the covers"""
#pybind11#        im = afwImage.MaskedImageF(10, 20)
#pybind11#        im[4, 10] = (10, 0x2, 100)
#pybind11#        im[-3:, -2:] = 100
#pybind11#        sim = im[1:4, 6:10]
#pybind11#        nan = -666  # a real NaN != NaN so tests fail
#pybind11#        sim[:] = (-1, 0x8, nan)
#pybind11#        im[0:4, 0:4] = im[2:6, 8:12]
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(im)
#pybind11#
#pybind11#        self.assertEqual(im.get(0, 6), (0, 0x0, 0))
#pybind11#        self.assertEqual(im.get(6, 17), (0, 0x0, 0))
#pybind11#        self.assertEqual(im.get(7, 18), (100, 0x0, 0))
#pybind11#        self.assertEqual(im.get(9, 19), (100, 0x0, 0))
#pybind11#        self.assertEqual(im.get(1, 6), (-1, 0x8, nan))
#pybind11#        self.assertEqual(im.get(3, 9), (-1, 0x8, nan))
#pybind11#        self.assertEqual(im.get(4, 10), (10, 0x2, 100))
#pybind11#        self.assertEqual(im.get(4, 9), (0, 0x0, 0))
#pybind11#        self.assertEqual(im.get(2, 2), (10, 0x2, 100))
#pybind11#        self.assertEqual(im.get(0, 0), (-1, 0x8, nan))
#pybind11#
#pybind11#    def testConversionToScalar(self):
#pybind11#        """Test that even 1-pixel MaskedImages can't be converted to scalars"""
#pybind11#        im = afwImage.MaskedImageF(10, 20)
#pybind11#
#pybind11#        self.assertRaises(TypeError, float, im)  # only single pixel images may be converted
#pybind11#        self.assertRaises(TypeError, float, im[0, 0])  # actually, can't convert (img, msk, var) to scalar
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

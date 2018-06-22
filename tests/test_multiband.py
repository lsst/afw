# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Tests for Multiband objects
"""
import unittest
import operator

import numpy as np

import lsst.utils
import lsst.utils.tests
from lsst.geom import Point2I, Box2I, Extent2I
from lsst.afw.geom import SpanSet, Stencil
from lsst.afw.detection import GaussianPsf, Footprint, makeHeavyFootprint, MultibandFootprint, HeavyFootprintF
from lsst.afw.image import ImageF, Mask, MaskPixel, MaskedImage, ExposureF, MaskedImageF
from lsst.afw.image.multiband import MultibandPixel, MultibandImage, MultibandMask, MultibandMaskedImage
from lsst.afw.image.multiband import MaskedPixel, MultibandMaskedPixel, MultibandExposure


def _testImageFilterSlicing(cls, mImage, singleType, bbox, value):
    assert isinstance(mImage[0], singleType)
    assert isinstance(mImage["R"], singleType)
    assert isinstance(mImage[:], type(mImage))
    idx = np.int32(2)
    assert isinstance(mImage[idx], singleType)

    cls.assertEqual(mImage[2], mImage["I"])
    cls.assertEqual(mImage[0, -1, -1], value)
    cls.assertEqual(mImage[0].array.shape, (100, 200))
    cls.assertEqual(mImage[:2].array.shape, (2, 100, 200))
    cls.assertEqual(mImage[:2].filters, ("G", "R"))
    cls.assertEqual(mImage[:2].getBBox(), bbox)
    cls.assertEqual(mImage[["G", "I"]].array.shape, (2, 100, 200))
    cls.assertEqual(mImage[["G", "I"]].filters, ("G", "I"))
    cls.assertEqual(mImage[["G", "I"]].getBBox(), bbox)

    cls.assertEqual(mImage[0:1].filters, ("G",))
    cls.assertEqual(mImage[1:3].filters, ("R", "I"))
    cls.assertEqual(mImage.filters, tuple(cls.filters))


def _testImageSlicing(cls, mImage, gVal, rVal, iVal):
    assert isinstance(mImage[:, -1, -1], MultibandPixel)
    cls.assertEqual(mImage["G", -1, -1], gVal)

    cls.assertEqual(mImage[:, 1100:, 2025:].getBBox(), Box2I(Point2I(1100, 2025), Extent2I(100, 75)))
    cls.assertEqual(mImage[:, -20:-10, -10:-5].getBBox(),
                    Box2I(Point2I(1179, 2089), Extent2I(10, 5)))
    cls.assertEqual(mImage[:, :1100, :2050].getBBox(), Box2I(Point2I(1000, 2000), Extent2I(100, 50)))
    coord = Point2I(1075, 2015)
    bbox = Box2I(Point2I(1050, 2010), coord)
    cls.assertEqual(mImage[:, bbox].getBBox(), bbox)
    cls.assertEqual(mImage[:, 1010, 2010].getBBox(), Point2I(1010, 2010))
    cls.assertEqual(mImage[:, Point2I(1075, 2015)].getBBox(), coord)

    cls.assertEqual(mImage[0, 1100:, 2025:].getBBox(), Box2I(Point2I(1100, 2025), Extent2I(100, 75)))
    cls.assertEqual(mImage["R", -20:-10, -10:-5].getBBox(),
                    Box2I(Point2I(1179, 2089), Extent2I(10, 5)))
    cls.assertEqual(mImage[2, :1100, :2050].getBBox(), Box2I(Point2I(1000, 2000), Extent2I(100, 50)))
    cls.assertEqual(mImage[1, bbox].getBBox(), bbox)
    cls.assertEqual(mImage["I", 1010, 2010], iVal)
    cls.assertEqual(mImage[1, Point2I(1075, 2015)], rVal)

    with cls.assertRaises(IndexError):
        mImage[:, 0]
    with cls.assertRaises(IndexError):
        mImage[:, 10:]
    with cls.assertRaises(IndexError):
        mImage[:, :10]
    with cls.assertRaises(IndexError):
        mImage[:, :, 0]
    with cls.assertRaises(IndexError):
        mImage[:, :, 10:]
    with cls.assertRaises(IndexError):
        mImage[:, :, :10]
    with cls.assertRaises(ValueError):
        mImage.setBBox(bbox)


def _testImageModification(cls, mImage1, mImage2, bbox1, bbox2, value1, value2):
    mImage1[:1, bbox2].array = value2
    cls.assertFloatsEqual(mImage1[0, bbox2].array, mImage2[0].array)
    cls.assertFloatsEqual(mImage1[1].array, value1)
    mImage1.setXY0(Point2I(500, 150))
    cls.assertEqual(mImage1.getBBox(), Box2I(Point2I(500, 150), Extent2I(bbox1.getDimensions())))

    mImage1[0].array[:] = value2
    cls.assertFloatsEqual(mImage1[0].array, value2)
    cls.assertFloatsEqual(mImage1.array[0], value2)

    mImage1[1:3].array[:] = 7
    cls.assertFloatsEqual(mImage1[2].array, 7)


def _testImageCopy(cls, mImage1, value1, value2):
    mImage2 = mImage1.copy(True)
    mImage2.setXY0(Point2I(11, 23))
    cls.assertEqual(mImage2.getBBox(), Box2I(Point2I(11, 23), Extent2I(200, 100)))
    cls.assertEqual(mImage1.getBBox(), Box2I(Point2I(1000, 2000), Extent2I(200, 100)))
    cls.assertTrue(np.all([s.getBBox() == mImage1.getBBox() for s in mImage1.singles]))
    cls.assertTrue(np.all([s.getBBox() == mImage2.getBBox() for s in mImage2.singles]))

    mImage2.array[:] = value2
    cls.assertFloatsEqual(mImage1.array, value1)
    cls.assertFloatsEqual(mImage2.array, value2)
    cls.assertFloatsEqual(mImage1[0].array, value1)
    cls.assertFloatsEqual(mImage2[0].array, value2)

    mImage2 = mImage1.copy()
    mImage2.setXY0(Point2I(11, 23))
    cls.assertEqual(mImage2.getBBox(), Box2I(Point2I(11, 23), Extent2I(200, 100)))
    cls.assertEqual(mImage1.getBBox(), Box2I(Point2I(1000, 2000), Extent2I(200, 100)))
    cls.assertTrue(np.all([s.getBBox() == mImage2.getBBox() for s in mImage1.singles]))
    cls.assertTrue(np.all([s.getBBox() == mImage2.getBBox() for s in mImage2.singles]))

    mImage2.array[:] = value2
    cls.assertFloatsEqual(mImage1.array, value2)
    cls.assertFloatsEqual(mImage2.array, value2)
    cls.assertFloatsEqual(mImage1[0].array, value2)
    cls.assertFloatsEqual(mImage2[0].array, value2)


class MultibandPixelTestCase(lsst.utils.tests.TestCase):
    """Test case for MultibandPixel
    """
    def setUp(self):
        np.random.seed(1)
        self.bbox = Point2I(101, 502)
        self.filters = ["G", "R", "I", "Z", "Y"]
        singles = np.arange(5, dtype=float)
        self.pixel = MultibandPixel(self.filters, singles, self.bbox)

    def tearDown(self):
        del self.bbox
        del self.filters
        del self.pixel

    def testFilterSlicing(self):
        pixel = self.pixel
        self.assertEqual(pixel[1], 1.)
        self.assertFloatsEqual(pixel.array, np.arange(5))
        self.assertFloatsEqual(pixel.singles, np.arange(5))
        self.assertFloatsEqual(pixel[["G", "I"]].array, [0, 2])

    def testPixelBBoxModification(self):
        pixel = self.pixel.copy(deep=True)
        otherPixel = pixel.copy(deep=True)
        pixel.getBBox().shift(Extent2I(9, -2))
        self.assertEqual(pixel.getBBox(), Point2I(110, 500))
        self.assertEqual(otherPixel.getBBox(), Point2I(101, 502))

        pixel = self.pixel.copy(deep=True)
        otherPixel = pixel.copy()
        pixel.getBBox().shift(Extent2I(9, -2))
        self.assertEqual(pixel.getBBox(), Point2I(110, 500))
        self.assertEqual(otherPixel.getBBox(), Point2I(110, 500))

    def testPixelModification(self):
        pixel = self.pixel
        otherPixel = pixel.copy(deep=True)
        otherPixel.array = np.arange(10, 15)
        self.assertFloatsEqual(otherPixel.array, np.arange(10, 15))
        self.assertFloatsEqual(pixel.array, np.arange(0, 5))


class MultibandImageTestCase(lsst.utils.tests.TestCase):
    """Test case for MultibandImage"""

    def setUp(self):
        np.random.seed(1)
        self.bbox1 = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
        self.filters = ["G", "R", "I", "Z", "Y"]
        self.value1, self.value2 = 10, 100
        images = [ImageF(self.bbox1, self.value1) for f in self.filters]
        self.mImage1 = MultibandImage(self.filters, images)
        self.bbox2 = Box2I(Point2I(1100, 2025), Extent2I(30, 50))
        images = [ImageF(self.bbox2, self.value2) for f in self.filters]
        self.mImage2 = MultibandImage(self.filters, images)

    def tearDown(self):
        del self.bbox1
        del self.bbox2
        del self.filters
        del self.value1
        del self.value2
        del self.mImage1
        del self.mImage2

    def testFilterSlicing(self):
        _testImageFilterSlicing(self, self.mImage1, ImageF, self.bbox1, self.value1)

    def testImageSlicing(self):
        _testImageSlicing(self, self.mImage1, self.value1, self.value1, self.value1)

    def testImageModification(self):
        _testImageModification(self, self.mImage1, self.mImage2, self.bbox1, self.bbox2,
                               self.value1, self.value2)

    def testImageCopy(self):
        _testImageCopy(self, self.mImage1, self.value1, 5.)


class MultibandMaskTestCase(lsst.utils.tests.TestCase):
    """A test case for Mask"""

    def setUp(self):
        np.random.seed(1)
        self.filters = ["G", "R", "I"]
        self.Mask = Mask[MaskPixel]

        # Store the default mask planes for later use
        maskPlaneDict = self.Mask().getMaskPlaneDict()
        self.defaultMaskPlanes = sorted(maskPlaneDict, key=maskPlaneDict.__getitem__)

        # reset so tests will be deterministic
        self.Mask.clearMaskPlaneDict()
        for p in ("BAD", "SAT", "INTRP", "CR", "EDGE"):
            self.Mask.addMaskPlane(p)

        self.BAD = self.Mask.getPlaneBitMask("BAD")
        self.CR = self.Mask.getPlaneBitMask("CR")
        self.EDGE = self.Mask.getPlaneBitMask("EDGE")

        self.values1 = [self.BAD | self.CR, self.BAD | self.EDGE, self.BAD | self.CR | self.EDGE]
        self.bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
        singles = [self.Mask(self.bbox) for f in range(len(self.filters))]
        for n in range(len(singles)):
            singles[n].set(self.values1[n])
        self.mMask1 = MultibandMask(self.filters, singles)

        self.values2 = [self.EDGE, self.BAD, self.CR | self.EDGE]
        singles = [self.Mask(self.bbox) for f in range(len(self.filters))]
        for n in range(len(singles)):
            singles[n].set(self.values2[n])
        self.mMask2 = MultibandMask(self.filters, singles)

    def tearDown(self):
        del self.mMask1
        del self.mMask2
        del self.bbox
        del self.values1
        del self.values2
        # Reset the mask plane to the default
        self.Mask.clearMaskPlaneDict()
        for p in self.defaultMaskPlanes:
            self.Mask.addMaskPlane(p)
        self.defaultMaskPlanes

    def testInitializeMasks(self):
        self.assertTrue(np.all([s.getBBox() == self.mMask1.getBBox() for s in self.mMask1.singles]))
        self.assertTrue(np.all([s.array == self.values1[n] for n, s in enumerate(self.mMask1.singles)]))

    def _bitOperator(self, op):
        op(self.mMask2, self.mMask1)
        for n in range(len(self.mMask1)):
            op(self.mMask1[n], self.values2[n])

        self.assertFloatsEqual(self.mMask1.array, self.mMask2.array)
        expect = np.empty_like(self.mMask1.array)
        expect[:] = op(np.array(self.values1), np.array(self.values2))[:, None, None]
        self.assertFloatsEqual(self.mMask1.array, expect)

    def testOrMasks(self):
        self._bitOperator(operator.ior)

    def testAndMasks(self):
        self._bitOperator(operator.iand)

    def testXorMasks(self):
        self._bitOperator(operator.ixor)

    def testSetMask(self):
        mMask = self.mMask1.copy(True)
        mMask.set(self.CR)
        self.assertFloatsEqual(mMask.array, self.CR)
        mMask.set(self.EDGE, "G")
        self.assertFloatsEqual(mMask[1:].array, self.CR)
        self.assertFloatsEqual(mMask[0].array, self.EDGE)
        mMask.set(self.BAD, slice(1, 3))
        self.assertFloatsEqual(mMask[1:].array, self.BAD)
        mMask.set(self.CR | self.EDGE, "R", 1100, 2050)
        self.assertEqual(mMask["R", 1100, 2050], self.CR | self.EDGE)
        self.assertEqual(mMask["R", 1101, 2051], self.BAD)

        with self.assertRaises(IndexError):
            mMask.set(self.CR, "G", 1100)
        with self.assertRaises(IndexError):
            mMask.set(self.CR, "G", y=2050)

    def testMaskPlanes(self):
        planes = self.mMask1.getMaskPlaneDict()
        self.assertEqual(len(planes), self.mMask1.getNumPlanesUsed())

        for k in sorted(planes.keys()):
            self.assertEqual(planes[k], self.mMask1.getMaskPlane(k))

    def testFilterSlicing(self):
        _testImageFilterSlicing(self, self.mMask1, Mask, self.bbox, self.values1[0])

    def testImageSlicing(self):
        _testImageSlicing(self, self.mMask1, *self.values1)

    def testImageModification(self):
        mMask1 = self.mMask1
        value1 = self.CR
        value2 = self.EDGE
        mMask1.set(value1)

        bbox2 = Box2I(Point2I(1100, 2025), Extent2I(30, 50))
        singles = [self.Mask(bbox2) for f in range(len(self.filters))]
        for n in range(len(singles)):
            singles[n].set(value2)
        mMask2 = MultibandMask(self.filters, singles)

        _testImageModification(self, mMask1, mMask2, self.bbox, bbox2, value1, value2)

    def testImageCopy(self):
        mMask = self.mMask1
        value1 = self.CR
        value2 = self.EDGE
        mMask.set(value1)
        _testImageCopy(self, mMask, value1, value2)


class MultibandMaskedPixelTestCase(lsst.utils.tests.TestCase):
    """Test case for MultibandPixel
    """
    def setUp(self):
        np.random.seed(1)
        self.bbox = Point2I(101, 502)
        self.filters = ["G", "R", "I", "Z", "Y"]
        self.image = np.ones((len(self.filters),), dtype=float) * 10
        self.mask = np.ones((len(self.filters),))
        self.variance = np.ones((len(self.filters),)) * 1e-2
        self.pixel = MultibandMaskedPixel(filters=self.filters, image=self.image, mask=self.mask,
                                          variance=self.variance, bbox=self.bbox)

    def tearDown(self):
        del self.bbox
        del self.filters
        del self.pixel

    def testFilterSlicing(self):
        pixel = self.pixel
        self.assertEqual(pixel[1].image, 10.)
        self.assertEqual(pixel[2].mask, 1)
        self.assertEqual(pixel[0].variance, 1e-2)
        self.assertFloatsEqual(pixel.image, np.ones((len(self.filters),))*10)
        self.assertFloatsEqual(pixel.mask, np.ones((len(self.filters),)))
        self.assertFloatsEqual(pixel.variance, np.ones((len(self.filters),))*1e-2)

        for single in pixel.singles:
            assert isinstance(single, MaskedPixel)
            self.assertEqual(single.image, 10)
            self.assertEqual(single.mask, 1)
            self.assertEqual(single.variance, 1e-2)

        self.assertFloatsEqual(pixel[["G", "I"]].image.array, np.array([10, 10]))
        self.assertFloatsEqual(pixel[["G", "R"]].mask.array, np.array([1, 1]))
        self.assertFloatsEqual(pixel[["R", "I"]].variance.array, np.array([1e-2, 1e-2]))

    def testPixelBBoxModification(self):
        pixel = self.pixel.copy(deep=True)
        otherPixel = pixel.copy(deep=True)
        pixel.getBBox().shift(Extent2I(9, -2))
        self.assertEqual(pixel.getBBox(), Point2I(110, 500))
        self.assertEqual(otherPixel.getBBox(), Point2I(101, 502))

        pixel = self.pixel.copy(deep=True)
        otherPixel = pixel.copy()
        pixel.getBBox().shift(Extent2I(9, -2))
        self.assertEqual(pixel.getBBox(), Point2I(110, 500))
        self.assertEqual(otherPixel.getBBox(), Point2I(110, 500))

    def testPixelModification(self):
        ones = np.ones((len(self.filters),))
        pixel = self.pixel
        otherPixel = pixel.copy(deep=True)
        otherPixel.image = MultibandPixel(self.filters, ones*8, pixel.getBBox())
        otherPixel.mask = MultibandPixel(self.filters, ones*2, pixel.getBBox())
        otherPixel.variance = MultibandPixel(self.filters, ones*1e-3, pixel.getBBox())

        self.assertFloatsEqual(pixel.image, ones * 10)
        # Stopped development of this test here, since this class is likely to be removed


def _testMaskedImageFilters(cls, maskedImage, singleType):
    assert isinstance(maskedImage[0], singleType)
    assert isinstance(maskedImage["R"], singleType)
    assert isinstance(maskedImage.image[0], ImageF)
    assert isinstance(maskedImage.mask[1], Mask)
    assert isinstance(maskedImage.variance[2], ImageF)

    cls.assertEqual(maskedImage[2], maskedImage["I"])
    cls.assertEqual(maskedImage[0].image.array.shape, (100, 200))
    cls.assertEqual(maskedImage[:2].mask.array.shape, (2, 100, 200))
    cls.assertEqual(maskedImage[:2].filters, ("G", "R"))
    cls.assertEqual(maskedImage[:2].getBBox(), cls.bbox)
    cls.assertEqual(maskedImage[["G", "I"]].getBBox(), cls.bbox)

    cls.assertEqual(maskedImage.filters, tuple(cls.filters))


def _testMaskedImageSlicing(cls, maskedImage):
    subBox = Box2I(Point2I(1100, 2025), Extent2I(30, 50))
    cls.assertEqual(maskedImage[:, subBox].getBBox(), subBox)
    cls.assertEqual(maskedImage[:, subBox].image.getBBox(), subBox)
    cls.assertEqual(maskedImage[:, subBox].mask.getBBox(), subBox)
    cls.assertEqual(maskedImage[:, subBox].variance.getBBox(), subBox)

    newBox = Box2I(Point2I(100, 500), Extent2I(200, 100))
    maskedImage.setBBox(newBox)
    cls.assertEqual(maskedImage.getBBox(), newBox)
    cls.assertEqual(maskedImage.image.getBBox(), newBox)
    cls.assertEqual(maskedImage.mask.getBBox(), newBox)
    cls.assertEqual(maskedImage.variance.getBBox(), newBox)
    cls.assertEqual(maskedImage[0].getBBox(), newBox)
    cls.assertEqual(maskedImage[0].image.getBBox(), newBox)
    cls.assertEqual(maskedImage[1].mask.getBBox(), newBox)
    cls.assertEqual(maskedImage[2].variance.getBBox(), newBox)

    with cls.assertRaises(ValueError):
        maskedImage.setBBox(subBox)


def _testMaskedmageModification(cls, maskedImage):
    images = [ImageF(cls.bbox, 10*cls.imgValue) for f in cls.filters]
    mImage = MultibandImage(cls.filters, images)
    maskedImage.image = mImage
    cls.assertFloatsEqual(maskedImage.image[0].array, mImage.array[0])
    cls.assertFloatsEqual(maskedImage[0].image.array, mImage.array[0])

    singles = [cls.Mask(cls.bbox) for f in range(len(cls.filters))]
    for n in range(len(singles)):
        singles[n].set(cls.maskValue*2)
    mMask = MultibandMask(cls.filters, singles)
    maskedImage.mask = mMask
    cls.assertFloatsEqual(maskedImage.mask[0].array, mMask.array[0])
    cls.assertFloatsEqual(maskedImage[0].mask.array, mMask.array[0])

    images = [ImageF(cls.bbox, .1 * cls.varValue) for f in cls.filters]
    mVariance = MultibandImage(cls.filters, images)
    maskedImage.variance = mVariance
    cls.assertFloatsEqual(maskedImage.variance[0].array, mVariance.array[0])
    cls.assertFloatsEqual(maskedImage[0].variance.array, mVariance.array[0])

    subBox = Box2I(Point2I(1100, 2025), Extent2I(30, 50))
    maskedImage.image[:, subBox].array = 12
    cls.assertFloatsEqual(maskedImage.image[0, subBox].array, 12)
    cls.assertFloatsEqual(maskedImage[0, subBox].image.array, 12)
    maskedImage[1, subBox].image.set(15)
    cls.assertFloatsEqual(maskedImage.image[1, subBox].array, 15)
    cls.assertFloatsEqual(maskedImage[1, subBox].image.array, 15)

    maskedImage.mask[:, subBox].array = 64
    cls.assertFloatsEqual(maskedImage.mask[0, subBox].array, 64)
    cls.assertFloatsEqual(maskedImage[0, subBox].mask.array, 64)
    maskedImage[1, subBox].mask.set(128)
    cls.assertFloatsEqual(maskedImage.mask[1, subBox].array, 128)
    cls.assertFloatsEqual(maskedImage[1, subBox].mask.array, 128)

    maskedImage.variance[:, subBox].array = 1e-6
    cls.assertFloatsEqual(maskedImage.variance[0, subBox].array, 1e-6)
    cls.assertFloatsEqual(maskedImage[0, subBox].variance.array, 1e-6)
    maskedImage[1, subBox].variance.set(1e-7)
    cls.assertFloatsEqual(maskedImage.variance[1, subBox].array, 1e-7)
    cls.assertFloatsEqual(maskedImage[1, subBox].variance.array, 1e-7)


def _testMaskedImageCopy(cls, maskedImage1):
    maskedImage2 = maskedImage1.copy(True)

    maskedImage2.setXY0(Point2I(11, 23))
    cls.assertEqual(maskedImage2.getBBox(), Box2I(Point2I(11, 23), Extent2I(200, 100)))
    cls.assertEqual(maskedImage1.getBBox(), Box2I(Point2I(1000, 2000), Extent2I(200, 100)))
    cls.assertTrue(np.all([img.getBBox() == maskedImage1.getBBox() for img in maskedImage1.image]))
    cls.assertTrue(np.all([img.getBBox() == maskedImage2.getBBox() for img in maskedImage2.image]))

    maskedImage2.image.array = 1
    cls.assertFloatsEqual(maskedImage1.image.array, cls.imgValue)
    cls.assertFloatsEqual(maskedImage2.image.array, 1)
    cls.assertFloatsEqual(maskedImage1[0].image.array, cls.imgValue)
    cls.assertFloatsEqual(maskedImage2[0].image.array, 1)

    maskedImage2 = maskedImage1.copy()
    maskedImage2.image.array = 1
    cls.assertFloatsEqual(maskedImage1.image.array, 1)
    cls.assertFloatsEqual(maskedImage2.image.array, 1)
    cls.assertFloatsEqual(maskedImage1[0].image.array, 1)
    cls.assertFloatsEqual(maskedImage2[0].image.array, 1)


class MultibandMaskedImageTestCase(lsst.utils.tests.TestCase):
    """Test case for MultibandMaskedImage"""

    def setUp(self):
        np.random.seed(1)
        self.bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
        self.filters = ["G", "R", "I"]

        self.imgValue = 10
        images = [ImageF(self.bbox, self.imgValue) for f in self.filters]
        mImage = MultibandImage(self.filters, images)

        self.Mask = Mask[MaskPixel]
        # Store the default mask planes for later use
        maskPlaneDict = self.Mask().getMaskPlaneDict()
        self.defaultMaskPlanes = sorted(maskPlaneDict, key=maskPlaneDict.__getitem__)

        # reset so tests will be deterministic
        self.Mask.clearMaskPlaneDict()
        for p in ("BAD", "SAT", "INTRP", "CR", "EDGE"):
            self.Mask.addMaskPlane(p)

        self.maskValue = self.Mask.getPlaneBitMask("BAD")
        singles = [self.Mask(self.bbox) for f in range(len(self.filters))]
        for n in range(len(singles)):
            singles[n].set(self.maskValue)
        mMask = MultibandMask(self.filters, singles)

        self.varValue = 1e-2
        images = [ImageF(self.bbox, self.varValue) for f in self.filters]
        mVariance = MultibandImage(self.filters, images)

        self.maskedImage = MultibandMaskedImage(image=mImage, mask=mMask, variance=mVariance,
                                                filters=self.filters)

    def tearDown(self):
        del self.maskedImage
        del self.bbox
        del self.imgValue
        del self.maskValue
        del self.varValue
        # Reset the mask plane to the default
        self.Mask.clearMaskPlaneDict()
        for p in self.defaultMaskPlanes:
            self.Mask.addMaskPlane(p)
        del self.defaultMaskPlanes

    def testFilterSlicing(self):
        _testMaskedImageFilters(self, self.maskedImage, MaskedImage)

    def testImageSlicing(self):
        _testMaskedImageSlicing(self, self.maskedImage)

    def testModification(self):
        _testMaskedmageModification(self, self.maskedImage)

    def testCopy(self):
        _testMaskedImageCopy(self, self.maskedImage)


class MultibandExposureTestCase(lsst.utils.tests.TestCase):
    """
    A test case for the Exposure Class
    """

    def setUp(self):
        np.random.seed(1)
        self.bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
        self.filters = ["G", "R", "I"]

        self.imgValue = 10
        images = [ImageF(self.bbox, self.imgValue) for f in self.filters]
        mImage = MultibandImage(self.filters, images)

        self.Mask = Mask[MaskPixel]
        # Store the default mask planes for later use
        maskPlaneDict = self.Mask().getMaskPlaneDict()
        self.defaultMaskPlanes = sorted(maskPlaneDict, key=maskPlaneDict.__getitem__)

        # reset so tests will be deterministic
        self.Mask.clearMaskPlaneDict()
        for p in ("BAD", "SAT", "INTRP", "CR", "EDGE"):
            self.Mask.addMaskPlane(p)

        self.maskValue = self.Mask.getPlaneBitMask("BAD")
        singles = [self.Mask(self.bbox) for f in range(len(self.filters))]
        for n in range(len(singles)):
            singles[n].set(self.maskValue)
        mMask = MultibandMask(self.filters, singles)

        self.varValue = 1e-2
        images = [ImageF(self.bbox, self.varValue) for f in self.filters]
        mVariance = MultibandImage(self.filters, images)

        self.kernelSize = 51
        self.psfs = [GaussianPsf(self.kernelSize, self.kernelSize, 4.0) for f in self.filters]
        self.psfImage = np.array([p.computeImage().array for p in self.psfs])

        self.exposure = MultibandExposure(image=mImage, mask=mMask, variance=mVariance,
                                          psfs=self.psfs, filters=self.filters)

    def tearDown(self):
        del self.exposure
        del self.psfs
        del self.bbox
        del self.imgValue
        del self.maskValue
        del self.varValue
        # Reset the mask plane to the default
        self.Mask.clearMaskPlaneDict()
        for p in self.defaultMaskPlanes:
            self.Mask.addMaskPlane(p)
        del self.defaultMaskPlanes

    def testFilterSlicing(self):
        _testMaskedImageFilters(self, self.exposure, ExposureF)

    def testImageSlicing(self):
        _testMaskedImageSlicing(self, self.exposure)

    def testModification(self):
        _testMaskedmageModification(self, self.exposure)

    def testCopy(self):
        _testMaskedImageCopy(self, self.exposure)

    def testPsf(self):
        psfImage = self.exposure.getPsfImage()
        self.assertFloatsAlmostEqual(psfImage, self.psfImage)

        newPsfs = [GaussianPsf(self.kernelSize, self.kernelSize, 1.0) for f in self.filters]
        newPsfImage = [p.computeImage().array for p in newPsfs]
        self.exposure.setAllPsfs(newPsfs)
        psfImage = self.exposure.getPsfImage()
        self.assertFloatsAlmostEqual(psfImage, newPsfImage)


class MultibandFootprintTestCase(lsst.utils.tests.TestCase):
    """
    A test case for the Exposure Class
    """

    def setUp(self):
        np.random.seed(1)
        self.spans = SpanSet.fromShape(2, Stencil.CIRCLE)
        self.footprint = Footprint(self.spans)
        self.footprint.addPeak(3, 4, 10)
        self.footprint.addPeak(8, 1, 2)
        fp = Footprint(self.spans)
        for peak in self.footprint.getPeaks():
            fp.addPeak(peak["f_x"], peak["f_y"], peak["peakValue"])
        self.peaks = fp.getPeaks()
        self.bbox = self.footprint.getBBox()
        self.filters = ("G", "R", "I")
        singles = []
        images = []
        for n, f in enumerate(self.filters):
            image = ImageF(self.spans.getBBox())
            image.set(n)
            images.append(image.array)
            maskedImage = MaskedImageF(image)
            heavy = makeHeavyFootprint(self.footprint, maskedImage)
            singles.append(heavy)
        self.image = np.array(images)
        self.mFoot = MultibandFootprint(self.filters, singles)

    def tearDown(self):
        del self.spans
        del self.footprint
        del self.peaks
        del self.bbox
        del self.filters
        del self.mFoot
        del self.image

    def verifyPeaks(self, peaks1, peaks2):
        self.assertEqual(len(peaks1), len(peaks2))
        for n in range(len(peaks1)):
            pk1 = peaks1[n]
            pk2 = peaks2[n]
            # self.assertEqual(pk1["id"], pk2["id"])
            self.assertEqual(pk1["f_x"], pk2["f_x"])
            self.assertEqual(pk1["f_y"], pk2["f_y"])
            self.assertEqual(pk1["i_x"], pk2["i_x"])
            self.assertEqual(pk1["i_y"], pk2["i_y"])
            self.assertEqual(pk1["peakValue"], pk2["peakValue"])

    def testConstructor(self):
        def projectSpans(radius, value, bbox, asArray):
            ss = SpanSet.fromShape(radius, Stencil.CIRCLE, offset=(10, 10))
            image = ImageF(bbox)
            ss.setImage(image, value)
            if asArray:
                return image.array
            else:
                return image

        def multibandProjectSpans(bbox):
            images = np.array([projectSpans(n, n, bbox, True) for n in range(2, 5)])
            return MultibandImage(array=images, bbox=bbox, filters=self.filters)

        def runTest(images=None, footprint=None, xy0=Point2I(5, 5),
                    peaks=self.peaks, thresh=0, footprintBBox=Box2I(Point2I(6, 6), Extent2I(9, 9))):
            mFoot = MultibandFootprint(
                self.filters,
                images=images,
                xy0=xy0,
                footprint=footprint,
                peaks=peaks,
                thresh=thresh
            )
            self.assertEqual(mFoot.getBBox(), footprintBBox)
            try:
                fpImage = np.array(images)[:, 1:-1, 1:-1]
            except IndexError:
                fpImage = np.array([img.array for img in images])[:, 1:-1, 1:-1]
            self.assertFloatsAlmostEqual(mFoot.getArray(), fpImage)
            if peaks is not None:
                self.verifyPeaks(mFoot.getPeaks(), peaks)

        bbox = Box2I(Point2I(5, 5), Extent2I(11, 11))
        runTest(images=np.array([projectSpans(n, 5-n, bbox, True) for n in range(2, 5)]))
        runTest(images=[projectSpans(n, 5-n, bbox, False) for n in range(2, 5)])
        runTest(images=multibandProjectSpans(bbox))
        runTest(images=np.array([projectSpans(n, 5-n, bbox, True) for n in range(2, 5)]),
                xy0=None, footprintBBox=Box2I(Point2I(1, 1), Extent2I(9, 9)), peaks=None)

        images = [projectSpans(n, 5-n, bbox, True) for n in range(2, 5)]
        thresh = [1, 2, 2.5]
        mFoot = MultibandFootprint(
            self.filters,
            images=images,
            xy0=bbox.getMin(),
            thresh=thresh
        )
        footprintBBox = Box2I(Point2I(8, 8), Extent2I(5, 5))
        self.assertEqual(mFoot.getBBox(), footprintBBox)
        fpImage = np.array(images)[:, 3:-3, 3:-3]
        mask = np.all(fpImage <= np.array(thresh)[:, None, None], axis=0)
        fpImage[:, mask] = 0
        self.assertFloatsAlmostEqual(mFoot.getArray(), fpImage)

    def testSlicing(self):
        assert isinstance(self.mFoot[0], HeavyFootprintF)
        assert isinstance(self.mFoot["R"], HeavyFootprintF)
        assert isinstance(self.mFoot[:], MultibandFootprint)

        self.assertEqual(self.mFoot[2], self.mFoot["I"])
        self.assertEqual(self.mFoot[:2].filters, ("G", "R"))
        self.assertEqual(self.mFoot[:2].getBBox(), self.bbox)
        self.assertEqual(self.mFoot[["G", "I"]].filters, ("G", "I"))
        self.assertEqual(self.mFoot[["G", "I"]].getBBox(), self.bbox)

        with self.assertRaises(IndexError):
            self.mFoot[2, 4, 5]
            self.mFoot[2, :, :]
            self.mFoot[:, :, :]

    def testSpans(self):
        self.assertEqual(self.mFoot.getSpans(), self.spans)
        for footprint in self.mFoot.singles:
            self.assertEqual(footprint.getSpans(), self.spans)

    def testPeaks(self):
        self.verifyPeaks(self.peaks, self.footprint.getPeaks())
        for footprint in self.mFoot.singles:
            self.verifyPeaks(footprint.getPeaks(), self.peaks)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

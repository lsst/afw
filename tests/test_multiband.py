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
from lsst.afw.image import ImageF, Mask, MaskPixel, MaskedImage, ExposureF, MaskedImageF, LOCAL
from lsst.afw.image import MultibandPixel, MultibandImage, MultibandMask, MultibandMaskedImage
from lsst.afw.image import MultibandExposure


def _testImageBandSlicing(testCase, mImage, singleType, bbox, value):
    """Test slicing by bands for image-like objects"""
    testCase.assertIsInstance(mImage["R"], singleType)
    testCase.assertIsInstance(mImage[:], type(mImage))

    testCase.assertEqual(mImage["G", -1, -1, LOCAL], value)
    testCase.assertEqual(mImage["G"].array.shape, (100, 200))
    testCase.assertEqual(mImage[:"I"].array.shape, (2, 100, 200))
    testCase.assertEqual(mImage[:"I"].bands, ("G", "R"))
    testCase.assertEqual(mImage[:"I"].getBBox(), bbox)
    testCase.assertEqual(mImage[["G", "I"]].array.shape, (2, 100, 200))
    testCase.assertEqual(mImage[["G", "I"]].bands, ("G", "I"))
    testCase.assertEqual(mImage[["G", "I"]].getBBox(), bbox)

    testCase.assertEqual(mImage["G":"R"].bands, ("G",))

    if "Z" in mImage.bands:
        bandSlice = slice("R", "Z")
    else:
        bandSlice = slice("R", None)
    testCase.assertEqual(mImage[bandSlice].bands, ("R", "I"))
    testCase.assertEqual(mImage.bands, tuple(testCase.bands))


def _testImageSlicing(testCase, mImage, gVal, rVal, iVal):
    """Test slicing in the spatial dimensions for image-like objects"""
    testCase.assertIsInstance(mImage[:, -1, -1, LOCAL], MultibandPixel)
    testCase.assertEqual(mImage["G", -1, -1, LOCAL], gVal)

    testCase.assertEqual(mImage[:, 1100:, 2025:].getBBox(), Box2I(Point2I(1100, 2025), Extent2I(100, 75)))
    testCase.assertEqual(mImage[:, -20:-10, -10:-5, LOCAL].getBBox(),
                         Box2I(Point2I(1180, 2090), Extent2I(10, 5)))
    testCase.assertEqual(mImage[:, :1100, :2050].getBBox(), Box2I(Point2I(1000, 2000), Extent2I(100, 50)))
    coord = Point2I(1075, 2015)
    bbox = Box2I(Point2I(1050, 2010), coord)
    testCase.assertEqual(mImage[:, bbox].getBBox(), bbox)
    testCase.assertEqual(mImage[:, 1010, 2010].getBBox().getMin(), Point2I(1010, 2010))
    testCase.assertEqual(mImage[:, Point2I(1075, 2015)].getBBox().getMin(), coord)

    testCase.assertEqual(mImage["G", 1100:, 2025:].getBBox(), Box2I(Point2I(1100, 2025), Extent2I(100, 75)))
    testCase.assertEqual(mImage["R", -20:-10, -10:-5, LOCAL].getBBox(),
                         Box2I(Point2I(1180, 2090), Extent2I(10, 5)))
    testCase.assertEqual(mImage["I", :1100, :2050].getBBox(), Box2I(Point2I(1000, 2000), Extent2I(100, 50)))
    testCase.assertEqual(mImage["R", bbox].getBBox(), bbox)
    testCase.assertEqual(mImage["I", 1010, 2010], iVal)
    testCase.assertEqual(mImage["R", Point2I(1075, 2015)], rVal)

    with testCase.assertRaises(TypeError):
        mImage[:, 0]
    with testCase.assertRaises(TypeError):
        mImage[:, 10:]
    with testCase.assertRaises(TypeError):
        mImage[:, :10]
    with testCase.assertRaises(TypeError):
        mImage[:, :, 0]


def _testImageModification(testCase, mImage1, mImage2, bbox1, bbox2, value1, value2):
    """Test the image-like objects can be modified"""
    mImage1[:"R", bbox2].array = value2
    testCase.assertFloatsEqual(mImage1["G", bbox2].array, mImage2["G"].array)
    testCase.assertFloatsEqual(mImage1["R"].array, value1)
    mImage1.setXY0(Point2I(500, 150))
    testCase.assertEqual(mImage1.getBBox(), Box2I(Point2I(500, 150), Extent2I(bbox1.getDimensions())))

    mImage1["G"].array[:] = value2
    testCase.assertFloatsEqual(mImage1["G"].array, value2)
    testCase.assertFloatsEqual(mImage1.array[0], value2)

    if "Z" in mImage1.bands:
        bandSlice = slice("R", "Z")
    else:
        bandSlice = slice("R", None)
    mImage1[bandSlice].array[:] = 7
    testCase.assertFloatsEqual(mImage1["I"].array, 7)
    newBBox = Box2I(Point2I(10000, 20000), mImage1.getBBox().getDimensions())
    mImage1.setXY0(newBBox.getMin())
    testCase.assertEqual(mImage1.getBBox(), newBBox)
    for image in mImage1:
        testCase.assertEqual(image.getBBox(), newBBox)

    # Uncomment this test when DM-10781 is implemented
    # offset = Extent2I(-9000, -18000)
    # mImage1.shiftedBy(offset)
    # newBBox = Box2I(Point2I(1000, 2000), newBBox.getDimensions())
    # testCase.assertEqual(mImage1.getBBox(), newBBox)
    # for image in mImage1:
    #    testCase.assertEqual(image.getBBox(), newBBox)


def _testImageCopy(testCase, mImage1, value1, value2):
    """Test copy and deep copy in image-like objects"""
    mImage2 = mImage1.clone()
    mImage2.setXY0(Point2I(11, 23))
    testCase.assertEqual(mImage2.getBBox(), Box2I(Point2I(11, 23), Extent2I(200, 100)))
    testCase.assertEqual(mImage1.getBBox(), Box2I(Point2I(1000, 2000), Extent2I(200, 100)))
    testCase.assertTrue(np.all([s.getBBox() == mImage1.getBBox() for s in mImage1.singles]))
    testCase.assertTrue(np.all([s.getBBox() == mImage2.getBBox() for s in mImage2.singles]))
    mImage2.array[:] = 17
    testCase.assertNotEqual(mImage1.array[0, 0, 0], 17)

    mImage2.array[:] = value2
    testCase.assertFloatsEqual(mImage1.array, value1)
    testCase.assertFloatsEqual(mImage2.array, value2)
    testCase.assertFloatsEqual(mImage1["G"].array, value1)
    testCase.assertFloatsEqual(mImage2["G"].array, value2)

    mImage2 = mImage1.clone(False)
    mImage2.setXY0(Point2I(11, 23))
    mImage2.array[:] = 17
    testCase.assertFloatsEqual(mImage2.array, mImage1.array)

    mImage2.array[:] = value2
    testCase.assertFloatsEqual(mImage1.array, value2)
    testCase.assertFloatsEqual(mImage2.array, value2)
    testCase.assertFloatsEqual(mImage1["G"].array, value2)
    testCase.assertFloatsEqual(mImage2["G"].array, value2)


class MultibandPixelTestCase(lsst.utils.tests.TestCase):
    """Test case for MultibandPixel
    """
    def setUp(self):
        np.random.seed(1)
        self.bbox = Point2I(101, 502)
        self.bands = ["G", "R", "I", "Z", "Y"]
        singles = np.arange(5, dtype=float)
        self.pixel = MultibandPixel(self.bands, singles, self.bbox)

    def tearDown(self):
        del self.bbox
        del self.bands
        del self.pixel

    def testbandSlicing(self):
        pixel = self.pixel
        self.assertEqual(pixel["R"], 1.)
        self.assertFloatsEqual(pixel.array, np.arange(5))
        self.assertFloatsEqual(pixel.singles, np.arange(5))
        self.assertFloatsEqual(pixel[["G", "I"]].array, [0, 2])

    def testPixelBBoxModification(self):
        pixel = self.pixel.clone()
        otherPixel = pixel.clone()
        pixel.getBBox().shift(Extent2I(9, -2))
        self.assertEqual(pixel.getBBox().getMin(), Point2I(110, 500))
        self.assertEqual(otherPixel.getBBox().getMin(), Point2I(101, 502))

        pixel = self.pixel.clone()
        otherPixel = pixel.clone(False)
        pixel.getBBox().shift(Extent2I(9, -2))
        self.assertEqual(pixel.getBBox().getMin(), Point2I(110, 500))
        self.assertEqual(otherPixel.getBBox().getMin(), Point2I(110, 500))

    def testPixelModification(self):
        pixel = self.pixel
        otherPixel = pixel.clone()
        otherPixel.array = np.arange(10, 15)
        self.assertFloatsEqual(otherPixel.array, np.arange(10, 15))
        self.assertFloatsEqual(pixel.array, np.arange(0, 5))


class MultibandImageTestCase(lsst.utils.tests.TestCase):
    """Test case for MultibandImage"""

    def setUp(self):
        np.random.seed(1)
        self.bbox1 = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
        self.bands = ["G", "R", "I", "Z", "Y"]
        self.value1, self.value2 = 10, 100
        images = [ImageF(self.bbox1, self.value1) for f in self.bands]
        self.mImage1 = MultibandImage.fromImages(self.bands, images)
        self.bbox2 = Box2I(Point2I(1100, 2025), Extent2I(30, 50))
        images = [ImageF(self.bbox2, self.value2) for f in self.bands]
        self.mImage2 = MultibandImage.fromImages(self.bands, images)

    def tearDown(self):
        del self.bbox1
        del self.bbox2
        del self.bands
        del self.value1
        del self.value2
        del self.mImage1
        del self.mImage2

    def testbandSlicing(self):
        _testImageBandSlicing(self, self.mImage1, ImageF, self.bbox1, self.value1)

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
        self.bands = ["G", "R", "I"]
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
        singles = [self.Mask(self.bbox) for f in range(len(self.bands))]
        for n in range(len(singles)):
            singles[n].set(self.values1[n])
        self.mMask1 = MultibandMask.fromMasks(self.bands, singles)

        self.values2 = [self.EDGE, self.BAD, self.CR | self.EDGE]
        singles = [self.Mask(self.bbox) for f in range(len(self.bands))]
        for n in range(len(singles)):
            singles[n].set(self.values2[n])
        self.mMask2 = MultibandMask.fromMasks(self.bands, singles)

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
        for n, mask in enumerate(self.mMask1):
            op(mask, self.values2[n])

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
        mMask = self.mMask1.clone()
        mMask[:] = self.CR
        self.assertFloatsEqual(mMask.array, self.CR)
        mMask["G"] = self.EDGE
        self.assertFloatsEqual(mMask["R":].array, self.CR)
        self.assertFloatsEqual(mMask["G"].array, self.EDGE)
        mMask["R":] = self.BAD
        self.assertFloatsEqual(mMask["R":].array, self.BAD)
        mMask["R", 1100, 2050] = self.CR | self.EDGE
        self.assertEqual(mMask["R", 1100, 2050], self.CR | self.EDGE)
        self.assertEqual(mMask["R", 1101, 2051], self.BAD)

    def testMaskPlanes(self):
        planes = self.mMask1.getMaskPlaneDict()
        self.assertEqual(len(planes), self.mMask1.getNumPlanesUsed())

        for k in sorted(planes.keys()):
            self.assertEqual(planes[k], self.mMask1.getMaskPlane(k))

    def testRemoveMaskPlane(self):
        mMask = self.mMask1
        # Add mask plane FOO and make sure it got added properly
        mMask.addMaskPlane("FOO")
        self.assertIn("FOO", mMask.getMaskPlaneDict())
        self.assertIn("FOO", Mask().getMaskPlaneDict())
        # Remove plane FOO, noting that removeMaskPlane removes it from the
        # default, but each instance remembers the version of the mask
        # dictionary that was current when it was created, so it will still
        # be in the mMask dict.
        mMask.removeMaskPlane("FOO")
        self.assertIn("FOO", mMask.getMaskPlaneDict())
        self.assertNotIn("FOO", Mask().getMaskPlaneDict())

    def testRemoveAndClearMaskPlane(self):
        mMask = self.mMask1
        # Add mask plane FOO and test clearing it without removing plane from
        # default dict
        mMask.addMaskPlane("FOO")
        mMask.removeAndClearMaskPlane("FOO")
        self.assertNotIn("FOO", mMask.getMaskPlaneDict())
        self.assertIn("FOO", Mask().getMaskPlaneDict())
        # Now also remove it from default dict
        mMask.addMaskPlane("FOO")
        mMask.removeAndClearMaskPlane("FOO", removeFromDefault=True)
        self.assertNotIn("FOO", mMask.getMaskPlaneDict())
        self.assertNotIn("FOO", Mask().getMaskPlaneDict())
        # Now remove and clear the EDGE mask plane and make sure all of the planes
        # in the MultibandMask (i.e. the "singles") got updated accordingly
        mMask.removeAndClearMaskPlane("EDGE", removeFromDefault=True)
        self.assertNotIn("EDGE", mMask.getMaskPlaneDict())
        self.assertNotIn("EDGE", Mask().getMaskPlaneDict())
        # Assert that all mask planes were updated (i.e. having EDGE removed)
        self.assertTrue(np.all([s.array == self.values1[n] & ~self.EDGE for
                                n, s in enumerate(mMask.singles)]))

    def testbandSlicing(self):
        _testImageBandSlicing(self, self.mMask1, Mask, self.bbox, self.values1[0])

    def testImageSlicing(self):
        _testImageSlicing(self, self.mMask1, *self.values1)

    def testImageModification(self):
        mMask1 = self.mMask1
        value1 = self.CR
        value2 = self.EDGE
        mMask1[:] = value1

        bbox2 = Box2I(Point2I(1100, 2025), Extent2I(30, 50))
        singles = [self.Mask(bbox2) for f in range(len(self.bands))]
        for n in range(len(singles)):
            singles[n].set(value2)
        mMask2 = MultibandMask.fromMasks(self.bands, singles)

        _testImageModification(self, mMask1, mMask2, self.bbox, bbox2, value1, value2)

    def testImageCopy(self):
        mMask = self.mMask1
        value1 = self.CR
        value2 = self.EDGE
        mMask[:] = value1
        _testImageCopy(self, mMask, value1, value2)


def _testMaskedImagebands(testCase, maskedImage, singleType):
    testCase.assertIsInstance(maskedImage["R"], singleType)
    testCase.assertIsInstance(maskedImage.image["G"], ImageF)
    testCase.assertIsInstance(maskedImage.mask["R"], Mask)
    testCase.assertIsInstance(maskedImage.variance["I"], ImageF)

    testCase.assertEqual(maskedImage["G"].image.array.shape, (100, 200))
    testCase.assertEqual(maskedImage[:"I"].mask.array.shape, (2, 100, 200))
    testCase.assertEqual(maskedImage[:"I"].bands, ("G", "R"))
    testCase.assertEqual(maskedImage[:"I"].getBBox(), testCase.bbox)
    testCase.assertEqual(maskedImage[["G", "I"]].getBBox(), testCase.bbox)

    testCase.assertEqual(maskedImage.bands, tuple(testCase.bands))


def _testMaskedImageSlicing(testCase, maskedImage):
    subBox = Box2I(Point2I(1100, 2025), Extent2I(30, 50))
    testCase.assertEqual(maskedImage[:, subBox].getBBox(), subBox)
    testCase.assertEqual(maskedImage[:, subBox].image.getBBox(), subBox)
    testCase.assertEqual(maskedImage[:, subBox].mask.getBBox(), subBox)
    testCase.assertEqual(maskedImage[:, subBox].variance.getBBox(), subBox)

    maskedPixel = maskedImage[:, 1100, 2025]
    testCase.assertFloatsEqual(maskedPixel[0].array, np.array([10., 10., 10.]))
    testCase.assertFloatsEqual(maskedPixel[1].array, np.array([1, 1, 1]))
    testCase.assertFloatsAlmostEqual(maskedPixel[2].array, np.array([.01, .01, .01]), 1e-6)

    newBox = Box2I(Point2I(100, 500), Extent2I(200, 100))
    maskedImage.setXY0(newBox.getMin())
    testCase.assertEqual(maskedImage.getBBox(), newBox)
    testCase.assertEqual(maskedImage.image.getBBox(), newBox)
    testCase.assertEqual(maskedImage.mask.getBBox(), newBox)
    testCase.assertEqual(maskedImage.variance.getBBox(), newBox)
    testCase.assertEqual(maskedImage["G"].getBBox(), newBox)
    testCase.assertEqual(maskedImage["G"].image.getBBox(), newBox)
    testCase.assertEqual(maskedImage["R"].mask.getBBox(), newBox)
    testCase.assertEqual(maskedImage["I"].variance.getBBox(), newBox)


def _testMaskedmageModification(testCase, maskedImage):
    images = [ImageF(testCase.bbox, 10*testCase.imgValue) for f in testCase.bands]
    mImage = MultibandImage.fromImages(testCase.bands, images)
    maskedImage.image.array = mImage.array
    testCase.assertFloatsEqual(maskedImage.image["G"].array, mImage.array[0])
    testCase.assertFloatsEqual(maskedImage["G"].image.array, mImage.array[0])

    singles = [testCase.Mask(testCase.bbox) for f in range(len(testCase.bands))]
    for n in range(len(singles)):
        singles[n].set(testCase.maskValue*2)
    mMask = MultibandMask.fromMasks(testCase.bands, singles)
    maskedImage.mask.array = mMask.array
    testCase.assertFloatsEqual(maskedImage.mask["G"].array, mMask.array[0])
    testCase.assertFloatsEqual(maskedImage["G"].mask.array, mMask.array[0])

    images = [ImageF(testCase.bbox, .1 * testCase.varValue) for f in testCase.bands]
    mVariance = MultibandImage.fromImages(testCase.bands, images)
    maskedImage.variance.array = mVariance.array
    testCase.assertFloatsEqual(maskedImage.variance["G"].array, mVariance.array[0])
    testCase.assertFloatsEqual(maskedImage["G"].variance.array, mVariance.array[0])

    subBox = Box2I(Point2I(1100, 2025), Extent2I(30, 50))
    maskedImage.image[:, subBox].array = 12
    testCase.assertFloatsEqual(maskedImage.image["G", subBox].array, 12)
    testCase.assertFloatsEqual(maskedImage["G", subBox].image.array, 12)
    maskedImage["R", subBox].image[:] = 15
    testCase.assertFloatsEqual(maskedImage.image["R", subBox].array, 15)
    testCase.assertFloatsEqual(maskedImage["R", subBox].image.array, 15)

    maskedImage.mask[:, subBox].array = 64
    testCase.assertFloatsEqual(maskedImage.mask["G", subBox].array, 64)
    testCase.assertFloatsEqual(maskedImage["G", subBox].mask.array, 64)
    maskedImage["R", subBox].mask[:] = 128
    testCase.assertFloatsEqual(maskedImage.mask["R", subBox].array, 128)
    testCase.assertFloatsEqual(maskedImage["R", subBox].mask.array, 128)

    maskedImage.variance[:, subBox].array = 1e-6
    testCase.assertFloatsEqual(maskedImage.variance["G", subBox].array, 1e-6)
    testCase.assertFloatsEqual(maskedImage["G", subBox].variance.array, 1e-6)
    maskedImage["R", subBox].variance[:] = 1e-7
    testCase.assertFloatsEqual(maskedImage.variance["R", subBox].array, 1e-7)
    testCase.assertFloatsEqual(maskedImage["R", subBox].variance.array, 1e-7)


def _testMaskedImageCopy(testCase, maskedImage1):
    maskedImage2 = maskedImage1.clone()

    maskedImage2.setXY0(Point2I(11, 23))
    testCase.assertEqual(maskedImage2.getBBox(), Box2I(Point2I(11, 23), Extent2I(200, 100)))
    testCase.assertEqual(maskedImage1.getBBox(), Box2I(Point2I(1000, 2000), Extent2I(200, 100)))
    testCase.assertTrue(np.all([img.getBBox() == maskedImage1.getBBox() for img in maskedImage1.image]))
    testCase.assertTrue(np.all([img.getBBox() == maskedImage2.getBBox() for img in maskedImage2.image]))

    maskedImage2.image.array = 1
    testCase.assertFloatsEqual(maskedImage1.image.array, testCase.imgValue)
    testCase.assertFloatsEqual(maskedImage2.image.array, 1)
    testCase.assertFloatsEqual(maskedImage1["G"].image.array, testCase.imgValue)
    testCase.assertFloatsEqual(maskedImage2["G"].image.array, 1)

    maskedImage2 = maskedImage1.clone(False)
    maskedImage2.image.array = 1
    testCase.assertFloatsEqual(maskedImage1.image.array, 1)
    testCase.assertFloatsEqual(maskedImage2.image.array, 1)
    testCase.assertFloatsEqual(maskedImage1["G"].image.array, 1)
    testCase.assertFloatsEqual(maskedImage2["G"].image.array, 1)


class MultibandMaskedImageTestCase(lsst.utils.tests.TestCase):
    """Test case for MultibandMaskedImage"""

    def setUp(self):
        np.random.seed(1)
        self.bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
        self.bands = ["G", "R", "I"]

        self.imgValue = 10
        images = [ImageF(self.bbox, self.imgValue) for f in self.bands]
        mImage = MultibandImage.fromImages(self.bands, images)

        self.Mask = Mask[MaskPixel]
        # Store the default mask planes for later use
        maskPlaneDict = self.Mask().getMaskPlaneDict()
        self.defaultMaskPlanes = sorted(maskPlaneDict, key=maskPlaneDict.__getitem__)

        # reset so tests will be deterministic
        self.Mask.clearMaskPlaneDict()
        for p in ("BAD", "SAT", "INTRP", "CR", "EDGE"):
            self.Mask.addMaskPlane(p)

        self.maskValue = self.Mask.getPlaneBitMask("BAD")
        singles = [self.Mask(self.bbox) for f in range(len(self.bands))]
        for n in range(len(singles)):
            singles[n].set(self.maskValue)
        mMask = MultibandMask.fromMasks(self.bands, singles)

        self.varValue = 1e-2
        images = [ImageF(self.bbox, self.varValue) for f in self.bands]
        mVariance = MultibandImage.fromImages(self.bands, images)

        self.maskedImage = MultibandMaskedImage(self.bands, image=mImage, mask=mMask, variance=mVariance)

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

    def testbandSlicing(self):
        _testMaskedImagebands(self, self.maskedImage, MaskedImage)

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
        self.bands = ["G", "R", "I"]

        self.imgValue = 10
        images = [ImageF(self.bbox, self.imgValue) for f in self.bands]
        mImage = MultibandImage.fromImages(self.bands, images)

        self.Mask = Mask[MaskPixel]
        # Store the default mask planes for later use
        maskPlaneDict = self.Mask().getMaskPlaneDict()
        self.defaultMaskPlanes = sorted(maskPlaneDict, key=maskPlaneDict.__getitem__)

        # reset so tests will be deterministic
        self.Mask.clearMaskPlaneDict()
        for p in ("BAD", "SAT", "INTRP", "CR", "EDGE"):
            self.Mask.addMaskPlane(p)

        self.maskValue = self.Mask.getPlaneBitMask("BAD")
        singles = [self.Mask(self.bbox) for f in range(len(self.bands))]
        for n in range(len(singles)):
            singles[n].set(self.maskValue)
        mMask = MultibandMask.fromMasks(self.bands, singles)

        self.varValue = 1e-2
        images = [ImageF(self.bbox, self.varValue) for f in self.bands]
        mVariance = MultibandImage.fromImages(self.bands, images)

        self.kernelSize = 51
        self.psfs = [GaussianPsf(self.kernelSize, self.kernelSize, 4.0) for f in self.bands]
        self.psfImage = np.array([
            p.computeKernelImage(p.getAveragePosition()).array for p in self.psfs
        ])

        self.exposure = MultibandExposure(image=mImage, mask=mMask, variance=mVariance,
                                          psfs=self.psfs, bands=self.bands)

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

    def testConstructor(self):
        exposures = self.exposure.singles
        bands = self.exposure.bands
        newExposure = MultibandExposure.fromExposures(bands, exposures)
        for exposure in newExposure.singles:
            self.assertIsNotNone(exposure.getPsf())

    def testbandSlicing(self):
        _testMaskedImagebands(self, self.exposure, ExposureF)

    def testImageSlicing(self):
        _testMaskedImageSlicing(self, self.exposure)

    def testModification(self):
        _testMaskedmageModification(self, self.exposure)

    def testCopy(self):
        _testMaskedImageCopy(self, self.exposure)

    def testPsf(self):
        psfImage = self.exposure.computePsfKernelImage(self.exposure.getBBox().getCenter())
        self.assertFloatsAlmostEqual(psfImage.array, self.psfImage)

        newPsfs = [GaussianPsf(self.kernelSize, self.kernelSize, 1.0) for f in self.bands]
        newPsfImage = [p.computeImage(p.getAveragePosition()).array for p in newPsfs]
        for psf, exposure in zip(newPsfs, self.exposure.singles):
            exposure.setPsf(psf)
        psfImage = self.exposure.computePsfKernelImage(self.exposure.getBBox().getCenter())
        self.assertFloatsAlmostEqual(psfImage.array, newPsfImage)

        psfImage = self.exposure.computePsfImage(self.exposure.getBBox().getCenter())["G"]
        self.assertFloatsAlmostEqual(
            psfImage.array,
            self.exposure["G"].getPsf().computeImage(
                self.exposure["G"].getPsf().getAveragePosition()
            ).array
        )


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
        self.bands = ("G", "R", "I")
        singles = []
        images = []
        for n, f in enumerate(self.bands):
            image = ImageF(self.spans.getBBox())
            image.set(n)
            images.append(image.array)
            maskedImage = MaskedImageF(image)
            heavy = makeHeavyFootprint(self.footprint, maskedImage)
            singles.append(heavy)
        self.image = np.array(images)
        self.mFoot = MultibandFootprint(self.bands, singles)

    def tearDown(self):
        del self.spans
        del self.footprint
        del self.peaks
        del self.bbox
        del self.bands
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

        def runTest(images, mFoot, peaks=self.peaks, footprintBBox=Box2I(Point2I(6, 6), Extent2I(9, 9))):
            self.assertEqual(mFoot.getBBox(), footprintBBox)
            try:
                fpImage = np.array(images)[:, 1:-1, 1:-1]
            except IndexError:
                fpImage = np.array([img.array for img in images])[:, 1:-1, 1:-1]
            # result = mFoot.getImage(fill=0).image.array
            self.assertFloatsAlmostEqual(mFoot.getImage(fill=0).image.array, fpImage)
            if peaks is not None:
                self.verifyPeaks(mFoot.getPeaks(), peaks)

        bbox = Box2I(Point2I(5, 5), Extent2I(11, 11))
        xy0 = Point2I(5, 5)

        images = np.array([projectSpans(n, 5-n, bbox, True) for n in range(2, 5)])
        mFoot = MultibandFootprint.fromArrays(self.bands, images, xy0=xy0, peaks=self.peaks)
        runTest(images, mFoot)

        mFoot = MultibandFootprint.fromArrays(self.bands, images)
        runTest(images, mFoot, None, Box2I(Point2I(1, 1), Extent2I(9, 9)))

        images = [projectSpans(n, 5-n, bbox, False) for n in range(2, 5)]
        mFoot = MultibandFootprint.fromImages(self.bands, images, peaks=self.peaks)
        runTest(images, mFoot)

        images = np.array([projectSpans(n, n, bbox, True) for n in range(2, 5)])
        mFoot = MultibandFootprint.fromArrays(self.bands, images, peaks=self.peaks, xy0=bbox.getMin())
        runTest(images, mFoot)

        images = np.array([projectSpans(n, 5-n, bbox, True) for n in range(2, 5)])
        thresh = [1, 2, 2.5]
        mFoot = MultibandFootprint.fromArrays(self.bands, images, xy0=bbox.getMin(), thresh=thresh)
        footprintBBox = Box2I(Point2I(8, 8), Extent2I(5, 5))
        self.assertEqual(mFoot.getBBox(), footprintBBox)

        fpImage = np.array(images)[:, 3:-3, 3:-3]
        mask = np.all(fpImage <= np.array(thresh)[:, None, None], axis=0)
        fpImage[:, mask] = 0
        self.assertFloatsAlmostEqual(mFoot.getImage(fill=0).image.array, fpImage)
        img = mFoot.getImage().image.array
        img[~np.isfinite(img)] = 1.1
        self.assertFloatsAlmostEqual(mFoot.getImage(fill=1.1).image.array, img)

    def testSlicing(self):
        self.assertIsInstance(self.mFoot["R"], HeavyFootprintF)
        self.assertIsInstance(self.mFoot[:], MultibandFootprint)

        self.assertEqual(self.mFoot["I"], self.mFoot["I"])
        self.assertEqual(self.mFoot[:"I"].bands, ("G", "R"))
        self.assertEqual(self.mFoot[:"I"].getBBox(), self.bbox)
        self.assertEqual(self.mFoot[["G", "I"]].bands, ("G", "I"))
        self.assertEqual(self.mFoot[["G", "I"]].getBBox(), self.bbox)

        with self.assertRaises(TypeError):
            self.mFoot["I", 4, 5]
            self.mFoot["I", :, :]
        with self.assertRaises(IndexError):
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

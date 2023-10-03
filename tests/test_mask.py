# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tests of the Mask and MaskDict objects."""

import unittest
import warnings

import numpy as np

import lsst.afw.image as afwImage
import lsst.utils.tests


class MaskDictTestCase(lsst.utils.tests.TestCase):
    """Test the python MaskDict interface to the underlying C_MaskDict .
    """
    def setUp(self):
        # Default mask planes, in bit number order.
        self.defaultPlanes = ("BAD", "SAT", "INTRP", "CR", "EDGE", "DETECTED",
                              "DETECTED_NEGATIVE", "SUSPECT", "NO_DATA")

    def testStr(self):
        default = afwImage.MaskDict(32)
        expect = ("Plane 0 -> BAD : This pixel is known to be bad (e.g. the amplifier is not working).\n"
                  "Plane 1 -> SAT : This pixel is saturated and has bloomed.\n"
                  "Plane 2 -> INTRP : This pixel has been interpolated over. Check other mask planes for the reason.\n"  # noqa: E501
                  "Plane 3 -> CR : This pixel is contaminated by a cosmic ray.\n"
                  "Plane 4 -> EDGE : This pixel is too close to the edge to be processed properly.\n"
                  "Plane 5 -> DETECTED : This pixel lies within an object's Footprint.\n"
                  "Plane 6 -> DETECTED_NEGATIVE : This pixel lies within an object's Footprint, and the detection was looking for pixels *below* a specified level.\n"  # noqa: E501
                  "Plane 7 -> SUSPECT : This pixel is untrustworthy (e.g. contains an instrumental flux in ADU above the correctable non-linear regime).\n"  # noqa: E501
                  "Plane 8 -> NO_DATA : There was no data at this pixel location (e.g. no input images at this location in a coadd, or extremely high vignetting, such that there is no incoming signal)."  # noqa: E501
                  )

        self.assertEqual(str(default), expect)

        # Remove and re-add the first numbered mask; printed order should not
        # change. This tests that the output order is sorted on bit number,
        # not on map insertion order.
        maskDict = afwImage.MaskDict(32)
        maskDict.remove("BAD")
        # TODO: have to actually implement `add` fully!
        maskDict.add("BAD", "Adding original first plane")
        temp = expect.splitlines()[1:]
        temp.insert(0, "Plane 0 -> BAD : Adding original first plane")
        expect = "\n".join(temp)
        self.assertEqual(str(maskDict), expect)

    def testConstructTwoDicts(self):
        """Test the constructor that takes two dicts.
        """
        bits = {"bit1": 1, "bit5": 5}
        docs = {"bit1": "doc for bit 1", "bit5": "doc for bit 5"}
        maskDict = afwImage.MaskDict(32, bits, docs)
        self.assertEqual(list(maskDict.items()), list(bits.items()))
        self.assertEqual(list(maskDict.docs.items()), list(docs.items()))

    def testDictLike(self):
        """Test the (frozen) dict-like behavior of MaskDict and the docs.

        We don't check eq/ne here, as they only apply to MaskDict-to-MaskDict
        comparisons, not MaskDict-to-dict, and eq/ne underpins most of the
        other tests in this class.
        """
        maskDict = afwImage.MaskDict(32)
        self.assertIn("BAD", maskDict)
        self.assertEqual(maskDict["BAD"], 0)
        self.assertEqual(maskDict.get("BAD"), 0)
        self.assertNotIn("NotIn", maskDict)
        with self.assertRaises(KeyError):
            maskDict["NotIn"]
        self.assertIsNone(maskDict.get("NotIn"))
        self.assertEqual(len(maskDict), 9)
        with self.assertRaises(TypeError):
            maskDict["CannotAdd"] = 10
        with self.assertRaises(TypeError):
            del maskDict["BAD"]

        # keys, values, items
        self.assertEqual(set(maskDict.keys()), set(self.defaultPlanes))
        self.assertEqual(set(maskDict.values()), set(range(len(self.defaultPlanes))))
        self.assertEqual(set(maskDict.items()), {(x, i) for i, x in enumerate(self.defaultPlanes)})

        # iteration
        self.assertEqual(set(x for x in maskDict), set(self.defaultPlanes))

        # Check that the docs behave like a dict, too.
        self.assertIn("BAD", maskDict.docs)
        self.assertNotIn("NotIn", maskDict.docs)
        self.assertEqual(len(maskDict.docs), 9)
        with self.assertRaises(TypeError):
            maskDict.docs["CannotAdd"] = "sometext"
        with self.assertRaises(TypeError):
            del maskDict.docs["BAD"]

    def testClone(self):
        """Check that clone() does not share internals.
        """
        maskDict = afwImage.MaskDict(32)
        cloned = maskDict.clone()
        self.assertEqual(maskDict, cloned)
        maskDict.add("new", "new doc")
        self.assertNotEqual(maskDict, cloned)

    def testAddPlanes(self):
        """Check the behavior of `mask.addPlanes`.

        Some of these tests may seem pedantic, but they are here to confirm
        the new behavior of the non-static/non-global MaskDict, in comparison
        with the old one (e.g. changing one plane doesn't affect others).
        """
        mask1 = afwImage.Mask(100, 200)
        mask2 = afwImage.Mask(100, 200)
        # Initially, all MaskDicts match the default one.
        self.assertEqual(mask1.maskDict, mask2.maskDict)
        self.assertEqual(mask1.maskDict, afwImage.MaskDict(32))
        self.assertEqual(mask1.getNumPlanesUsed(), 9)

        # Normal behavior when adding a new plane that doesn't exist: keep the
        # same object, but it does not affect other planes.
        bit = mask1.addPlane("TEST", "some docs")
        self.assertEqual(bit, 9)
        self.assertEqual(mask1.getNumPlanesUsed(), 10)
        self.assertNotEqual(mask1.maskDict, mask2.maskDict)
        self.assertIn("TEST", mask1.maskDict)
        self.assertNotIn("TEST", mask2.maskDict)

        # Adding the same plane with an empty doc does not change anything.
        cloned = mask1.maskDict.clone()
        bit = mask1.addPlane("TEST", "")
        self.assertEqual(bit, 9)
        self.assertEqual(mask1.getNumPlanesUsed(), 10)
        self.assertEqual(mask1.maskDict, cloned)

        # Adding the same plane with the same doc does not change anything.
        cloned = mask1.maskDict.clone()
        bit = mask1.addPlane("TEST", "some docs")
        self.assertEqual(bit, 9)
        self.assertEqual(mask1.getNumPlanesUsed(), 10)
        self.assertEqual(mask1.maskDict, cloned)

        # Adding the same plane name with new docs will not modify and raise.
        cloned = mask1.maskDict.clone()
        with self.assertRaisesRegex(RuntimeError, "Not changing existing docstring for plane 'TEST'"):
            mask1.addPlane("TEST", "new docs")
        self.assertEqual(mask1.maskDict, cloned)

        # TODO: redo c++ check of internal coverage: I think the below is unnecessary.

        # Adding an empty doc and then adding a non-empty docstring on the
        # same key just fills in the empty value, without bifurcating.
        bit = mask2.addPlane("EMPTYDOC", "")
        self.assertEqual(bit, 9)
        self.assertEqual(mask2.getNumPlanesUsed(), 10)
        self.assertEqual(mask2.maskDict.docs["EMPTYDOC"], "")
        bit = mask2.addPlane("EMPTYDOC", "non empty doc")
        self.assertEqual(bit, 9)
        self.assertEqual(mask2.getNumPlanesUsed(), 10)
        self.assertEqual(mask2.maskDict.docs["EMPTYDOC"], "non empty doc")

    def testExceedMaxPlanes(self):
        """Test raising when adding more planes than the maximum allowed.
        """
        maxBits = 32
        maskDict = afwImage.MaskDict(maxBits)
        for i in range(maxBits - len(maskDict)):
            maskDict.add(f"plane{i}", f"docstring {i}")
        self.assertEqual(len(maskDict), maxBits)
        with self.assertRaisesRegex(RuntimeError, "Max number of planes"):
            maskDict.add("TooBig", "not enough bits for this one")

    def testRemove(self):
        """Test remove() on maskDict independent of a Mask.

        Some of these tests may seem pedantic, but they are here to confirm
        the new behavior of the non-static/non-global MaskDict, in comparison
        with the old one (e.g. changing one plane doesn't affect others).
        """
        maskDict1 = afwImage.MaskDict(32)
        maskDict2 = afwImage.MaskDict(32)
        maskDict1.add("TEST", "some docs")
        maskDict2.add("TEST", "other docs")
        # consistency checks
        self.assertEqual(maskDict1.docs["TEST"], "some docs")
        self.assertEqual(maskDict2.docs["TEST"], "other docs")

        # Does not modify mask2.
        maskDict1.remove("TEST")
        self.assertNotIn("TEST", maskDict1)
        self.assertNotIn("TEST", maskDict1.docs)
        self.assertEqual(maskDict2.docs["TEST"], "other docs")

        # Removing a plane in the middle should re-use that bit.
        originalBit = maskDict2["EDGE"]
        maskDict2.remove("EDGE")
        self.assertNotIn("EDGE", maskDict2)
        maskDict2.add("EDGE", "new edge")
        self.assertEqual(originalBit, maskDict2["EDGE"])


class MaskTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        np.random.seed(1)  # for consistency
        self.mask1 = afwImage.Mask(100, 200)
        self.mask2 = afwImage.Mask(self.mask1.getDimensions())

        BAD = self.mask1.getBitMask("BAD")
        CR = self.mask1.getBitMask("CR")
        EDGE = self.mask1.getBitMask("EDGE")
        SAT = self.mask1.getBitMask("SAT")

        # So we can both AND and OR the masks in tests.
        self.value1 = BAD | CR
        self.value2 = CR | EDGE

        self.mask1.set(self.value1)
        self.mask2.set(self.value2)

    def testSubsetMakeDict(self):
        """Check that a subset cutout shares the MaskDict with the parent.
        """
        box = lsst.geom.Box2I(lsst.geom.Point2I(10, 10),
                              lsst.geom.Extent2I(10, 20))
        self.mask1.addPlane("new", "new docstring")
        cutout = self.mask1.subset(box)
        self.assertEqual(cutout.maskDict, self.mask1.maskDict)
        self.mask1.addPlane("other", "other docstring")
        self.assertEqual(cutout.maskDict, self.mask1.maskDict)

        # Trying to remove a plane from a shared maskDict raises:
        with self.assertRaisesRegex(RuntimeError,
                                    "Cannot remove plane 'new'; there are '3' entities sharing this"):
            self.mask1.maskDict.remove("new")

    def testGetBitMask(self):
        """Test getting mask planes from a Mask.
        """
        self.assertEqual(self.mask1.getBitMask("BAD"), 2**0)
        self.assertEqual(self.mask1.getBitMask("EDGE"), 2**4)
        self.assertEqual(self.mask1.getBitMask(("BAD", "EDGE")), 2**0 + 2**4)

    # def testAddMaskPlaneDeprecation(self):
    #     """Remove this method on DM-XXXXX once the doc is non-optional.
    #     """
    #     with self.assertWarnsRegex(FutureWarning, "Doc field will become non-optional"):
    #         self.mask1.addMaskPlane("NODOCSTRING")

    #     # This should not raise
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("error", category=FutureWarning)
    #         self.mask1.addMaskPlane("YESDOCSTRING", "some docs")

    def testArrays(self):
        """Tests of the ndarray interface to the underlying Mask array.
        """
        # could use Mask(5, 6) but check extent(5, 6) form too
        image1 = afwImage.Mask(lsst.geom.ExtentI(5, 6))
        self.assertEqual(image1.array.shape[0], image1.getHeight())
        self.assertEqual(image1.array.shape[1], image1.getWidth())

        image2 = afwImage.Mask(image1.array, False)
        self.assertEqual(image1.array.shape[0], image2.getHeight())
        self.assertEqual(image1.array.shape[1], image2.getWidth())

        image3 = afwImage.makeMaskFromArray(image1.array)
        self.assertEqual(image1.array.shape[0], image2.getHeight())
        self.assertEqual(image1.array.shape[1], image2.getWidth())
        self.assertEqual(type(image3), afwImage.Mask[afwImage.MaskPixel])
        image1.array[:, :] = np.random.uniform(low=0, high=10, size=image1.array.shape)
        self.assertMasksEqual(image1, image1.array)
        array3 = np.random.uniform(low=0, high=10, size=image1.array.shape).astype(image1.array.dtype)
        image1.array = array3
        np.testing.assert_array_equal(image1.array, array3)

    def testInitializeMask(self):
        value = 0x1234
        mask = afwImage.Mask(lsst.geom.ExtentI(10, 10), value)
        self.assertEqual(mask[0, 0], value)

    def testOrMasks(self):
        expect = np.empty_like(self.mask1.array)
        expect[:] = self.value2 | self.value1

        # Check or-ing with another mask array.
        self.mask2 |= self.mask1
        self.assertMasksEqual(self.mask2, expect)

        # Check or-ing with a number.
        self.mask1 |= self.value2
        self.assertMasksEqual(self.mask1, expect)

    def testAndMasks(self):
        expect = np.empty_like(self.mask1.array)
        expect[:] = self.value1 & self.value2

        # Check and-ing with another mask array.
        self.mask2 &= self.mask1
        self.assertMasksEqual(self.mask2, expect)

        # Check and-ing with a number.
        self.mask1 &= self.value2
        self.assertMasksEqual(self.mask1, expect)

    def testXorMasks(self):
        expect = np.empty_like(self.mask1.array)
        expect[:] = self.value1 ^ self.value2

        # Check xor-ing with another mask array.
        self.mask2 ^= self.mask1
        self.assertMasksEqual(self.mask2, expect)

        # Check xor-ing with a number.
        self.mask1 ^= self.value2
        self.assertMasksEqual(self.mask1, expect)

    def testLogicalMasksMismatch(self):
        """Logical operations on Masks of different sizes should raise.
        """
        i1 = afwImage.Mask(lsst.geom.ExtentI(100, 100))
        i1.set(100)
        i2 = afwImage.Mask(lsst.geom.ExtentI(10, 10))
        i2.set(10)

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            i1 |= i2

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            i1 &= i2

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            i1 ^= i2


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

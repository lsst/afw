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

import numpy as np

import lsst.afw.image as afwImage
import lsst.utils.tests


class MaskDictTestCase(lsst.utils.tests.TestCase):
    """Test the python MaskDict interface to the underlying C_MaskDict .
    """
    def setUp(self):
        np.random.seed(1)  # for consistency
        # Default mask planes, in bit number order.
        self.defaultPlanes = ('BAD', 'CR', 'DETECTED', 'DETECTED_NEGATIVE', 'EDGE',
                              'INTRP', 'NO_DATA', 'SAT', 'SUSPECT')
        # Ensure the planes are always the same at the start of each test.
        afwImage.Mask.restoreDefaultMaskDict()

    def testDefaults(self):
        """Test behavior of the default and canonical mask plane dicts."""
        # Check that the default masks exist.
        mask = afwImage.Mask(100, 200)
        default = afwImage.MaskDict.getDefault()
        self.assertEqual(tuple(default.keys()), self.defaultPlanes)
        # Check that all of the default docstrings are non-empty.
        for doc in default.doc:
            self.assertNotEqual(doc, "")

        # Check clearing of default mask while leaving canonical planes.
        afwImage.Mask.clearDefaultMaskDict()
        self.assertEqual(default, {})
        # Haven't cleared the canonical planes, so new bits don't start at 0.
        bit = mask.addPlane("NEW", "a new plane")
        self.assertEqual(bit, len(self.defaultPlanes))
        # TODO: should this be changing the default list?
        self.assertEqual(default["NEW"], len(self.defaultPlanes))
        self.assertEqual(mask.maskDict["NEW"], len(self.defaultPlanes))

        afwImage.Mask.restoreDefaultMaskDict()
        self.assertEqual(tuple(default.keys()), self.defaultPlanes)

        # Check clearing of canonical planes.
        afwImage.Mask.clearDefaultMaskDict(clearCanonical=True)
        self.assertEqual(default, {})
        bit = mask.addPlane("NEW0", "a new plane for bit 0")
        self.assertEqual(bit, 0)
        # TODO: should this be changing the default list?
        self.assertEqual(list(default.items()), [("NEW0", 0)])
        self.assertEqual(list(mask.maskDict.items()), [("NEW0", 0)])

        afwImage.Mask.restoreDefaultMaskDict()
        self.assertEqual(tuple(default.keys()), self.defaultPlanes)
        self.assertEqual(tuple(mask.maskDict.keys()), self.defaultPlanes)

    def testStr(self):
        default = afwImage.MaskDict.getDefault()
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

    def testDictLike(self):
        """Test the (frozen) dict-like behavior of MaskDict and the docs.
        """
        mask = afwImage.Mask(100, 200)
        self.assertIn("BAD", mask.maskDict)
        self.assertNotIn("NotIn", mask.maskDict)
        self.assertEqual(len(mask.maskDict), 9)
        with self.assertRaises(TypeError):
            mask.maskDict["CannotAdd"] = 10
        with self.assertRaises(TypeError):
            del mask.maskDict["BAD"]

        self.assertIn("BAD", mask.maskDict.doc)
        self.assertNotIn("NotIn", mask.maskDict.doc)
        self.assertEqual(len(mask.maskDict.doc), 9)
        with self.assertRaises(TypeError):
            mask.maskDict.doc["CannotAdd"] = "sometext"
        with self.assertRaises(TypeError):
            del mask.maskDict.doc["BAD"]

    def testAddPlanes(self):
        mask1 = afwImage.Mask(100, 200)
        mask2 = afwImage.Mask(100, 200)
        # Initially, all MaskDicts are the default one.
        self.assertIs(mask1.maskDict, mask2.maskDict)
        self.assertIs(mask1.maskDict, afwImage.MaskDict.getDefault())
        self.assertEqual(mask1.getNumPlanesUsed(), 9)

        # Normal behavior when adding a new plane that doesn't exist: keep the
        # same object, thus adding it to the default dict as well.
        bit = mask1.addPlane("TEST", "some docs")
        self.assertEqual(bit, 9)
        self.assertEqual(mask1.getNumPlanesUsed(), 10)
        self.assertIs(mask1.maskDict, mask2.maskDict)
        self.assertIs(mask1.maskDict, afwImage.MaskDict.getDefault())
        self.assertIn("TEST", mask1.maskDict)
        self.assertIn("TEST", mask2.maskDict)

        # Adding the same plane with an empty doc does not change anything.
        bit = mask1.addPlane("TEST", "")
        self.assertEqual(bit, 9)
        self.assertEqual(mask1.getNumPlanesUsed(), 10)
        self.assertIs(mask1.maskDict, mask2.maskDict)

        # Adding the same plane with the same doc does not change anything.
        bit = mask1.addPlane("TEST", "some docs")
        self.assertEqual(bit, 9)
        self.assertEqual(mask1.getNumPlanesUsed(), 10)
        self.assertIs(mask1.maskDict, mask2.maskDict)

        # Adding the same plane name with new docs will bifurcate this object
        # from the others.
        bit = mask1.addPlane("TEST", "new docs")
        self.assertEqual(bit, 9)
        self.assertEqual(mask1.getNumPlanesUsed(), 10)
        self.assertEqual(mask2.getNumPlanesUsed(), 10)
        self.assertIsNot(mask1.maskDict, mask2.maskDict)
        self.assertEqual(mask1.maskDict.doc["TEST"], "new docs")
        self.assertEqual(mask2.maskDict.doc["TEST"], "some docs")
        # The default list retains the original definition.
        self.assertEqual(afwImage.MaskDict.getDefault().doc["TEST"], "some docs")

        # Adding the same plane with an empty doc does not change anything.
        bit = mask1.addPlane("TEST", "")
        self.assertEqual(bit, 9)
        self.assertEqual(mask1.getNumPlanesUsed(), 10)
        self.assertIsNot(mask1.maskDict, mask2.maskDict)
        self.assertEqual(mask1.maskDict.doc["TEST"], "new docs")
        self.assertEqual(mask2.maskDict.doc["TEST"], "some docs")

        # Adding the same plane with the same doc does not change anything.
        bit = mask1.addPlane("TEST", "new docs")
        self.assertEqual(bit, 9)
        self.assertEqual(mask1.getNumPlanesUsed(), 10)
        self.assertIsNot(mask1.maskDict, mask2.maskDict)
        self.assertEqual(mask1.maskDict.doc["TEST"], "new docs")
        self.assertEqual(mask2.maskDict.doc["TEST"], "some docs")

        # Adding an empty doc and then adding a non-empty docstring on the
        # same key just fills in the empty value, without bifurcating.
        bit = mask2.addPlane("EMPTYDOC", "")
        self.assertEqual(bit, 10)
        self.assertEqual(mask1.getNumPlanesUsed(), 10)
        self.assertEqual(mask2.getNumPlanesUsed(), 11)
        self.assertIs(mask2.maskDict, afwImage.MaskDict.getDefault())
        self.assertEqual(mask2.maskDict.doc["EMPTYDOC"], "")
        bit = mask2.addPlane("EMPTYDOC", "non empty doc")
        self.assertEqual(bit, 10)
        self.assertEqual(mask1.getNumPlanesUsed(), 10)
        self.assertEqual(mask2.getNumPlanesUsed(), 11)
        self.assertIs(mask2.maskDict, afwImage.MaskDict.getDefault())
        self.assertEqual(mask2.maskDict.doc["EMPTYDOC"], "non empty doc")

    def testAddPlanesMax(self):
        """Test raising when adding more planes than the maximum allowed.
        """
        mask = afwImage.Mask(100, 200)
        maxbits = mask.array.dtype.type().nbytes * 8
        for i in range(maxbits - len(mask.maskDict)):
            mask.addPlane(f"plane{i}", f"docstring {i}")
        self.assertEqual(mask.getNumPlanesUsed(), maxbits)
        with self.assertRaisesRegex(RuntimeError, "Max number of planes"):
            mask.addPlane("TooBig", "not enough bits for this one")

    def testRemoveAndClearMaskPlane(self):
        mask1 = afwImage.Mask(100, 200)
        mask2 = afwImage.Mask(100, 200)
        mask1.addPlane("TEST", "some docs")
        mask1.addPlane("TEST", "new docs")
        # consistency checks
        self.assertEqual(mask1.maskDict.doc["TEST"], "new docs")
        self.assertEqual(mask2.maskDict.doc["TEST"], "some docs")
        self.assertIs(mask2.maskDict, afwImage.MaskDict.getDefault())

        # Does not modify the default dict, and thus does not modify mask2.
        mask1.removeAndClearMaskPlane("TEST")
        self.assertNotIn("TEST", mask1.maskDict)
        self.assertEqual(mask2.maskDict.doc["TEST"], "some docs")
        self.assertIs(mask2.maskDict, afwImage.MaskDict.getDefault())

        mask1.addPlane("TEST", "new docs")
        # consistency checks
        self.assertEqual(mask1.maskDict.doc["TEST"], "new docs")
        self.assertEqual(mask2.maskDict.doc["TEST"], "some docs")

        # Does modify the default dict, but still does not change mask2;
        # mask2 and the default have now bifurcated.
        mask1.removeAndClearMaskPlane("TEST", removeFromDefault=True)
        self.assertNotIn("TEST", mask1.maskDict)
        self.assertNotIn("TEST", afwImage.MaskDict.getDefault())
        self.assertEqual(mask2.maskDict.doc["TEST"], "some docs")
        self.assertIsNot(mask2.maskDict, afwImage.MaskDict.getDefault())
        self.assertIsNot(mask1.maskDict, afwImage.MaskDict.getDefault())


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

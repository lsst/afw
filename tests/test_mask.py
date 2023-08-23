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
        # self.Mask = afwImage.Mask[afwImage.MaskPixel]
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


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

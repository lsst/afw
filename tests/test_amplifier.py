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

from types import SimpleNamespace
import unittest
import numpy as np

import lsst.utils.tests
import lsst.geom
from lsst.afw.cameraGeom import Amplifier, AmplifierGeometryComparison, ReadoutCorner
import lsst.afw.cameraGeom.testUtils  # for assert methods injected into TestCase


class AmplifierTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.data = SimpleNamespace(
            name="Amp1",
            gain=1.2345,
            saturation=65535,
            readNoise=-0.523,
            linearityCoeffs=np.array([1.1, 2.2, 3.3, 4.4], dtype=float),
            linearityType="Polynomial",
            bbox=lsst.geom.Box2I(lsst.geom.Point2I(3, -2), lsst.geom.Extent2I(231, 320)),
            rawFlipX=True,
            rawFlipY=False,
            readoutCorner=ReadoutCorner.UL,
            rawBBox=lsst.geom.Box2I(lsst.geom.Point2I(-25, 2), lsst.geom.Extent2I(550, 629)),
            rawXYOffset=lsst.geom.Extent2I(-97, 253),
            rawDataBBox=lsst.geom.Box2I(lsst.geom.Point2I(-2, 29), lsst.geom.Extent2I(123, 307)),
            rawHorizontalOverscanBBox=lsst.geom.Box2I(
                lsst.geom.Point2I(150, 29),
                lsst.geom.Extent2I(25, 307),
            ),
            rawVerticalOverscanBBox=lsst.geom.Box2I(
                lsst.geom.Point2I(-2, 201),
                lsst.geom.Extent2I(123, 6),
            ),
            rawPrescanBBox=lsst.geom.Box2I(
                lsst.geom.Point2I(-20, 2),
                lsst.geom.Extent2I(5, 307),
            ),
        )
        builder = Amplifier.Builder()
        builder.setBBox(self.data.bbox)
        builder.setName(self.data.name)
        builder.setGain(self.data.gain)
        builder.setSaturation(self.data.saturation)
        builder.setReadNoise(self.data.readNoise)
        builder.setReadoutCorner(self.data.readoutCorner)
        builder.setLinearityCoeffs(self.data.linearityCoeffs)
        builder.setLinearityType(self.data.linearityType)
        builder.setRawFlipX(self.data.rawFlipX)
        builder.setRawFlipY(self.data.rawFlipY)
        builder.setRawBBox(self.data.rawBBox)
        builder.setRawXYOffset(self.data.rawXYOffset)
        builder.setRawDataBBox(self.data.rawDataBBox)
        builder.setRawHorizontalOverscanBBox(self.data.rawHorizontalOverscanBBox)
        builder.setRawVerticalOverscanBBox(self.data.rawVerticalOverscanBBox)
        builder.setRawPrescanBBox(self.data.rawPrescanBBox)
        self.amplifier = builder.finish()

    def testBasics(self):
        self.assertEqual(self.data.name, self.amplifier.getName())
        self.assertEqual(self.data.gain, self.amplifier.getGain())
        self.assertEqual(self.data.saturation, self.amplifier.getSaturation())
        self.assertEqual(self.data.readNoise, self.amplifier.getReadNoise())
        self.assertEqual(self.data.readoutCorner, self.amplifier.getReadoutCorner())
        self.assertEqual(list(self.data.linearityCoeffs), list(self.amplifier.getLinearityCoeffs()))
        self.assertEqual(self.data.linearityType, self.amplifier.getLinearityType())
        self.assertEqual(self.data.bbox, self.amplifier.getBBox())
        self.assertEqual(self.data.rawBBox, self.amplifier.getRawBBox())
        self.assertEqual(self.data.rawDataBBox, self.amplifier.getRawDataBBox())
        self.assertEqual(self.data.rawHorizontalOverscanBBox, self.amplifier.getRawHorizontalOverscanBBox())
        self.assertEqual(self.data.rawVerticalOverscanBBox, self.amplifier.getRawVerticalOverscanBBox())
        self.assertEqual(self.data.rawPrescanBBox, self.amplifier.getRawPrescanBBox())
        self.assertEqual(self.data.rawHorizontalOverscanBBox, self.amplifier.getRawSerialOverscanBBox())
        self.assertEqual(self.data.rawVerticalOverscanBBox, self.amplifier.getRawParallelOverscanBBox())
        self.assertEqual(self.data.rawPrescanBBox, self.amplifier.getRawSerialPrescanBBox())
        self.assertEqual(self.data.rawPrescanBBox, self.amplifier.getRawHorizontalPrescanBBox())
        self.assertEqual(self.data.rawFlipX, self.amplifier.getRawFlipX())
        self.assertEqual(self.data.rawFlipY, self.amplifier.getRawFlipY())
        self.assertEqual(self.data.rawXYOffset, self.amplifier.getRawXYOffset())

        # Test get/set methods for overscan/prescan alias names.
        # Change slightly, don't care about contiguity, make smaller.
        newHorizontalOverscanBBox = lsst.geom.Box2I(
            lsst.geom.Point2I(150, 29), lsst.geom.Extent2I(25, 306))
        newVerticalOverscanBBox = lsst.geom.Box2I(
            lsst.geom.Point2I(-2, 201), lsst.geom.Extent2I(123, 5))
        newPrescanBBox = lsst.geom.Box2I(
            lsst.geom.Point2I(-20, 2), lsst.geom.Extent2I(4, 306))

        builder = self.amplifier.rebuild()
        builder.setRawSerialOverscanBBox(newHorizontalOverscanBBox)
        builder.setRawParallelOverscanBBox(newVerticalOverscanBBox)
        builder.setRawSerialPrescanBBox(newPrescanBBox)
        amplifier = builder.finish()

        self.assertEqual(newHorizontalOverscanBBox,
                         amplifier.getRawHorizontalOverscanBBox())
        self.assertEqual(newVerticalOverscanBBox,
                         amplifier.getRawVerticalOverscanBBox())
        self.assertEqual(newPrescanBBox,
                         amplifier.getRawPrescanBBox())

        newPrescanBBox2 = lsst.geom.Box2I(
            lsst.geom.Point2I(-20, 2), lsst.geom.Extent2I(5, 306))
        builder.setRawHorizontalPrescanBBox(newPrescanBBox2)
        amplifier = builder.finish()
        self.assertEqual(newPrescanBBox2,
                         amplifier.getRawPrescanBBox())

    def test_compareGeometry(self):
        """Test the `Amplifier.compareGeometry` method.

        This doesn't handle the case where the amplifiers have the same
        regions but different assembly state (and hence the regions look
        different until they are transformed).  That test is in test_transform.
        """
        Cmp = AmplifierGeometryComparison

        def test_combos(rhs, expected):
            """Test all combinations of flag kwargs to compareGeometry.

            Parameters
            ----------
            rhs : `Amplifier`
                RHS of comparison (LHS is always self.amplifier).
            expected : `list` of `AmplifierGeometryComparison`
                Expected results for comparison (just look at the code for
                order).
            """
            self.assertIs(self.amplifier.compareGeometry(rhs), expected[0])
            self.assertIs(self.amplifier.compareGeometry(rhs, assembly=False), expected[1])
            self.assertIs(self.amplifier.compareGeometry(rhs, regions=False), expected[2])
            self.assertIs(self.amplifier.compareGeometry(rhs, assembly=False, regions=False), Cmp.EQUAL)

        def modified(**kwargs):
            """Return an Amplifier by modifying a copy of ``self.amplifier``.

            Keyword arguments encode the name of an `Amplifier.Builder` setter
            method in the key and the value for that setter in the value.
            """
            builder = self.amplifier.rebuild()
            for k, v in kwargs.items():
                getattr(builder, k)(v)
            return builder.rebuild()

        # Comparing an amplifier to itself always returns EQUAL.
        test_combos(self.amplifier, [Cmp.EQUAL]*4)
        # If we modify a flip or shift parameter we notice iff assembly=true.
        # If regions=True as well, then those compare as different because
        # the method tries to transform to the same assembly state and that
        # makes them different.
        test_combos(modified(setRawFlipX=not self.data.rawFlipX),
                    [Cmp.FLIPPED_X | Cmp.REGIONS_DIFFER, Cmp.EQUAL, Cmp.FLIPPED_X])
        test_combos(modified(setRawFlipY=not self.data.rawFlipY),
                    [Cmp.FLIPPED_Y | Cmp.REGIONS_DIFFER, Cmp.EQUAL, Cmp.FLIPPED_Y])
        test_combos(modified(setRawXYOffset=self.data.rawXYOffset + lsst.geom.Extent2I(4, 5)),
                    [Cmp.SHIFTED_X | Cmp.SHIFTED_Y | Cmp.REGIONS_DIFFER, Cmp.EQUAL,
                     Cmp.SHIFTED_X | Cmp.SHIFTED_Y])
        # If we modify or data box we notice iff regions=True.  We modify them
        # all by shifting by the same amount because the Amplifier constructor
        # would be within its rights to raise if these didn't line up properly,
        # and nontrivial modifications that are valid are a lot harder to
        # write.
        offset = lsst.geom.Extent2I(-2, 3)
        test_combos(
            modified(
                setRawBBox=self.data.rawBBox.shiftedBy(offset),
                setRawDataBBox=self.data.rawDataBBox.shiftedBy(offset),
                setRawHorizontalOverscanBBox=self.data.rawHorizontalOverscanBBox.shiftedBy(offset),
                setRawVerticalOverscanBBox=self.data.rawVerticalOverscanBBox.shiftedBy(offset),
                setRawPrescanBBox=self.data.rawPrescanBBox.shiftedBy(offset),
            ),
            [Cmp.REGIONS_DIFFER, Cmp.REGIONS_DIFFER, Cmp.EQUAL, Cmp.EQUAL],
        )

    def test_transform(self):
        """Test the `Amplifier.Builder.transform` method and
        `Amplifier.compareGeometry` cases that involve post-transform
        comparisons.
        """
        # Standard case: no kwargs to transform, so we apply flips and
        # rawXYOffset shifts.  This should leave no flip or offset in the
        # returned amplifier.
        assembled_amp = self.amplifier.rebuild().transform().finish()
        self.assertFalse(assembled_amp.getRawFlipX())
        self.assertFalse(assembled_amp.getRawFlipY())
        self.assertEqual(assembled_amp.getRawXYOffset(), lsst.geom.Extent2I(0, 0))
        comparison = self.amplifier.compareGeometry(assembled_amp)
        self.assertEqual(bool(comparison & comparison.FLIPPED_X), self.data.rawFlipX)
        self.assertEqual(bool(comparison & comparison.FLIPPED_Y), self.data.rawFlipY)
        self.assertEqual(bool(comparison & comparison.SHIFTED),
                         self.data.rawXYOffset != lsst.geom.Extent2I(0, 0))
        self.assertFalse(comparison & comparison.REGIONS_DIFFER)
        # Use transform to round-trip the assembled amp back to the original.
        roundtripped = assembled_amp.rebuild().transform(
            outOffset=self.data.rawXYOffset,
            outFlipX=self.data.rawFlipX,
            outFlipY=self.data.rawFlipY,
        ).finish()
        self.assertAmplifiersEqual(self.amplifier, roundtripped)
        self.assertIs(self.amplifier.compareGeometry(roundtripped), AmplifierGeometryComparison.EQUAL)
        # Transform to completely different offsets and the inverse of the
        # flips we tried before, just to make sure we didn't mix up X and Y
        # somewhere.
        different = self.amplifier.rebuild().transform(
            outOffset=lsst.geom.Extent2I(7, 8),
            outFlipX=not self.data.rawFlipX,
            outFlipY=not self.data.rawFlipY,
        ).finish()
        self.assertEqual(different.getRawFlipX(), not self.data.rawFlipX)
        self.assertEqual(different.getRawFlipY(), not self.data.rawFlipY)
        self.assertEqual(different.getRawXYOffset(), lsst.geom.Extent2I(7, 8))
        comparison = self.amplifier.compareGeometry(different)
        self.assertTrue(comparison & comparison.ASSEMBLY_DIFFERS)
        self.assertFalse(comparison & comparison.REGIONS_DIFFER)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

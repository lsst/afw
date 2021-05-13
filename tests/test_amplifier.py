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
from lsst.afw.cameraGeom import Amplifier, ReadoutCorner


class AmplifierTestCase(unittest.TestCase):

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


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

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

import unittest
import numpy as np

import lsst.utils.tests
import lsst.geom
from lsst.afw.cameraGeom import Amplifier, ReadoutCorner


class AmplifierTestCase(unittest.TestCase):

    def testBasics(self):
        name = "Amp1"
        gain = 1.2345
        saturation = 65535
        readNoise = -0.523
        linearityCoeffs = np.array([1.1, 2.2, 3.3, 4.4], dtype=float)
        linearityType = "Polynomial"
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(3, -2),
                               lsst.geom.Extent2I(231, 320))
        rawFlipX = True
        rawFlipY = False
        readoutCorner = ReadoutCorner.UL
        rawBBox = lsst.geom.Box2I(lsst.geom.Point2I(-25, 2),
                                  lsst.geom.Extent2I(550, 629))
        rawXYOffset = lsst.geom.Extent2I(-97, 253)
        rawDataBBox = lsst.geom.Box2I(lsst.geom.Point2I(-2, 29),
                                      lsst.geom.Extent2I(123, 307))
        rawHorizontalOverscanBBox = lsst.geom.Box2I(
            lsst.geom.Point2I(150, 29), lsst.geom.Extent2I(25, 307))
        rawVerticalOverscanBBox = lsst.geom.Box2I(
            lsst.geom.Point2I(-2, 201), lsst.geom.Extent2I(123, 6))
        rawPrescanBBox = lsst.geom.Box2I(
            lsst.geom.Point2I(-20, 2), lsst.geom.Extent2I(5, 307))

        builder = Amplifier.Builder()
        builder.setBBox(bbox)
        builder.setName(name)
        builder.setGain(gain)
        builder.setSaturation(saturation)
        builder.setReadNoise(readNoise)
        builder.setReadoutCorner(readoutCorner)
        builder.setLinearityCoeffs(linearityCoeffs)
        builder.setLinearityType(linearityType)
        builder.setRawFlipX(rawFlipX)
        builder.setRawFlipY(rawFlipY)
        builder.setRawBBox(rawBBox)
        builder.setRawXYOffset(rawXYOffset)
        builder.setRawDataBBox(rawDataBBox)
        builder.setRawHorizontalOverscanBBox(rawHorizontalOverscanBBox)
        builder.setRawVerticalOverscanBBox(rawVerticalOverscanBBox)
        builder.setRawPrescanBBox(rawPrescanBBox)
        amplifier = builder.finish()

        self.assertEqual(name, amplifier.getName())
        self.assertEqual(gain, amplifier.getGain())
        self.assertEqual(saturation, amplifier.getSaturation())
        self.assertEqual(readNoise, amplifier.getReadNoise())
        self.assertEqual(readoutCorner, amplifier.getReadoutCorner())
        self.assertEqual(list(linearityCoeffs),
                         list(amplifier.getLinearityCoeffs()))
        self.assertEqual(linearityType, amplifier.getLinearityType())
        self.assertEqual(bbox, amplifier.getBBox())
        self.assertEqual(rawBBox, amplifier.getRawBBox())
        self.assertEqual(rawDataBBox, amplifier.getRawDataBBox())
        self.assertEqual(rawHorizontalOverscanBBox,
                         amplifier.getRawHorizontalOverscanBBox())
        self.assertEqual(rawVerticalOverscanBBox,
                         amplifier.getRawVerticalOverscanBBox())
        self.assertEqual(rawPrescanBBox, amplifier.getRawPrescanBBox())
        self.assertEqual(rawFlipX, amplifier.getRawFlipX())
        self.assertEqual(rawFlipY, amplifier.getRawFlipY())
        self.assertEqual(rawXYOffset, amplifier.getRawXYOffset())


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

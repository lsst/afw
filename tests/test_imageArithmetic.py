# LSST Data Management System
# Copyright 2019 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.

import unittest

import lsst.utils.tests
from lsst.geom import Box2I, Point2I, Extent2I
import lsst.afw.image


class ImageArithmeticTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.bbox = Box2I(Point2I(12345, 56789), Extent2I(12, 34))

    def tearDown(self):
        del self.bbox

    def testImage(self):
        for cls in (lsst.afw.image.ImageI,
                    lsst.afw.image.ImageL,
                    lsst.afw.image.ImageU,
                    lsst.afw.image.ImageF,
                    lsst.afw.image.ImageD,
                    lsst.afw.image.DecoratedImageI,
                    lsst.afw.image.DecoratedImageL,
                    lsst.afw.image.DecoratedImageU,
                    lsst.afw.image.DecoratedImageF,
                    lsst.afw.image.DecoratedImageD,
                    lsst.afw.image.MaskedImageI,
                    lsst.afw.image.MaskedImageL,
                    lsst.afw.image.MaskedImageU,
                    lsst.afw.image.MaskedImageF,
                    lsst.afw.image.MaskedImageD,
                    lsst.afw.image.ExposureI,
                    lsst.afw.image.ExposureL,
                    lsst.afw.image.ExposureU,
                    lsst.afw.image.ExposureF,
                    lsst.afw.image.ExposureD,
                    ):
            im1 = cls(self.bbox)
            im2 = cls(self.bbox)

            # Image and image
            with self.assertRaises(NotImplementedError):
                im1 + im2
            with self.assertRaises(NotImplementedError):
                im1 - im2
            with self.assertRaises(NotImplementedError):
                im1 * im2
            with self.assertRaises(NotImplementedError):
                im1 / im2

            # Image and scalar
            with self.assertRaises(NotImplementedError):
                im1 + 12345
            with self.assertRaises(NotImplementedError):
                im1 - 12345
            with self.assertRaises(NotImplementedError):
                im1 * 12345
            with self.assertRaises(NotImplementedError):
                im1 / 12345

            # Scalar and image
            with self.assertRaises(NotImplementedError):
                54321 + im2
            with self.assertRaises(NotImplementedError):
                54321 - im2
            with self.assertRaises(NotImplementedError):
                54321 * im2
            with self.assertRaises(NotImplementedError):
                54321 / im2

    def testMask(self):
        for cls in (lsst.afw.image.MaskX,
                    ):
            im1 = cls(self.bbox)
            im2 = cls(self.bbox)

            # Image and image
            with self.assertRaises(NotImplementedError):
                im1 | im2
            with self.assertRaises(NotImplementedError):
                im1 & im2
            with self.assertRaises(NotImplementedError):
                im1 ^ im2

            # Image and scalar
            with self.assertRaises(NotImplementedError):
                im1 | 12345
            with self.assertRaises(NotImplementedError):
                im1 & 12345
            with self.assertRaises(NotImplementedError):
                im1 ^ 12345

            # Scalar and image
            with self.assertRaises(NotImplementedError):
                54321 | im2
            with self.assertRaises(NotImplementedError):
                54321 & im2
            with self.assertRaises(NotImplementedError):
                54321 ^ im2


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

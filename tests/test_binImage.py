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
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display as afwDisplay

try:
    type(display)
except NameError:
    display = False


class BinImageTestCase(unittest.TestCase):
    """A test case for binning images.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testBin(self):
        """Test that we can bin images.
        """
        inImage = afwImage.ImageF(203, 131)
        inImage.set(1)
        bin = 4

        outImage = afwMath.binImage(inImage, bin)

        self.assertEqual(outImage.getWidth(), inImage.getWidth()//bin)
        self.assertEqual(outImage.getHeight(), inImage.getHeight()//bin)

        stats = afwMath.makeStatistics(outImage, afwMath.MAX | afwMath.MIN)
        self.assertEqual(stats.getValue(afwMath.MIN), 1)
        self.assertEqual(stats.getValue(afwMath.MAX), 1)

    def testBin2(self):
        """Test that we can bin images anisotropically.
        """
        inImage = afwImage.ImageF(203, 131)
        val = 1
        inImage.set(val)
        binX, binY = 2, 4

        outImage = afwMath.binImage(inImage, binX, binY)

        self.assertEqual(outImage.getWidth(), inImage.getWidth()//binX)
        self.assertEqual(outImage.getHeight(), inImage.getHeight()//binY)

        stats = afwMath.makeStatistics(outImage, afwMath.MAX | afwMath.MIN)
        self.assertEqual(stats.getValue(afwMath.MIN), val)
        self.assertEqual(stats.getValue(afwMath.MAX), val)

        inImage.set(0)
        subImg = inImage.Factory(inImage, lsst.geom.BoxI(lsst.geom.PointI(4, 4), lsst.geom.ExtentI(4, 8)),
                                 afwImage.LOCAL)
        subImg.set(100)
        del subImg
        outImage = afwMath.binImage(inImage, binX, binY)

        if display:
            afwDisplay.Display(frame=2).mtv(inImage, title="unbinned")
            afwDisplay.Display(frame=3).mtv(outImage, title=f"binned {binX}x{binY}")

    def testBinValues(self):
        """Test that the binned values are correct.
        """
        image_i = afwImage.ImageI(6, 6)
        image_d = afwImage.ImageD(6, 6)

        # Add an offset that's a little under maxint/4 to test that integers
        # won't overflow, and doubles don't accumulate large errors
        for offset in (0, 536870847):
            image_d.array = np.arange(36).reshape(6, 6) + offset
            image_i.array = image_d.array
            binned = (
                (
                    2, 2,
                    np.array([
                        [3.5, 5.5, 7.5],
                        [15.5, 17.5, 19.5],
                        [27.5, 29.5, 31.5],
                    ]),
                ),
                (
                    2, 3,
                    np.array([
                        [6.5, 8.5, 10.5],
                        [24.5, 26.5, 28.5],
                    ]),
                ),
                (
                    3, 2,
                    np.array([
                        [4., 7.],
                        [16., 19.],
                        [28., 31.],
                    ])
                ),
            )
            for bin_x, bin_y, truth in binned:
                # This actually tests exactly equal with ImageD (not ImageF)
                # but this could be compiler/optimization-dependent
                np.testing.assert_allclose(
                    afwMath.binImage(image_d, bin_x, bin_y).array,
                    truth + offset,
                    rtol=1e-14,
                    atol=1e-14,
                )
                np.testing.assert_array_equal(
                    afwMath.binImage(image_i, bin_x, bin_y).array,
                    truth//1 + offset,
                )


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

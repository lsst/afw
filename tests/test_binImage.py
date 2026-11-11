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

# An integer slightly less than np.iinfo(np.int32).max/4, so it can be binned
# only 2x2 before overflowing.
maxint_d4 = 536870847


def divide_nearest(array: np.ndarray, divisor: int):
    """Divide an array by an integer while rounding to the nearest even int."""
    exact = (array % divisor) == 0
    array[exact] = array[exact] // divisor
    array[~exact] = np.round(array[~exact]/np.double(divisor)).astype(array.dtype)
    return array


def rebin_array(array: np.ndarray, bin_x: int, bin_y: int) -> np.ndarray:
    """Rebin an array by summing pixel values."""
    n_y, n_x = array.shape
    result = array.reshape(
        n_y//bin_y, bin_y, n_x//bin_x, bin_x
    ).sum(3).sum(1)
    return result


def rebin_mask(array: np.ndarray, bin_x: int, bin_y: int) -> np.ndarray:
    """Rebin a mask array by summing pixel values."""
    n_y, n_x = array.shape
    result = array.reshape(
        n_y//bin_y, bin_y, n_x//bin_x, bin_x
    )
    result = np.bitwise_or.reduce(result, axis=3)
    result = np.bitwise_or.reduce(result, axis=1)
    return result


class BinImageTestCase(unittest.TestCase):
    """A test case for binning images.
    """

    def setUp(self):
        np.random.seed(1)

        image_d = afwImage.ImageD(35, 63)
        image_d.array[:, :] = np.random.uniform(low=-1, high=1, size=image_d.array.shape)

        self.image_d = image_d

    def tearDown(self):
        del self.image_d

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

    def testBinValuesMasked(self):
        """Test that binning works correctly for small masked images.
        """
        image_i = afwImage.ImageI(8, 8)
        image_i.array.flat = np.arange(64)

        # Trivial binning
        binned = afwMath.binImage(image_i, 1)
        np.testing.assert_array_equal(image_i.array, binned.array)

        # Make sum odd in some bins and test rounding to even
        image_i.array += (image_i.array % 4) == 0
        binned = afwMath.binImage(image_i, 2)
        truth = divide_nearest(rebin_array(image_i.array, 2, 2), 4)
        np.testing.assert_array_equal(binned.array, truth)

        value = 100
        image_i.set(value)

        # Basic test of maskImage binning
        mimage_i = afwImage.makeMaskedImage(image_i)
        mimage_i.variance.set(value)

        binned = afwMath.binImage(mimage_i, 3)

        np.testing.assert_array_equal(
            binned.image.array,
            np.full((2, 2), value),
        )
        np.testing.assert_allclose(
            binned.variance.array,
            np.full((2, 2), value/9).astype(np.float32),
            rtol=1e-8,
            atol=1e-8,
        )

    def testBinValuesInteger(self):
        """Test that binning works correctly for small (half-)integer values.
        """
        shape = (6, 6)
        image_d = afwImage.ImageD(*shape)
        image_f = afwImage.ImageF(*shape)

        image_i = afwImage.ImageI(*shape)
        image_u = afwImage.ImageU(*shape)
        # ImageL is not yet supported by binImage.

        for offset in (0, 1, maxint_d4):
            image_d.array = np.arange(shape[0]*shape[1]).reshape(shape) + offset
            image_f.array = image_d.array
            image_i.array = image_d.array
            image_u.array = image_i.array
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
                (
                    3, 3,
                    np.array([
                        [7., 10.],
                        [25., 28.],
                    ])
                ),
            )
            for bin_x, bin_y, truth in binned:
                truth += offset
                # Small offsets actually tests exactly equal with ImageD (not ImageF)
                # but this could be compiler/optimization-dependent
                for image, rtol, atol in ((image_d, 1e-14, 1e-14), (image_f, 1e-9, 1e-10)):
                    # These adjustments are somewhat arbitrary and may need
                    # to be more principled
                    if (offset > 1e8) and ((atol >= 1e-12) or (rtol >= 1e-12)):
                        atol *= 1000*offset
                        rtol *= 100
                    np.testing.assert_allclose(
                        afwMath.binImage(image, bin_x, bin_y).array,
                        truth,
                        rtol=rtol,
                        atol=atol,
                    )
                for image in (image_u, image_i):
                    # image_u will overflow for large offsets, but it should
                    # still overflow predictably
                    truth_t = np.round(truth).astype(image.array.dtype)
                    np.testing.assert_array_equal(
                        afwMath.binImage(image, bin_x, bin_y).array,
                        truth_t,
                    )

    def testBinValuesFloat(self):
        """Test that binning works correctly for a wide range of floats.
        """
        image_d_original = self.image_d
        n_y, n_x = image_d_original.array.shape

        mask = afwImage.makeMaskFromArray(np.zeros((0, 0), dtype=np.int32))
        mask = afwImage.makeMaskFromArray(
            np.random.uniform(
                high=2**(1 + max(mask.getMaskPlaneDict().values())) - 1,
                size=(n_y, n_x),
            ).astype(np.int32)
        )

        bin_info = tuple(
            (bin_x, bin_y, rebin_mask(mask.array, bin_x, bin_y))
            for bin_x, bin_y in (
                (5, 21),
                (7, 7),
            )
        )

        # Test that rebinning works on a larger array with a wider range
        # of floating point values

        for multiple, rtol_d, rtol_f, rtol_v in (
            (1e-32, 1e-12, None, None),
            (1./maxint_d4, 1e-13, 1e-8, 5e-5),
            (1./32564, 5e-14, 1e-7, 5e-7,),
            (1, 1e-14, 1e-6, 5e-7),
            (32564, 5e-14, 1e-6, 5e-6),
            (maxint_d4, 1e-12, 1e-6, 5e-5),
            (1e32, 1e-12, None, None),
        ):
            image_d = image_d_original.clone()
            image_d.array *= multiple

            if rtol_f is not None:
                image_f = afwImage.ImageF(image_d, deep=True)
            else:
                image_f = None
            if maxint_d4 >= multiple >= 1:
                image_i = afwImage.ImageI(image_d, deep=True)
            else:
                image_i = None

            for bin_x, bin_y, mask_truth in bin_info:
                bin_xy = bin_x*bin_y
                truth = rebin_array(image_d.array/bin_xy, bin_x, bin_y)
                binned_image = afwMath.binImage(image_d, bin_x, bin_y).array
                np.testing.assert_allclose(
                    binned_image,
                    truth,
                    rtol=rtol_d,
                    atol=rtol_d,
                )
                variance_large = multiple > 32564
                self._compareBinnedMaskedImage(
                    image_d, mask, bin_x, bin_y, binned_image, mask_truth, rtol_v, rtol_v,
                    variance_large=variance_large,
                )

                if image_f is not None:
                    binned_image = afwMath.binImage(image_f, bin_x, bin_y).array
                    np.testing.assert_allclose(
                        binned_image,
                        truth.astype(np.float32),
                        rtol=rtol_f,
                        atol=multiple*max(rtol_f, 1e-6),
                    )
                    self._compareBinnedMaskedImage(
                        image_f, mask, bin_x, bin_y, binned_image, mask_truth, rtol_v, rtol_v,
                        variance_large=variance_large,
                    )
                if image_i is not None:
                    truth = rebin_array(image_i.array.astype(np.int64), bin_x, bin_y)
                    divide_nearest(truth, bin_xy)
                    binned_image = afwMath.binImage(image_i, bin_x, bin_y).array

                    np.testing.assert_array_equal(binned_image, truth)
                    # The utility of an integer image with floating variance
                    # is dubious, but it should still work.
                    self._compareBinnedMaskedImage(
                        image_i, mask, bin_x, bin_y, binned_image, mask_truth, rtol_v, rtol_v,
                        variance_large=variance_large,
                    )

    @staticmethod
    def _compareBinnedMaskedImage(
        image, mask, bin_x, bin_y, binned_image, mask_truth, rtol, atol,
        variance=None, variance_large=False,
    ):
        """Assert that the results of binning a masked image are correct.

        This is a convenience function that
        """
        bin_xy = bin_x*bin_y
        variance = (image.array[::-1] if variance is None else variance).astype(np.float32)
        masked = afwImage.makeMaskedImageFromArrays(image.array, mask.array, variance)

        binned = afwMath.binImage(masked, bin_x, bin_y)
        # Assert that the binning gives the same result without a mask
        np.testing.assert_array_equal(binned_image, binned.image.array)
        np.testing.assert_array_equal(binned.mask.array, mask_truth)
        if rtol is not None and atol is not None:
            truth = rebin_array(
                masked.variance.array/(bin_xy if variance_large else 1), bin_x, bin_y
            )
            truth /= bin_xy*(1 if variance_large else bin_xy)
            truth = truth.astype(np.float32)
            np.testing.assert_allclose(
                binned.variance.array,
                truth,
                rtol=rtol,
                atol=atol,
            )


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

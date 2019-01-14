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

"""
Tests for offsetting images in (dx, dy)

Run with:
   python offsetImage.py
or
   python
   >>> import offsetImage; offsetImage.run()
"""
import math
import unittest

import numpy as np

import lsst.utils.tests
import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display as afwDisplay

try:
    type(display)
except NameError:
    display = False


class OffsetImageTestCase(unittest.TestCase):
    """A test case for offsetImage.
    """

    def setUp(self):
        self.inImage = afwImage.ImageF(200, 100)
        self.background = 200
        self.inImage.set(self.background)

    def tearDown(self):
        del self.inImage

    def testSetFluxConvervation(self):
        """Test that flux is preserved.
        """
        for algorithm in ("lanczos5", "bilinear", "nearest"):
            outImage = afwMath.offsetImage(self.inImage, 0, 0, algorithm)
            self.assertEqual(outImage[50, 50, afwImage.LOCAL], self.background)

            outImage = afwMath.offsetImage(self.inImage, 0.5, 0, algorithm)
            self.assertAlmostEqual(outImage[50, 50, afwImage.LOCAL], self.background, 4)

            outImage = afwMath.offsetImage(self.inImage, 0.5, 0.5, algorithm)
            self.assertAlmostEqual(outImage[50, 50, afwImage.LOCAL], self.background, 4)

    def testSetIntegerOffset(self):
        """Test that we can offset by positive and negative amounts.
        """
        self.inImage[50, 50, afwImage.LOCAL] = 400

        if False and display:
            frame = 0
            disp = afwDisplay.Display(frame=frame)
            disp.mtv(self.inImage, title="Image for Integer Offset Test")
            disp.pan(50, 50)
            disp.dot("+", 50, 50)

        for algorithm in ("lanczos5", "bilinear", "nearest"):
            frame = 1
            for delta in [-0.49, 0.51]:
                for dx, dy in [(2, 3), (-2, 3), (-2, -3), (2, -3)]:
                    outImage = afwMath.offsetImage(
                        self.inImage, dx + delta, dy + delta, algorithm)

                    if False and display:
                        frame += 1
                        disp = afwDisplay.Display(frame=frame)
                        disp.mtv(outImage, title=algorithm + ": offset image (dx, dy) = (%d, %d)" % (dx, dy))

                        disp.pan(50, 50)
                        disp.dot("+", 50 + dx + delta - outImage.getX0(), 50 + dy + delta - outImage.getY0())

    def calcGaussian(self, im, x, y, amp, sigma1):
        """Insert a Gaussian into the image centered at (x, y).
        """
        x = x - im.getX0()
        y = y - im.getY0()

        for ix in range(im.getWidth()):
            for iy in range(im.getHeight()):
                r2 = math.pow(x - ix, 2) + math.pow(y - iy, 2)
                val = math.exp(-r2/(2.0*pow(sigma1, 2)))
                im[ix, iy, afwImage.LOCAL] = amp*val

    def testOffsetGaussian(self):
        """Insert a Gaussian, offset, and check the residuals.
        """
        size = 50
        refIm = afwImage.ImageF(size, size)
        unshiftedIm = afwImage.ImageF(size, size)

        xc, yc = size/2.0, size/2.0

        amp, sigma1 = 1.0, 3

        #
        # Calculate Gaussian directly at (xc, yc)
        #
        self.calcGaussian(refIm, xc, yc, amp, sigma1)

        for dx in (-55.5, -1.500001, -1.5, -1.499999, -1.00001, -1.0, -0.99999, -0.5,
                   0.0, 0.5, 0.99999, 1.0, 1.00001, 1.499999, 1.5, 1.500001, 99.3):
            for dy in (-3.7, -1.500001, -1.5, -1.499999, -1.00001, -1.0, -0.99999, -0.5,
                       0.0, 0.5, 0.99999, 1.0, 1.00001, 1.499999, 1.5, 1.500001, 2.99999):
                dOrigX, dOrigY, dFracX, dFracY = getOrigFracShift(dx, dy)
                self.calcGaussian(unshiftedIm, xc - dFracX,
                                  yc - dFracY, amp, sigma1)

                for algorithm, maxMean, maxLim in (
                    ("lanczos5", 1e-8, 0.0015),
                    ("bilinear", 1e-8, 0.03),
                    ("nearest", 1e-8, 0.2),
                ):
                    im = afwImage.ImageF(size, size)
                    im = afwMath.offsetImage(unshiftedIm, dx, dy, algorithm)

                    if display:
                        afwDisplay.Display(frame=0).mtv(im, title=algorithm + ": image")

                    im -= refIm

                    if display:
                        afwDisplay.Display(frame=1).mtv(im, title=algorithm +
                                                        ": diff image (dx, dy) = (%f, %f)" % (dx, dy))

                    imArr = im.getArray()
                    imGoodVals = np.ma.array(
                        imArr, copy=False, mask=np.isnan(imArr)).compressed()

                    try:
                        imXY0 = tuple(im.getXY0())
                        self.assertEqual(imXY0, (dOrigX, dOrigY))
                        self.assertLess(abs(imGoodVals.mean()), maxMean*amp)
                        self.assertLess(abs(imGoodVals.max()), maxLim*amp)
                        self.assertLess(abs(imGoodVals.min()), maxLim*amp)
                    except Exception:
                        print("failed on algorithm=%s; dx = %s; dy = %s" %
                              (algorithm, dx, dy))
                        raise

# the following would be preferable if there was an easy way to NaN pixels
#
#         stats = afwMath.makeStatistics(im, afwMath.MEAN | afwMath.MAX | afwMath.MIN)
#
#         if not False:
#             print "mean = %g, min = %g, max = %g" % (stats.getValue(afwMath.MEAN),
#                                                      stats.getValue(afwMath.MIN),
#                                                      stats.getValue(afwMath.MAX))
#
#         self.assertTrue(abs(stats.getValue(afwMath.MEAN)) < 1e-7)
#         self.assertTrue(abs(stats.getValue(afwMath.MIN)) < 1.2e-3*amp)
#         self.assertTrue(abs(stats.getValue(afwMath.MAX)) < 1.2e-3*amp)


def getOrigFracShift(dx, dy):
    """Return the predicted integer shift to XY0 and the fractional shift that offsetImage will use

    offsetImage preserves the origin if dx and dy both < 1 pixel; larger shifts are to the nearest pixel.
    """
    if (abs(dx) < 1) and (abs(dy) < 1):
        return (0, 0, dx, dy)

    dOrigX = math.floor(dx + 0.5)
    dOrigY = math.floor(dy + 0.5)
    dFracX = dx - dOrigX
    dFracY = dy - dOrigY
    return (int(dOrigX), int(dOrigY), dFracX, dFracY)


class TransformImageTestCase(unittest.TestCase):
    """A test case for rotating images.
    """

    def setUp(self):
        self.inImage = afwImage.ImageF(20, 10)
        self.inImage[0, 0, afwImage.LOCAL] = 100
        self.inImage[10, 0, afwImage.LOCAL] = 50

    def tearDown(self):
        del self.inImage

    def testRotate(self):
        """Test that we end up with the correct image after rotating by 90 degrees.
        """
        for nQuarter, x, y in [(0, 0, 0),
                               (1, 9, 0),
                               (2, 19, 9),
                               (3, 0, 19)]:
            outImage = afwMath.rotateImageBy90(self.inImage, nQuarter)
            if display:
                afwDisplay.Display(frame=nQuarter).mtv(outImage, title="out %d" % nQuarter)
            self.assertEqual(self.inImage[0, 0, afwImage.LOCAL], outImage[x, y, afwImage.LOCAL])

    def testFlip(self):
        """Test that we end up with the correct image after flipping it.
        """
        frame = 2
        for flipLR, flipTB, x, y in [(True, False, 19, 0),
                                     (True, True, 19, 9),
                                     (False, True, 0, 9),
                                     (False, False, 0, 0)]:
            outImage = afwMath.flipImage(self.inImage, flipLR, flipTB)
            if display:
                afwDisplay.Display(frame=frame).mtv(outImage, title="%s %s" % (flipLR, flipTB))
                frame += 1
            self.assertEqual(self.inImage[0, 0, afwImage.LOCAL], outImage[x, y, afwImage.LOCAL])

    def testMask(self):
        """Test that we can flip a Mask.
        """
        mask = afwImage.Mask(10, 20)
        # for a while, swig couldn't handle the resulting std::shared_ptr<Mask>
        afwMath.flipImage(mask, True, False)


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
            afwDisplay.Display(frame=3).mtv(outImage, title="binned %dx%d" % (binX, binY))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

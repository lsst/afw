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
Tests for RGB Images

Run with:
   ./rgb.py
or
   python
   >>> import rgb; rgb.run()
"""
import os
import math
import unittest

import numpy as np

import lsst.utils.tests
import lsst.geom
import lsst.afw.detection as afwDetect
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.rgb as rgb

ver1, ver2, ver3 = 1, 3, 1
NO_MATPLOTLIB_STRING = "Requires matplotlib >= %d.%d.%d" % (ver1, ver2, ver3)
try:
    import matplotlib
    mplVersion = matplotlib.__version__
    # Split at + to check for development version (PEP 440)
    mplVersion = mplVersion.split('+')
    versionInfo = tuple(int(s.strip("rc")) for s in mplVersion[0].split("."))
    HAVE_MATPLOTLIB = versionInfo >= (ver1, ver2, ver3)
except ImportError:
    HAVE_MATPLOTLIB = False

try:
    import scipy.misc
    scipy.misc.imresize
    HAVE_SCIPY_MISC = True
except (ImportError, AttributeError):
    HAVE_SCIPY_MISC = False

try:
    type(display)
except NameError:
    display = False


def saturate(image, satValue):
    """Simulate saturation on an image, so we can test 'replaceSaturatedPixels'

    Takes an Image, sets saturated pixels to NAN and masks them, returning
    a MaskedImage.
    """
    image = afwImage.makeMaskedImage(image)
    afwDetect.FootprintSet(image, afwDetect.Threshold(satValue), "SAT")
    arr = image.getImage().getArray()
    arr[np.where(arr >= satValue)] = np.nan
    return image


R, G, B = 2, 1, 0


class RgbTestCase(unittest.TestCase):
    """A test case for Rgb"""

    def setUp(self):
        self.min, self.range, self.Q = 0, 5, 20  # asinh

        width, height = 85, 75
        self.images = []
        self.images.append(afwImage.ImageF(lsst.geom.ExtentI(width, height)))
        self.images.append(afwImage.ImageF(lsst.geom.ExtentI(width, height)))
        self.images.append(afwImage.ImageF(lsst.geom.ExtentI(width, height)))

        for (x, y, A, g_r, r_i) in [(15, 15, 1000, 1.0, 2.0),
                                    (50, 45, 5500, -1.0, -0.5),
                                    (30, 30, 600, 1.0, 2.5),
                                    (45, 15, 20000, 1.0, 1.0),
                                    ]:
            for i in range(len(self.images)):
                if i == B:
                    amp = A
                elif i == G:
                    amp = A*math.pow(10, 0.4*g_r)
                elif i == R:
                    amp = A*math.pow(10, 0.4*r_i)

                self.images[i][x, y, afwImage.LOCAL] = amp

        psf = afwMath.AnalyticKernel(
            15, 15, afwMath.GaussianFunction2D(2.5, 1.5, 0.5))

        convolvedImage = type(self.images[0])(self.images[0].getDimensions())
        randomImage = type(self.images[0])(self.images[0].getDimensions())
        rand = afwMath.Random("MT19937", 666)
        for i in range(len(self.images)):
            afwMath.convolve(convolvedImage, self.images[i], psf, True, True)
            afwMath.randomGaussianImage(randomImage, rand)
            randomImage *= 2
            convolvedImage += randomImage
            self.images[i][:] = convolvedImage
        del convolvedImage
        del randomImage

    def tearDown(self):
        for im in self.images:
            del im
        del self.images

    def testStarsAsinh(self):
        """Test creating an RGB image using an asinh stretch"""
        asinhMap = rgb.AsinhMapping(self.min, self.range, self.Q)
        rgbImage = asinhMap.makeRgbImage(
            self.images[R], self.images[G], self.images[B])

        if display:
            rgb.displayRGB(rgbImage)

    def testStarsAsinhZscale(self):
        """Test creating an RGB image using an asinh stretch estimated using zscale"""

        rgbImages = [self.images[R], self.images[G], self.images[B]]

        map = rgb.AsinhZScaleMapping(rgbImages[0])
        rgbImage = map.makeRgbImage(*rgbImages)

        if display:
            rgb.displayRGB(rgbImage)

    def testStarsAsinhZscaleIntensity(self):
        """Test creating an RGB image using an asinh stretch estimated using zscale on the intensity"""

        rgbImages = [self.images[R], self.images[G], self.images[B]]

        map = rgb.AsinhZScaleMapping(rgbImages)
        rgbImage = map.makeRgbImage(*rgbImages)

        if display:
            rgb.displayRGB(rgbImage)

    def testStarsAsinhZscaleIntensityPedestal(self):
        """Test creating an RGB image using an asinh stretch estimated using zscale on the intensity
        where the images each have a pedestal added"""

        rgbImages = [self.images[R], self.images[G], self.images[B]]

        pedestal = [100, 400, -400]
        for i, ped in enumerate(pedestal):
            rgbImages[i] += ped

        map = rgb.AsinhZScaleMapping(rgbImages, pedestal=pedestal)
        rgbImage = map.makeRgbImage(*rgbImages)

        if display:
            rgb.displayRGB(rgbImage)

    def testStarsAsinhZscaleIntensityBW(self):
        """Test creating a black-and-white image using an asinh stretch estimated
        using zscale on the intensity"""

        rgbImage = rgb.AsinhZScaleMapping(self.images[R]).makeRgbImage()

        if display:
            rgb.displayRGB(rgbImage)

    @unittest.skipUnless(HAVE_MATPLOTLIB, NO_MATPLOTLIB_STRING)
    def testMakeRGB(self):
        """Test the function that does it all"""
        satValue = 1000.0
        with lsst.utils.tests.getTempFilePath(".png") as fileName:
            red = saturate(self.images[R], satValue)
            green = saturate(self.images[G], satValue)
            blue = saturate(self.images[B], satValue)
            rgb.makeRGB(red, green, blue, self.min, self.range, self.Q, fileName=fileName,
                        saturatedBorderWidth=1, saturatedPixelValue=2000)
            self.assertTrue(os.path.exists(fileName))

    def testLinear(self):
        """Test using a specified linear stretch"""

        rgbImage = rgb.LinearMapping(-8.45, 13.44).makeRgbImage(self.images[R])

        if display:
            rgb.displayRGB(rgbImage)

    def testLinearMinMax(self):
        """Test using a min/max linear stretch

        N.b. also checks that an image passed to the ctor is used as the default in makeRgbImage()
        """

        rgbImage = rgb.LinearMapping(image=self.images[R]).makeRgbImage()

        if display:
            rgb.displayRGB(rgbImage)

    def testZScale(self):
        """Test using a zscale stretch"""

        rgbImage = rgb.ZScaleMapping(self.images[R]).makeRgbImage()

        if display:
            plt = rgb.displayRGB(rgbImage, False)
            plt.title("zscale")
            plt.show()

    @unittest.skipUnless(HAVE_MATPLOTLIB, NO_MATPLOTLIB_STRING)
    def testWriteStars(self):
        """Test writing RGB files to disk"""
        asinhMap = rgb.AsinhMapping(self.min, self.range, self.Q)
        rgbImage = asinhMap.makeRgbImage(
            self.images[R], self.images[G], self.images[B])
        with lsst.utils.tests.getTempFilePath(".png") as fileName:
            rgb.writeRGB(fileName, rgbImage)
            self.assertTrue(os.path.exists(fileName))

    def testSaturated(self):
        """Test interpolating saturated pixels"""

        satValue = 1000.0
        for f in [R, G, B]:
            self.images[f] = saturate(self.images[f], satValue)

        rgb.replaceSaturatedPixels(
            self.images[R], self.images[G], self.images[B], 1, 2000)
        #
        # Check that we replaced those NaNs with some reasonable value
        #
        for f in [R, G, B]:
            self.assertTrue(np.isfinite(
                self.images[f].getImage().getArray()).all())

        if False:
            ds9.mtv(self.images[B], frame=0, title="B")
            ds9.mtv(self.images[G], frame=1, title="G")
            ds9.mtv(self.images[R], frame=2, title="R")
        #
        # Prepare for generating an output file
        #
        for f in [R, G, B]:
            self.images[f] = self.images[f].getImage()

        asinhMap = rgb.AsinhMapping(self.min, self.range, self.Q)
        rgbImage = asinhMap.makeRgbImage(
            self.images[R], self.images[G], self.images[B])

        if display:
            rgb.displayRGB(rgbImage)

    @unittest.skipUnless(HAVE_SCIPY_MISC, "Resizing images requires scipy.misc")
    def testStarsResizeToSize(self):
        """Test creating an RGB image of a specified size"""

        xSize = self.images[R].getWidth()//2
        ySize = self.images[R].getHeight()//2
        for rgbImages in ([self.images[R], self.images[G], self.images[B]],
                          [afwImage.ImageU(_.getArray().astype('uint16')) for _ in [
                              self.images[R], self.images[G], self.images[B]]]):
            rgbImage = rgb.AsinhZScaleMapping(rgbImages[0]).makeRgbImage(*rgbImages,
                                                                         xSize=xSize, ySize=ySize)

            if display:
                rgb.displayRGB(rgbImage)

    @unittest.skipUnless(HAVE_SCIPY_MISC, "Resizing images requires scipy.misc")
    def testStarsResizeSpecifications(self):
        """Test creating an RGB image changing the output """

        rgbImages = [self.images[R], self.images[G], self.images[B]]
        map = rgb.AsinhZScaleMapping(rgbImages[0])

        for xSize, ySize, frac in [(self.images[R].getWidth()//2, self.images[R].getHeight()//2, None),
                                   (2*self.images[R].getWidth(), None, None),
                                   (self.images[R].getWidth()//2, None, None),
                                   (None, self.images[R].getHeight()//2, None),
                                   (None, None, 0.5),
                                   (None, None, 2),
                                   ]:
            rgbImage = map.makeRgbImage(
                *rgbImages, xSize=xSize, ySize=ySize, rescaleFactor=frac)

            h, w = rgbImage.shape[0:2]
            self.assertTrue(xSize is None or xSize == w)
            self.assertTrue(ySize is None or ySize == h)
            self.assertTrue(frac is None or w == int(frac*self.images[R].getWidth()),
                            "%g == %g" % (w, int((frac if frac else 1)*self.images[R].getWidth())))

            if display:
                rgb.displayRGB(rgbImage)

    @unittest.skipUnless(HAVE_SCIPY_MISC, "Resizing images requires scipy.misc")
    @unittest.skipUnless(HAVE_MATPLOTLIB, NO_MATPLOTLIB_STRING)
    def testMakeRGBResize(self):
        """Test the function that does it all, including rescaling"""
        rgb.makeRGB(self.images[R], self.images[G],
                    self.images[B], xSize=40, ySize=60)

        with lsst.utils.tests.getTempFilePath(".png") as fileName:
            rgb.makeRGB(self.images[R], self.images[G],
                        self.images[B], fileName=fileName, rescaleFactor=0.5)
            self.assertTrue(os.path.exists(fileName))

    def writeFileLegacyAPI(self, fileName):
        """Test that the legacy API still works, although it's deprecated"""
        asinh = rgb.asinhMappingF(self.min, self.range, self.Q)
        rgbImage = rgb.RgbImageF(
            self.images[R], self.images[G], self.images[B], asinh)
        if False:
            ds9.mtv(self.images[B], frame=0, title="B")
            ds9.mtv(self.images[G], frame=1, title="G")
            ds9.mtv(self.images[R], frame=2, title="R")

        rgbImage.write(fileName)

    @unittest.skipUnless(HAVE_MATPLOTLIB, NO_MATPLOTLIB_STRING)
    def testWriteStarsLegacyAPI(self):
        with lsst.utils.tests.getTempFilePath(".png") as fileName:
            self.writeFileLegacyAPI(fileName)
            self.assertTrue(os.path.exists(fileName))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

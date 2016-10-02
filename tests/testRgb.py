#pybind11##!/usr/bin/env python
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2015-2016 LSST/AURA
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <https://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11#"""
#pybind11#Tests for RGB Images
#pybind11#
#pybind11#Run with:
#pybind11#   ./rgb.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import rgb; rgb.run()
#pybind11#"""
#pybind11#from __future__ import division
#pybind11#from builtins import range
#pybind11#
#pybind11#import os
#pybind11#import math
#pybind11#import unittest
#pybind11#
#pybind11#import numpy as np
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.detection as afwDetect
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#import lsst.afw.display.rgb as rgb
#pybind11#
#pybind11#ver1, ver2, ver3 = 1, 3, 1
#pybind11#NO_MATPLOTLIB_STRING = "Requires matplotlib >= %d.%d.%d" % (ver1, ver2, ver3)
#pybind11#try:
#pybind11#    import matplotlib
#pybind11#    mplVersion = matplotlib.__version__
#pybind11#    # Split at + to check for development version (PEP 440)
#pybind11#    mplVersion = mplVersion.split('+')
#pybind11#    versionInfo = tuple(int(s.strip("rc")) for s in mplVersion[0].split("."))
#pybind11#    HAVE_MATPLOTLIB = versionInfo >= (ver1, ver2, ver3)
#pybind11#except ImportError:
#pybind11#    HAVE_MATPLOTLIB = False
#pybind11#
#pybind11#try:
#pybind11#    import scipy.misc
#pybind11#    scipy.misc.imresize
#pybind11#    HAVE_SCIPY_MISC = True
#pybind11#except (ImportError, AttributeError):
#pybind11#    HAVE_SCIPY_MISC = False
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#def saturate(image, satValue):
#pybind11#    """Simulate saturation on an image, so we can test 'replaceSaturatedPixels'
#pybind11#
#pybind11#    Takes an Image, sets saturated pixels to NAN and masks them, returning
#pybind11#    a MaskedImage.
#pybind11#    """
#pybind11#    image = afwImage.makeMaskedImage(image)
#pybind11#    afwDetect.FootprintSet(image, afwDetect.Threshold(satValue), "SAT")
#pybind11#    arr = image.getImage().getArray()
#pybind11#    arr[np.where(arr >= satValue)] = np.nan
#pybind11#    return image
#pybind11#
#pybind11#R, G, B = 2, 1, 0
#pybind11#
#pybind11#
#pybind11#class RgbTestCase(unittest.TestCase):
#pybind11#    """A test case for Rgb"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.min, self.range, self.Q = 0, 5, 20  # asinh
#pybind11#
#pybind11#        width, height = 85, 75
#pybind11#        self.images = []
#pybind11#        self.images.append(afwImage.ImageF(afwGeom.ExtentI(width, height)))
#pybind11#        self.images.append(afwImage.ImageF(afwGeom.ExtentI(width, height)))
#pybind11#        self.images.append(afwImage.ImageF(afwGeom.ExtentI(width, height)))
#pybind11#
#pybind11#        for (x, y, A, g_r, r_i) in [(15, 15, 1000, 1.0, 2.0),
#pybind11#                                    (50, 45, 5500, -1.0, -0.5),
#pybind11#                                    (30, 30, 600, 1.0, 2.5),
#pybind11#                                    (45, 15, 20000, 1.0, 1.0),
#pybind11#                                    ]:
#pybind11#            for i in range(len(self.images)):
#pybind11#                if i == B:
#pybind11#                    amp = A
#pybind11#                elif i == G:
#pybind11#                    amp = A*math.pow(10, 0.4*g_r)
#pybind11#                elif i == R:
#pybind11#                    amp = A*math.pow(10, 0.4*r_i)
#pybind11#
#pybind11#                self.images[i].set(x, y, amp)
#pybind11#
#pybind11#        psf = afwMath.AnalyticKernel(15, 15, afwMath.GaussianFunction2D(2.5, 1.5, 0.5))
#pybind11#
#pybind11#        convolvedImage = type(self.images[0])(self.images[0].getDimensions())
#pybind11#        randomImage = type(self.images[0])(self.images[0].getDimensions())
#pybind11#        rand = afwMath.Random("MT19937", 666)
#pybind11#        for i in range(len(self.images)):
#pybind11#            afwMath.convolve(convolvedImage, self.images[i], psf, True, True)
#pybind11#            afwMath.randomGaussianImage(randomImage, rand)
#pybind11#            randomImage *= 2
#pybind11#            convolvedImage += randomImage
#pybind11#            self.images[i][:] = convolvedImage
#pybind11#        del convolvedImage
#pybind11#        del randomImage
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        for im in self.images:
#pybind11#            del im
#pybind11#        del self.images
#pybind11#
#pybind11#    def testStarsAsinh(self):
#pybind11#        """Test creating an RGB image using an asinh stretch"""
#pybind11#        asinhMap = rgb.AsinhMapping(self.min, self.range, self.Q)
#pybind11#        rgbImage = asinhMap.makeRgbImage(self.images[R], self.images[G], self.images[B])
#pybind11#
#pybind11#        if display:
#pybind11#            rgb.displayRGB(rgbImage)
#pybind11#
#pybind11#    def testStarsAsinhZscale(self):
#pybind11#        """Test creating an RGB image using an asinh stretch estimated using zscale"""
#pybind11#
#pybind11#        rgbImages = [self.images[R], self.images[G], self.images[B]]
#pybind11#
#pybind11#        map = rgb.AsinhZScaleMapping(rgbImages[0])
#pybind11#        rgbImage = map.makeRgbImage(*rgbImages)
#pybind11#
#pybind11#        if display:
#pybind11#            rgb.displayRGB(rgbImage)
#pybind11#
#pybind11#    def testStarsAsinhZscaleIntensity(self):
#pybind11#        """Test creating an RGB image using an asinh stretch estimated using zscale on the intensity"""
#pybind11#
#pybind11#        rgbImages = [self.images[R], self.images[G], self.images[B]]
#pybind11#
#pybind11#        map = rgb.AsinhZScaleMapping(rgbImages)
#pybind11#        rgbImage = map.makeRgbImage(*rgbImages)
#pybind11#
#pybind11#        if display:
#pybind11#            rgb.displayRGB(rgbImage)
#pybind11#
#pybind11#    def testStarsAsinhZscaleIntensityPedestal(self):
#pybind11#        """Test creating an RGB image using an asinh stretch estimated using zscale on the intensity
#pybind11#        where the images each have a pedestal added"""
#pybind11#
#pybind11#        rgbImages = [self.images[R], self.images[G], self.images[B]]
#pybind11#
#pybind11#        pedestal = [100, 400, -400]
#pybind11#        for i, ped in enumerate(pedestal):
#pybind11#            rgbImages[i] += ped
#pybind11#
#pybind11#        map = rgb.AsinhZScaleMapping(rgbImages, pedestal=pedestal)
#pybind11#        rgbImage = map.makeRgbImage(*rgbImages)
#pybind11#
#pybind11#        if display:
#pybind11#            rgb.displayRGB(rgbImage)
#pybind11#
#pybind11#    def testStarsAsinhZscaleIntensityBW(self):
#pybind11#        """Test creating a black-and-white image using an asinh stretch estimated
#pybind11#        using zscale on the intensity"""
#pybind11#
#pybind11#        rgbImage = rgb.AsinhZScaleMapping(self.images[R]).makeRgbImage()
#pybind11#
#pybind11#        if display:
#pybind11#            rgb.displayRGB(rgbImage)
#pybind11#
#pybind11#    @unittest.skipUnless(HAVE_MATPLOTLIB, NO_MATPLOTLIB_STRING)
#pybind11#    def testMakeRGB(self):
#pybind11#        """Test the function that does it all"""
#pybind11#        satValue = 1000.0
#pybind11#        with lsst.utils.tests.getTempFilePath(".png") as fileName:
#pybind11#            red = saturate(self.images[R], satValue)
#pybind11#            green = saturate(self.images[G], satValue)
#pybind11#            blue = saturate(self.images[B], satValue)
#pybind11#            rgb.makeRGB(red, green, blue, self.min, self.range, self.Q, fileName=fileName,
#pybind11#                        saturatedBorderWidth=1, saturatedPixelValue=2000)
#pybind11#            self.assertTrue(os.path.exists(fileName))
#pybind11#
#pybind11#    def testLinear(self):
#pybind11#        """Test using a specified linear stretch"""
#pybind11#
#pybind11#        rgbImage = rgb.LinearMapping(-8.45, 13.44).makeRgbImage(self.images[R])
#pybind11#
#pybind11#        if display:
#pybind11#            rgb.displayRGB(rgbImage)
#pybind11#
#pybind11#    def testLinearMinMax(self):
#pybind11#        """Test using a min/max linear stretch
#pybind11#
#pybind11#        N.b. also checks that an image passed to the ctor is used as the default in makeRgbImage()
#pybind11#        """
#pybind11#
#pybind11#        rgbImage = rgb.LinearMapping(image=self.images[R]).makeRgbImage()
#pybind11#
#pybind11#        if display:
#pybind11#            rgb.displayRGB(rgbImage)
#pybind11#
#pybind11#    def testZScale(self):
#pybind11#        """Test using a zscale stretch"""
#pybind11#
#pybind11#        rgbImage = rgb.ZScaleMapping(self.images[R]).makeRgbImage()
#pybind11#
#pybind11#        if display:
#pybind11#            plt = rgb.displayRGB(rgbImage, False)
#pybind11#            plt.title("zscale")
#pybind11#            plt.show()
#pybind11#
#pybind11#    @unittest.skipUnless(HAVE_MATPLOTLIB, NO_MATPLOTLIB_STRING)
#pybind11#    def testWriteStars(self):
#pybind11#        """Test writing RGB files to disk"""
#pybind11#        asinhMap = rgb.AsinhMapping(self.min, self.range, self.Q)
#pybind11#        rgbImage = asinhMap.makeRgbImage(self.images[R], self.images[G], self.images[B])
#pybind11#        with lsst.utils.tests.getTempFilePath(".png") as fileName:
#pybind11#            rgb.writeRGB(fileName, rgbImage)
#pybind11#            self.assertTrue(os.path.exists(fileName))
#pybind11#
#pybind11#    def testSaturated(self):
#pybind11#        """Test interpolating saturated pixels"""
#pybind11#
#pybind11#        satValue = 1000.0
#pybind11#        for f in [R, G, B]:
#pybind11#            self.images[f] = saturate(self.images[f], satValue)
#pybind11#
#pybind11#        rgb.replaceSaturatedPixels(self.images[R], self.images[G], self.images[B], 1, 2000)
#pybind11#        #
#pybind11#        # Check that we replaced those NaNs with some reasonable value
#pybind11#        #
#pybind11#        for f in [R, G, B]:
#pybind11#            self.assertTrue(np.isfinite(self.images[f].getImage().getArray()).all())
#pybind11#
#pybind11#        if False:
#pybind11#            ds9.mtv(self.images[B], frame=0, title="B")
#pybind11#            ds9.mtv(self.images[G], frame=1, title="G")
#pybind11#            ds9.mtv(self.images[R], frame=2, title="R")
#pybind11#        #
#pybind11#        # Prepare for generating an output file
#pybind11#        #
#pybind11#        for f in [R, G, B]:
#pybind11#            self.images[f] = self.images[f].getImage()
#pybind11#
#pybind11#        asinhMap = rgb.AsinhMapping(self.min, self.range, self.Q)
#pybind11#        rgbImage = asinhMap.makeRgbImage(self.images[R], self.images[G], self.images[B])
#pybind11#
#pybind11#        if display:
#pybind11#            rgb.displayRGB(rgbImage)
#pybind11#
#pybind11#    @unittest.skipUnless(HAVE_SCIPY_MISC, "Resizing images requires scipy.misc")
#pybind11#    def testStarsResizeToSize(self):
#pybind11#        """Test creating an RGB image of a specified size"""
#pybind11#
#pybind11#        xSize = self.images[R].getWidth()//2
#pybind11#        ySize = self.images[R].getHeight()//2
#pybind11#        for rgbImages in ([self.images[R], self.images[G], self.images[B]],
#pybind11#                          [afwImage.ImageU(_.getArray().astype('uint16')) for _ in [
#pybind11#                              self.images[R], self.images[G], self.images[B]]]):
#pybind11#            rgbImage = rgb.AsinhZScaleMapping(rgbImages[0]).makeRgbImage(*rgbImages,
#pybind11#                                                                         xSize=xSize, ySize=ySize)
#pybind11#
#pybind11#            if display:
#pybind11#                rgb.displayRGB(rgbImage)
#pybind11#
#pybind11#    @unittest.skipUnless(HAVE_SCIPY_MISC, "Resizing images requires scipy.misc")
#pybind11#    def testStarsResizeSpecifications(self):
#pybind11#        """Test creating an RGB image changing the output """
#pybind11#
#pybind11#        rgbImages = [self.images[R], self.images[G], self.images[B]]
#pybind11#        map = rgb.AsinhZScaleMapping(rgbImages[0])
#pybind11#
#pybind11#        for xSize, ySize, frac in [(self.images[R].getWidth()//2, self.images[R].getHeight()//2, None),
#pybind11#                                   (2*self.images[R].getWidth(), None, None),
#pybind11#                                   (self.images[R].getWidth()//2, None, None),
#pybind11#                                   (None, self.images[R].getHeight()//2, None),
#pybind11#                                   (None, None, 0.5),
#pybind11#                                   (None, None, 2),
#pybind11#                                   ]:
#pybind11#            rgbImage = map.makeRgbImage(*rgbImages, xSize=xSize, ySize=ySize, rescaleFactor=frac)
#pybind11#
#pybind11#            h, w = rgbImage.shape[0:2]
#pybind11#            self.assertTrue(xSize is None or xSize == w)
#pybind11#            self.assertTrue(ySize is None or ySize == h)
#pybind11#            self.assertTrue(frac is None or w == int(frac*self.images[R].getWidth()),
#pybind11#                            "%g == %g" % (w, int((frac if frac else 1)*self.images[R].getWidth())))
#pybind11#
#pybind11#            if display:
#pybind11#                rgb.displayRGB(rgbImage)
#pybind11#
#pybind11#    @unittest.skipUnless(HAVE_SCIPY_MISC, "Resizing images requires scipy.misc")
#pybind11#    @unittest.skipUnless(HAVE_MATPLOTLIB, NO_MATPLOTLIB_STRING)
#pybind11#    def testMakeRGBResize(self):
#pybind11#        """Test the function that does it all, including rescaling"""
#pybind11#        rgb.makeRGB(self.images[R], self.images[G], self.images[B], xSize=40, ySize=60)
#pybind11#
#pybind11#        with lsst.utils.tests.getTempFilePath(".png") as fileName:
#pybind11#            rgb.makeRGB(self.images[R], self.images[G], self.images[B], fileName=fileName, rescaleFactor=0.5)
#pybind11#            self.assertTrue(os.path.exists(fileName))
#pybind11#
#pybind11#    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#    #
#pybind11#    # Test that the legacy API still works, although it's deprecated
#pybind11#    #
#pybind11#    def writeFileLegacyAPI(self, fileName):
#pybind11#        asinh = rgb.asinhMappingF(self.min, self.range, self.Q)
#pybind11#        rgbImage = rgb.RgbImageF(self.images[R], self.images[G], self.images[B], asinh)
#pybind11#        if False:
#pybind11#            ds9.mtv(self.images[B], frame=0, title="B")
#pybind11#            ds9.mtv(self.images[G], frame=1, title="G")
#pybind11#            ds9.mtv(self.images[R], frame=2, title="R")
#pybind11#
#pybind11#        rgbImage.write(fileName)
#pybind11#
#pybind11#    @unittest.skipUnless(HAVE_MATPLOTLIB, NO_MATPLOTLIB_STRING)
#pybind11#    def testWriteStarsLegacyAPI(self):
#pybind11#        with lsst.utils.tests.getTempFilePath(".png") as fileName:
#pybind11#            self.writeFileLegacyAPI(fileName)
#pybind11#            self.assertTrue(os.path.exists(fileName))
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

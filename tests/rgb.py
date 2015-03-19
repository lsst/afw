#!/usr/bin/env python
"""
Tests for RGB Images

Run with:
   ./rgb.py
or
   python
   >>> import rgb; rgb.run()
"""

import math
import os
import numpy as np
import unittest

import lsst.utils.tests as utilsTests
import lsst.afw.detection as afwDetect
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.rgb as rgb

try:
    import matplotlib
    versionInfo = tuple(int(s.strip("rc")) for s in matplotlib.__version__.split("."))
    HAVE_MATPLOTLIB = versionInfo >= (1, 3, 1)
except ImportError:
    HAVE_MATPLOTLIB = False

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# A context object to clean up temporary files
#
from contextlib import contextmanager

@contextmanager
def Tempfile(fileName, remove=True):
    pass                                # entry to context block
    yield
    if remove:                          # exit from context block
        os.remove(fileName)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

R, G, B = 2, 1, 0

class RgbTestCase(unittest.TestCase):
    """A test case for Rgb"""
    def setUp(self):
        self.min, self.range, self.Q = 0, 5, 20 # asinh

        width, height = 85, 75
        self.images = []
        self.images.append(afwImage.ImageF(afwGeom.ExtentI(width, height)))
        self.images.append(afwImage.ImageF(afwGeom.ExtentI(width, height)))
        self.images.append(afwImage.ImageF(afwGeom.ExtentI(width, height)))

        for (x, y, A, g_r, r_i) in [(15, 15, 1000,  1.0,  2.0),
                                    (50, 45, 5500, -1.0, -0.5),
                                    (30, 30,  600,  1.0,  2.5),
                                    (45, 15, 20000,  1.0,  1.0),
                                    ]:
            for i in range(len(self.images)):
                if i == B:
                    amp = A
                elif i == G:
                    amp = A*math.pow(10, 0.4*g_r)
                elif i == R:
                    amp = A*math.pow(10, 0.4*r_i)

                self.images[i].set(x, y, amp)

        psf = afwMath.AnalyticKernel(15, 15, afwMath.GaussianFunction2D(2.5, 1.5, 0.5))

        convolvedImage = type(self.images[0])(self.images[0].getDimensions())
        randomImage = type(self.images[0])(self.images[0].getDimensions())
        rand = afwMath.Random("MT19937", 666)
        for i in range(len(self.images)):
            afwMath.convolve(convolvedImage, self.images[i], psf, True, True)
            afwMath.randomGaussianImage(randomImage, rand)
            randomImage *= 2
            convolvedImage += randomImage
            self.images[i] <<= convolvedImage
        del convolvedImage; del randomImage

    def tearDown(self):
        for im in self.images:
            del im
        del self.images

    def testStars(self):
        """Test creating an RGB image"""
        asinhMap = rgb.AsinhMapping(self.min, self.range, self.Q)
        rgbImage = asinhMap.makeRgbImage(self.images[R], self.images[G], self.images[B])

        if display:
            rgb.displayRGB(rgbImage)            

    @unittest.skipUnless(HAVE_MATPLOTLIB, "Requires matplotlib >= 1.3.1")
    def testWriteStars(self):
        """Test writing RGB files to disk"""
        asinhMap = rgb.AsinhMapping(self.min, self.range, self.Q)
        rgbImage = asinhMap.makeRgbImage(self.images[R], self.images[G], self.images[B])

        for ext in ("jpeg", "jpg", "png", "tif", "tiff"):
            fileName = "rgb.%s" % ext
            with Tempfile(fileName, remove=True):
                rgb.writeRGB(fileName, rgbImage)

                if False:               # you'll also want to set remove=False in Tempfile manager
                    os.system("open %s > /dev/null 2>&1" % fileName)

    def testSaturated(self):
        """Test interpolating saturated pixels"""

        feet = {}
        for f in [R, G, B]:
            self.images[f] = afwImage.makeMaskedImage(self.images[f])

            ds = afwDetect.FootprintSet(self.images[f], afwDetect.Threshold(1000), "SAT")
            feet[f] = ds.getFootprints()

            arr = self.images[f].getImage().getArray()
            arr[np.where(arr >= 1000)] = np.nan

        rgb.replaceSaturatedPixels(self.images[R], self.images[G], self.images[B], 1, 2000)
        #
        # Check that we replaced those NaNs with some reasonable value
        #
        f0 = [k for k, v in feet.items() if v][0] # find a filter with a saturated region
        foot = feet[f0][0]
        s = foot.getSpans()[0]
        for f in [R, G, B]:
            self.assertTrue(np.isfinite(self.images[f].getImage().getArray()).all())
        
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
        rgbImage = asinhMap.makeRgbImage(self.images[R], self.images[G], self.images[B])

        if display:
            rgb.displayRGB(rgbImage)            

    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    # Test that the legacy API still works, although it's deprecated
    #
    def writeFileLegacyAPI(self, fileName):
        asinh = rgb.asinhMappingF(self.min, self.range, self.Q)
        rgbImage = rgb.RgbImageF(self.images[R], self.images[G], self.images[B], asinh)
        if False:
            ds9.mtv(self.images[B], frame=0, title="B")
            ds9.mtv(self.images[G], frame=1, title="G")
            ds9.mtv(self.images[R], frame=2, title="R")

        rgbImage.write(fileName)

    @unittest.skipUnless(HAVE_MATPLOTLIB, "Requires matplotlib >= 1.3.1")
    def testWriteStarsLegacyAPI(self):
        for ext in ("jpeg", "jpg", "png", "tif", "tiff"):
            fileName = "rgb_legacyAPI.%s" % ext
            with Tempfile(fileName, remove=True):
                self.writeFileLegacyAPI(fileName)

                if False:
                    os.system("open %s > /dev/null 2>&1" % fileName)

        def tst():
            self.writeFileLegacyAPI("rgb.unknown")
        self.assertRaises(ValueError, tst)
        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(RgbTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

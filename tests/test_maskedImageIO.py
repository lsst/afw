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
Tests for MaskedImages

Run with:
   python test_maskedImageIO.py
or
   pytest test_maskedImageIO.py
"""

import contextlib
import itertools
import os.path
import unittest
import shutil
import tempfile

import numpy as np
import astropy.io.fits

import lsst.utils
import lsst.utils.tests
import lsst.daf.base as dafBase
import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display as afwDisplay
import lsst.pex.exceptions as pexEx

try:
    dataDir = lsst.utils.getPackageDir("afwdata")
except LookupError:
    dataDir = None

try:
    type(display)
    afwDisplay.setDefaultMaskTransparency(75)
except NameError:
    display = False


class MaskedImageTestCase(lsst.utils.tests.TestCase):
    """A test case for MaskedImage"""

    def setUp(self):
        # Set a (non-standard) initial Mask plane definition
        #
        # Ideally we'd use the standard dictionary and a non-standard file, but
        # a standard file's what we have
        #
        self.Mask = afwImage.Mask
        mask = self.Mask()

        # Store the default mask planes for later use
        maskPlaneDict = self.Mask().getMaskPlaneDict()
        self.defaultMaskPlanes = sorted(
            maskPlaneDict, key=maskPlaneDict.__getitem__)

        # reset so tests will be deterministic
        self.Mask.clearMaskPlaneDict()
        for p in ("ZERO", "BAD", "SAT", "INTRP", "CR", "EDGE"):
            mask.addMaskPlane(p)

        if dataDir is not None:
            if False:
                self.fileName = os.path.join(dataDir, "Small_MI.fits")
            else:
                self.fileName = os.path.join(
                    dataDir, "CFHT", "D4", "cal-53535-i-797722_1.fits")
            self.mi = afwImage.MaskedImageF(self.fileName)

    def tearDown(self):
        if dataDir is not None:
            del self.mi
        # Reset the mask plane to the default
        self.Mask.clearMaskPlaneDict()
        for p in self.defaultMaskPlanes:
            self.Mask.addMaskPlane(p)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testFitsRead(self):
        """Check if we read MaskedImages"""

        image = self.mi.getImage()
        mask = self.mi.getMask()

        if display:
            afwDisplay.Display(frame=0).mtv(self.mi, title="Image")

        self.assertEqual(image[32, 1, afwImage.LOCAL], 3728)
        self.assertEqual(mask[0, 0, afwImage.LOCAL], 2)  # == BAD

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testFitsReadImage(self):
        """Check if we can read a single-HDU image as a MaskedImage, setting the mask and variance
        planes to zero."""
        filename = os.path.join(dataDir, "data", "small_img.fits")
        image = afwImage.ImageF(filename)
        maskedImage = afwImage.MaskedImageF(filename)
        exposure = afwImage.ExposureF(filename)
        self.assertEqual(image[0, 0, afwImage.LOCAL], maskedImage.image[0, 0, afwImage.LOCAL])
        self.assertEqual(
            image[0, 0, afwImage.LOCAL], exposure.getMaskedImage().image[0, 0, afwImage.LOCAL])
        self.assertTrue(np.all(maskedImage.getMask().getArray() == 0))
        self.assertTrue(
            np.all(exposure.getMaskedImage().getMask().getArray() == 0))
        self.assertTrue(np.all(maskedImage.getVariance().getArray() == 0.0))
        self.assertTrue(
            np.all(exposure.getMaskedImage().getVariance().getArray() == 0.0))

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testFitsReadConform(self):
        """Check if we read MaskedImages and make them replace Mask's plane dictionary"""

        metadata, bbox, conformMasks = None, lsst.geom.Box2I(), True
        self.mi = afwImage.MaskedImageF(
            self.fileName, metadata, bbox, afwImage.LOCAL, conformMasks)

        image = self.mi.getImage()
        mask = self.mi.getMask()

        self.assertEqual(image[32, 1, afwImage.LOCAL], 3728)
        # i.e. not shifted 1 place to the right
        self.assertEqual(mask[0, 0, afwImage.LOCAL], 1)

        self.assertEqual(mask.getMaskPlane("CR"), 3,
                         "Plane CR has value specified in FITS file")

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testFitsReadNoConform2(self):
        """Check that reading a mask doesn't invalidate the plane dictionary"""

        testMask = afwImage.Mask(self.fileName, hdu=2)

        mask = self.mi.getMask()
        mask |= testMask

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testFitsReadConform2(self):
        """Check that conforming a mask invalidates the plane dictionary"""

        hdu, metadata, bbox, conformMasks = 2, None, lsst.geom.Box2I(), True
        testMask = afwImage.Mask(self.fileName,
                                 hdu, metadata, bbox, afwImage.LOCAL, conformMasks)

        mask = self.mi.getMask()

        def tst(mask=mask):
            mask |= testMask

        self.assertRaises(pexEx.RuntimeError, tst)

    def testTicket617(self):
        """Test reading an F64 image and converting it to a MaskedImage"""
        im = afwImage.ImageD(lsst.geom.Extent2I(100, 100))
        im.set(666)
        afwImage.MaskedImageD(im)

    def testReadWriteXY0(self):
        """Test that we read and write (X0, Y0) correctly"""
        im = afwImage.MaskedImageF(lsst.geom.Extent2I(10, 20))

        x0, y0 = 1, 2
        im.setXY0(x0, y0)
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            im.writeFits(tmpFile)

            im2 = im.Factory(tmpFile)
            self.assertEqual(im2.getX0(), x0)
            self.assertEqual(im2.getY0(), y0)

            self.assertEqual(im2.getImage().getX0(), x0)
            self.assertEqual(im2.getImage().getY0(), y0)

            self.assertEqual(im2.getMask().getX0(), x0)
            self.assertEqual(im2.getMask().getY0(), y0)

            self.assertEqual(im2.getVariance().getX0(), x0)
            self.assertEqual(im2.getVariance().getY0(), y0)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testReadFitsWithOptions(self):
        xy0Offset = lsst.geom.Extent2I(7, 5)
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(10, 11), lsst.geom.Extent2I(31, 22))

        with lsst.utils.tests.getTempFilePath(".fits") as filepath:
            # write a temporary version of the image with non-zero XY0
            imagePath = os.path.join(dataDir, "data", "med.fits")
            maskedImage = afwImage.MaskedImageD(imagePath)
            maskedImage.setXY0(lsst.geom.Point2I(xy0Offset))
            maskedImage.writeFits(filepath)

            for ImageClass, imageOrigin in itertools.product(
                (afwImage.MaskedImageF, afwImage.MaskedImageD),
                (None, "LOCAL", "PARENT"),
            ):
                with self.subTest(ImageClass=str(ImageClass), imageOrigin=repr(imageOrigin)):
                    fullImage = ImageClass(filepath)
                    options = dafBase.PropertySet()
                    options.set("llcX", bbox.getMinX())
                    options.set("llcY", bbox.getMinY())
                    options.set("width", bbox.getWidth())
                    options.set("height", bbox.getHeight())
                    if imageOrigin is not None:
                        options.set("imageOrigin", imageOrigin)
                    image1 = ImageClass.readFitsWithOptions(filepath, options)
                    readBBoxParent = lsst.geom.Box2I(bbox)
                    if imageOrigin == "LOCAL":
                        readBBoxParent.shift(xy0Offset)
                    self.assertMaskedImagesEqual(image1, ImageClass(fullImage, readBBoxParent))

                    for name in ("llcY", "width", "height"):
                        badOptions = options.deepCopy()
                        badOptions.remove(name)
                        with self.assertRaises(LookupError):
                            ImageClass.readFitsWithOptions(filepath, badOptions)

                        badOptions = options.deepCopy()
                        badOptions.set("imageOrigin", "INVALID")
                        with self.assertRaises(RuntimeError):
                            ImageClass.readFitsWithOptions(filepath, badOptions)


@contextlib.contextmanager
def tmpFits(*hdus):
    # Given a list of numpy arrays, create a temporary FITS file that
    # contains them as consecutive HDUs. Yield it, then remove it.
    hdus = [astropy.io.fits.PrimaryHDU(hdus[0])] + \
        [astropy.io.fits.ImageHDU(hdu) for hdu in hdus[1:]]
    hdulist = astropy.io.fits.HDUList(hdus)
    tempdir = tempfile.mkdtemp()
    try:
        filename = os.path.join(tempdir, 'test.fits')
        hdulist.writeto(filename)
        yield filename
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


class MultiExtensionTestCase:
    """Base class for testing that we correctly read multi-extension FITS files.

    MEF files may be read to either MaskedImage or Exposure objects. We apply
    the same set of tests to each by subclassing and defining _constructImage
    and _checkImage.
    """
    # When persisting a MaskedImage (or derivative, e.g. Exposure) to FITS, we impose a data
    # model which the combination of the limits of the FITS structure and the desire to maintain
    # backwards compatibility make it hard to express. We attempt to make this as safe as
    # possible by handling the following situations and logging appropriate warnings:
    #
    # Note that Exposures always set needAllHdus to False.
    #
    # 1. If needAllHdus is true:
    #    1.1 If the user has specified a non-default HDU, we throw.
    #    1.2 If the user has not specified an HDU (or has specified one equal to the default):
    #        1.2.1 If any of the image, mask or variance is unreadable (eg because they don't
    #              exist, or they have the wrong data type), we throw.
    #        1.2.2 Otherwise, we return the MaskedImage with image/mask/variance set as
    #              expected.
    # 2. If needAllHdus is false:
    #    2.1 If the user has specified a non-default HDU:
    #        2.1.1 If the user specified HDU is unreadable, we throw.
    #        2.1.2 Otherwise, we return the contents of that HDU as the image and default
    #              (=empty) mask & variance.
    #    2.2 If the user has not specified an HDU, or has specified one equal to the default:
    #        2.2.1 If the default HDU is unreadable, we throw.
    #        2.2.2 Otherwise, we attempt to read both mask and variance from the FITS file,
    #              and return them together with the image. If one or both are unreadable,
    #              we fall back to an empty default for the missing data and return the
    #              remainder..
    #
    # See also the discussion at DM-2599.

    def _checkMaskedImage(self, mim, width, height, val1, val2, val3):
        # Check that the input image has dimensions width & height and that the image, mask and
        # variance have mean val1, val2 & val3 respectively.
        self.assertEqual(mim.getWidth(), width)
        self.assertEqual(mim.getHeight(), width)
        self.assertEqual(
            afwMath.makeStatistics(mim.getImage(), afwMath.MEAN).getValue(),
            val1)
        s = afwMath.makeStatistics(mim.getMask(), afwMath.SUM | afwMath.NPOINT)
        self.assertEqual(float(s.getValue(afwMath.SUM)) / s.getValue(afwMath.NPOINT),
                         val2)
        self.assertEqual(
            afwMath.makeStatistics(mim.getVariance(), afwMath.MEAN).getValue(),
            val3)

    def testUnreadableExtensionAsImage(self):
        # Test for case 2.1.1 above.
        with tmpFits(None, np.array([[1]]), np.array([[2]], dtype=np.int16), None) as fitsfile:
            self.assertRaises(Exception, self._constructImage, fitsfile, 3)

    def testReadableExtensionAsImage(self):
        # Test for case 2.1.2 above.
        with tmpFits(None, np.array([[1]]), np.array([[2]], dtype=np.int16),
                     np.array([[3]])) as fitsfile:
            self._checkImage(self._constructImage(fitsfile, 3), 1, 1, 3, 0, 0)

    def testUnreadbleDefaultAsImage(self):
        # Test for case 2.2.1 above.
        with tmpFits(None, None, np.array([[2]], dtype=np.int16), np.array([[3]])) as fitsfile:
            self.assertRaises(Exception, self._constructImage, fitsfile)

    def testUnreadbleOptionalExtensions(self):
        # Test for case 2.2.2 above.
        # Unreadable mask.
        with tmpFits(None, np.array([[1]]), None, np.array([[3]])) as fitsfile:
            self._checkImage(self._constructImage(fitsfile), 1, 1, 1, 0, 3)
        # Unreadable variance.
        with tmpFits(None, np.array([[1]]), np.array([[2]], dtype=np.int16), None) as fitsfile:
            self._checkImage(self._constructImage(fitsfile, needAllHdus=False),
                             1, 1, 1, 2, 0)


class MaskedMultiExtensionTestCase(MultiExtensionTestCase, lsst.utils.tests.TestCase):
    """Derived version of MultiExtensionTestCase for MaskedImages."""

    def _constructImage(self, filename, hdu=None, needAllHdus=False):
        # Construct an instance of MaskedImageF by loading from filename. If hdu
        # is specified, load that HDU specifically. Pass through needAllHdus
        # to the MaskedImageF constructor.  This function exists only to stub
        # default arguments into the constructor for parameters which we are
        # not exercising in this test.
        if hdu:
            filename = f"{filename}[{hdu}]"
        return afwImage.MaskedImageF(filename, None, lsst.geom.Box2I(), afwImage.PARENT, False, needAllHdus)

    def _checkImage(self, *args, **kwargs):
        self._checkMaskedImage(*args, **kwargs)

    def testNeedAllHdus(self):
        # Tests for cases 1.1 & 1.2.2 above.
        # We'll regard it as ok for the user to specify any of:
        # * No HDU;
        # * The "zeroeth" (primary) HDU;
        # * The first (first extension) HDU.
        # Any others should raise when needAllHdus is true
        with tmpFits(None, np.array([[1]]), np.array([[2]], dtype=np.int16),
                     np.array([[3]])) as fitsfile:
            # No HDU specified -> ok.
            self._checkImage(self._constructImage(fitsfile, needAllHdus=True),
                             1, 1, 1, 2, 3)
            # First HDU -> ok.
            self._checkImage(
                self._constructImage(fitsfile, 0, needAllHdus=True),
                1, 1, 1, 2, 3)
            # First HDU -> ok.
            self._checkImage(
                self._constructImage(fitsfile, 1, needAllHdus=True),
                1, 1, 1, 2, 3)
            # Second HDU -> raises.
            self.assertRaises(Exception, self._constructImage,
                              fitsfile, 2, needAllHdus=True)

    def testUnreadableImage(self):
        # Test for case 1.2.1 above.
        with tmpFits(None, None, np.array([[2]], dtype=np.int16), np.array([[3]])) as fitsfile:
            self.assertRaises(Exception, self._constructImage,
                              fitsfile, None, needAllHdus=True)

    def testUnreadableMask(self):
        # Test for case 1.2.1 above.
        with tmpFits(None, np.array([[1]]), None, np.array([[3]])) as fitsfile:
            self.assertRaises(Exception, self._constructImage,
                              fitsfile, None, needAllHdus=True)

    def testUnreadableVariance(self):
        # Test for case 1.2.1 above.
        with tmpFits(None, np.array([[1]]), np.array([[2]], dtype=np.int16), None) as fitsfile:
            self.assertRaises(Exception, self._constructImage,
                              fitsfile, None, needAllHdus=True)


class ExposureMultiExtensionTestCase(MultiExtensionTestCase, lsst.utils.tests.TestCase):
    """Derived version of MultiExtensionTestCase for Exposures."""

    def _constructImage(self, filename, hdu=None, needAllHdus=False):
        # Construct an instance of ExposureF by loading from filename. If hdu
        # is specified, load that HDU specifically. needAllHdus exists for API
        # compatibility, but should always be False. This function exists only
        # to stub default arguments into the constructor for parameters which
        # we are not exercising in this test.
        if hdu:
            filename = f"{filename}[{hdu}]"
        if needAllHdus:
            raise ValueError("Cannot needAllHdus with Exposure")
        return afwImage.ExposureF(filename)

    def _checkImage(self, im, width, height, val1, val2, val3):
        self._checkMaskedImage(im.getMaskedImage(), width,
                               height, val1, val2, val3)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

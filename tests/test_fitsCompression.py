#
# LSST Data Management System
# Copyright 2017 LSST Corporation.
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
#

import os
import unittest
import itertools

import numpy as np
import astropy.io.fits

import lsst.utils
import lsst.daf.base
import lsst.daf.persistence
import lsst.geom
import lsst.afw.geom
import lsst.afw.image
import lsst.afw.fits
import lsst.utils.tests
from lsst.afw.image import LOCAL
from lsst.afw.fits import ImageScalingOptions, ImageCompressionOptions


def checkAstropy(image, filename, hduNum=0):
    """Check that astropy can read our file

    We don't insist on equality for low BITPIX (8, 16) when the original
    type is double-precision: astropy will (quite reasonably) read that
    into a single-precision image and apply bscale,bzero to that so it's
    going to be subject to roundoff.

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image read by our own code.
    filename : `str`
        Filename of FITS file to read with astropy.
    hduNum : `int`
        HDU number of interest.
    """
    print("Astropy currently doesn't read our compressed images perfectly.")
    return

    def parseVersion(version):
        return tuple(int(vv) for vv in np.array(version.split(".")))

    if parseVersion(astropy.__version__) <= parseVersion("2.0.1"):
        # astropy 2.0.1 and earlier have problems:
        # * Doesn't support GZIP_2: https://github.com/astropy/astropy/pull/6486
        # * Uses the wrong array type: https://github.com/astropy/astropy/pull/6492
        print("Refusing to check with astropy version %s due to astropy bugs" % (astropy.__version__,))
        return
    hdu = astropy.io.fits.open(filename)[hduNum]
    if hdu.header["BITPIX"] in (8, 16) and isinstance(image, lsst.afw.image.ImageD):
        return
    dtype = image.getArray().dtype
    theirs = hdu.data.astype(dtype)
    # Allow for minor differences due to arithmetic: +/- 1 in the last place
    np.testing.assert_array_max_ulp(theirs, image.getArray())


class ImageScalingTestCase(lsst.utils.tests.TestCase):
    """Tests of image scaling

    The pattern here is to create an image, write it out with a
    specific scaling algorithm, read it back in and test that everything
    is as we expect. We do this for each scaling algorithm in its own
    test, and within that test iterate over various parameters (input
    image type, BITPIX, etc.). The image we create has a few features
    (low, high and masked pixels) that we check.
    """
    def setUp(self):
        self.bbox = lsst.geom.Box2I(lsst.geom.Point2I(123, 456), lsst.geom.Extent2I(7, 8))
        self.base = 456  # Base value for pixels
        self.highValue = 789  # Value for high pixel
        self.lowValue = 123  # Value for low pixel
        self.maskedValue = 12345  # Value for masked pixel (to throw off statistics)
        self.highPixel = lsst.geom.Point2I(1, 1)  # Location of high pixel
        self.lowPixel = lsst.geom.Point2I(2, 2)  # Location of low pixel
        self.maskedPixel = lsst.geom.Point2I(3, 3)  # Location of masked pixel
        self.badMask = "BAD"  # Mask plane to set for masked pixel
        self.stdev = 5.0  # Noise stdev to add to image

    def makeImage(self, ImageClass, scaling, addNoise=True):
        """Make an image for testing

        We create an image, persist and unpersist it, returning
        some data to the caller.

        Parameters
        ----------
        ImageClass : `type`, an `lsst.afw.image.Image` class
            Class of image to create.
        scaling : `lsst.afw.fits.ImageScalingOptions`
            Scaling to apply during persistence.
        addNoise : `bool`
            Add noise to image?

        Returns
        -------
        image : `lsst.afw.image.Image` (ImageClass)
            Created image.
        unpersisted : `lsst.afw.image.Image` (ImageClass)
            Unpersisted image.
        bscale, bzero : `float`
            FITS scale factor and zero used.
        minValue, maxValue : `float`
            Minimum and maximum value given the nominated scaling.
        """
        image = ImageClass(self.bbox)
        mask = lsst.afw.image.Mask(self.bbox)
        mask.addMaskPlane(self.badMask)
        bad = mask.getPlaneBitMask(self.badMask)
        image.set(self.base)
        image[self.highPixel, LOCAL] = self.highValue
        image[self.lowPixel, LOCAL] = self.lowValue
        image[self.maskedPixel, LOCAL] = self.maskedValue
        mask[self.maskedPixel, LOCAL] = bad

        rng = np.random.RandomState(12345)
        dtype = image.getArray().dtype
        if addNoise:
            image.getArray()[:] += rng.normal(0.0, self.stdev, image.getArray().shape).astype(dtype)

        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            with lsst.afw.fits.Fits(filename, "w") as fits:
                options = lsst.afw.fits.ImageWriteOptions(scaling)
                header = lsst.daf.base.PropertyList()
                image.writeFits(fits, options, header, mask)
            unpersisted = ImageClass(filename)
            self.assertEqual(image.getBBox(), unpersisted.getBBox())

            header = lsst.afw.fits.readMetadata(filename)
            bscale = header.getScalar("BSCALE")
            bzero = header.getScalar("BZERO")

            if scaling.algorithm != ImageScalingOptions.NONE:
                self.assertEqual(header.getScalar("BITPIX"), scaling.bitpix)

            if scaling.bitpix == 8:  # unsigned, says FITS
                maxValue = bscale*(2**scaling.bitpix - 1) + bzero
                minValue = bzero
            else:
                maxValue = bscale*(2**(scaling.bitpix - 1) - 1) + bzero
                if scaling.bitpix == 32:
                    # cfitsio pads 10 values, and so do we
                    minValue = -bscale*(2**(scaling.bitpix - 1) - 10) + bzero
                else:
                    minValue = -bscale*(2**(scaling.bitpix - 1)) + bzero

            # Convert scalars to the appropriate type
            maxValue = np.array(maxValue, dtype=image.getArray().dtype)
            minValue = np.array(minValue, dtype=image.getArray().dtype)

            checkAstropy(unpersisted, filename)

        return image, unpersisted, bscale, bzero, minValue, maxValue

    def checkPixel(self, unpersisted, original, xy, expected, rtol=None, atol=None):
        """Check one of the special pixels

        After checking, we set this pixel to the original value so
        it's then easy to compare the entire image.

        Parameters
        ----------
        unpersisted : `lsst.afw.image.Image`
            Unpersisted image.
        original : `lsst.afw.image.Image`
            Original image.
        xy : `tuple` of two `int`s
            Position of pixel to check.
        expected : scalar
            Expected value of pixel.
        rtol, atol : `float` or `None`
            Relative/absolute tolerance for comparison.
        """
        if np.isnan(expected):
            self.assertTrue(np.isnan(unpersisted[xy, LOCAL]))
        else:
            self.assertFloatsAlmostEqual(unpersisted[xy, LOCAL], expected, rtol=rtol, atol=atol)
        unpersisted[xy, LOCAL] = original[xy, LOCAL]  # for ease of comparison of the whole image

    def checkSpecialPixels(self, original, unpersisted, maxValue, minValue, rtol=None, atol=None):
        """Check the special pixels

        Parameters
        ----------
        original : `lsst.afw.image.Image`
            Original image.
        unpersisted : `lsst.afw.image.Image`
            Unpersisted image.
        minValue, maxValue : `float`
            Minimum and maximum value given the nominated scaling.
        rtol, atol : `float` or `None`
            Relative/absolute tolerance for comparison.
        """
        highValue = original[self.highPixel, LOCAL]
        lowValue = original[self.lowPixel, LOCAL]
        maskedValue = original[self.maskedPixel, LOCAL]

        expectHigh = min(highValue, maxValue)
        expectLow = max(lowValue, minValue)
        expectMasked = min(maskedValue, maxValue)

        if unpersisted.getArray().dtype in (np.float32, np.float64):
            if highValue >= maxValue:
                expectHigh = np.nan
            if maskedValue >= maxValue:
                expectMasked = np.nan

        self.checkPixel(unpersisted, original, self.highPixel, expectHigh, rtol=rtol, atol=atol)
        self.checkPixel(unpersisted, original, self.lowPixel, expectLow, rtol=rtol, atol=atol)
        self.checkPixel(unpersisted, original, self.maskedPixel, expectMasked, rtol=rtol, atol=atol)

    def checkRange(self, ImageClass, bitpix):
        """Check that the RANGE scaling works

        Parameters
        ----------
        ImageClass : `type`, an `lsst.afw.image.Image` class
            Class of image to create.
        bitpix : `int`
            Bits per pixel for FITS image.
        """
        scaling = ImageScalingOptions(ImageScalingOptions.RANGE, bitpix, [u"BAD"], fuzz=False)
        original, unpersisted, bscale, bzero, minValue, maxValue = self.makeImage(ImageClass, scaling, False)

        numValues = 2**bitpix - 1
        numValues -= 2  # Padding on either end
        if bitpix == 32:
            numValues -= 10
        bscaleExpect = (self.highValue - self.lowValue)/numValues
        self.assertFloatsAlmostEqual(bscale, bscaleExpect, atol=1.0e-6)  # F32 resolution

        rtol = 1.0/2**(bitpix - 1)
        self.checkSpecialPixels(original, unpersisted, maxValue, minValue, atol=bscale)
        self.assertImagesAlmostEqual(original, unpersisted, rtol=rtol)

    def checkStdev(self, ImageClass, bitpix, algorithm, quantizeLevel, quantizePad):
        """Check that one of the STDEV scaling algorithms work

        Parameters
        ----------
        ImageClass : `type`, an `lsst.afw.image.Image` class
            Class of image to create.
        bitpix : `int`
            Bits per pixel for FITS image.
        algorithm : `lsst.afw.fits.ImageScalingOptions.ScalingAlgorithm`
            Scaling algorithm to apply (one of the STDEV_*).
        quantizeLevel : `float`
            Quantization level.
        quantizePad : `float`
            Quantization padding.
        """
        scaling = lsst.afw.fits.ImageScalingOptions(algorithm, bitpix, [u"BAD"], fuzz=False,
                                                    quantizeLevel=quantizeLevel, quantizePad=quantizePad)

        makeImageResults = self.makeImage(ImageClass, scaling)
        original, unpersisted, bscale, bzero, minValue, maxValue = makeImageResults

        self.assertFloatsAlmostEqual(bscale, self.stdev/quantizeLevel, rtol=3.0/quantizeLevel)
        self.checkSpecialPixels(original, unpersisted, maxValue, minValue, atol=bscale)
        self.assertImagesAlmostEqual(original, unpersisted, atol=bscale)

    def testRange(self):
        """Test that the RANGE scaling works on floating-point inputs

        We deliberately don't include BITPIX=64 because int64 provides
        a larger dynamic range than 'double BSCALE' can handle.
        """
        classList = (lsst.afw.image.ImageF, lsst.afw.image.ImageD)
        bitpixList = (8, 16, 32)
        for cls, bitpix in itertools.product(classList, bitpixList):
            self.checkRange(cls, bitpix)

    def testStdev(self):
        """Test that the STDEV scalings work on floating-point inputs

        We deliberately don't include BITPIX=64 because int64 provides
        a larger dynamic range than 'double BSCALE' can handle.

        We deliberately don't include BITPIX=8 because that provides
        only a tiny dynamic range where everything goes out of range easily.
        """
        classList = (lsst.afw.image.ImageF, lsst.afw.image.ImageD)
        bitpixList = (16, 32)
        algorithmList = (ImageScalingOptions.STDEV_POSITIVE, ImageScalingOptions.STDEV_NEGATIVE,
                         ImageScalingOptions.STDEV_BOTH)
        quantizeLevelList = (2.0, 10.0, 100.0)
        quantizePadList = (5.0, 10.0, 100.0)
        for values in itertools.product(classList, bitpixList, algorithmList,
                                        quantizeLevelList, quantizePadList):
            self.checkStdev(*values)

    def testRangeFailures(self):
        """Test that the RANGE scaling fails on integer inputs"""
        classList = (lsst.afw.image.ImageU, lsst.afw.image.ImageI, lsst.afw.image.ImageL)
        bitpixList = (8, 16, 32)
        for cls, bitpix in itertools.product(classList, bitpixList):
            with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
                self.checkRange(cls, bitpix)

    def testStdevFailures(self):
        """Test that the STDEV scalings fail on integer inputs"""
        classList = (lsst.afw.image.ImageU, lsst.afw.image.ImageI, lsst.afw.image.ImageL)
        bitpixList = (16, 32)
        algorithmList = (ImageScalingOptions.STDEV_POSITIVE, ImageScalingOptions.STDEV_NEGATIVE,
                         ImageScalingOptions.STDEV_BOTH)
        for cls, bitpix, algorithm in itertools.product(classList, bitpixList, algorithmList):
            with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
                self.checkStdev(cls, bitpix, algorithm, 10.0, 10.0)

    def checkNone(self, ImageClass, bitpix):
        """Check that the NONE scaling algorithm works

        Parameters
        ----------
        ImageClass : `type`, an `lsst.afw.image.Image` class
            Class of image to create.
        bitpix : `int`
            Bits per pixel for FITS image.
        """
        scaling = ImageScalingOptions(ImageScalingOptions.NONE, bitpix, [u"BAD"], fuzz=False)
        original, unpersisted, bscale, bzero, minValue, maxValue = self.makeImage(ImageClass, scaling)
        self.assertFloatsAlmostEqual(bscale, 1.0, atol=0.0)
        self.assertFloatsAlmostEqual(bzero, 0.0, atol=0.0)
        self.assertImagesAlmostEqual(original, unpersisted, atol=0.0)

    def testNone(self):
        """Test that the NONE scaling works on floating-point inputs"""
        classList = (lsst.afw.image.ImageF, lsst.afw.image.ImageD)
        bitpixList = (8, 16, 32)
        for cls, bitpix in itertools.product(classList, bitpixList):
            self.checkNone(cls, bitpix)

    def checkManual(self, ImageClass, bitpix):
        """Check that the MANUAL scaling algorithm works

        Parameters
        ----------
        ImageClass : `type`, an `lsst.afw.image.Image` class
            Class of image to create.
        bitpix : `int`
            Bits per pixel for FITS image.
        """
        bscaleSet = 1.2345
        bzeroSet = self.base
        scaling = ImageScalingOptions(ImageScalingOptions.MANUAL, bitpix, [u"BAD"], bscale=bscaleSet,
                                      bzero=bzeroSet, fuzz=False)
        original, unpersisted, bscale, bzero, minValue, maxValue = self.makeImage(ImageClass, scaling)
        self.assertFloatsAlmostEqual(bscale, bscaleSet, atol=0.0)
        self.assertFloatsAlmostEqual(bzero, bzeroSet, atol=0.0)
        self.assertImagesAlmostEqual(original, unpersisted, atol=bscale)

    def testManual(self):
        """Test that the MANUAL scaling works on floating-point inputs"""
        classList = (lsst.afw.image.ImageF, lsst.afw.image.ImageD)
        bitpixList = (16, 32)
        for cls, bitpix in itertools.product(classList, bitpixList):
            self.checkNone(cls, bitpix)


class ImageCompressionTestCase(lsst.utils.tests.TestCase):
    """Tests of image compression

    We test compression both with and without loss (quantisation/scaling).

    The pattern here is to create an image, write it out with a
    specific compression algorithm, read it back in and test that everything
    is as we expect. We do this for each compression algorithm in its own
    test, and within that test iterate over various parameters (input
    image type, BITPIX, etc.).

    We print the (inverse) compression ratio for interest. Note that
    these should not be considered to be representative of the
    compression that will be achieved on scientific data, since the
    images created here have different qualities than scientific data
    that will affect the compression ratio (e.g., size, noise properties).
    """
    def setUp(self):
        self.bbox = lsst.geom.Box2I(lsst.geom.Point2I(123, 456), lsst.geom.Extent2I(7, 8))
        self.background = 12345.6789  # Background value
        self.noise = 67.89  # Noise (stdev)
        self.maskPlanes = ["FOO", "BAR"]  # Mask planes to add
        self.extension = "." + self.__class__.__name__ + ".fits"  # extension name for temp files

    def readWriteImage(self, ImageClass, image, filename, options, *args):
        """Read the image after it has been written

        This implementation does the persistence using methods on the
        ImageClass.

        Parameters
        ----------
        ImageClass : `type`, an `lsst.afw.image.Image` class
            Class of image to create.
        image : `lsst.afw.image.Image`
            Image to compress.
        filename : `str`
            Filename to which to write.
        options : `lsst.afw.fits.ImageWriteOptions`
            Options for writing.
        """
        image.writeFits(filename, options, *args)
        return ImageClass(filename)

    def makeImage(self, ImageClass):
        """Create an image

        Parameters
        ----------
        ImageClass : `type`, an `lsst.afw.image.Image` class
            Class of image to create.

        Returns
        -------
        image : `ImageClass`
            The created image.
        """
        image = ImageClass(self.bbox)
        rng = np.random.RandomState(12345)
        dtype = image.getArray().dtype
        noise = rng.normal(0.0, self.noise, image.getArray().shape).astype(dtype)
        image.getArray()[:] = np.array(self.background, dtype=dtype) + noise
        return image

    def makeMask(self):
        """Create a mask

        Note that we generate a random distribution of mask pixel values,
        which is very different from the usual distribution in science images.

        Returns
        -------
        mask : `lsst.afw.image.Mask`
            The created mask.
        """
        mask = lsst.afw.image.Mask(self.bbox)
        rng = np.random.RandomState(12345)
        dtype = mask.getArray().dtype
        mask.getArray()[:] = rng.randint(0, 2**(dtype.itemsize*8 - 1), mask.getArray().shape, dtype=dtype)
        for plane in self.maskPlanes:
            mask.addMaskPlane(plane)
        return mask

    def checkCompressedImage(self, ImageClass, image, compression, scaling=None, atol=0.0):
        """Check that compression works on an image

        Parameters
        ----------
        ImageClass : `type`, an `lsst.afw.image.Image` class
            Class of image.
        image : `lsst.afw.image.Image`
            Image to compress.
        compression : `lsst.afw.fits.ImageCompressionOptions`
            Compression parameters.
        scaling : `lsst.afw.fits.ImageScalingOptions` or `None`
            Scaling parameters for lossy compression (optional).
        atol : `float`
            Absolute tolerance for comparing unpersisted image.

        Returns
        -------
        unpersisted : `ImageClass`
            The unpersisted image.
        """
        with lsst.utils.tests.getTempFilePath(self.extension) as filename:
            if scaling:
                options = lsst.afw.fits.ImageWriteOptions(compression, scaling)
            else:
                options = lsst.afw.fits.ImageWriteOptions(compression)
            unpersisted = self.readWriteImage(ImageClass, image, filename, options)

            fileSize = os.stat(filename).st_size
            fitsBlockSize = 2880  # All sizes in FITS are a multiple of this
            numBlocks = 1 + np.ceil(self.bbox.getArea()*image.getArray().dtype.itemsize/fitsBlockSize)
            uncompressedSize = fitsBlockSize*numBlocks
            print(ImageClass, compression.algorithm, fileSize, uncompressedSize, fileSize/uncompressedSize)

            self.assertEqual(image.getBBox(), unpersisted.getBBox())
            self.assertImagesAlmostEqual(unpersisted, image, atol=atol)

            checkAstropy(unpersisted, filename, 1)

            return unpersisted

    def testLosslessFloat(self):
        """Test lossless compression of floating-point image"""
        classList = (lsst.afw.image.ImageF, lsst.afw.image.ImageD)
        algorithmList = ("GZIP", "GZIP_SHUFFLE")  # Lossless float compression requires GZIP
        for cls, algorithm in itertools.product(classList, algorithmList):
            image = self.makeImage(cls)
            compression = ImageCompressionOptions(lsst.afw.fits.compressionAlgorithmFromString(algorithm))
            self.checkCompressedImage(cls, image, compression, atol=0.0)

    def testLosslessInt(self):
        """Test lossless compression of integer image

        We deliberately don't test `lsst.afw.image.ImageL` because
        compression of LONGLONG images is unsupported by cfitsio.
        """
        classList = (lsst.afw.image.ImageU, lsst.afw.image.ImageI)
        algorithmList = ("GZIP", "GZIP_SHUFFLE", "RICE")
        for cls, algorithm in itertools.product(classList, algorithmList):
            compression = ImageCompressionOptions(lsst.afw.fits.compressionAlgorithmFromString(algorithm))
            image = self.makeImage(cls)
            self.checkCompressedImage(cls, image, compression, atol=0.0)

    def testLongLong(self):
        """Test graceful failure when compressing ImageL

        We deliberately don't test `lsst.afw.image.ImageL` because
        compression of LONGLONG images is unsupported by cfitsio.
        """
        algorithmList = ("GZIP", "GZIP_SHUFFLE", "RICE")
        for algorithm in algorithmList:
            compression = ImageCompressionOptions(lsst.afw.fits.compressionAlgorithmFromString(algorithm))
            cls = lsst.afw.image.ImageL
            image = self.makeImage(cls)
            with self.assertRaises(lsst.afw.fits.FitsError):
                self.checkCompressedImage(cls, image, compression)

    def testMask(self):
        """Test compression of mask

        We deliberately don't test PLIO compression (which is designed for
        masks) because our default mask type (32) has too much dynamic range
        for PLIO (limit of 24 bits).
        """
        for algorithm in ("GZIP", "GZIP_SHUFFLE", "RICE"):
            compression = ImageCompressionOptions(lsst.afw.fits.compressionAlgorithmFromString(algorithm))
            mask = self.makeMask()
            unpersisted = self.checkCompressedImage(lsst.afw.image.Mask, mask, compression, atol=0.0)
            for mp in mask.getMaskPlaneDict():
                self.assertIn(mp, unpersisted.getMaskPlaneDict())
                unpersisted.getPlaneBitMask(mp)

    def testLossyFloatCfitsio(self):
        """Test lossy compresion of floating-point images with cfitsio

        cfitsio does the compression, controlled through the 'quantizeLevel'
        parameter. Note that cfitsio doesn't have access to our masks when
        it does its statistics.
        """
        classList = (lsst.afw.image.ImageF, lsst.afw.image.ImageD)
        algorithmList = ("GZIP", "GZIP_SHUFFLE", "RICE")
        quantizeList = (4.0, 10.0)
        for cls, algorithm, quantizeLevel in itertools.product(classList, algorithmList, quantizeList):
            compression = ImageCompressionOptions(lsst.afw.fits.compressionAlgorithmFromString(algorithm),
                                                  quantizeLevel=quantizeLevel)
            image = self.makeImage(cls)
            self.checkCompressedImage(cls, image, compression, atol=self.noise/quantizeLevel)

    def testLossyFloatOurs(self):
        """Test lossy compression of floating-point images ourselves

        We do lossy compression by scaling first. We have full control over
        the scaling (multiple scaling algorithms), and we have access to our
        own masks when we do statistics.
        """
        classList = (lsst.afw.image.ImageF, lsst.afw.image.ImageD)
        algorithmList = ("GZIP", "GZIP_SHUFFLE", "RICE")
        bitpixList = (16, 32)
        quantizeList = (4.0, 10.0)
        for cls, algorithm, bitpix, quantize in itertools.product(classList, algorithmList, bitpixList,
                                                                  quantizeList):
            compression = ImageCompressionOptions(lsst.afw.fits.compressionAlgorithmFromString(algorithm),
                                                  quantizeLevel=0.0)
            scaling = ImageScalingOptions(ImageScalingOptions.STDEV_BOTH, bitpix, quantizeLevel=quantize,
                                          fuzz=True)
            image = self.makeImage(cls)
            self.checkCompressedImage(cls, image, compression, scaling, atol=self.noise/quantize)

    def readWriteMaskedImage(self, image, filename, imageOptions, maskOptions, varianceOptions):
        """Read the MaskedImage after it has been written

        This implementation does the persistence using methods on the
        MaskedImage class.

        Parameters
        ----------
        image : `lsst.afw.image.Image`
            Image to compress.
        filename : `str`
            Filename to which to write.
        imageOptions, maskOptions, varianceOptions : `lsst.afw.fits.ImageWriteOptions`
            Options for writing the image, mask and variance planes.
        """
        image.writeFits(filename, imageOptions, maskOptions, varianceOptions)
        if hasattr(image, "getMaskedImage"):
            image = image.getMaskedImage()
        return lsst.afw.image.MaskedImageF(filename)

    def checkCompressedMaskedImage(self, image, imageOptions, maskOptions, varianceOptions, atol=0.0):
        """Check that compression works on a MaskedImage

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage` or `lsst.afw.image.Exposure`
            MaskedImage or exposure to compress.
        imageOptions, maskOptions, varianceOptions : `lsst.afw.fits.ImageWriteOptions`
            Parameters for writing (compression and scaling) the image, mask
            and variance planes.
        atol : `float`
            Absolute tolerance for comparing unpersisted image.
        """
        with lsst.utils.tests.getTempFilePath(self.extension) as filename:
            self.readWriteMaskedImage(image, filename, imageOptions, maskOptions, varianceOptions)
            unpersisted = type(image)(filename)
            if hasattr(image, "getMaskedImage"):
                image = image.getMaskedImage()
                unpersisted = unpersisted.getMaskedImage()
            self.assertEqual(image.getBBox(), unpersisted.getBBox())
            self.assertImagesAlmostEqual(unpersisted.getImage(), image.getImage(), atol=atol)
            self.assertImagesAlmostEqual(unpersisted.getMask(), image.getMask(), atol=atol)
            self.assertImagesAlmostEqual(unpersisted.getVariance(), image.getVariance(), atol=atol)

            for mp in image.getMask().getMaskPlaneDict():
                self.assertIn(mp, unpersisted.getMask().getMaskPlaneDict())
                unpersisted.getMask().getPlaneBitMask(mp)

    def checkMaskedImage(self, imageOptions, maskOptions, varianceOptions, atol=0.0):
        """Check that we can compress a MaskedImage and Exposure

        Parameters
        ----------
        imageOptions, maskOptions, varianceOptions : `lsst.afw.fits.ImageWriteOptions`
            Parameters for writing (compression and scaling) the image, mask
            and variance planes.
        atol : `float`
            Absolute tolerance for comparing unpersisted image.
        """
        image = lsst.afw.image.makeMaskedImage(self.makeImage(lsst.afw.image.ImageF),
                                               self.makeMask(), self.makeImage(lsst.afw.image.ImageF))
        self.checkCompressedMaskedImage(image, imageOptions, maskOptions, varianceOptions, atol=atol)
        exp = lsst.afw.image.makeExposure(image)
        self.checkCompressedMaskedImage(exp, imageOptions, maskOptions, varianceOptions, atol=atol)

    def testMaskedImage(self):
        """Test compression of MaskedImage

        We test lossless, lossy cfitsio and lossy LSST compression.
        """
        # Lossless
        lossless = lsst.afw.fits.ImageCompressionOptions(ImageCompressionOptions.GZIP_SHUFFLE)
        options = lsst.afw.fits.ImageWriteOptions(lossless)
        self.checkMaskedImage(options, options, options, atol=0.0)

        # Lossy cfitsio compression
        quantize = 4.0
        cfitsio = lsst.afw.fits.ImageCompressionOptions(ImageCompressionOptions.GZIP_SHUFFLE, True, quantize)
        imageOptions = lsst.afw.fits.ImageWriteOptions(cfitsio)
        maskOptions = lsst.afw.fits.ImageWriteOptions(lossless)
        self.checkMaskedImage(imageOptions, maskOptions, imageOptions, atol=self.noise/quantize)

        # Lossy our compression
        quantize = 10.0
        compression = lsst.afw.fits.ImageCompressionOptions(ImageCompressionOptions.RICE, True, 0.0)
        scaling = lsst.afw.fits.ImageScalingOptions(ImageScalingOptions.STDEV_BOTH, 32,
                                                    quantizeLevel=quantize)
        imageOptions = lsst.afw.fits.ImageWriteOptions(compression, scaling)
        maskOptions = lsst.afw.fits.ImageWriteOptions(compression)
        self.checkMaskedImage(imageOptions, maskOptions, imageOptions, atol=self.noise/quantize)

    def testQuantization(self):
        """Test that our quantization produces the same values as cfitsio

        Our quantization is more configurable (e.g., choice of scaling algorithm,
        specifying mask planes) and extensible (logarithmic, asinh scalings)
        than cfitsio's. However, cfitsio uses its own fuzz ("subtractive dithering")
        when reading the data, so if we don't want to add random values twice,
        we need to be sure that we're using the same random values. To check that,
        we write one image with our scaling+compression, and one with cfitsio's
        compression using exactly the BSCALE and dither seed we used for our own.
        That way, the two codes will quantize independently, and we can compare
        the results.
        """
        bscaleSet = 1.0
        bzeroSet = self.background - 10*self.noise
        algorithm = ImageCompressionOptions.GZIP
        classList = (lsst.afw.image.ImageF, lsst.afw.image.ImageD)
        tilesList = ((4, 5), (0, 0), (0, 5), (4, 0), (0, 1))
        for cls, tiles in itertools.product(classList, tilesList):
            tiles = np.array(tiles, dtype=np.int64)
            compression = ImageCompressionOptions(algorithm, tiles, -bscaleSet)
            original = self.makeImage(cls)
            with lsst.utils.tests.getTempFilePath(self.extension) as filename:
                with lsst.afw.fits.Fits(filename, "w") as fits:
                    options = lsst.afw.fits.ImageWriteOptions(compression)
                    original.writeFits(fits, options)
                cfitsio = cls(filename)
                header = lsst.afw.fits.readMetadata(filename, 1)
                seed = header.getScalar("ZDITHER0")
                self.assertEqual(header.getScalar("BSCALE"), bscaleSet)

            compression = ImageCompressionOptions(algorithm, tiles, 0.0)
            scaling = ImageScalingOptions(ImageScalingOptions.MANUAL, 32, [u"BAD"], bscale=bscaleSet,
                                          bzero=bzeroSet, fuzz=True, seed=seed)
            unpersisted = self.checkCompressedImage(cls, original, compression, scaling, atol=bscaleSet)
            oursDiff = unpersisted.getArray() - original.getArray()
            cfitsioDiff = cfitsio.getArray() - original.getArray()
            self.assertImagesAlmostEqual(oursDiff, cfitsioDiff, atol=0.0)


def optionsToPropertySet(options):
    """Convert the ImageWriteOptions to a PropertySet

    This allows us to pass the options into the persistence framework
    as the "additionalData".
    """
    ps = lsst.daf.base.PropertySet()
    ps.set("compression.algorithm", lsst.afw.fits.compressionAlgorithmToString(options.compression.algorithm))
    ps.set("compression.columns", options.compression.tiles[0])
    ps.set("compression.rows", options.compression.tiles[1])
    ps.set("compression.quantizeLevel", options.compression.quantizeLevel)

    ps.set("scaling.algorithm", lsst.afw.fits.scalingAlgorithmToString(options.scaling.algorithm))
    ps.set("scaling.bitpix", options.scaling.bitpix)
    ps.setString("scaling.maskPlanes", options.scaling.maskPlanes)
    ps.set("scaling.fuzz", options.scaling.fuzz)
    ps.set("scaling.seed", options.scaling.seed)
    ps.set("scaling.quantizeLevel", options.scaling.quantizeLevel)
    ps.set("scaling.quantizePad", options.scaling.quantizePad)
    ps.set("scaling.bscale", options.scaling.bscale)
    ps.set("scaling.bzero", options.scaling.bzero)
    return ps


def persistUnpersist(ImageClass, image, filename, additionalData):
    """Use read/writeFitsWithOptions to persist and unpersist an image

    Parameters
    ----------
    ImageClass : `type`, an `lsst.afw.image.Image` class
        Class of image.
    image : `lsst.afw.image.Image` or `lsst.afw.image.MaskedImage`
        Image to compress.
    filename : `str`
        Filename to write.
    additionalData : `lsst.daf.base.PropertySet`
        Additional data for persistence framework.

    Returns
    -------
    unpersisted : `ImageClass`
        The unpersisted image.
    """
    additionalData.set("visit", 12345)
    additionalData.set("ccd", 67)

    image.writeFitsWithOptions(filename, additionalData)
    return ImageClass.readFits(filename)


class PersistenceTestCase(ImageCompressionTestCase):
    """Test compression using the persistence framework

    We override the I/O methods to use the persistence framework.
    """
    def testQuantization(self):
        """Not appropriate --- disable"""
        pass

    def readWriteImage(self, ImageClass, image, filename, options):
        """Read the image after it has been written

        This implementation uses the persistence framework.

        Parameters
        ----------
        ImageClass : `type`, an `lsst.afw.image.Image` class
            Class of image to create.
        image : `lsst.afw.image.Image`
            Image to compress.
        filename : `str`
            Filename to which to write.
        options : `lsst.afw.fits.ImageWriteOptions`
            Options for writing.
        """
        additionalData = lsst.daf.base.PropertySet()
        additionalData.set("image", optionsToPropertySet(options))
        return persistUnpersist(ImageClass, image, filename, additionalData)

    def readWriteMaskedImage(self, image, filename, imageOptions, maskOptions, varianceOptions):
        """Read the MaskedImage after it has been written

        This implementation uses the persistence framework.

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Image to compress.
        filename : `str`
            Filename to which to write.
        imageOptions, maskOptions, varianceOptions : `lsst.afw.fits.ImageWriteOptions`
            Options for writing the image, mask and variance planes.
        """
        additionalData = lsst.daf.base.PropertySet()
        additionalData.set("image", optionsToPropertySet(imageOptions))
        additionalData.set("mask", optionsToPropertySet(maskOptions))
        additionalData.set("variance", optionsToPropertySet(varianceOptions))
        return persistUnpersist(lsst.afw.image.MaskedImageF, image, filename, additionalData)


class EmptyExposureTestCase(lsst.utils.tests.TestCase):
    """Test that an empty image can be written

    We sometimes use an empty lsst.afw.image.Exposure as a vehicle for
    persisting other things, e.g., Wcs, Calib. cfitsio compression will
    choke on an empty image, so make sure we're dealing with that.
    """
    def checkEmptyExposure(self, algorithm):
        """Check that we can persist an empty Exposure

        Parameters
        ----------
        algorithm : `lsst.afw.fits.ImageCompressionOptions.CompressionAlgorithm`
            Compression algorithm to try.
        """
        exp = lsst.afw.image.ExposureF(0, 0)
        degrees = lsst.geom.degrees
        cdMatrix = np.array([[1.0e-4, 0.0], [0.0, 1.0e-4]], dtype=float)
        exp.setWcs(lsst.afw.geom.makeSkyWcs(crval=lsst.geom.SpherePoint(0*degrees, 0*degrees),
                                            crpix=lsst.geom.Point2D(0.0, 0.0),
                                            cdMatrix=cdMatrix))
        imageOptions = lsst.afw.fits.ImageWriteOptions(ImageCompressionOptions(algorithm))
        maskOptions = lsst.afw.fits.ImageWriteOptions(exp.getMaskedImage().getMask())
        varianceOptions = lsst.afw.fits.ImageWriteOptions(ImageCompressionOptions(algorithm))
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            exp.writeFits(filename, imageOptions, maskOptions, varianceOptions)
            unpersisted = type(exp)(filename)
        self.assertEqual(unpersisted.getMaskedImage().getDimensions(), lsst.geom.Extent2I(0, 0))
        self.assertEqual(unpersisted.getWcs(), exp.getWcs())

    def testEmptyExposure(self):
        """Persist an empty Exposure with compression"""
        algorithmList = ("GZIP", "GZIP_SHUFFLE", "RICE")
        for algorithm in algorithmList:
            self.checkEmptyExposure(lsst.afw.fits.compressionAlgorithmFromString(algorithm))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    import sys
    setup_module(sys.modules[__name__])
    unittest.main()

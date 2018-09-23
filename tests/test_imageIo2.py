#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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

import unittest

import numpy as np
import astropy.io.fits
import lsst.utils.tests
from lsst.afw.geom import Box2I, Point2I
from lsst.afw.fits import ImageCompressionOptions, ImageScalingOptions, ImageWriteOptions
import lsst.afw.image as afwImage


class ImageIoTestCase(lsst.utils.tests.TestCase):
    """A test case for Image FITS I/O"""

    def assertImagesEqual(self, image, original):
        self.assertEqual(image.getBBox(), original.getBBox())
        super().assertImagesEqual(image, original)

    def setUp(self):
        self.IntegerImages = (afwImage.ImageU, afwImage.ImageI, afwImage.ImageL)
        self.FloatImages = (afwImage.ImageF, afwImage.ImageD)
        self.bbox = Box2I(minimum=Point2I(3, 4), maximum=Point2I(9, 7))

    def doRoundTrip(self, image, compression=None, scaling=None):
        if compression is None:
            compression = dict(algorithm=ImageCompressionOptions.NONE)
        if scaling is None:
            scaling = dict(algorithm=ImageScalingOptions.NONE, bitpix=0)
        options = ImageWriteOptions(compression=ImageCompressionOptions(**compression),
                                    scaling=ImageScalingOptions(**scaling))
        isCompressed = (compression.get("algorithm", ImageCompressionOptions.NONE) !=
                        ImageCompressionOptions.NONE)
        with lsst.utils.tests.getTempFilePath("_%s.fits" % (type(image).__name__,)) as filename:
            image.writeFits(filename, options=options)
            readImage = type(image)(filename)
            with astropy.io.fits.open(filename) as hduList:
                hdu = hduList[1 if isCompressed else 0]
                if hdu.data.dtype.byteorder != '=':
                    hdu.data = hdu.data.byteswap().newbyteorder()
        return readImage, hdu

    def runRoundTripTest(self, cls, compression=None, scaling=None, addNaN=False, checkAstropy=True, rtol=0):
        original = cls(self.bbox)
        original.array[:, :] = np.random.randint(size=original.array.shape, low=1, high=255, dtype=np.uint8)
        if addNaN:
            original[5, 6] = np.nan

        readImage, hdu = self.doRoundTrip(original, compression=compression, scaling=scaling)
        self.assertImagesAlmostEqual(original, readImage, rtol=rtol, atol=0)

        # Integer LSST images never have missing pixels; FITS floating-point images always use NaN
        self.assertNotIn("BLANK", hdu.header.keys())

        if checkAstropy:
            # Compare to what astropy reads, to more-or-less check that we're not abusing FITS
            hduImage = cls(hdu.data, deep=False, xy0=self.bbox.getMin())
            self.assertImagesAlmostEqual(original, hduImage, rtol=rtol, atol=0)

    def testIntegerUncompression(self):
        """Test round-tripping integer images with no compression or scaling.
        """
        for cls in self.IntegerImages:
            with self.subTest(cls=cls.__name__):
                self.runRoundTripTest(cls)

    def testIntegerCompression(self):
        """Test round-tripping integer images with compression (and no scaling).
        """
        for cls in self.IntegerImages:
            with self.subTest(cls=cls.__name__):
                self.runRoundTripTest(cls, compression=dict(algorithm=ImageCompressionOptions.RICE))

    def testFloatUncompressed(self):
        """Test round-tripping floating-point images with no compression."""
        for cls in self.FloatImages:
            with self.subTest(cls=cls.__name__):
                self.runRoundTripTest(cls, addNaN=True)

    def testFloatCompressedLossless(self):
        """Test round-tripping floating-point images with lossless compression."""
        for cls in self.FloatImages:
            with self.subTest(cls=cls.__name__):
                self.runRoundTripTest(
                    cls,
                    compression=dict(algorithm=ImageCompressionOptions.GZIP, quantizeLevel=0),
                    addNaN=True
                )

    @unittest.skip("Fix deferred to DM-15644")
    def testFloatCompressedRange(self):
        """Test round-tripping floating-point images with lossy compression
        and RANGE scaling."""
        for cls in self.FloatImages:
            with self.subTest(cls=cls.__name__):
                self.runRoundTripTest(
                    cls,
                    compression=dict(algorithm=ImageCompressionOptions.GZIP, quantizeLevel=1),
                    scaling=dict(algorithm=ImageScalingOptions.RANGE, bitpix=32, fuzz=False),
                    addNaN=True,
                    checkAstropy=True
                )

    @unittest.skip("Fix deferred to DM-15644")
    def testFloatCompressedManual(self):
        """Test round-tripping floating-point images with lossy compression
        and MANUAL scaling."""
        for cls in self.FloatImages:
            with self.subTest(cls=cls.__name__):
                self.runRoundTripTest(
                    cls,
                    compression=dict(algorithm=ImageCompressionOptions.GZIP, quantizeLevel=1),
                    scaling=dict(algorithm=ImageScalingOptions.MANUAL, bitpix=32, fuzz=False,
                                 bzero=3, bscale=2),
                    addNaN=True,
                    checkAstropy=True
                )


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

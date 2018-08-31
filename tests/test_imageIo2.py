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
import lsst.afw.image as afwImage


class ImageIoTestCase(lsst.utils.tests.TestCase):
    """A test case for Image Persistence"""

    def checkImages(self, image, original):
        # Check that two images are identical
        self.assertEqual(image.getHeight(), original.getHeight())
        self.assertEqual(image.getWidth(), original.getWidth())
        self.assertEqual(image.getY0(), original.getY0())
        self.assertEqual(image.getX0(), original.getX0())
        for x in range(0, original.getWidth()):
            for y in range(0, image.getHeight()):
                self.assertEqual(image[x, y, afwImage.LOCAL], original[x, y, afwImage.LOCAL])

    def setUp(self):
        # Create the additionalData PropertySet
        self.cols = 4
        self.rows = 4

    def testIo(self):
        for Image in (afwImage.ImageU,
                      afwImage.ImageL,
                      afwImage.ImageI,
                      afwImage.ImageF,
                      afwImage.ImageD,
                      ):
            image = Image(self.cols, self.rows)
            for x in range(0, self.cols):
                for y in range(0, self.rows):
                    image[x, y] = x + y

            with lsst.utils.tests.getTempFilePath("_%s.fits" % (Image.__name__,)) as filename:
                image.writeFits(filename)
                readImage = Image(filename)
                with astropy.io.fits.open(filename) as astropyFits:
                    astropyData = astropyFits[0].data

            self.checkImages(readImage, image)

            dt1 = astropyData.dtype
            dt2 = image.array.dtype
            # dtypes won't be equal, because astropy doesn't byteswap,
            # so we just compare kind (uint vs. int vs. float) and
            # size.
            if Image is not afwImage.ImageL:
                # reading int64 into uint64 via nonzero BZERO is a
                # CFITSIO special that astropy doesn't support.
                self.assertEqual(dt1.kind, dt2.kind)
            self.assertEqual(dt1.itemsize, dt2.itemsize)
            self.assertTrue(np.all(astropyData == image.array))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

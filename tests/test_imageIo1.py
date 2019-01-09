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
Test cases to test image I/O
"""
import itertools
import os.path
import unittest

import lsst.utils
import lsst.daf.base as dafBase
import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.fits as afwFits
import lsst.utils.tests
import lsst.afw.display.ds9 as ds9
import lsst.pex.exceptions as pexExcept

try:
    type(verbose)
except NameError:
    verbose = 0

try:
    dataDir = os.path.join(lsst.utils.getPackageDir("afwdata"), "data")
except pexExcept.NotFoundError:
    dataDir = None


class ReadFitsTestCase(lsst.utils.tests.TestCase):
    """A test case for reading FITS images"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testU16(self):
        """Test reading U16 image"""

        im = afwImage.ImageD(os.path.join(dataDir, "small_img.fits"))

        col, row, val = 0, 0, 1154
        self.assertEqual(im[col, row, afwImage.LOCAL], val)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testS16(self):
        """Test reading S16 image"""
        im = afwImage.ImageD(os.path.join(dataDir, "871034p_1_img.fits"))

        if False:
            ds9.mtv(im)

        col, row, val = 32, 1, 62
        self.assertEqual(im[col, row, afwImage.LOCAL], val)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testF32(self):
        """Test reading F32 image"""
        im = afwImage.ImageD(os.path.join(dataDir, "871034p_1_MI.fits"), 3)

        col, row, val = 32, 1, 39.11672
        self.assertAlmostEqual(im[col, row, afwImage.LOCAL], val, 4)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testF64(self):
        """Test reading a U16 file into a F64 image"""
        im = afwImage.ImageD(os.path.join(dataDir, "small_img.fits"))
        col, row, val = 0, 0, 1154
        self.assertEqual(im[col, row, afwImage.LOCAL], val)

        # print "IM = ", im
    def testWriteReadF64(self):
        """Test writing then reading an F64 image"""
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            im = afwImage.ImageD(lsst.geom.Extent2I(100, 100))
            im.set(666)
            im.writeFits(tmpFile)
            afwImage.ImageD(tmpFile)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testSubimage(self):
        """Test reading a subimage image"""
        fileName, hdu = os.path.join(dataDir, "871034p_1_MI.fits"), 3
        im = afwImage.ImageF(fileName, hdu)

        bbox = lsst.geom.Box2I(lsst.geom.Point2I(110, 120),
                               lsst.geom.Extent2I(20, 15))
        sim = im.Factory(im, bbox, afwImage.LOCAL)

        im2 = afwImage.ImageF(fileName, hdu, None, bbox, afwImage.LOCAL)

        self.assertEqual(im2.getDimensions(), sim.getDimensions())
        self.assertEqual(im2[1, 1, afwImage.LOCAL], sim[1, 1, afwImage.LOCAL])

        self.assertEqual(im2.getX0(), sim.getX0())
        self.assertEqual(im2.getY0(), sim.getY0())

    def testMEF(self):
        """Test writing a set of images to an MEF fits file, and then reading them back

        We disable compression to avoid the empty PHU that comes when writing FITS
        compressed images.
        """
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile, afwFits.imageCompressionDisabled():
            im = afwImage.ImageF(lsst.geom.Extent2I(20, 20))

            for hdu in range(4):
                im.set(100*hdu)
                if hdu == 0:
                    mode = "w"
                else:
                    mode = "a"
                im.writeFits(tmpFile, None, mode)

            for hdu in range(4):
                im = afwImage.ImageF(tmpFile, hdu)
                self.assertEqual(im[0, 0, afwImage.LOCAL], 100*hdu)

    def testWriteBool(self):
        """Test that we can read and write bools"""
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            im = afwImage.ImageF(lsst.geom.ExtentI(10, 20))
            md = dafBase.PropertySet()
            keys = {"BAD": False,
                    "GOOD": True,
                    }
            for k, v in keys.items():
                md.add(k, v)

            im.writeFits(tmpFile, md)

            jim = afwImage.DecoratedImageF(tmpFile)

            for k, v in keys.items():
                self.assertEqual(jim.getMetadata().getScalar(k), v)

    def testLongStrings(self):
        keyWord = 'ZZZ'
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            longString = ' '.join(['This is a long string.'] * 8)

            expOrig = afwImage.ExposureF(100, 100)
            mdOrig = expOrig.getMetadata()
            mdOrig.set(keyWord, longString)
            expOrig.writeFits(tmpFile)

            expNew = afwImage.ExposureF(tmpFile)
            self.assertEqual(expNew.getMetadata().getScalar(keyWord), longString)

    def checkBBoxFromMetadata(self, filename, expected, hdu=0):
        metadata = afwFits.readMetadata(filename, hdu)
        bbox = afwImage.bboxFromMetadata(metadata)
        self.assertEqual(bbox, expected)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testBBoxFromMetadata(self):
        self.checkBBoxFromMetadata(os.path.join(dataDir, "871034p_1_img.fits"),
                                   lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                                                   lsst.geom.Extent2I(2112, 4644)))
        for hdu in range(1, 4):
            self.checkBBoxFromMetadata(os.path.join(dataDir, "871034p_1_MI.fits"),
                                       lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                                                       lsst.geom.Extent2I(2112, 4644)),
                                       hdu=hdu)
            self.checkBBoxFromMetadata(os.path.join(dataDir, "medsub.fits"),
                                       lsst.geom.Box2I(lsst.geom.Point2I(40, 150),
                                                       lsst.geom.Extent2I(145, 200)),
                                       hdu=hdu)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testReadFitsWithOptions(self):
        xy0Offset = lsst.geom.Extent2I(7, 5)
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(10, 11), lsst.geom.Extent2I(31, 22))

        with lsst.utils.tests.getTempFilePath(".fits") as filepath:
            # write a temporary version of the image with non-zero XY0
            imagePath = os.path.join(dataDir, "med.fits")
            maskedImage = afwImage.MaskedImageD(imagePath)
            maskedImage.setXY0(lsst.geom.Point2I(xy0Offset))
            maskedImage.writeFits(filepath)

            for ImageClass, imageOrigin in itertools.product(
                (afwImage.ImageF, afwImage.ImageD),
                (None, "LOCAL", "PARENT"),
            ):
                with self.subTest(ImageClass=ImageClass, imageOrigin=imageOrigin):
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
                    self.assertImagesEqual(image1, ImageClass(fullImage, readBBoxParent))

                    for name in ("llcY", "width", "height"):
                        badOptions = options.deepCopy()
                        badOptions.remove(name)
                        with self.assertRaises(pexExcept.NotFoundError):
                            ImageClass.readFitsWithOptions(filepath, badOptions)

                        badOptions = options.deepCopy()
                        badOptions.set("imageOrigin", "INVALID")
                        with self.assertRaises(RuntimeError):
                            ImageClass.readFitsWithOptions(filepath, badOptions)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

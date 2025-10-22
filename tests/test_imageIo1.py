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

import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
import lsst.utils.tests

try:
    dataDir = os.path.join(lsst.utils.getPackageDir("afwdata"), "data")
except LookupError:
    dataDir = None


class ReadFitsTestCase(lsst.utils.tests.TestCase):
    """A test case for reading FITS images"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

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
                with self.subTest(ImageClass=str(ImageClass), imageOrigin=imageOrigin):
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
                        with self.assertRaises(LookupError):
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

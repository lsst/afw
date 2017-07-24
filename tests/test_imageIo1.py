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

"""
Test cases to test image I/O
"""
from __future__ import absolute_import, division, print_function
import os.path
import unittest

from builtins import range

import lsst.utils
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
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
        self.assertEqual(im.get(col, row), val)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testS16(self):
        """Test reading S16 image"""
        im = afwImage.ImageD(os.path.join(dataDir, "871034p_1_img.fits"))

        if False:
            ds9.mtv(im)

        col, row, val = 32, 1, 62
        self.assertEqual(im.get(col, row), val)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testF32(self):
        """Test reading F32 image"""
        im = afwImage.ImageD(os.path.join(dataDir, "871034p_1_MI.fits"), 3)

        col, row, val = 32, 1, 39.11672
        self.assertAlmostEqual(im.get(col, row), val, 4)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testF64(self):
        """Test reading a U16 file into a F64 image"""
        im = afwImage.ImageD(os.path.join(dataDir, "small_img.fits"))
        col, row, val = 0, 0, 1154
        self.assertEqual(im.get(col, row), val)

        # print "IM = ", im
    def testWriteReadF64(self):
        """Test writing then reading an F64 image"""
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            im = afwImage.ImageD(afwGeom.Extent2I(100, 100))
            im.set(666)
            im.writeFits(tmpFile)
            afwImage.ImageD(tmpFile)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testSubimage(self):
        """Test reading a subimage image"""
        fileName, hdu = os.path.join(dataDir, "871034p_1_MI.fits"), 3
        im = afwImage.ImageF(fileName, hdu)

        bbox = afwGeom.Box2I(afwGeom.Point2I(110, 120),
                             afwGeom.Extent2I(20, 15))
        sim = im.Factory(im, bbox, afwImage.LOCAL)

        im2 = afwImage.ImageF(fileName, hdu, None, bbox, afwImage.LOCAL)

        self.assertEqual(im2.getDimensions(), sim.getDimensions())
        self.assertEqual(im2.get(1, 1), sim.get(1, 1))

        self.assertEqual(im2.getX0(), sim.getX0())
        self.assertEqual(im2.getY0(), sim.getY0())

    def testMEF(self):
        """Test writing a set of images to an MEF fits file, and then reading them back"""

        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            im = afwImage.ImageF(afwGeom.Extent2I(20, 20))

            for hdu in range(4):
                im.set(100*hdu)
                if hdu == 0:
                    mode = "w"
                else:
                    mode = "a"
                im.writeFits(tmpFile, None, mode)

            for hdu in range(4):
                im = afwImage.ImageF(tmpFile, hdu)
                self.assertEqual(im.get(0, 0), 100*hdu)

    def testWriteBool(self):
        """Test that we can read and write bools"""
        import lsst.afw.image as afwImage
        import lsst.daf.base as dafBase

        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            im = afwImage.ImageF(afwGeom.ExtentI(10, 20))
            md = dafBase.PropertySet()
            keys = {"BAD": False,
                    "GOOD": True,
                    }
            for k, v in keys.items():
                md.add(k, v)

            im.writeFits(tmpFile, md)

            jim = afwImage.DecoratedImageF(tmpFile)

            for k, v in keys.items():
                self.assertEqual(jim.getMetadata().get(k), v)

    def testLongStrings(self):
        keyWord = 'ZZZ'
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            longString = ' '.join(['This is a long string.'] * 8)

            expOrig = afwImage.ExposureF(100, 100)
            mdOrig = expOrig.getMetadata()
            mdOrig.set(keyWord, longString)
            expOrig.writeFits(tmpFile)

            expNew = afwImage.ExposureF(tmpFile)
            self.assertEqual(expNew.getMetadata().get(keyWord), longString)

    def checkBBoxFromMetadata(self, filename, expected, hdu=0):
        metadata = afwImage.readMetadata(filename, hdu)
        bbox = afwImage.bboxFromMetadata(metadata)
        self.assertEqual(bbox, expected)

    def testBBoxFromMetadata(self):
        self.checkBBoxFromMetadata(os.path.join(dataDir, "871034p_1_img.fits"),
                                   afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(2112, 4644)))
        for hdu in range(1, 4):
            self.checkBBoxFromMetadata(os.path.join(dataDir, "871034p_1_MI.fits"),
                                       afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(2112, 4644)),
                                       hdu=hdu)
            self.checkBBoxFromMetadata(os.path.join(dataDir, "medsub.fits"),
                                       afwGeom.Box2I(afwGeom.Point2I(40, 150), afwGeom.Extent2I(145, 200)),
                                       hdu=hdu)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

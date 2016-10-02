#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
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
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11#"""
#pybind11#Test cases to test image I/O
#pybind11#"""
#pybind11#import os.path
#pybind11#
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#
#pybind11#try:
#pybind11#    type(verbose)
#pybind11#except NameError:
#pybind11#    verbose = 0
#pybind11#
#pybind11#try:
#pybind11#    dataDir = os.path.join(lsst.utils.getPackageDir("afwdata"), "data")
#pybind11#except pexExcept.NotFoundError:
#pybind11#    dataDir = None
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class ReadFitsTestCase(lsst.utils.tests.TestCase):
#pybind11#    """A test case for reading FITS images"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        pass
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        pass
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testU16(self):
#pybind11#        """Test reading U16 image"""
#pybind11#
#pybind11#        im = afwImage.ImageD(os.path.join(dataDir, "small_img.fits"))
#pybind11#
#pybind11#        col, row, val = 0, 0, 1154
#pybind11#        self.assertEqual(im.get(col, row), val)
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testS16(self):
#pybind11#        """Test reading S16 image"""
#pybind11#        im = afwImage.ImageD(os.path.join(dataDir, "871034p_1_img.fits"))
#pybind11#
#pybind11#        if False:
#pybind11#            ds9.mtv(im)
#pybind11#
#pybind11#        col, row, val = 32, 1, 62
#pybind11#        self.assertEqual(im.get(col, row), val)
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testF32(self):
#pybind11#        """Test reading F32 image"""
#pybind11#        im = afwImage.ImageD(os.path.join(dataDir, "871034p_1_MI.fits"), 4)
#pybind11#
#pybind11#        col, row, val = 32, 1, 39.11672
#pybind11#        self.assertAlmostEqual(im.get(col, row), val, 5)
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testF64(self):
#pybind11#        """Test reading a U16 file into a F64 image"""
#pybind11#        im = afwImage.ImageD(os.path.join(dataDir, "small_img.fits"))
#pybind11#        col, row, val = 0, 0, 1154
#pybind11#        self.assertEqual(im.get(col, row), val)
#pybind11#
#pybind11#        # print "IM = ", im
#pybind11#    def testWriteReadF64(self):
#pybind11#        """Test writing then reading an F64 image"""
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            im = afwImage.ImageD(afwGeom.Extent2I(100, 100))
#pybind11#            im.set(666)
#pybind11#            im.writeFits(tmpFile)
#pybind11#            afwImage.ImageD(tmpFile)
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testSubimage(self):
#pybind11#        """Test reading a subimage image"""
#pybind11#        fileName, hdu = os.path.join(dataDir, "871034p_1_MI.fits"), 4
#pybind11#        im = afwImage.ImageF(fileName, hdu)
#pybind11#
#pybind11#        bbox = afwGeom.Box2I(afwGeom.Point2I(110, 120), afwGeom.Extent2I(20, 15))
#pybind11#        sim = im.Factory(im, bbox, afwImage.LOCAL)
#pybind11#
#pybind11#        im2 = afwImage.ImageF(fileName, hdu, None, bbox, afwImage.LOCAL)
#pybind11#
#pybind11#        self.assertEqual(im2.getDimensions(), sim.getDimensions())
#pybind11#        self.assertEqual(im2.get(1, 1), sim.get(1, 1))
#pybind11#
#pybind11#        self.assertEqual(im2.getX0(), sim.getX0())
#pybind11#        self.assertEqual(im2.getY0(), sim.getY0())
#pybind11#
#pybind11#    def testMEF(self):
#pybind11#        """Test writing a set of images to an MEF fits file, and then reading them back"""
#pybind11#
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            im = afwImage.ImageF(afwGeom.Extent2I(20, 20))
#pybind11#
#pybind11#            for hdu in range(1, 5):
#pybind11#                im.set(100*hdu)
#pybind11#                if hdu == 1:
#pybind11#                    mode = "w"
#pybind11#                else:
#pybind11#                    mode = "a"
#pybind11#                im.writeFits(tmpFile, None, mode)
#pybind11#
#pybind11#            for hdu in range(1, 5):
#pybind11#                im = afwImage.ImageF(tmpFile, hdu)
#pybind11#                self.assertEqual(im.get(0, 0), 100*hdu)
#pybind11#
#pybind11#    def testWriteBool(self):
#pybind11#        """Test that we can read and write bools"""
#pybind11#        import lsst.afw.image as afwImage
#pybind11#        import lsst.daf.base as dafBase
#pybind11#
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            im = afwImage.ImageF(afwGeom.ExtentI(10, 20))
#pybind11#            md = dafBase.PropertySet()
#pybind11#            keys = {"BAD": False,
#pybind11#                    "GOOD": True,
#pybind11#                    }
#pybind11#            for k, v in keys.items():
#pybind11#                md.add(k, v)
#pybind11#
#pybind11#            im.writeFits(tmpFile, md)
#pybind11#
#pybind11#            jim = afwImage.DecoratedImageF(tmpFile)
#pybind11#
#pybind11#            for k, v in keys.items():
#pybind11#                self.assertEqual(jim.getMetadata().get(k), v)
#pybind11#
#pybind11#    def testLongStrings(self):
#pybind11#        keyWord = 'ZZZ'
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            longString = ' '.join(['This is a long string.'] * 8)
#pybind11#
#pybind11#            expOrig = afwImage.ExposureF(100, 100)
#pybind11#            mdOrig = expOrig.getMetadata()
#pybind11#            mdOrig.set(keyWord, longString)
#pybind11#            expOrig.writeFits(tmpFile)
#pybind11#
#pybind11#            expNew = afwImage.ExposureF(tmpFile)
#pybind11#            self.assertEqual(expNew.getMetadata().get(keyWord), longString)
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

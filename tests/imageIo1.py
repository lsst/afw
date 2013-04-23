#!/usr/bin/env python

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
import os
import os.path

import unittest

import eups
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.utils.tests as utilsTests
import lsst.afw.display.ds9 as ds9

try:
    type(verbose)
except NameError:
    verbose = 0

dataDir = os.path.join(eups.productDir("afwdata"), "data")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ReadFitsTestCase(unittest.TestCase):
    """A test case for reading FITS images"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testU16(self):
        """Test reading U16 image"""

        im = afwImage.ImageD(os.path.join(dataDir, "small_img.fits"))
        
        col, row, val = 0, 0, 1154
        self.assertEqual(im.get(col, row), val)

    def testS16(self):
        """Test reading S16 image"""
        im = afwImage.ImageD(os.path.join(dataDir, "871034p_1_img.fits"))

        if False:
            ds9.mtv(im)
        
        col, row, val = 32, 1, 62
        self.assertEqual(im.get(col, row), val)

    def testF32(self):
        """Test reading F32 image"""
        im = afwImage.ImageD(os.path.join(dataDir, "871034p_1_MI.fits"), 4)
        
        col, row, val = 32, 1, 39.11672
        self.assertAlmostEqual(im.get(col, row), val, 5)

    def testF64(self):
        """Test reading a U16 file into a F64 image"""
        im = afwImage.ImageD(os.path.join(dataDir, "small_img.fits"))
        col, row, val = 0, 0, 1154
        self.assertEqual(im.get(col, row), val)
        
        #print "IM = ", im
    def testWriteReadF64(self):
        """Test writing then reading an F64 image"""

        imPath = "data"
        if os.path.exists("tests"):
            imPath = os.path.join("tests", imPath)
        imPath = os.path.join(imPath, "smallD.fits")
        
        im = afwImage.ImageD(afwGeom.Extent2I(100, 100))
        im.set(666)
        im.writeFits(imPath)
        newIm = afwImage.ImageD(imPath)
        os.remove(imPath)

    def testSubimage(self):
        """Test reading a subimage image"""
        fileName, hdu = os.path.join(dataDir, "871034p_1_MI.fits"), 4
        im = afwImage.ImageF(fileName, hdu)

        bbox = afwGeom.Box2I(afwGeom.Point2I(110, 120), afwGeom.Extent2I(20, 15))
        sim = im.Factory(im, bbox, afwImage.LOCAL) 

        im2 = afwImage.ImageF(fileName, hdu, None, bbox, afwImage.LOCAL)

        self.assertEqual(im2.getDimensions(), sim.getDimensions())
        self.assertEqual(im2.get(1, 1), sim.get(1, 1))

        self.assertEqual(im2.getX0(), sim.getX0())
        self.assertEqual(im2.getY0(), sim.getY0())

    def testMEF(self):
        """Test writing a set of images to an MEF fits file, and then reading them back"""
        
        imPath = "data"
        if os.path.exists("tests"):
            imPath = os.path.join("tests", imPath)
        imPath = os.path.join(imPath, "MEF.fits")

        im = afwImage.ImageF(afwGeom.Extent2I(20, 20))

        for hdu in range(1, 5):
            im.set(100*hdu)
            if hdu == 1:
                mode = "w"
            else:
                mode = "a"
            im.writeFits(imPath, None, mode)

        for hdu in range(1, 5):
            im = afwImage.ImageF(imPath, hdu)
            self.assertEqual(im.get(0, 0), 100*hdu)

        os.remove(imPath)

    def testWriteBool(self):
        """Test that we can read and write bools"""
        import lsst.afw.image as afwImage
        import lsst.daf.base as dafBase

        imPath = "data"
        if os.path.exists("tests"):
            imPath = os.path.join("tests", imPath)
        imPath = os.path.join(imPath, "tmp.fits")

        im = afwImage.ImageF(afwGeom.ExtentI(10,20))
        md = dafBase.PropertySet()
        keys = {"BAD" : False,
                "GOOD" : True,
                }
        for k, v in keys.items():
            md.add(k, v)
        
        im.writeFits(imPath, md)

        jim = afwImage.DecoratedImageF(imPath)
        os.remove(imPath)

        for k, v in keys.items():
            self.assertEqual(jim.getMetadata().get(k), v)

    def testLongStrings(self):
        keyWord = 'ZZZ'
        fitsName = 'zzz.fits'
        longString = ' '.join(['This is a long string.'] * 8)

        expOrig = afwImage.ExposureF(100,100)
        mdOrig = expOrig.getMetadata()
        mdOrig.set(keyWord, longString)
        expOrig.writeFits(fitsName)

        expNew = afwImage.ExposureF(fitsName)
        self.assertEqual(expNew.getMetadata().get(keyWord), longString)
        os.remove(fitsName)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ReadFitsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)
 
if __name__ == "__main__":
    run(True)

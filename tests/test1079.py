#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
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
#pybind11#
#pybind11## test1079
#pybind11## \brief Test that the wcs of sub-images are written and read from disk correctly
#pybind11## $Id$
#pybind11## \author Fergal Mullally
#pybind11#
#pybind11#import os.path
#pybind11#import unittest
#pybind11#import numbers
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#
#pybind11#try:
#pybind11#    type(verbose)
#pybind11#except NameError:
#pybind11#    verbose = 0
#pybind11#
#pybind11#
#pybind11#class SavingSubImagesTest(unittest.TestCase):
#pybind11#    """
#pybind11#    Tests for changes made for ticket #1079. In the LSST wcs transformations are done in terms
#pybind11#    of pixel position, which is measured from the lower left hand corner of the parent image from
#pybind11#    which this sub-image is drawn. However, when saving a sub-image to disk, the fits standards
#pybind11#    has no concept of parent- and sub- images, and specifies that the wcs is measured relative to
#pybind11#    the pixel index (i.e the lower left hand corner of the sub-image). This test makes sure
#pybind11#    we're saving and reading wcs headers from sub-images correctly.
#pybind11#    """
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        path = lsst.utils.getPackageDir("afw")
#pybind11#        self.parentFile = os.path.join(path, "tests", "data", "parent.fits")
#pybind11#
#pybind11#        self.parent = afwImage.ExposureF(self.parentFile)
#pybind11#        self.llcParent = self.parent.getMaskedImage().getXY0()
#pybind11#        self.oParent = self.parent.getWcs().getPixelOrigin()
#pybind11#
#pybind11#        # A list of pixel positions to test
#pybind11#        self.testPositions = []
#pybind11#        self.testPositions.append(afwGeom.Point2D(128, 128))
#pybind11#        self.testPositions.append(afwGeom.Point2D(0, 0))
#pybind11#        self.testPositions.append(afwGeom.Point2D(20, 30))
#pybind11#        self.testPositions.append(afwGeom.Point2D(60, 50))
#pybind11#        self.testPositions.append(afwGeom.Point2D(80, 80))
#pybind11#        self.testPositions.append(afwGeom.Point2D(255, 255))
#pybind11#
#pybind11#        self.parent.getMaskedImage().set(0)
#pybind11#        for p in self.testPositions:
#pybind11#            self.parent.getMaskedImage().set(int(p[0]), int(p[1]), (10 + p[0],))
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.parent
#pybind11#        del self.oParent
#pybind11#        del self.testPositions
#pybind11#
#pybind11#    def testInvarianceOfCrpix1(self):
#pybind11#        """Test that crpix is the same for parent and sub-image. Also tests that llc of sub-image
#pybind11#        saved correctly"""
#pybind11#
#pybind11#        llc = afwGeom.Point2I(20, 30)
#pybind11#        bbox = afwGeom.Box2I(llc, afwGeom.Extent2I(60, 50))
#pybind11#        subImg = afwImage.ExposureF(self.parent, bbox, afwImage.LOCAL)
#pybind11#
#pybind11#        subImgLlc = subImg.getMaskedImage().getXY0()
#pybind11#        oSubImage = subImg.getWcs().getPixelOrigin()
#pybind11#
#pybind11#        # Useful for debugging
#pybind11#        if False:
#pybind11#            print(self.parent.getMaskedImage().getXY0())
#pybind11#            print(subImg.getMaskedImage().getXY0())
#pybind11#            print(self.parent.getWcs().getFitsMetadata().toString())
#pybind11#            print(subImg.getWcs().getFitsMetadata().toString())
#pybind11#            print(self.oParent, oSubImage)
#pybind11#
#pybind11#        for i in range(2):
#pybind11#            self.assertEqual(llc[i], subImgLlc[i], "Corner of sub-image not correct")
#pybind11#            self.assertAlmostEqual(self.oParent[i], oSubImage[i], 6, "Crpix of sub-image not correct")
#pybind11#
#pybind11#    def testInvarianceOfCrpix2(self):
#pybind11#        """For sub-images loaded from disk, test that crpix is the same for parent and sub-image.
#pybind11#        Also tests that llc of sub-image saved correctly"""
#pybind11#
#pybind11#        # Load sub-image directly off of disk
#pybind11#        llc = afwGeom.Point2I(20, 30)
#pybind11#        bbox = afwGeom.Box2I(llc, afwGeom.Extent2I(60, 50))
#pybind11#        subImg = afwImage.ExposureF(self.parentFile, bbox, afwImage.LOCAL)
#pybind11#        oSubImage = subImg.getWcs().getPixelOrigin()
#pybind11#        subImgLlc = subImg.getMaskedImage().getXY0()
#pybind11#
#pybind11#        # Useful for debugging
#pybind11#        if False:
#pybind11#            print(self.parent.getMaskedImage().getXY0())
#pybind11#            print(subImg.getMaskedImage().getXY0())
#pybind11#            print(self.parent.getWcs().getFitsMetadata().toString())
#pybind11#            print(subImg.getWcs().getFitsMetadata().toString())
#pybind11#            print(self.oParent, oSubImage)
#pybind11#
#pybind11#        for i in range(2):
#pybind11#            self.assertEqual(llc[i], subImgLlc[i], "Corner of sub-image not correct")
#pybind11#            self.assertAlmostEqual(self.oParent[i], oSubImage[i], 6, "Crpix of sub-image not correct")
#pybind11#
#pybind11#    def testInvarianceOfPixelToSky(self):
#pybind11#
#pybind11#        for deep in (True, False):
#pybind11#            llc = afwGeom.Point2I(20, 30)
#pybind11#            bbox = afwGeom.Box2I(llc, afwGeom.Extent2I(60, 50))
#pybind11#            subImg = afwImage.ExposureF(self.parent, bbox, afwImage.LOCAL, deep)
#pybind11#
#pybind11#            xy0 = subImg.getMaskedImage().getXY0()
#pybind11#
#pybind11#            if False:
#pybind11#                ds9.mtv(self.parent, frame=0)
#pybind11#                ds9.mtv(subImg, frame=1)
#pybind11#
#pybind11#            for p in self.testPositions:
#pybind11#                subP = p - afwGeom.Extent2D(llc[0], llc[1])  # pixel in subImg
#pybind11#
#pybind11#                if \
#pybind11#                        subP[0] < 0 or subP[0] >= bbox.getWidth() or \
#pybind11#                        subP[1] < 0 or subP[1] >= bbox.getHeight():
#pybind11#                    continue
#pybind11#
#pybind11#                adParent = self.parent.getWcs().pixelToSky(p)
#pybind11#                adSub = subImg.getWcs().pixelToSky(subP + afwGeom.Extent2D(xy0[0], xy0[1]))
#pybind11#                #
#pybind11#                # Check that we're talking about the same pixel
#pybind11#                #
#pybind11#                self.assertEqual(self.parent.getMaskedImage().get(int(p[0]), int(p[1])),
#pybind11#                                 subImg.getMaskedImage().get(int(subP[0]), int(subP[1])))
#pybind11#
#pybind11#                self.assertEqual(adParent[0], adSub[0], "RAs are equal; deep = %s" % deep)
#pybind11#                self.assertEqual(adParent[1], adSub[1], "DECs are equal; deep = %s" % deep)
#pybind11#
#pybind11#    def testSubSubImage(self):
#pybind11#        """Check that a sub-image of a sub-image is equivalent to a sub image, i.e
#pybind11#        that the parent is an invarient"""
#pybind11#
#pybind11#        llc1 = afwGeom.Point2I(20, 30)
#pybind11#        bbox = afwGeom.Box2I(llc1, afwGeom.Extent2I(60, 50))
#pybind11#        subImg = afwImage.ExposureF(self.parentFile, bbox, afwImage.LOCAL)
#pybind11#
#pybind11#        llc2 = afwGeom.Point2I(22, 23)
#pybind11#
#pybind11#        # This subsub image should fail. Although it's big enough to fit in the parent image
#pybind11#        # it's too small for the sub-image
#pybind11#        bbox = afwGeom.Box2I(llc2, afwGeom.Extent2I(100, 110))
#pybind11#        self.assertRaises(lsst.pex.exceptions.Exception, afwImage.ExposureF, subImg, bbox, afwImage.LOCAL)
#pybind11#
#pybind11#        bbox = afwGeom.Box2I(llc2, afwGeom.Extent2I(10, 11))
#pybind11#        subSubImg = afwImage.ExposureF(subImg, bbox, afwImage.LOCAL)
#pybind11#
#pybind11#        sub0 = subImg.getMaskedImage().getXY0()
#pybind11#        subsub0 = subSubImg.getMaskedImage().getXY0()
#pybind11#
#pybind11#        if False:
#pybind11#            print(sub0)
#pybind11#            print(subsub0)
#pybind11#
#pybind11#        for i in range(2):
#pybind11#            self.assertEqual(llc1[i], sub0[i], "XY0 don't match (1)")
#pybind11#            self.assertEqual(llc1[i] + llc2[i], subsub0[i], "XY0 don't match (2)")
#pybind11#
#pybind11#        subCrpix = subImg.getWcs().getPixelOrigin()
#pybind11#        subsubCrpix = subSubImg.getWcs().getPixelOrigin()
#pybind11#
#pybind11#        for i in range(2):
#pybind11#            self.assertAlmostEqual(subCrpix[i], subsubCrpix[i], 6, "crpix don't match")
#pybind11#
#pybind11#    def testRoundTrip(self):
#pybind11#        """Test that saving and retrieving an image doesn't alter the metadata"""
#pybind11#        llc = afwGeom.Point2I(20, 30)
#pybind11#        bbox = afwGeom.Box2I(llc, afwGeom.Extent2I(60, 50))
#pybind11#        for deep in (False, True):
#pybind11#            subImg = afwImage.ExposureF(self.parent, bbox, afwImage.LOCAL, deep)
#pybind11#
#pybind11#            with lsst.utils.tests.getTempFilePath("_%s.fits" % (deep,)) as outFile:
#pybind11#                subImg.writeFits(outFile)
#pybind11#                newImg = afwImage.ExposureF(outFile)
#pybind11#
#pybind11#                subXY0 = subImg.getMaskedImage().getXY0()
#pybind11#                newXY0 = newImg.getMaskedImage().getXY0()
#pybind11#
#pybind11#                self.parent.getWcs().getPixelOrigin()
#pybind11#                subCrpix = subImg.getWcs().getPixelOrigin()
#pybind11#                newCrpix = newImg.getWcs().getPixelOrigin()
#pybind11#
#pybind11#                if False:
#pybind11#                    print(self.parent.getWcs().getFitsMetadata().toString())
#pybind11#                    print(subImg.getWcs().getFitsMetadata().toString())
#pybind11#                    print(newImg.getWcs().getFitsMetadata().toString())
#pybind11#
#pybind11#                for i in range(2):
#pybind11#                    self.assertEqual(subXY0[i], newXY0[i], "Origin has changed; deep = %s" % deep)
#pybind11#                    self.assertAlmostEqual(subCrpix[i], newCrpix[i], 6, "crpix has changed; deep = %s" % deep)
#pybind11#
#pybind11#    def testFitsHeader(self):
#pybind11#        """Test that XY0 and crpix are written to the header as expected"""
#pybind11#
#pybind11#        # getPixelOrigin() returns origin in lsst coordinates, so need to add 1 to
#pybind11#        # compare to values stored in fits headers
#pybind11#        parentCrpix = self.parent.getWcs().getPixelOrigin()
#pybind11#
#pybind11#        # Make a sub-image
#pybind11#        x0, y0 = 20, 30
#pybind11#        llc = afwGeom.Point2I(x0, y0)
#pybind11#        bbox = afwGeom.Box2I(llc, afwGeom.Extent2I(60, 50))
#pybind11#        deep = False
#pybind11#        subImg = afwImage.ExposureF(self.parent, bbox, afwImage.LOCAL, deep)
#pybind11#
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as outFile:
#pybind11#            subImg.writeFits(outFile)
#pybind11#            hdr = afwImage.readMetadata(outFile)
#pybind11#
#pybind11#            def checkLtvHeader(hdr, name, value):
#pybind11#                # Per DM-4133, LTVn headers are required to be floating point
#pybind11#                self.assertTrue(hdr.exists(name), name + " not saved to FITS header")
#pybind11#                self.assertIsInstance(hdr.get(name), numbers.Real, name + " is not numeric")
#pybind11#                self.assertNotIsInstance(hdr.get(name), numbers.Integral, name + " is an int")
#pybind11#                self.assertEqual(hdr.get(name), value, name + " has wrong value")
#pybind11#
#pybind11#            checkLtvHeader(hdr, "LTV1", -1*x0)
#pybind11#            checkLtvHeader(hdr, "LTV2", -1*y0)
#pybind11#
#pybind11#            self.assertTrue(hdr.exists("CRPIX1"), "CRPIX1 not saved to fits header")
#pybind11#            self.assertTrue(hdr.exists("CRPIX2"), "CRPIX2 not saved to fits header")
#pybind11#
#pybind11#            fitsCrpix = [hdr.get("CRPIX1"), hdr.get("CRPIX2")]
#pybind11#            self.assertAlmostEqual(fitsCrpix[0] - hdr.get("LTV1"), parentCrpix[0]+1, 6, "CRPIX1 saved wrong")
#pybind11#            self.assertAlmostEqual(fitsCrpix[1] - hdr.get("LTV2"), parentCrpix[1]+1, 6, "CRPIX2 saved wrong")
#pybind11#
#pybind11######
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

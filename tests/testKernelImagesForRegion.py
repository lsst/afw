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
#pybind11#import math
#pybind11#import unittest
#pybind11#
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.math.detail as mathDetail
#pybind11#from lsst.log import Log
#pybind11#
#pybind11## Change the level to Log.DEBUG to see debug messages
#pybind11#Log.getLogger("TRACE5.afw.math.convolve").setLevel(Log.INFO)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#LocNameDict = {
#pybind11#    mathDetail.KernelImagesForRegion.BOTTOM_LEFT: "BOTTOM_LEFT",
#pybind11#    mathDetail.KernelImagesForRegion.BOTTOM_RIGHT: "BOTTOM_RIGHT",
#pybind11#    mathDetail.KernelImagesForRegion.TOP_LEFT: "TOP_LEFT",
#pybind11#    mathDetail.KernelImagesForRegion.TOP_RIGHT: "TOP_RIGHT",
#pybind11#}
#pybind11#
#pybind11#NameLocDict = dict((name, loc) for (loc, name) in LocNameDict.items())
#pybind11#
#pybind11#
#pybind11#class KernelImagesForRegion(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        boxCorner = afwGeom.Point2I(11, 50)
#pybind11#        boxExtent = afwGeom.Extent2I(100, 99)
#pybind11#        self.bbox = afwGeom.Box2I(boxCorner, boxExtent)
#pybind11#        self.xy0 = afwGeom.Point2I(100, 251)
#pybind11#        self.kernel = self.makeKernel()
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        self.bbox = None
#pybind11#        self.kernel = None
#pybind11#
#pybind11#    def assertRegionCorrect(self, region):
#pybind11#        """Assert that a region has correct corner images
#pybind11#
#pybind11#        This test is only relevant for operations that try to reuse the image array data
#pybind11#        """
#pybind11#        regionCopy = mathDetail.KernelImagesForRegion(
#pybind11#            region.getKernel(), region.getBBox(), region.getXY0(), region.getDoNormalize())
#pybind11#
#pybind11#        for location in (
#pybind11#            region.BOTTOM_LEFT,
#pybind11#            region.BOTTOM_RIGHT,
#pybind11#            region.TOP_LEFT,
#pybind11#            region.TOP_RIGHT,
#pybind11#        ):
#pybind11#            actImage = region.getImage(location)
#pybind11#            actImArr = actImage.getArray().transpose().copy()
#pybind11#            desImage = regionCopy.getImage(location)
#pybind11#            desImArr = desImage.getArray().transpose().copy()
#pybind11#            actImArr -= desImArr
#pybind11#            if not numpy.allclose(actImArr, 0):
#pybind11#                actImage.writeFits("actImage%s.fits" % (location,))
#pybind11#                desImage.writeFits("desImage%s.fits" % (location,))
#pybind11#                self.fail("failed on location %s" % (location,))
#pybind11#
#pybind11#    def makeKernel(self):
#pybind11#        kCols = 7
#pybind11#        kRows = 6
#pybind11#
#pybind11#        # create spatial model
#pybind11#        sFunc = afwMath.PolynomialFunction2D(1)
#pybind11#
#pybind11#        minSigma = 0.1
#pybind11#        maxSigma = 3.0
#pybind11#
#pybind11#        # spatial parameters are a list of entries, one per kernel parameter;
#pybind11#        # each entry is a list of spatial parameters
#pybind11#        xSlope = (maxSigma - minSigma) / self.bbox.getWidth()
#pybind11#        ySlope = (maxSigma - minSigma) / self.bbox.getHeight()
#pybind11#        xOrigin = minSigma - (self.xy0[0] * xSlope)
#pybind11#        yOrigin = minSigma - (self.xy0[1] * ySlope)
#pybind11#        sParams = (
#pybind11#            (xOrigin, xSlope, 0.0),
#pybind11#            (yOrigin, 0.0, ySlope),
#pybind11#            (0.0, 0.0, 0.0),
#pybind11#        )
#pybind11#
#pybind11#        kFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        kernel = afwMath.AnalyticKernel(kCols, kRows, kFunc, sFunc)
#pybind11#        kernel.setSpatialParameters(sParams)
#pybind11#        return kernel
#pybind11#
#pybind11#    def testDoNormalize(self):
#pybind11#        """Test getDoNormalize
#pybind11#        """
#pybind11#        kernel = self.makeKernel()
#pybind11#        for doNormalize in (False, True):
#pybind11#            region = mathDetail.KernelImagesForRegion(kernel, self.bbox, self.xy0, doNormalize)
#pybind11#            self.assertEqual(region.getDoNormalize(), doNormalize)
#pybind11#
#pybind11#    def testGetPixelIndex(self):
#pybind11#        """Test getPixelIndex method
#pybind11#        """
#pybind11#        region = mathDetail.KernelImagesForRegion(self.kernel, self.bbox, self.xy0, False)
#pybind11#        leftInd = self.bbox.getMinX()
#pybind11#        rightInd = self.bbox.getMaxX() + 1
#pybind11#        bottomInd = self.bbox.getMinY()
#pybind11#        topInd = self.bbox.getMaxY() + 1
#pybind11#        int(round((leftInd + rightInd) / 2.0))
#pybind11#        int(round((bottomInd + topInd) / 2.0))
#pybind11#
#pybind11#        for location, desIndex in (
#pybind11#            (region.BOTTOM_LEFT, (leftInd, bottomInd)),
#pybind11#            (region.BOTTOM_RIGHT, (rightInd, bottomInd)),
#pybind11#            (region.TOP_LEFT, (leftInd, topInd)),
#pybind11#            (region.TOP_RIGHT, (rightInd, topInd)),
#pybind11#        ):
#pybind11#            desPixIndex = afwGeom.Point2I(desIndex[0], desIndex[1])
#pybind11#            self.assertEqual(region.getPixelIndex(location), desPixIndex,
#pybind11#                         "getPixelIndex(%s) = %s != %s" % (LocNameDict[location], region.getPixelIndex(location),
#pybind11#                                                           desPixIndex)
#pybind11#                         )
#pybind11#
#pybind11#    def testComputeNextRow(self):
#pybind11#        """Test computeNextRow method and the resulting RowOfKernelImagesForRegion
#pybind11#        """
#pybind11#        nx = 6
#pybind11#        ny = 5
#pybind11#        regionRow = mathDetail.RowOfKernelImagesForRegion(nx, ny)
#pybind11#        self.assertFalse(regionRow.hasData())
#pybind11#        self.assertFalse(regionRow.isLastRow())
#pybind11#        self.assertEqual(regionRow.getYInd(), -1)
#pybind11#
#pybind11#        region = mathDetail.KernelImagesForRegion(self.kernel, self.bbox, self.xy0, False)
#pybind11#        floatWidth = self.bbox.getWidth() / float(nx)
#pybind11#        validWidths = (int(math.floor(floatWidth)), int(math.ceil(floatWidth)))
#pybind11#        floatHeight = self.bbox.getHeight() / float(ny)
#pybind11#        validHeights = (int(math.floor(floatHeight)), int(math.ceil(floatHeight)))
#pybind11#
#pybind11#        totalHeight = 0
#pybind11#        prevBBox = None
#pybind11#        prevFirstBBox = None
#pybind11#        for yInd in range(ny):
#pybind11#            rowWidth = 0
#pybind11#            isOK = region.computeNextRow(regionRow)
#pybind11#            self.assertTrue(isOK)
#pybind11#            self.assertTrue(regionRow.hasData())
#pybind11#            self.assertEqual(regionRow.isLastRow(), (yInd + 1 >= ny))
#pybind11#            self.assertEqual(regionRow.getYInd(), yInd)
#pybind11#            firstBBox = regionRow.getRegion(0).getBBox()
#pybind11#            self.assertEqual(firstBBox.getMinX(), self.bbox.getMinX())
#pybind11#            if yInd == 0:
#pybind11#                self.assertEqual(firstBBox.getMinY(), self.bbox.getMinY())
#pybind11#            firstBBoxHeight = firstBBox.getHeight()
#pybind11#            self.assertTrue(firstBBoxHeight in validHeights)
#pybind11#            totalHeight += firstBBoxHeight
#pybind11#            if yInd > 0:
#pybind11#                self.assertEqual(firstBBox.getMinY(), prevFirstBBox.getMaxY() + 1)
#pybind11#                if yInd == ny - 1:
#pybind11#                    self.assertEqual(firstBBox.getMaxY(), self.bbox.getMaxY())
#pybind11#            prevFirstBBox = firstBBox
#pybind11#            for xInd in range(nx):
#pybind11#                subregion = regionRow.getRegion(xInd)
#pybind11#                try:
#pybind11#                    self.assertRegionCorrect(subregion)
#pybind11#                except:
#pybind11#                    print("failed on xInd=%s, yInd=%s" % (xInd, yInd))
#pybind11#                    raise
#pybind11#                bbox = subregion.getBBox()
#pybind11#                rowWidth += bbox.getWidth()
#pybind11#                self.assertTrue(bbox.getWidth() in validWidths)
#pybind11#                self.assertEqual(bbox.getHeight(), firstBBoxHeight)
#pybind11#                if xInd > 0:
#pybind11#                    self.assertEqual(bbox.getMinX(), prevBBox.getMaxX() + 1)
#pybind11#                    self.assertEqual(bbox.getMinY(), prevBBox.getMinY())
#pybind11#                    self.assertEqual(bbox.getMaxY(), prevBBox.getMaxY())
#pybind11#                    if xInd == nx - 1:
#pybind11#                        self.assertEqual(bbox.getMaxX(), self.bbox.getMaxX())
#pybind11#                prevBBox = bbox
#pybind11#            self.assertEqual(rowWidth, self.bbox.getWidth())
#pybind11#        self.assertEqual(totalHeight, self.bbox.getHeight())
#pybind11#        self.assertTrue(not region.computeNextRow(regionRow))
#pybind11#
#pybind11#    def testExactImages(self):
#pybind11#        """Confirm that kernel image at each location is correct
#pybind11#        """
#pybind11#        desImage = afwImage.ImageD(afwGeom.Extent2I(self.kernel.getWidth(), self.kernel.getHeight()))
#pybind11#
#pybind11#        for doNormalize in (False, True):
#pybind11#            region = mathDetail.KernelImagesForRegion(self.kernel, self.bbox, self.xy0, doNormalize)
#pybind11#            for location in (
#pybind11#                region.BOTTOM_LEFT,
#pybind11#                region.BOTTOM_RIGHT,
#pybind11#                region.TOP_LEFT,
#pybind11#                region.TOP_RIGHT,
#pybind11#            ):
#pybind11#                pixelIndex = region.getPixelIndex(location)
#pybind11#                xPos = afwImage.indexToPosition(pixelIndex[0] + self.xy0[0])
#pybind11#                yPos = afwImage.indexToPosition(pixelIndex[1] + self.xy0[1])
#pybind11#                self.kernel.computeImage(desImage, doNormalize, xPos, yPos)
#pybind11#
#pybind11#                actImage = region.getImage(location)
#pybind11#                msg = "exact image(%s) incorrect" % (LocNameDict[location],)
#pybind11#                self.assertImagesNearlyEqual(actImage, desImage, msg=msg)
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
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

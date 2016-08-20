#!/usr/bin/env python
from __future__ import absolute_import, division
from __future__ import print_function
from builtins import range

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
import math
import unittest

import numpy

import lsst.utils.tests
import lsst.pex.logging as pexLog
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.math.detail as mathDetail

VERBOSITY = 0  # increase to see trace

pexLog.Debug("lsst.afw", VERBOSITY)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

LocNameDict = {
    mathDetail.KernelImagesForRegion.BOTTOM_LEFT: "BOTTOM_LEFT",
    mathDetail.KernelImagesForRegion.BOTTOM_RIGHT: "BOTTOM_RIGHT",
    mathDetail.KernelImagesForRegion.TOP_LEFT: "TOP_LEFT",
    mathDetail.KernelImagesForRegion.TOP_RIGHT: "TOP_RIGHT",
}

NameLocDict = dict((name, loc) for (loc, name) in LocNameDict.items())


class KernelImagesForRegion(lsst.utils.tests.TestCase):

    def setUp(self):
        boxCorner = afwGeom.Point2I(11, 50)
        boxExtent = afwGeom.Extent2I(100, 99)
        self.bbox = afwGeom.Box2I(boxCorner, boxExtent)
        self.xy0 = afwGeom.Point2I(100, 251)
        self.kernel = self.makeKernel()

    def tearDown(self):
        self.bbox = None
        self.kernel = None

    def assertRegionCorrect(self, region):
        """Assert that a region has correct corner images

        This test is only relevant for operations that try to reuse the image array data
        """
        regionCopy = mathDetail.KernelImagesForRegion(
            region.getKernel(), region.getBBox(), region.getXY0(), region.getDoNormalize())

        for location in (
            region.BOTTOM_LEFT,
            region.BOTTOM_RIGHT,
            region.TOP_LEFT,
            region.TOP_RIGHT,
        ):
            actImage = region.getImage(location)
            actImArr = actImage.getArray().transpose().copy()
            desImage = regionCopy.getImage(location)
            desImArr = desImage.getArray().transpose().copy()
            actImArr -= desImArr
            if not numpy.allclose(actImArr, 0):
                actImage.writeFits("actImage%s.fits" % (location,))
                desImage.writeFits("desImage%s.fits" % (location,))
                self.fail("failed on location %s" % (location,))

    def makeKernel(self):
        kCols = 7
        kRows = 6

        # create spatial model
        sFunc = afwMath.PolynomialFunction2D(1)

        minSigma = 0.1
        maxSigma = 3.0

        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        xSlope = (maxSigma - minSigma) / self.bbox.getWidth()
        ySlope = (maxSigma - minSigma) / self.bbox.getHeight()
        xOrigin = minSigma - (self.xy0[0] * xSlope)
        yOrigin = minSigma - (self.xy0[1] * ySlope)
        sParams = (
            (xOrigin, xSlope, 0.0),
            (yOrigin, 0.0, ySlope),
            (0.0, 0.0, 0.0),
        )

        kFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        kernel = afwMath.AnalyticKernel(kCols, kRows, kFunc, sFunc)
        kernel.setSpatialParameters(sParams)
        return kernel

    def testDoNormalize(self):
        """Test getDoNormalize
        """
        kernel = self.makeKernel()
        for doNormalize in (False, True):
            region = mathDetail.KernelImagesForRegion(kernel, self.bbox, self.xy0, doNormalize)
            self.assertEqual(region.getDoNormalize(), doNormalize)

    def testGetPixelIndex(self):
        """Test getPixelIndex method
        """
        region = mathDetail.KernelImagesForRegion(self.kernel, self.bbox, self.xy0, False)
        leftInd = self.bbox.getMinX()
        rightInd = self.bbox.getMaxX() + 1
        bottomInd = self.bbox.getMinY()
        topInd = self.bbox.getMaxY() + 1
        int(round((leftInd + rightInd) / 2.0))
        int(round((bottomInd + topInd) / 2.0))

        for location, desIndex in (
            (region.BOTTOM_LEFT, (leftInd, bottomInd)),
            (region.BOTTOM_RIGHT, (rightInd, bottomInd)),
            (region.TOP_LEFT, (leftInd, topInd)),
            (region.TOP_RIGHT, (rightInd, topInd)),
        ):
            desPixIndex = afwGeom.Point2I(desIndex[0], desIndex[1])
            self.assertEqual(region.getPixelIndex(location), desPixIndex,
                         "getPixelIndex(%s) = %s != %s" % (LocNameDict[location], region.getPixelIndex(location),
                                                           desPixIndex)
                         )

    def testComputeNextRow(self):
        """Test computeNextRow method and the resulting RowOfKernelImagesForRegion
        """
        nx = 6
        ny = 5
        regionRow = mathDetail.RowOfKernelImagesForRegion(nx, ny)
        self.assertTrue(not regionRow.hasData())
        self.assertTrue(not regionRow.isLastRow())
        self.assertEqual(regionRow.getYInd(), -1)

        region = mathDetail.KernelImagesForRegion(self.kernel, self.bbox, self.xy0, False)
        floatWidth = self.bbox.getWidth() / float(nx)
        validWidths = (int(math.floor(floatWidth)), int(math.ceil(floatWidth)))
        floatHeight = self.bbox.getHeight() / float(ny)
        validHeights = (int(math.floor(floatHeight)), int(math.ceil(floatHeight)))

        totalHeight = 0
        prevBBox = None
        prevFirstBBox = None
        for yInd in range(ny):
            rowWidth = 0
            isOK = region.computeNextRow(regionRow)
            self.assertTrue(isOK)
            self.assertTrue(regionRow.hasData())
            self.assertEqual(regionRow.isLastRow(), (yInd + 1 >= ny))
            self.assertEqual(regionRow.getYInd(), yInd)
            firstBBox = regionRow.getRegion(0).getBBox()
            self.assertEqual(firstBBox.getMinX(), self.bbox.getMinX())
            if yInd == 0:
                self.assertEqual(firstBBox.getMinY(), self.bbox.getMinY())
            firstBBoxHeight = firstBBox.getHeight()
            self.assertTrue(firstBBoxHeight in validHeights)
            totalHeight += firstBBoxHeight
            if yInd > 0:
                self.assertEqual(firstBBox.getMinY(), prevFirstBBox.getMaxY() + 1)
                if yInd == ny - 1:
                    self.assertEqual(firstBBox.getMaxY(), self.bbox.getMaxY())
            prevFirstBBox = firstBBox
            for xInd in range(nx):
                subregion = regionRow.getRegion(xInd)
                try:
                    self.assertRegionCorrect(subregion)
                except:
                    print("failed on xInd=%s, yInd=%s" % (xInd, yInd))
                    raise
                bbox = subregion.getBBox()
                rowWidth += bbox.getWidth()
                self.assertTrue(bbox.getWidth() in validWidths)
                self.assertEqual(bbox.getHeight(), firstBBoxHeight)
                if xInd > 0:
                    self.assertEqual(bbox.getMinX(), prevBBox.getMaxX() + 1)
                    self.assertEqual(bbox.getMinY(), prevBBox.getMinY())
                    self.assertEqual(bbox.getMaxY(), prevBBox.getMaxY())
                    if xInd == nx - 1:
                        self.assertEqual(bbox.getMaxX(), self.bbox.getMaxX())
                prevBBox = bbox
            self.assertEqual(rowWidth, self.bbox.getWidth())
        self.assertEqual(totalHeight, self.bbox.getHeight())
        self.assertTrue(not region.computeNextRow(regionRow))

    def testExactImages(self):
        """Confirm that kernel image at each location is correct
        """
        desImage = afwImage.ImageD(afwGeom.Extent2I(self.kernel.getWidth(), self.kernel.getHeight()))

        for doNormalize in (False, True):
            region = mathDetail.KernelImagesForRegion(self.kernel, self.bbox, self.xy0, doNormalize)
            for location in (
                region.BOTTOM_LEFT,
                region.BOTTOM_RIGHT,
                region.TOP_LEFT,
                region.TOP_RIGHT,
            ):
                pixelIndex = region.getPixelIndex(location)
                xPos = afwImage.indexToPosition(pixelIndex[0] + self.xy0[0])
                yPos = afwImage.indexToPosition(pixelIndex[1] + self.xy0[1])
                self.kernel.computeImage(desImage, doNormalize, xPos, yPos)

                actImage = region.getImage(location)
                msg = "exact image(%s) incorrect" % (LocNameDict[location],)
                self.assertImagesNearlyEqual(actImage, desImage, msg=msg)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass

def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
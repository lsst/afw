#!/usr/bin/env python2
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

import lsst.utils.tests as utilsTests
import lsst.pex.logging as pexLog
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.math.detail as mathDetail

VERBOSITY = 0 # increase to see trace

pexLog.Debug("lsst.afw", VERBOSITY)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

LocNameDict = {
    mathDetail.KernelImagesForRegion.BOTTOM_LEFT: "BOTTOM_LEFT",
    mathDetail.KernelImagesForRegion.BOTTOM_RIGHT: "BOTTOM_RIGHT",
    mathDetail.KernelImagesForRegion.TOP_LEFT: "TOP_LEFT",
    mathDetail.KernelImagesForRegion.TOP_RIGHT: "TOP_RIGHT",
}

NameLocDict = dict((name, loc) for (loc, name) in LocNameDict.items())

class KernelImagesForRegion(utilsTests.TestCase):
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
        regionCopy = mathDetail.KernelImagesForRegion(region.getKernel(), region.getBBox(), region.getXY0(), region.getDoNormalize())

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
            (yOrigin, 0.0,    ySlope),
            (0.0, 0.0, 0.0),
        )

        kFunc =  afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        kernel = afwMath.AnalyticKernel(kCols, kRows, kFunc, sFunc)
        kernel.setSpatialParameters(sParams)
        return kernel

    def testDoNormalize(self):
        """Test getDoNormalize
        """
        kernel = self.makeKernel()
        for doNormalize in (False, True):
            region = mathDetail.KernelImagesForRegion(kernel, self.bbox, self.xy0, doNormalize)
            self.assert_(region.getDoNormalize() == doNormalize)

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
            (region.BOTTOM_LEFT,  (leftInd,  bottomInd)),
            (region.BOTTOM_RIGHT, (rightInd, bottomInd)),
            (region.TOP_LEFT,     (leftInd,  topInd)),
            (region.TOP_RIGHT,    (rightInd, topInd)),
        ):
            desPixIndex = afwGeom.Point2I(desIndex[0], desIndex[1])
            self.assert_(region.getPixelIndex(location) == desPixIndex,
                "getPixelIndex(%s) = %s != %s" % (LocNameDict[location], region.getPixelIndex(location),
                    desPixIndex)
            )

    def testComputeNextRow(self):
        """Test computeNextRow method and the resulting RowOfKernelImagesForRegion
        """
        nx = 6
        ny = 5
        regionRow = mathDetail.RowOfKernelImagesForRegion(nx, ny)
        self.assert_(not regionRow.hasData())
        self.assert_(not regionRow.isLastRow())
        self.assert_(regionRow.getYInd() == -1)

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
            self.assert_(isOK)
            self.assert_(regionRow.hasData())
            self.assert_(regionRow.isLastRow() == (yInd + 1 >= ny))
            self.assert_(regionRow.getYInd() == yInd)
            firstBBox = regionRow.getRegion(0).getBBox()
            self.assert_(firstBBox.getMinX() == self.bbox.getMinX())
            if yInd == 0:
                self.assert_(firstBBox.getMinY() == self.bbox.getMinY())
            firstBBoxHeight = firstBBox.getHeight()
            self.assert_(firstBBoxHeight in validHeights)
            totalHeight += firstBBoxHeight
            if yInd > 0:
                self.assert_(firstBBox.getMinY() == prevFirstBBox.getMaxY() + 1)
                if yInd == ny - 1:
                    self.assert_(firstBBox.getMaxY() == self.bbox.getMaxY())
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
                self.assert_(bbox.getWidth() in validWidths)
                self.assert_(bbox.getHeight() == firstBBoxHeight)
                if xInd > 0:
                    self.assert_(bbox.getMinX() == prevBBox.getMaxX() + 1)
                    self.assert_(bbox.getMinY() == prevBBox.getMinY())
                    self.assert_(bbox.getMaxY() == prevBBox.getMaxY())
                    if xInd == nx - 1:
                        self.assert_(bbox.getMaxX() == self.bbox.getMaxX())
                prevBBox = bbox
            self.assert_(rowWidth == self.bbox.getWidth())
        self.assert_(totalHeight == self.bbox.getHeight())
        self.assert_(not region.computeNextRow(regionRow))

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

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(KernelImagesForRegion)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(doExit=False):
    """Run the tests"""
    utilsTests.run(suite(), doExit)

if __name__ == "__main__":
    run(True)

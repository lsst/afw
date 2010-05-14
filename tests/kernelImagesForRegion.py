#!/usr/bin/env python
import unittest

import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.pex.logging as pexLog
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.math.detail as mathDetail
import lsst.afw.image.testUtils as imTestUtils

VERBOSITY = 0 # increase to see trace

pexLog.Debug("lsst.afw", VERBOSITY)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

LocNameDict = {
    mathDetail.KernelImagesForRegion.BOTTOM_LEFT: "BOTTOM_LEFT",
    mathDetail.KernelImagesForRegion.BOTTOM: "BOTTOM",
    mathDetail.KernelImagesForRegion.BOTTOM_RIGHT: "BOTTOM_RIGHT",
    mathDetail.KernelImagesForRegion.LEFT: "LEFT",
    mathDetail.KernelImagesForRegion.CENTER: "CENTER",
    mathDetail.KernelImagesForRegion.RIGHT: "RIGHT",
    mathDetail.KernelImagesForRegion.TOP_LEFT: "TOP_LEFT",
    mathDetail.KernelImagesForRegion.TOP: "TOP",
    mathDetail.KernelImagesForRegion.TOP_RIGHT: "TOP_RIGHT",
}

NameLocDict = dict((name, loc) for (loc, name) in LocNameDict.iteritems())

class KernelImagesForRegion(unittest.TestCase):
    def setUp(self):
        boxCorner = afwGeom.makePointI(11, 50)
        boxExtent = afwGeom.makeExtentI(100, 99)
        self.bbox = afwGeom.BoxI(boxCorner, boxExtent)
        self.imWidth = 200
        self.imHeight = 200
        self.kernel = self.makeKernel()

    def tearDown(self):
        self.bbox = None
        self.kernel = None

    def makeKernel(self):
        kCols = 7
        kRows = 6

        # create spatial model
        sFunc = afwMath.PolynomialFunction2D(1)
        
        minSigma = 0.1
        maxSigma = 3.0

        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (minSigma, (maxSigma - minSigma) / self.imWidth, 0.0),
            (minSigma, 0.0,  (maxSigma - minSigma) / self.imHeight),
            (0.0, 0.0, 0.0),
        )
   
        kFunc =  afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        kernel = afwMath.AnalyticKernel(kCols, kRows, kFunc, sFunc)
        kernel.setSpatialParameters(sParams)
        return kernel

    def runInterpTest(self, region, location, im0Arr, im1Arr, ind0, indCtr, ind1):
        """Run one interpolation test
        
        Inputs:
        - region: a KernelImagesForRegion
        - location: a location on the region (e.g. region.BOTTOM)
        - im0Arr: image array at one edge of the desired location (e.g. BOTTOM_LEFT for location=BOTTOM)
        - im1Arr: image array at the other edge of the location (e.g. BOTTOM_RIGHT for location=BOTTOM)
        - ind0: index of im0Arr along appropriate axis (e.g. LEFT for loation=BOTTOM)
        - indCtr: index of center position along appropriate axis (e.g. CENTER for location=BOTTOM)
        - ind1: index of im1Arr along appropriate axis (e.g. RIGHT for location=BOTTOM)
        """
        actImage = afwImage.ImageD(self.kernel.getWidth(), self.kernel.getHeight())
        region.interpolateImage(actImage, location)
        actImArr = imTestUtils.arrayFromImage(actImage)
        fracDist = float(indCtr - ind0) / float(ind1 - ind0)
        desImArr = (im0Arr * (1.0 - fracDist)) + (im1Arr * fracDist)
        errStr = imTestUtils.imagesDiffer(actImArr, desImArr)
        if errStr:
            self.fail("interpolateImage(%s) failed:\n%s" % (LocNameDict[location], errStr))
        
    def testDoNormalize(self):
        """Test getDoNormalize
        """
        kernel = self.makeKernel()
        for doNormalize in (False, True):
            region = mathDetail.KernelImagesForRegion(kernel, self.bbox, doNormalize)
            self.assert_(region.getDoNormalize() == doNormalize)

    def testGetPixelIndex(self):
        """Test getPixelIndex method
        """
        region = mathDetail.KernelImagesForRegion(self.kernel, self.bbox, False)
        leftInd = self.bbox.getMinX()
        rightInd = self.bbox.getMaxX() + 1
        bottomInd = self.bbox.getMinY()
        topInd = self.bbox.getMaxY() + 1
        ctrXInd = int(round((leftInd + rightInd) / 2.0))
        ctrYInd = int(round((bottomInd + topInd) / 2.0))
        
        for location, desIndex in (
            (region.BOTTOM_LEFT,  (leftInd,  bottomInd)),
            (region.BOTTOM,       (ctrXInd,  bottomInd)),
            (region.BOTTOM_RIGHT, (rightInd, bottomInd)),
            (region.LEFT,         (leftInd,  ctrYInd)),
            (region.CENTER,       (ctrXInd,  ctrYInd)),
            (region.RIGHT,        (rightInd, ctrYInd)),
            (region.TOP_LEFT,     (leftInd,  topInd)),
            (region.TOP,          (ctrXInd,  topInd)),
            (region.TOP_RIGHT,    (rightInd, topInd)),
        ):
            desPixIndex = afwGeom.makePointI(desIndex[0], desIndex[1])
            self.assert_(region.getPixelIndex(location) == desPixIndex,
                "getPixelIndex(%s) = %s != %s" % (LocNameDict[location], region.getPixelIndex(location),
                    desPixIndex)
            )
    
    def XXXtestGetSubregions(self):
        """Would like to test getSubregions(); but the returned object is opaque so can't do much
        """
        region = mathDetail.KernelImagesForRegion(self.kernel, self.bbox, False)
        subregionList = region.getSubregions()

    def testInterpolateImage(self):
        region = mathDetail.KernelImagesForRegion(self.kernel, self.bbox, False)
        actImage = afwImage.ImageD(self.kernel.getWidth(), self.kernel.getHeight())
        
        bottomLeftImArr = imTestUtils.arrayFromImage(region.getImage(region.BOTTOM_LEFT))
        bottomRightImArr = imTestUtils.arrayFromImage(region.getImage(region.BOTTOM_RIGHT))
        topLeftImArr = imTestUtils.arrayFromImage(region.getImage(region.TOP_LEFT))
        topRightImArr = imTestUtils.arrayFromImage(region.getImage(region.TOP_RIGHT))

        leftInd, bottomInd = region.getPixelIndex(region.BOTTOM_LEFT)
        rightInd, topInd = region.getPixelIndex(region.TOP_RIGHT)
        ctrXInd, ctrYInd = region.getPixelIndex(region.CENTER)
        
        self.runInterpTest(region, region.BOTTOM, bottomLeftImArr, bottomRightImArr, leftInd, ctrXInd, rightInd)
        self.runInterpTest(region, region.TOP,    topLeftImArr,    topRightImArr,    leftInd, ctrXInd, rightInd)
        self.runInterpTest(region, region.LEFT,  bottomLeftImArr,  topLeftImArr,  bottomInd, ctrYInd, topInd)
        self.runInterpTest(region, region.RIGHT, bottomRightImArr, topRightImArr, bottomInd, ctrYInd, topInd)

        bottomIm = afwImage.ImageD(self.kernel.getWidth(), self.kernel.getHeight())
        region.interpolateImage(bottomIm, region.BOTTOM)
        bottomImArr = imTestUtils.arrayFromImage(bottomIm)
        topIm = afwImage.ImageD(self.kernel.getWidth(), self.kernel.getHeight())
        region.interpolateImage(topIm, region.TOP)
        topImArr = imTestUtils.arrayFromImage(topIm)
        self.runInterpTest(region, region.CENTER, bottomImArr, topImArr, bottomInd, ctrYInd, topInd)
        

    def testIsInterpolateOk(self):
        region = mathDetail.KernelImagesForRegion(self.kernel, self.bbox, True)
        
        # compute max error
        interpImage = afwImage.ImageD(self.kernel.getWidth(), self.kernel.getHeight())
        maxDiff = 0.0
        for location in (region.BOTTOM, region.LEFT, region.CENTER, region.RIGHT, region.TOP):
            region.interpolateImage(interpImage, location)
            interpImArr = imTestUtils.arrayFromImage(interpImage)
            actImage = region.getImage(location)
            actImArr = imTestUtils.arrayFromImage(actImage)
            diffImArr = interpImArr - actImArr
            maxDiff = max(maxDiff, numpy.max(numpy.abs(diffImArr)))
        for epsilon in (1.0e-99, 0.0, 1.0e-99):
            if epsilon >= 0.0:
                self.assert_(region.isInterpolationOk(maxDiff + epsilon))
            else:
                self.assert_(not region.isInterpolationOk(maxDiff + epsilon))


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

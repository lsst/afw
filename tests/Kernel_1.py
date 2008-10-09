#!/usr/bin/env python
import os
import math
import pdb                          # we may want to say pdb.set_trace()
import unittest

import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions.exceptions as pexExcept
import lsst.pex.logging as pexLog
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.image.testUtils as imTestUtils

Verbosity = 0 # increase to see trace
pexLog.Trace_setVerbosity("lsst.afw", Verbosity)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class KernelTestCase(unittest.TestCase):
    """A test case for Kernels"""
    def testFixedKernel(self):
        """Test FixedKernel using a ramp function
        """
        kWidth = 5
        kHeight = 6
        
        inArr = numpy.arange(kWidth * kHeight, dtype=float)
        inArr.shape = [kWidth, kHeight]

        inImage = afwImage.ImageD(kWidth, kHeight)
        for row in range(inImage.getHeight()):
            for col in range(inImage.getWidth()):
                inImage.set(col, row, inArr[col, row])
        
        k = afwMath.FixedKernel(inImage);
        outImage = k.computeNewImage(False)[0]
        outArr = imTestUtils.arrayFromImage(outImage)
        if not numpy.allclose(inArr, outArr):
            self.fail("%s = %s != %s (not normalized)" % \
                (k.__class__.__name__, inArr, outArr))
        
        normInArr = inArr / inArr.sum()
        normOutImage = k.computeNewImage(True)[0]
        normOutArr = imTestUtils.arrayFromImage(normOutImage)
        if not numpy.allclose(normOutArr, normInArr):
            self.fail("%s = %s != %s (normalized)" % \
                (k.__class__.__name__, normInArr, normOutArr))

    def testAnalyticKernel(self):
        """Test AnalyticKernel using a Gaussian function
        """
        kWidth = 5
        kHeight = 8

        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0)
        k = afwMath.AnalyticKernel(gaussFunc, kWidth, kHeight)
        fArr = numpy.zeros(shape=[k.getWidth(), k.getHeight()], dtype=float)
        for xsigma in (0.1, 1.0, 3.0):
            for ysigma in (0.1, 1.0, 3.0):
                gaussFunc.setParameters((xsigma, ysigma))
                # compute array of function values and normalize
                for row in range(k.getHeight()):
                    y = row - k.getCtrY()
                    for col in range(k.getWidth()):
                        x = col - k.getCtrX()
                        fArr[col, row] = gaussFunc(x, y)
                fArr /= fArr.sum()
                
                k.setKernelParameters((xsigma, ysigma))
                kImage = k.computeNewImage(True)[0]
                kArr = imTestUtils.arrayFromImage(kImage)
                if not numpy.allclose(fArr, kArr):
                    self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" % \
                        (k.__class__.__name__, kArr, fArr, xsigma, ysigma))
    
    def testDeltaFunctionKernel(self):
        """Test DeltaFunctionKernel
        """
        for kWidth in range(1, 4):
            for kHeight in range(1, 4):
                for activeCol in range(kWidth):
                    for activeRow in range(kHeight):
                        kernel = afwMath.DeltaFunctionKernel(activeCol, activeRow, kWidth, kHeight)
                        kImage, kSum = kernel.computeNewImage(False)
                        self.assertEqual(kSum, 1.0)
                        kArr = imTestUtils.arrayFromImage(kImage)
                        self.assertEqual(kArr[activeCol, activeRow], 1.0)
                        kArr[activeCol, activeRow] = 0.0
                        self.assertEqual(kArr.sum(), 0.0)
                self.assertRaises(
                    pexExcept.LsstInvalidParameter,
                    afwMath.DeltaFunctionKernel, 0, kHeight, kWidth, kHeight)
                self.assertRaises(
                    pexExcept.LsstInvalidParameter,
                    afwMath.DeltaFunctionKernel, kWidth, 0, kWidth, kHeight)
                            

    def testSeparableKernel(self):
        """Test SeparableKernel using a Gaussian function
        """
        kWidth = 5
        kHeight = 8

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        k = afwMath.SeparableKernel(gaussFunc1, gaussFunc1, kWidth, kHeight)
        fArr = numpy.zeros(shape=[k.getWidth(), k.getHeight()], dtype=float)
        gArr = numpy.zeros(shape=[k.getWidth(), k.getHeight()], dtype=float)
        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0)
        for xsigma in (0.1, 1.0, 3.0):
            gaussFunc1.setParameters((xsigma,))
            for ysigma in (0.1, 1.0, 3.0):
                gaussFunc.setParameters((xsigma, ysigma))
                # compute array of function values and normalize
                for row in range(k.getHeight()):
                    y = row - k.getCtrY()
                    for col in range(k.getWidth()):
                        x = col - k.getCtrX()
                        fArr[col, row] = gaussFunc(x, y)
                fArr /= fArr.sum()
                
                k.setKernelParameters((xsigma, ysigma))
                kImage = k.computeNewImage(True)[0]
                kArr = imTestUtils.arrayFromImage(kImage)
                if not numpy.allclose(fArr, kArr):
                    self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" % \
                        (k.__class__.__name__, kArr, fArr, xsigma, ysigma))
    
    def testLinearCombinationKernel(self):
        """Test LinearCombinationKernel using a set of delta basis functions
        """
        kWidth = 3
        kHeight = 2
        
        # create list of kernels
        basisImArrList = []
        kVec = afwMath.KernelListD()
        for row in range(kHeight):
            for col in range(kWidth):
                kPtr = afwMath.DeltaFunctionKernel(col, row, kWidth, kHeight)
                basisImage = afwImage.ImageF(kPtr.dimensions())
                computeImage(basicImage, True)
                basisImArrList.append(imTestUtils.arrayFromImage(basisImage))
                kVec.append(kPtr)
        
        kParams = [0.0]*len(kVec)
        k = afwMath.LinearCombinationKernel(kVec, kParams)
        for ii in range(len(kVec)):
            kParams = [0.0]*len(kVec)
            kParams[ii] = 1.0
            k.setKernelParameters(kParams)
            kIm = k.computeNewImage(True)[0]
            kImArr = imTestUtils.arrayFromImage(kIm)
            if not numpy.allclose(kImArr, basisImArrList[ii]):
                self.fail("%s = %s != %s for the %s'th basis kernel" % \
                    (k.__class__.__name__, kImArr, basisImArrList[ii], ii))

    def testSVLinearCombinationKernel(self):
        """Test a spatially varying LinearCombinationKernel
        """
        kWidth = 3
        kHeight = 2

        # create image arrays for the basis kernels
        basisImArrList = []
        imArr = numpy.zeros((kWidth, kHeight), dtype=float)
        imArr += 0.1
        imArr[kWidth//2, :] = 0.9
        basisImArrList.append(imArr)
        imArr = numpy.zeros((kWidth, kHeight), dtype=float)
        imArr += 0.2
        imArr[:, kHeight//2] = 0.8
        basisImArrList.append(imArr)
        
        # create a list of basis kernels from the images
        kVec = afwMath.KernelListD()
        for basisImArr in basisImArrList:
            basisImage = imTestUtils.imageFromArray(basisImArr)
            kPtr = afwMath.FixedKernel(basisImage)
            kVec.append(kPtr)

        # create spatially varying linear combination kernel
        spFunc = afwMath.PolynomialFunction2D(1)
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        
        k = afwMath.LinearCombinationKernel(kVec, spFunc)
        k.setSpatialParameters(sParams)
        kImage = afwImage.ImageD(kWidth, kHeight)
        for colPos, rowPos, coeff0, coeff1 in [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5, 0.5),
        ]:
            k.computeImage(kImage, False, colPos, rowPos)
            kImArr = imTestUtils.arrayFromImage(kImage)
            refKImArr = (basisImArrList[0] * coeff0) + (basisImArrList[1] * coeff1)
            if not numpy.allclose(kImArr, refKImArr):
                self.fail("%s = %s != %s at colPos=%s, rowPos=%s" % \
                    (k.__class__.__name__, kImArr, refKImArr, colPos, rowPos))
    
    def testSetCtr(self):
        """Test setCtrCol/Row"""
        kWidth = 3
        kHeight = 4

        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0)
        k = afwMath.AnalyticKernel(gaussFunc, kWidth, kHeight)
        for xCtr in range(kWidth):
            k.setCtrX(xCtr)
            for yCtr in range(kHeight):
                k.setCtrY(yCtr)
                self.assertEqual(k.getCtrX(), colCtr)
                self.assertEqual(k.getCtrY(), rowCtr)
        

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(KernelTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

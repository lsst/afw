#!/usr/bin/env python
import os
import math
import pdb                          # we may want to say pdb.set_trace()
import unittest

import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
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
        
        kernel = afwMath.FixedKernel(inImage);
        self.basicTests(kernel, 0)
        outImage = afwImage.ImageD(kernel.getDimensions())
        kernel.computeImage(outImage, False)
        
        outArr = imTestUtils.arrayFromImage(outImage)
        if not numpy.allclose(inArr, outArr):
            self.fail("%s = %s != %s (not normalized)" % \
                (kernel.__class__.__name__, inArr, outArr))
        
        normInArr = inArr / inArr.sum()
        normOutImage = afwImage.ImageD(kernel.getDimensions())
        kernel.computeImage(normOutImage, True)
        normOutArr = imTestUtils.arrayFromImage(normOutImage)
        if not numpy.allclose(normOutArr, normInArr):
            self.fail("%s = %s != %s (normalized)" % \
                (kernel.__class__.__name__, normInArr, normOutArr))

    def testAnalyticKernel(self):
        """Test AnalyticKernel using a Gaussian function
        """
        kWidth = 5
        kHeight = 8

        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0)
        kernel = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc)
        self.basicTests(kernel, 2)
        fArr = numpy.zeros(shape=[kernel.getWidth(), kernel.getHeight()], dtype=float)
        for xsigma in (0.1, 1.0, 3.0):
            for ysigma in (0.1, 1.0, 3.0):
                gaussFunc.setParameters((xsigma, ysigma))
                # compute array of function values and normalize
                for row in range(kernel.getHeight()):
                    y = row - kernel.getCtrY()
                    for col in range(kernel.getWidth()):
                        x = col - kernel.getCtrX()
                        fArr[col, row] = gaussFunc(x, y)
                fArr /= fArr.sum()
                
                kernel.setKernelParameters((xsigma, ysigma))
                kImage = afwImage.ImageD(kernel.getDimensions())
                kernel.computeImage(kImage, True)
                
                kArr = imTestUtils.arrayFromImage(kImage)
                if not numpy.allclose(fArr, kArr):
                    self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" % \
                        (kernel.__class__.__name__, kArr, fArr, xsigma, ysigma))
    
    def testDeltaFunctionKernel(self):
        """Test DeltaFunctionKernel
        """
        for kWidth in range(1, 4):
            for kHeight in range(1, 4):
                for activeCol in range(kWidth):
                    for activeRow in range(kHeight):
                        kernel = afwMath.DeltaFunctionKernel(kWidth, kHeight,
                                                             afwImage.PointI(activeCol, activeRow))
                        kImage = afwImage.ImageD(kernel.getDimensions())
                        kSum = kernel.computeImage(kImage, False)
                        self.assertEqual(kSum, 1.0)
                        kArr = imTestUtils.arrayFromImage(kImage)
                        self.assertEqual(kArr[activeCol, activeRow], 1.0)
                        kArr[activeCol, activeRow] = 0.0
                        self.assertEqual(kArr.sum(), 0.0)
                utilsTests.assertRaisesLsstCpp(self, pexExcept.InvalidParameterException,
                    afwMath.DeltaFunctionKernel, 0, kHeight, afwImage.PointI(kWidth, kHeight))
                utilsTests.assertRaisesLsstCpp(self, pexExcept.InvalidParameterException,
                    afwMath.DeltaFunctionKernel, kWidth, 0, afwImage.PointI(kWidth, kHeight))
                            
        kernel = afwMath.DeltaFunctionKernel(kWidth, kHeight, afwImage.PointI(1, 1))
        self.basicTests(kernel, 0)

    def testSeparableKernel(self):
        """Test SeparableKernel using a Gaussian function
        """
        kWidth = 5
        kHeight = 8

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        kernel = afwMath.SeparableKernel(kWidth, kHeight, gaussFunc1, gaussFunc1)
        self.basicTests(kernel, 2)
        fArr = numpy.zeros(shape=[kernel.getWidth(), kernel.getHeight()], dtype=float)
        gArr = numpy.zeros(shape=[kernel.getWidth(), kernel.getHeight()], dtype=float)
        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0)
        for xsigma in (0.1, 1.0, 3.0):
            gaussFunc1.setParameters((xsigma,))
            for ysigma in (0.1, 1.0, 3.0):
                gaussFunc.setParameters((xsigma, ysigma))
                # compute array of function values and normalize
                for row in range(kernel.getHeight()):
                    y = row - kernel.getCtrY()
                    for col in range(kernel.getWidth()):
                        x = col - kernel.getCtrX()
                        fArr[col, row] = gaussFunc(x, y)
                fArr /= fArr.sum()
                
                kernel.setKernelParameters((xsigma, ysigma))
                kImage = afwImage.ImageD(kernel.getDimensions())
                kernel.computeImage(kImage, True)
                kArr = imTestUtils.arrayFromImage(kImage)
                if not numpy.allclose(fArr, kArr):
                    self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" % \
                        (kernel.__class__.__name__, kArr, fArr, xsigma, ysigma))
    
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
                kernel = afwMath.DeltaFunctionKernel(kWidth, kHeight, afwImage.PointI(col, row))
                basisImage = afwImage.ImageD(kernel.getDimensions())
                kernel.computeImage(basisImage, True)
                basisImArrList.append(imTestUtils.arrayFromImage(basisImage))
                kVec.append(kernel)

        kParams = [0.0]*len(kVec)
        kernel = afwMath.LinearCombinationKernel(kVec, kParams)
        self.basicTests(kernel, len(kParams))
        for ii in range(len(kVec)):
            kParams = [0.0]*len(kVec)
            kParams[ii] = 1.0
            kernel.setKernelParameters(kParams)
            kIm = afwImage.ImageD(kernel.getDimensions())
            kernel.computeImage(kIm, True)
            kImArr = imTestUtils.arrayFromImage(kIm)
            if not numpy.allclose(kImArr, basisImArrList[ii]):
                self.fail("%s = %s != %s for the %s'th basis kernel" % \
                    (kernel.__class__.__name__, kImArr, basisImArrList[ii], ii))

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
            basisImage = imTestUtils.imageFromArray(basisImArr, retType=afwImage.ImageD)
            kernel = afwMath.FixedKernel(basisImage)
            kVec.append(kernel)

        # create spatially varying linear combination kernel
        spFunc = afwMath.PolynomialFunction2D(1)
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        
        kernel = afwMath.LinearCombinationKernel(kVec, spFunc)
        self.basicTests(kernel, 2, 3)
        kernel.setSpatialParameters(sParams)
        kImage = afwImage.ImageD(kWidth, kHeight)
        for colPos, rowPos, coeff0, coeff1 in [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5, 0.5),
        ]:
            kernel.computeImage(kImage, False, colPos, rowPos)
            kImArr = imTestUtils.arrayFromImage(kImage)
            refKImArr = (basisImArrList[0] * coeff0) + (basisImArrList[1] * coeff1)
            if not numpy.allclose(kImArr, refKImArr):
                self.fail("%s = %s != %s at colPos=%s, rowPos=%s" % \
                    (kernel.__class__.__name__, kImArr, refKImArr, colPos, rowPos))
    
    def testSetCtr(self):
        """Test setCtrCol/Row"""
        kWidth = 3
        kHeight = 4

        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0)
        kernel = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc)
        for xCtr in range(kWidth):
            kernel.setCtrX(xCtr)
            for yCtr in range(kHeight):
                kernel.setCtrY(yCtr)
                self.assertEqual(kernel.getCtrX(), xCtr)
                self.assertEqual(kernel.getCtrY(), yCtr)

    def basicTests(self, kernel, nKernelParams, nSpatialParams=0):
        """Basic tests of a kernel"""
        self.assert_(kernel.getNSpatialParameters() == nSpatialParams)
        self.assert_(kernel.getNKernelParameters() == nKernelParams)
        if nSpatialParams == 0:
            self.assert_(not kernel.isSpatiallyVarying())
            for ii in range(nKernelParams+5):
                utilsTests.assertRaisesLsstCpp(self, pexExcept.InvalidParameterException,
                    kernel.getSpatialFunction, ii)
        else:
            self.assert_(kernel.isSpatiallyVarying())
            for ii in range(nKernelParams):
                kernel.getSpatialFunction(ii)
            for ii in range(nKernelParams, nKernelParams+5):
                utilsTests.assertRaisesLsstCpp(self, pexExcept.InvalidParameterException,
                    kernel.getSpatialFunction, ii)
        for nsp in range(nSpatialParams + 2):
            spatialParamsForOneKernel = (1.0,)*nsp
            for nkp in range(nKernelParams + 2):
                spatialParams = (spatialParamsForOneKernel,)*nkp
                if ((nkp == nKernelParams) and ((nsp == nSpatialParams) or (nkp == 0))):
                    kernel.setSpatialParameters(spatialParams)
                    self.assert_(numpy.alltrue(numpy.equal(kernel.getSpatialParameters(), spatialParams)))
                else:
                    try:
                        kernel.setSpatialParameters(spatialParams)
                        self.fail("test failed: spatialParams=%s" % (spatialParams,))
                    except pexExcept.LsstCppException:
                        pass
#                     utilsTests.assertRaisesLsstCpp(self, pexExcept.InvalidParameterException,
#                         kernel.setSpatialParameters, spatialParams)
        
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

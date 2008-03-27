import os
import math
import pdb                          # we may want to say pdb.set_trace()
import unittest

import numpy

import lsst.daf.tests as dafTests
import lsst.daf.utils as dafUtils
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.image.testUtils as imTestUtils

verbosity = 0 # increase to see trace
dafUtils.Trace_setVerbosity("lsst.afw", verbosity)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class KernelTestCase(unittest.TestCase):
    """A test case for Kernels"""
    def testFixedKernel(self):
        """Test FixedKernel using a ramp function
        """
        kCols = 5
        kRows = 6
        
        inArr = numpy.arange(kCols * kRows, dtype=float)
        inArr.shape = [kCols, kRows]

        inImage = afwImage.ImageD(kCols, kRows)
        for row in range(inImage.getRows()):
            for col in range(inImage.getCols()):
                inImage.set(col, row, inArr[col, row])
        
        fixedKernel = afw.FixedKernel(inImage);
        outImage = fixedKernel.computeNewImage(0.0, 0.0, False)[0]
        outArr = imTestUtils.arrayFromImage(outImage)
        if not numpy.allclose(inArr, outArr):
            self.fail("%s = %s != %s (not normalized)" % \
                (k.__class__.__name__, inArr, outArr))
        
        normInArr = inArr / inArr.sum()
        normOutImage = fixedKernel.computeNewImage(0.0, 0.0, True)[0]
        normOutArr = imTestUtils.arrayFromImage(normOutImage)
        if not numpy.allclose(normOutArr, normInArr):
            self.fail("%s = %s != %s (normalized)" % \
                (k.__class__.__name__, normInArr, normOutArr))

    def testGaussianKernel(self):
        """Test AnalyticKernel using a Gaussian function
        """
        kCols = 5
        kRows = 8

        fPtr =  afwMath.Function2DPtr(afw.GaussianFunction2D(1.0, 1.0))
        k = afw.AnalyticKernel(fPtr, kCols, kRows)
        fArr = numpy.zeros(shape=[k.getCols(), k.getRows()], dtype=float)
        for xsigma in (0.1, 1.0, 3.0):
            for ysigma in (0.1, 1.0, 3.0):
                fPtr.setParameters((xsigma, ysigma))
                # compute array of function values and normalize
                for row in range(k.getRows()):
                    y = row - k.getCtrRow()
                    for col in range(k.getCols()):
                        x = col - k.getCtrCol()
                        fArr[col, row] = fPtr(x, y)
                fArr /= fArr.sum()
                
                k.setKernelParameters((xsigma, ysigma))
                kImage = k.computeNewImage(0.0, 0.0, True)[0]
                kArr = imTestUtils.arrayFromImage(kImage)
                if not numpy.allclose(fArr, kArr):
                    self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" % \
                        (k.__class__.__name__, kArr, fArr, xsigma, ysigma))
        
    def testLinearCombinationKernel(self):
        """Test LinearCombinationKernel using a set of delta basis functions
        """
        kCols = 3
        kRows = 2
        
        # create list of kernels
        basisImArrList = []
        kVec = afwMath.KernelListD()
        ctrCol = (kCols - 1) // 2
        ctrRow = (kRows - 1) // 2
        for row in range(kRows):
            y = float(row - ctrRow)
            for col in range(kCols):
                x = float(col - ctrCol)
                fPtr = afwMath.Function2DPtr(afw.IntegerDeltaFunction2D(x, y))
                kPtr = afwMath.KernelPtr(afw.AnalyticKernel(fPtr, kCols, kRows))
                basisImage = kPtr.computeNewImage()[0]
                basisImArrList.append(imTestUtils.arrayFromImage(basisImage))
                kVec.append(kPtr)
        
        kParams = [0.0]*len(kVec)
        k = afw.LinearCombinationKernel(kVec, kParams)
        for ii in range(len(kVec)):
            kParams = [0.0]*len(kVec)
            kParams[ii] = 1.0
            k.setKernelParameters(kParams)
            kIm = k.computeNewImage()[0]
            kImArr = imTestUtils.arrayFromImage(kIm)
            if not numpy.allclose(kImArr, basisImArrList[ii]):
                self.fail("%s = %s != %s for the %s'th basis kernel" % \
                    (k.__class__.__name__, kImArr, basisImArrList[ii], ii))

    def testSVLinearCombinationKernel(self):
        """Test a spatially varying LinearCombinationKernel
        """
        kCols = 3
        kRows = 2

        # create image arrays for the basis kernels
        basisImArrList = []
        imArr = numpy.zeros((kCols, kRows), dtype=float)
        imArr += 0.1
        imArr[kCols//2, :] = 0.9
        basisImArrList.append(imArr)
        imArr = numpy.zeros((kCols, kRows), dtype=float)
        imArr += 0.2
        imArr[:, kRows//2] = 0.8
        basisImArrList.append(imArr)
        
        # create a list of basis kernels from the images
        kVec = afwMath.KernelListD()
        for basisImArr in basisImArrList:
            basisImage = imTestUtils.imageFromArray(basisImArr)
            kPtr = afwMath.KernelPtr(afw.FixedKernel(basisImage))
            kVec.append(kPtr)

        # create spatially varying linear combination kernel
        sFuncPtr =  afwMath.Function2DPtr(afw.PolynomialFunction2D(1))
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        
        k = afw.LinearCombinationKernel(kVec, sFuncPtr, sParams)
        kImage = afwImage.ImageD(kCols, kRows)
        for colPos, rowPos, coeff0, coeff1 in [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5, 0.5),
        ]:
            k.computeImage(kImage, colPos, rowPos, False)
            kImArr = imTestUtils.arrayFromImage(kImage)
            refKImArr = (basisImArrList[0] * coeff0) + (basisImArrList[1] * coeff1)
            if not numpy.allclose(kImArr, refKImArr):
                self.fail("%s = %s != %s at colPos=%s, rowPos=%s" % \
                    (k.__class__.__name__, kImArr, refKImArr, colPos, rowPos))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    dafTests.init()

    suites = []
    suites += unittest.makeSuite(KernelTestCase)
    suites += unittest.makeSuite(dafTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    dafTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

"""Test lsst.afwMath.convolve

The convolve function is overloaded in two flavors:
- in-place convolve: user supplies the output image as an argument
- new-image convolve: the convolve function returns the convolved image

All tests use the new-image version unless otherwise noted.
"""
import os
import math
import pdb                          # we may want to say pdb.set_trace()
import sys
import unittest

import numpy

import eups
import lsst.utils.tests as utilsTest
import lsst.pex.logging as pexLog
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.image.testUtils as imTestUtils

Verbosity = 0 # increase to see trace
pexLog.Trace_setVerbosity("lsst.afw", Verbosity)

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests")
InputImagePath = os.path.join(dataDir, "871034p_1_MI_img.fits")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def refConvolve(image, kernel, doNormalize):
    """Reference code to convolve a kernel with image data.
    
    Does NOT normalize the kernel.

    Warning: slow (especially for spatially varying kernels).
    
    Inputs:
    - image: image to be convolved (a numpy array)
    - kernel: lsst::afw::Core.Kernel object
    - doNormalize: normalize the kernel
    
    Border pixels (pixels too close to the edge to compute) are copied from the input.
    """
    # copy input data, handling the outer border and edge bit
    retImage = image.copy()
    
    kCols = kernel.getCols()
    kRows = kernel.getRows()
    numCols = image.shape[0] + 1 - kCols
    numRows = image.shape[1] + 1 - kRows
    if numCols < 0 or numRows < 0:
        raise RuntimeError("image must be larger than kernel in both dimensions")
    colRange = range(numCols)

    isSpatiallyVarying = kernel.isSpatiallyVarying()
    if not isSpatiallyVarying:
        kImArr = imTestUtils.arrayFromImage(kernel.computeNewImage(doNormalize)[0])
    else:
        kImage = afwImage.ImageD(kCols, kRows)

    retRow = kernel.getCtrRow()
    for inRowBeg in range(numRows):
        inRowEnd = inRowBeg + kRows
        retCol = kernel.getCtrCol()
        if isSpatiallyVarying:
            rowPos = afwImage.indexToPosition(retRow)
        for inColBeg in colRange:
            if isSpatiallyVarying:
                colPos = afwImage.indexToPosition(retCol)
                kernel.computeImage(kImage, doNormalize, colPos, rowPos)
                kImArr = imTestUtils.arrayFromImage(kImage)
            inColEnd = inColBeg + kCols
            subImage = image[inColBeg:inColEnd, inRowBeg:inRowEnd]
            retImage[retCol, retRow] = numpy.add.reduce((kImArr * subImage).flat)
            

            retCol += 1
        retRow += 1
    return retImage

def makeGaussianKernelVec(kCols, kRows):
    """Create a afwImage.VectorKernel of gaussian kernels.

    This is useful for constructing a LinearCombinationKernel.
    """
    xySigmaList = [
        (1.5, 1.5),
        (1.5, 2.5),
        (2.5, 1.5),
    ]
    kVec = afwMath.KernelListD()
    for xSigma, ySigma in xySigmaList:
        kFunc = afwMath.GaussianFunction2D(1.5, 2.5)
        basisKernelPtr = afwMath.KernelPtr(afwMath.AnalyticKernel(kFunc, kCols, kRows))
        kVec.append(basisKernelPtr)
    return kVec

class ConvolveTestCase(unittest.TestCase):
    def testUnityConvolution(self):
        """Verify that convolution with a centered delta function reproduces the original.
        """
        imCols = 45
        imRows = 55
        
        fullImage = afwImage.ImageF()
        fullImage.readFits(InputImagePath)
        
        # pick a small piece of the image to save time
        bbox = afwImage.BBox2i(50, 50, imCols, imRows)
        subImagePtr = fullImage.getSubImage(bbox)
        inImage = subImagePtr.get()
        inImage.this.disown()
        
        # create a delta function kernel that has 1,1 in the center
        kFunc = afwMath.IntegerDeltaFunction2D(0.0, 0.0)
        k = afwMath.AnalyticKernel(kFunc, 3, 3)
        
        cnvImage = afwMath.convolve(inImage, k, True)
    
        origImageArr = imTestUtils.arrayFromImage(inImage)
        cnvImageArr = imTestUtils.arrayFromImage(cnvImage)
        for name, ind in (("image", 0), ("variance", 1)): # , ("mask", 2)):
            if not numpy.allclose(origImageArr, cnvImageArr):
                self.fail("Convolved image does not match reference")
 
    def testSpatiallyInvariantInPlaceConvolve(self):
        """Test in-place version of convolve with a spatially invariant Gaussian function
        """
        kCols = 6
        kRows = 7
        imCols = 45
        imRows = 55
        doNormalize = False

        kFunc =  afwMath.GaussianFunction2D(1.5, 2.5)
        k = afwMath.AnalyticKernel(kFunc, kCols, kRows)
        
        fullImage = afwImage.ImageF()
        fullImage.readFits(InputImagePath)
        
        # pick a small piece of the image to save time
        bbox = afwImage.BBox2i(50, 50, imCols, imRows)
        subImagePtr = fullImage.getSubImage(bbox)
        inImage = subImagePtr.get()
        inImage.this.disown()
        
        cnvImage = afwImage.ImageF(imCols, imRows)
        for doNormalize in (False, True):
            afwMath.convolve(cnvImage, inImage, k, doNormalize)
            cnvImageArr = imTestUtils.arrayFromImage(cnvImage)

            inImageArr = imTestUtils.arrayFromImage(inImage)
            refCnvImageArr = refConvolve(inImageArr, k, doNormalize)
    
            if not numpy.allclose(cnvImageArr, refCnvImageArr):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)
                
    
    def testSpatiallyInvariantConvolve(self):
        """Test convolution with a spatially invariant Gaussian function
        """
        kCols = 7
        kRows = 6
        imCols = 55
        imRows = 45

        kFunc =  afwMath.GaussianFunction2D(1.5, 2.5)
        k = afwMath.AnalyticKernel(kFunc, kCols, kRows)
        
        fullImage = afwImage.ImageF()
        fullImage.readFits(InputImagePath)
        
        # pick a small piece of the image to save time
        bbox = afwImage.BBox2i(50, 50, imCols, imRows)
        subImagePtr = fullImage.getSubImage(bbox)
        inImage = subImagePtr.get()
        inImage.this.disown()
        
        for doNormalize in (False, True):
            cnvImage = afwMath.convolve(inImage, k, doNormalize)
            cnvImageArr = imTestUtils.arrayFromImage(cnvImage)
    
            inImageArr = imTestUtils.arrayFromImage(inImage)
            refCnvImageArr = refConvolve(inImageArr, k, doNormalize)
    
            if not numpy.allclose(cnvImageArr, refCnvImageArr):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)

    def testSpatiallyVaryingInPlaceConvolve(self):
        """Test in-place convolution with a spatially varying Gaussian function
        """
        kCols = 7
        kRows = 6
        imCols = 55
        imRows = 45

        # create spatially varying linear combination kernel
        sFunc = afwMath.PolynomialFunction2D(1)
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, 1.0 / imCols, 0.0),
            (1.0, 0.0,  1.0 / imRows),
        )
   
        kFunc =  afwMath.GaussianFunction2D(1.0, 1.0)
        k = afwMath.AnalyticKernel(kFunc, kCols, kRows, sFunc)
        k.setSpatialParameters(sParams)
        
        fullImage = afwImage.ImageF()
        fullImage.readFits(InputImagePath)
        
        # pick a small piece of the image to save time
        bbox = afwImage.BBox2i(50, 50, imCols, imRows)
        subImagePtr = fullImage.getSubImage(bbox)
        inImage = subImagePtr.get()
        inImage.this.disown()
        
        cnvImage = afwImage.ImageF(imCols, imRows)
        for doNormalize in (False, True):
            afwMath.convolve(cnvImage, inImage, k, doNormalize)
            cnvImageArr = imTestUtils.arrayFromImage(cnvImage)
    
            inImageArr = imTestUtils.arrayFromImage(inImage)
            refCnvImageArr= refConvolve(inImageArr, k, doNormalize)
    
            if not numpy.allclose(cnvImageArr, refCnvImageArr):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)

    def testSpatiallyVaryingSeparableInPlaceConvolve(self):
        """Test in-place separable convolution with a spatially varying Gaussian function
        """
        sys.stderr.write("Test convolution with SeparableKernel\n")
        kCols = 7
        kRows = 6
        imCols = 55
        imRows = 45

        # create spatially varying linear combination kernel
        sFunc = afwMath.PolynomialFunction2D(1)
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, 1.0 / imCols, 0.0),
            (1.0, 0.0,  1.0 / imRows),
        )

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        gaussFunc2 = afwMath.GaussianFunction2D(1.0, 1.0)
        separableKernel = afwMath.SeparableKernel(gaussFunc1, gaussFunc1, kCols, kRows, sFunc)
        analyticKernel = afwMath.AnalyticKernel(gaussFunc2, kCols, kRows, sFunc)
        separableKernel.setSpatialParameters(sParams)
        analyticKernel.setSpatialParameters(sParams)
        
        fullImage = afwImage.ImageF()
        fullImage.readFits(InputImagePath)
        
        # pick a small piece of the image to save time
        bbox = afwImage.BBox2i(50, 50, imCols, imRows)
        subImagePtr = fullImage.getSubImage(bbox)
        inImage = subImagePtr.get()
        inImage.this.disown()
        
        isFirst = True
        cnvImage = afwImage.ImageF(imCols, imRows)
        for doNormalize in (False, True):
            if isFirst and Verbosity < 3:
                pexLog.Trace_setVerbosity("lsst.afw", 3)
            afwMath.convolve(cnvImage, inImage, separableKernel, doNormalize)
            if isFirst:
                pexLog.Trace_setVerbosity("lsst.afw", Verbosity)
                isFirst = False
            cnvImageArr = imTestUtils.arrayFromImage(cnvImage)
    
            inImageArr = imTestUtils.arrayFromImage(inImage)
            refCnvImageArr = refConvolve(inImageArr, analyticKernel, doNormalize)
    
            if not numpy.allclose(cnvImageArr, refCnvImageArr):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)
    
    def testDeltaConvolve(self):
        """Test convolution with various delta function kernels using optimized code
        """
        sys.stderr.write("Test convolution with DeltaFunctionKernel\n")
        imCols = 20
        imRows = 12
        doNormalize = True

        fullImage = afwImage.ImageF()
        fullImage.readFits(InputImagePath)
        
        # pick a small piece of the image to save time
        bbox = afwImage.BBox2i(50, 50, imCols, imRows)
        subImagePtr = fullImage.getSubImage(bbox)
        inImage = subImagePtr.get()
        inImage.this.disown()
        
        isFirst = True
        for kCols in range(1, 4):
            for kRows in range(1, 4):
                for activeCol in range(kCols):
                    for activeRow in range(kRows):
                        kernel = afwMath.DeltaFunctionKernel(activeCol, activeRow, kCols, kRows)
                        
                        if isFirst and Verbosity < 3:
                            pexLog.Trace_setVerbosity("lsst.afw", 3)
                        refCnvImage = afwMath.convolve(inImage, kernel, doNormalize)
                        if isFirst:
                            pexLog.Trace_setVerbosity("lsst.afw", Verbosity)
                            isFirst = False
                        refCnvImageArr= imTestUtils.arrayFromImage(refCnvImage)
                
                        inImageArr = imTestUtils.arrayFromImage(inImage)
                        ref2CnvImageArr = refConvolve(inImageArr, kernel, doNormalize)
                
                        if not numpy.allclose(refCnvImageArr, ref2CnvImageArr):
                            print "kCols=%s, kRows=%s, refCnvImageArr=%r, ref2CnvImageArr=%r" % (kCols, kRows, refCnvImageArr, ref2CnvImageArr)
                            self.fail("Image from afwMath.convolve does not match image from refConvolve")
        

    def testConvolveLinear(self):
        """Test convolution with a spatially varying LinearCombinationKernel
        by comparing the results of convolveLinear to afwMath.convolve or refConvolve,
        depending on the value of compareToFwConvolve.
        """
        kCols = 5
        kRows = 5
        imCols = 50
        imRows = 55
        doNormalize = False # must be false because convolveLinear cannot normalize

        fullImage = afwImage.ImageF()
        fullImage.readFits(InputImagePath)
        
        # pick a small piece of the image to save time
        bbox = afwImage.BBox2i(50, 50, imCols, imRows)
        subImagePtr = fullImage.getSubImage(bbox)
        inImage = subImagePtr.get()
        inImage.this.disown()

        # create spatially varying linear combination kernel
        sFunc = afwMath.PolynomialFunction2D(1)
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, -0.5 / imCols, -0.5 / imRows),
            (0.0,  1.0 / imCols,  0.0 / imRows),
            (0.0,  0.0 / imCols,  1.0 / imRows),
        )
        
        kVec = makeGaussianKernelVec(kCols, kRows)
        lcKernel = afwMath.LinearCombinationKernel(kVec, sFunc)
        lcKernel.setSpatialParameters(sParams)

        refCnvImage = afwMath.convolve(inImage, lcKernel, doNormalize)
        refCnvImageArr = imTestUtils.arrayFromImage(refCnvImage)

        inImageArr = imTestUtils.arrayFromImage(inImage)
        ref2CnvImageArr = refConvolve(inImageArr, lcKernel, doNormalize)

        if not numpy.allclose(refCnvImageArr, ref2CnvImageArr):
            self.fail("Image from afwMath.convolve does not match image from refConvolve")

        # compute twice, to be sure cnvImage is properly reset
        cnvImage = afwImage.ImageF(imCols, imRows)
        for ii in range(2):        
            afwMath.convolveLinear(cnvImage, inImage, lcKernel)
            cnvImageArr = imTestUtils.arrayFromImage(cnvImage)
    
            if not numpy.allclose(cnvImageArr, ref2CnvImageArr):
                self.fail("Image from afwMath.convolveLinear does not match image from refConvolve in iter %d" % ii)

    def testConvolveLinearNewImage(self):
        """Test variant of convolveLinear that returns a new image
        """
        kCols = 5
        kRows = 5
        imCols = 50
        imRows = 55
        doNormalize = False # must be false because convolveLinear cannot normalize

        fullImage = afwImage.ImageF()
        fullImage.readFits(InputImagePath)
        
        # pick a small piece of the image to save time
        bbox = afwImage.BBox2i(50, 50, imCols, imRows)
        subImagePtr = fullImage.getSubImage(bbox)
        inImage = subImagePtr.get()
        inImage.this.disown()

        # create spatially varying linear combination kernel
        sFunc = afwMath.PolynomialFunction2D(1)
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, -0.5 / imCols, -0.5 / imRows),
            (0.0,  1.0 / imCols,  0.0 / imRows),
            (0.0,  0.0 / imCols,  1.0 / imRows),
        )
        
        kVec = makeGaussianKernelVec(kCols, kRows)
        lcKernel = afwMath.LinearCombinationKernel(kVec, sFunc)
        lcKernel.setSpatialParameters(sParams)

        refCnvImage = afwMath.convolve(inImage, lcKernel, doNormalize)
        refCnvImageArr = imTestUtils.arrayFromImage(refCnvImage)

        inImageArr = imTestUtils.arrayFromImage(inImage)
        ref2CnvImageArr = refConvolve(inImageArr, lcKernel, doNormalize)

        if not numpy.allclose(refCnvImageArr, ref2CnvImageArr):
            self.fail("Image from afwMath.convolve does not match image from refConvolve")

        # compute twice, to be sure cnvImage is properly reset
        for ii in range(2):        
            cnvImage = afwMath.convolveLinear(inImage, lcKernel)
            cnvImageArr = imTestUtils.arrayFromImage(cnvImage)
    
            if not numpy.allclose(cnvImageArr, ref2CnvImageArr):
                self.fail("Image from afwMath.convolveLinear does not match image from refConvolve in iter %d" % ii)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTest.init()

    suites = []
    suites += unittest.makeSuite(ConvolveTestCase)
    suites += unittest.makeSuite(utilsTest.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTest.run(suite(), exit)

if __name__ == "__main__":
    run(True)

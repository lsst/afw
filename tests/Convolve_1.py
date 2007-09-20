"""Test lsst.fw.Core.fwLib.convolve

The convolve function is overloaded in two flavors:
- in-place convolve: user supplies the output image as an argument
- new-image convolve: the convolve function returns the convolved image

All tests use the new-image version unless otherwise noted.
"""
import os
import math
import pdb                          # we may want to say pdb.set_trace()
import unittest

import numpy

import lsst.fw.Core.fwLib as fw
import lsst.fw.Core.imageTestUtils as iUtils
import lsst.mwi.tests as tests
import lsst.mwi.utils as mwiu


try:
    type(verbose)
except NameError:
    verbose = 0
    mwiu.Trace_setVerbosity("fw.kernel", verbose)

InputMaskedImageName = "871034p_1_MI"

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def refConvolve(imVarMask, kernel, threshold=0, edgeBit=-1):
    """Reference code to convolve a kernel with masked image data.

    Warning: slow (especially for spatially varying kernels).
    
    Inputs:
    - imVarMask: (image, variance, mask) numpy arrays
    - kernel: lsst::fw::Core.Kernel object
    - threshold: kernel pixels above this threshold are used to or in mask bits
    - edgeBit: this bit is set in the output mask for border pixels; no bit set if < 0
    
    Border pixels (pixels too close to the edge to compute) are copied from the input,
    and if edgeBit >= 0 then border mask pixels have the edgeBit bit set
    """
    image, variance, mask = imVarMask
    
    # copy input data, handling the outer border and edge bit
    retImage = image.copy()
    retVariance = variance.copy()
    retMask = mask.copy()
    if (edgeBit >= 0):
         retMask |= 2 ** edgeBit
    
    kCols = kernel.getCols()
    kRows = kernel.getRows()
    numCols = image.shape[0] + 1 - kCols
    numRows = image.shape[1] + 1 - kRows
    if numCols < 0 or numRows < 0:
        raise RuntimeError("image must be larger than kernel in both dimensions")
    colRange = range(numCols)

    isSpatiallyVarying = kernel.isSpatiallyVarying()
    if not isSpatiallyVarying:
        kImArr = iUtils.arrayFromImage(kernel.computeNewImage()[0])
    else:
        kImage = fw.ImageD(kCols, kRows)

    retRow = kernel.getCtrRow()
    for inRowBeg in range(numRows):
        inRowEnd = inRowBeg + kRows
        retCol = kernel.getCtrCol()
        if isSpatiallyVarying:
            rowPos = fw.indexToPosition(retRow)
        for inColBeg in colRange:
            if isSpatiallyVarying:
                colPos = fw.indexToPosition(retCol)
                kernel.computeImage(kImage, colPos, rowPos)
                kImArr = iUtils.arrayFromImage(kImage)
            inColEnd = inColBeg + kCols
            subImage = image[inColBeg:inColEnd, inRowBeg:inRowEnd]
            subVariance = variance[inColBeg:inColEnd, inRowBeg:inRowEnd]
            subMask = mask[inColBeg:inColEnd, inRowBeg:inRowEnd]
            retImage[retCol, retRow] = numpy.add.reduce((kImArr * subImage).flat)
            retVariance[retCol, retRow] = numpy.add.reduce((kImArr * kImArr * subVariance).flat)
            retMask[retCol, retRow] = numpy.bitwise_or.reduce(((kImArr > threshold) * subMask).flat)

            retCol += 1
        retRow += 1
    return (retImage, retVariance, retMask)

def makeGaussianKernelVec(kCols, kRows):
    """Create a fw.vectorKernelPtrD of gaussian kernels.

    This is useful for constructing a LinearCombinationKernel.
    """
    xySigmaList = [
        (1.5, 1.5),
        (1.5, 2.5),
        (2.5, 1.5),
    ]
    kVec = fw.vectorKernelPtrD()
    for xSigma, ySigma in xySigmaList:
        f = fw.GaussianFunction2D(1.5, 2.5)
        fPtr =  fw.Function2PtrTypeD(f)
        f.this.disown() # Only the shared pointer now owns f
        basisKernel = fw.AnalyticKernelD(fPtr, kCols, kRows)
        basisKernelPtr = fw.KernelPtrTypeD(basisKernel)
        basisKernel.this.disown() # only the shared pointer now owns basisKernel
        kVec.append(basisKernelPtr)
    return kVec

class ConvolveTestCase(unittest.TestCase):
    def testUnityConvolution(self):
        """Verify that convolution with a centered delta function reproduces the original"""
        threshold = 0.0
        edgeBit = -1
        
        currDir = os.path.abspath(os.path.dirname(__file__))
        inFilePath = os.path.join(currDir, "data", "small")
    
        maskedImage = fw.MaskedImageF()
        maskedImage.readFits(inFilePath)
        
        # create a delta function kernel that has 1,1 in the center
        f = fw.IntegerDeltaFunction2D(0.0, 0.0)
        fPtr =  fw.Function2PtrTypeD(f)
        f.this.disown() # Only the shared pointer now owns f
        k = fw.AnalyticKernelD(fPtr, 3, 3)
        
        cnvMaskedImage = fw.convolve(maskedImage, k, threshold, edgeBit)
    
        origImVarMaskArrays = iUtils.arraysFromMaskedImage(maskedImage)
        cnvImVarMaskArrays = iUtils.arraysFromMaskedImage(cnvMaskedImage)
        for name, ind in (("image", 0), ("variance", 1), ("mask", 2)):
            if not numpy.allclose(origImVarMaskArrays[ind], cnvImVarMaskArrays[ind]):
                self.fail("Convolved %s does not match reference" % (name,))

    def testSpatiallyInvariantInPlaceConvolve(self):
        """Test in-place version of convolve with a spatially invariant Gaussian function
        """
        kCols = 7
        kRows = 7
        threshold = 0.0
        edgeBit = 7

        f = fw.GaussianFunction2D(1.5, 2.5)
        fPtr =  fw.Function2PtrTypeD(f)
        f.this.disown() # Only the shared pointer now owns f
        k = fw.AnalyticKernelD(fPtr, kCols, kRows)
        fArr = numpy.zeros(shape=[k.getCols(), k.getRows()], dtype=float)
        
        currDir = os.path.abspath(os.path.dirname(__file__))
        inFilePath = os.path.join(currDir, "data", InputMaskedImageName)
        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(inFilePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(0, 0, 45, 55)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()

        cnvMaskedImage = fw.MaskedImageF(maskedImage.getCols(), maskedImage.getRows())
        fw.convolve(cnvMaskedImage, maskedImage, k, threshold, edgeBit)
        cnvImage, cnvVariance, cnvMask = iUtils.arraysFromMaskedImage(cnvMaskedImage)

        imVarMask = iUtils.arraysFromMaskedImage(maskedImage)
        refCnvImage, refCnvVariance, refCnvMask = \
            refConvolve(imVarMask, k, threshold, edgeBit)

        if not numpy.allclose(cnvImage, refCnvImage):
            self.fail("Convolved image does not match reference")
        if not numpy.allclose(cnvVariance, refCnvVariance):
            self.fail("Convolved variance does not match reference")
        if not numpy.allclose(cnvMask, refCnvMask):
            self.fail("Convolved mask does not match reference")
    
    def testSpatiallyInvariantNewConvolve(self):
        """Test convolution with a spatially invariant Gaussian function
        """
        kCols = 7
        kRows = 7
        threshold = 0.0
        edgeBit = 7

        f = fw.GaussianFunction2D(1.5, 2.5)
        fPtr =  fw.Function2PtrTypeD(f)
        f.this.disown() # Only the shared pointer now owns f
        k = fw.AnalyticKernelD(fPtr, kCols, kRows)
        fArr = numpy.zeros(shape=[k.getCols(), k.getRows()], dtype=float)
        
        currDir = os.path.abspath(os.path.dirname(__file__))
        inFilePath = os.path.join(currDir, "data", InputMaskedImageName)
        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(inFilePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(0, 0, 45, 55)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()
        
        cnvMaskedImage = fw.convolve(maskedImage, k, threshold, edgeBit)
        cnvImage, cnvVariance, cnvMask = iUtils.arraysFromMaskedImage(cnvMaskedImage)

        imVarMask = iUtils.arraysFromMaskedImage(maskedImage)
        refCnvImage, refCnvVariance, refCnvMask = \
            refConvolve(imVarMask, k, threshold, edgeBit)

        if not numpy.allclose(cnvImage, refCnvImage):
            self.fail("Convolved image does not match reference")
        if not numpy.allclose(cnvVariance, refCnvVariance):
            self.fail("Convolved variance does not match reference")
        if not numpy.allclose(cnvMask, refCnvMask):
            self.fail("Convolved mask does not match reference")

    def testSpatiallyInvariantLinearCombinationNewConvolve(self):
        """Test convolution with a spatially invariant LinearCombinationKernel
        """
        kCols = 7
        kRows = 7
        threshold = 0.0
        edgeBit = 7
        imCols = 45
        imRows = 55

        currDir = os.path.abspath(os.path.dirname(__file__))
        inFilePath = os.path.join(currDir, "data", InputMaskedImageName)
        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(inFilePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(0, 0, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()

        kVec = makeGaussianKernelVec(kCols, kRows)
        lcKernel = fw.LinearCombinationKernelD(kVec, (1.0, 1.0, 1.0))
        cnvMaskedImage = fw.convolve(maskedImage, lcKernel, threshold, edgeBit)
        cnvImage, cnvVariance, cnvMask = iUtils.arraysFromMaskedImage(cnvMaskedImage)

        imVarMask = iUtils.arraysFromMaskedImage(maskedImage)
        refCnvImage, refCnvVariance, refCnvMask = \
            refConvolve(imVarMask, lcKernel, threshold, edgeBit)

        if not numpy.allclose(cnvImage, refCnvImage):
            self.fail("Convolved image does not match reference")
        if not numpy.allclose(cnvVariance, refCnvVariance):
            self.fail("Convolved variance does not match reference")
        if not numpy.allclose(cnvMask, refCnvMask):
            self.fail("Convolved mask does not match reference")

    def testConvolveLinear(self):
        """Test convolution with a spatially varying LinearCombinationKernel
        by comparing the results of calling convolve with convolveLinear
        or refConvolve, depending on the value of compareToConvolve.
        """
        compareToConvolve = True
        
        kCols = 7
        kRows = 7
        threshold = 0.0
        edgeBit = 7
        imCols = 45
        imRows = 55

        currDir = os.path.abspath(os.path.dirname(__file__))
        inFilePath = os.path.join(currDir, "data", InputMaskedImageName)
        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(inFilePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(0, 0, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()

        # create spatially varying linear combination kernel
        sFunc = fw.PolynomialFunction2D(1)
        sFuncPtr =  fw.Function2PtrTypeD(sFunc)
        sFunc.this.disown() # Only the shared pointer now owns sFunc
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, -0.5 / imCols, -0.5 / imRows),
            (0.0,  1.0 / imCols,  0.0 / imRows),
            (0.0,  0.0 / imCols,  1.0 / imRows),
        )
        
        kVec = makeGaussianKernelVec(kCols, kRows)
        lcKernel = fw.LinearCombinationKernelD(kVec, sFuncPtr, sParams)
        cnvMaskedImage = fw.convolve(maskedImage, lcKernel, 0.0, edgeBit)
        cnvImage, cnvVariance, cnvMask = iUtils.arraysFromMaskedImage(cnvMaskedImage)
        
        if compareToConvolve:
            refCnvMaskedImage = fw.MaskedImageF(imCols, imRows)
            fw.convolveLinear(cnvMaskedImage, maskedImage, lcKernel, edgeBit)
            refCnvImage, refCnvVariance, refCnvMask = \
                iUtils.arraysFromMaskedImage(cnvMaskedImage)
        else:
            imVarMask = iUtils.arraysFromMaskedImage(maskedImage)
            refCnvImage, refCnvVariance, refCnvMask = \
               refConvolve(imVarMask, lcKernel, threshold, edgeBit)

        if not numpy.allclose(cnvImage, refCnvImage):
            self.fail("Convolved image does not match reference")
        if not numpy.allclose(cnvVariance, refCnvVariance):
            self.fail("Convolved variance does not match reference")
        if not numpy.allclose(cnvMask, refCnvMask):
            self.fail("Convolved mask does not match reference")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(ConvolveTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)

    return unittest.TestSuite(suites)

if __name__ == "__main__":
    tests.run(suite())

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

import eups; dataDir = eups.productDir("fwData")

if not dataDir:
    raise RuntimeError("Must set up fwData to run these tests")
InputMaskedImagePath = os.path.join(dataDir, "871034p_1_MI")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def refConvolve(imVarMask, kernel, threshold=0, edgeBit=-1, doNormalize=True):
    """Reference code to convolve a kernel with masked image data.
    
    Does NOT normalize the kernel.

    Warning: slow (especially for spatially varying kernels).
    
    Inputs:
    - imVarMask: (image, variance, mask) numpy arrays
    - kernel: lsst::fw::Core.Kernel object
    - threshold: kernel pixels above this threshold are used to or in mask bits
    - edgeBit: this bit is set in the output mask for border pixels; no bit set if < 0
    - doNormalize: normalize the kernel
    
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
        kImArr = iUtils.arrayFromImage(kernel.computeNewImage(0, 0, doNormalize)[0])
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
                kernel.computeImage(kImage, colPos, rowPos, doNormalize)
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
        imCols = 45
        imRows = 55
        threshold = 0.0
        edgeBit = -1
        
        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(InputMaskedImagePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(50, 50, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()
        
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
        kCols = 6
        kRows = 7
        imCols = 45
        imRows = 55
        threshold = 0.0
        edgeBit = 7
        doNormalize = False

        f = fw.GaussianFunction2D(1.5, 2.5)
        fPtr =  fw.Function2PtrTypeD(f)
        f.this.disown() # Only the shared pointer now owns f
        k = fw.AnalyticKernelD(fPtr, kCols, kRows)
        
        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(InputMaskedImagePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(50, 50, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()


        cnvMaskedImage = fw.MaskedImageF(imCols, imRows)
        for doNormalize in (False, True):
            fw.convolve(cnvMaskedImage, maskedImage, k, threshold, edgeBit, doNormalize)
            cnvImage, cnvVariance, cnvMask = iUtils.arraysFromMaskedImage(cnvMaskedImage)

            imVarMask = iUtils.arraysFromMaskedImage(maskedImage)
            refCnvImage, refCnvVariance, refCnvMask = \
                refConvolve(imVarMask, k, threshold, edgeBit, doNormalize)
    
            if not numpy.allclose(cnvImage, refCnvImage):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvVariance, refCnvVariance):
                self.fail("Convolved variance does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvMask, refCnvMask):
                self.fail("Convolved mask does not match reference for doNormalize=%s" % doNormalize)
    
    def testSpatiallyInvariantConvolve(self):
        """Test convolution with a spatially invariant Gaussian function
        """
        kCols = 7
        kRows = 6
        imCols = 55
        imRows = 45
        threshold = 0.0
        edgeBit = 7

        f = fw.GaussianFunction2D(1.5, 2.5)
        fPtr =  fw.Function2PtrTypeD(f)
        f.this.disown() # Only the shared pointer now owns f
        k = fw.AnalyticKernelD(fPtr, kCols, kRows)
        
        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(InputMaskedImagePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(50, 50, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()
        
        for doNormalize in (False, True):
            cnvMaskedImage = fw.convolve(maskedImage, k, threshold, edgeBit, doNormalize)
            cnvImage, cnvVariance, cnvMask = iUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            imVarMask = iUtils.arraysFromMaskedImage(maskedImage)
            refCnvImage, refCnvVariance, refCnvMask = \
                refConvolve(imVarMask, k, threshold, edgeBit, doNormalize)
    
            if not numpy.allclose(cnvImage, refCnvImage):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvVariance, refCnvVariance):
                self.fail("Convolved variance does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvMask, refCnvMask):
                self.fail("Convolved mask does not match reference for doNormalize=%s" % doNormalize)

    def testSpatiallyVaryingInPlaceConvolve(self):
        """Test in-place convolution with a spatially varying Gaussian function
        """
        kCols = 7
        kRows = 6
        imCols = 55
        imRows = 45
        threshold = 0.0
        edgeBit = 7

        # create spatially varying linear combination kernel
        sFunc = fw.PolynomialFunction2D(1)
        sFuncPtr =  fw.Function2PtrTypeD(sFunc)
        sFunc.this.disown() # Only the shared pointer now owns sFunc
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, 1.0 / imCols, 0.0),
            (1.0, 0.0,  1.0 / imRows),
        )
   
        f = fw.GaussianFunction2D(1.0, 1.0)
        fPtr =  fw.Function2PtrTypeD(f)
        f.this.disown() # Only the shared pointer now owns f
        k = fw.AnalyticKernelD(fPtr, kCols, kRows, sFuncPtr, sParams)
        
        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(InputMaskedImagePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(50, 50, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()
        
        cnvMaskedImage = fw.MaskedImageF(imCols, imRows)
        for doNormalize in (False, True):
            fw.convolve(cnvMaskedImage, maskedImage, k, threshold, edgeBit, doNormalize)
            cnvImage, cnvVariance, cnvMask = iUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            imVarMask = iUtils.arraysFromMaskedImage(maskedImage)
            refCnvImage, refCnvVariance, refCnvMask = \
                refConvolve(imVarMask, k, threshold, edgeBit, doNormalize)
    
            if not numpy.allclose(cnvImage, refCnvImage):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvVariance, refCnvVariance):
                self.fail("Convolved variance does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvMask, refCnvMask):
                self.fail("Convolved mask does not match reference for doNormalize=%s" % doNormalize)


    def testConvolveLinear(self):
        """Test convolution with a spatially varying LinearCombinationKernel
        by comparing the results of convolveLinear to fw.convolve or refConvolve,
        depending on the value of compareToFwConvolve.
        """
        kCols = 5
        kRows = 5
        edgeBit = 7
        imCols = 50
        imRows = 55
        threshold = 0.0 # must be 0 because convolveLinear only does threshold = 0
        doNormalize = False # must be false because convolveLinear cannot normalize

        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(InputMaskedImagePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(50, 50, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()
        maskedImage.writeFits("Src")

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

        refCnvMaskedImage = fw.convolve(maskedImage, lcKernel, threshold, edgeBit, doNormalize)
        refCnvImage, refCnvVariance, refCnvMask = \
            iUtils.arraysFromMaskedImage(refCnvMaskedImage)

        imVarMask = iUtils.arraysFromMaskedImage(maskedImage)
        ref2CnvImage, ref2CnvVariance, ref2CnvMask = \
           refConvolve(imVarMask, lcKernel, threshold, edgeBit, doNormalize)

        if not numpy.allclose(refCnvImage, ref2CnvImage):
            self.fail("Image from fw.convolve does not match image from refConvolve")
        if not numpy.allclose(refCnvVariance, ref2CnvVariance):
            self.fail("Variance from fw.convolve does not match image from refConvolve")
        if not numpy.allclose(refCnvMask, ref2CnvMask):
            self.fail("Mask from fw.convolve does not match image from refCconvolve")

        # compute twice, to be sure cnvMaskedImage is properly reset
        cnvMaskedImage = fw.MaskedImageF(imCols, imRows)
        for ii in range(2):        
            fw.convolveLinear(cnvMaskedImage, maskedImage, lcKernel, edgeBit)
            cnvImage, cnvVariance, cnvMask = iUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            if not numpy.allclose(cnvImage, ref2CnvImage):
                self.fail("Image from fw.convolveLinear does not match image from refConvolve in iter %d" % ii)
            if not numpy.allclose(cnvVariance, ref2CnvVariance):
                self.fail("Variance from fw.convolveLinear does not match image from refConvolve in iter %d" % ii)
            if not numpy.allclose(cnvMask, ref2CnvMask):
                self.fail("Mask from fw.convolveLinear does not match image from refConvolve in iter %d" % ii)

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

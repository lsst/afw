"""Test lsst.fw.Core.fwLib.convolve

To do:
- Fix memory leak in the version of the convolve function that returns a new MaskedImage
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

def refSpatiallyInvariantConvolve(imVarMask, kernel, threshold=0, edgeBit=-1):
    """Reference code to convolve a spatially invariant kernel with masked image data.
    
    Inputs:
    - imVarMask: (image, variance, mask) numpy arrays
    - kernel: lsst::fw::Core.Kernel object
    - threshold: kernel pixels above this threshold are used to or in mask bits
    - edgeBit: this bit is set in the output mask for border pixels; no bit set if < 0
    
    Border pixels (pixels too close to the edge to compute) are copied from the input,
    and if edgeBit >= 0 then border mask pixels have the edgeBit bit set
    """
    image, variance, mask = imVarMask
    kImage = iUtils.arrayFromImage(kernel.computeNewImage()[0])
    
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

    retRow = kernel.getCtrRow()
    for inRowBeg in range(numRows):
        inRowEnd = inRowBeg + kRows
        retCol = kernel.getCtrCol()
        for inColBeg in colRange:
            inColEnd = inColBeg + kCols
            subImage = image[inColBeg:inColEnd, inRowBeg:inRowEnd]
            subVariance = variance[inColBeg:inColEnd, inRowBeg:inRowEnd]
            subMask = mask[inColBeg:inColEnd, inRowBeg:inRowEnd]
            retImage[retCol, retRow] = numpy.add.reduce((kImage * subImage).flat)
            retVariance[retCol, retRow] = numpy.add.reduce((kImage * kImage * subVariance).flat)
            retMask[retCol, retRow] = numpy.bitwise_or.reduce(((kImage > threshold) * subMask).flat)

            retCol += 1
        retRow += 1
    return (retImage, retVariance, retMask)


class ConvolveTestCase(unittest.TestCase):
    def testUnityConvolution(self):
        """Verify that convolution with a centered delta function reproduces the original"""
        threshold = 0.0
        edgeBit = -1
        
        currDir = os.path.abspath(os.path.dirname(__file__))
        inFilePath = os.path.join(currDir, "data", "small")
    
        maskedImage = fw.MaskedImageD()
        maskedImage.readFits(inFilePath)
        
        # create a delta function kernel that has 1,1 in the center
        f = fw.IntegerDeltaFunction2F(0.0, 0.0)
        fPtr =  fw.Function2PtrTypeF(f)
        f.this.disown() # Only the shared pointer now owns f
        k = fw.AnalyticKernelF(fPtr, 3, 3)
        
        cnvMaskedImage = fw.MaskedImageD(maskedImage.getCols(), maskedImage.getRows())
        fw.convolveF(cnvMaskedImage, maskedImage, k, threshold, edgeBit)
    
        origImVarMaskArrays = iUtils.arraysFromMaskedImage(maskedImage)
        cnvImVarMaskArrays = iUtils.arraysFromMaskedImage(cnvMaskedImage)
        for name, ind in (("image", 0), ("variance", 1), ("mask", 2)):
            if not numpy.allclose(origImVarMaskArrays[ind], cnvImVarMaskArrays[ind]):
                self.fail("Convolved %s does not match reference" % (name,))

    def testSpatiallyInvariantInPlaceConvolve(self):
        """Test in-place convolution with a spatially invariant Gaussian function
        """
        kCols = 7
        kRows = 7
        threshold = 0.0
        edgeBit = 7

        f = fw.GaussianFunction2F(1.5, 2.5)
        fPtr =  fw.Function2PtrTypeF(f)
        f.this.disown() # Only the shared pointer now owns f
        k = fw.AnalyticKernelF(fPtr, kCols, kRows)
        fArr = numpy.zeros(shape=[k.getCols(), k.getRows()], dtype=float)
        
        currDir = os.path.abspath(os.path.dirname(__file__))
        inFilePath = os.path.join(currDir, "data", InputMaskedImageName)
        fullMaskedImage = fw.MaskedImageD()
        fullMaskedImage.readFits(inFilePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(0, 0, 45, 55)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()

        cnvMaskedImage = fw.MaskedImageD(maskedImage.getCols(), maskedImage.getRows())
        fw.convolveF(cnvMaskedImage, maskedImage, k, threshold, edgeBit)
        cnvImage, cnvVariance, cnvMask = iUtils.arraysFromMaskedImage(cnvMaskedImage)

        imVarMask = iUtils.arraysFromMaskedImage(maskedImage)
        refCnvImage, refCnvVariance, refCnvMask = \
            refSpatiallyInvariantConvolve(imVarMask, k, threshold, edgeBit)

        if not numpy.allclose(cnvImage, refCnvImage):
            self.fail("Convolved image does not match reference")
        if not numpy.allclose(cnvVariance, refCnvVariance):
            self.fail("Convolved variance does not match reference")
        if not numpy.allclose(cnvMask, refCnvMask):
            self.fail("Convolved mask does not match reference")
    
    def testSpatiallyInvariantNewConvolve(self):
        """Test convolution that returns a new MaskedImage with a spatially invariant Gaussian function
        In this case, just make sure there are no memory leaks.
        
        This presently triggers a memory leak. Don't use the new-image version
        of the convolve function until we get this straightened out.
        """
        kCols = 7
        kRows = 7
        threshold = 0.0
        edgeBit = 7

        f = fw.GaussianFunction2F(1.5, 2.5)
        fPtr =  fw.Function2PtrTypeF(f)
        f.this.disown() # Only the shared pointer now owns f
        k = fw.AnalyticKernelF(fPtr, kCols, kRows)
        fArr = numpy.zeros(shape=[k.getCols(), k.getRows()], dtype=float)
        
        currDir = os.path.abspath(os.path.dirname(__file__))
        inFilePath = os.path.join(currDir, "data", InputMaskedImageName)
        fullMaskedImage = fw.MaskedImageD()
        fullMaskedImage.readFits(inFilePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(0, 0, 45, 55)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()
        
        # this fails because of wrong # of arguments
        # so the function overloading is not working correctly (no surprise)
        cnvMaskedImage = fw.convolveF(maskedImage, k, threshold, edgeBit)
        cnvImage, cnvVariance, cnvMask = iUtils.arraysFromMaskedImage(cnvMaskedImage)

        imVarMask = iUtils.arraysFromMaskedImage(maskedImage)
        refCnvImage, refCnvVariance, refCnvMask = \
            refSpatiallyInvariantConvolve(imVarMask, k, threshold, edgeBit)

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

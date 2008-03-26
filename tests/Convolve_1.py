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
import eups

import numpy

import lsst.fw.Core.fwLib as fw
import lsst.fw.Core.imageTestUtils as imTestUtils
import lsst.mwi.tests as tests
import lsst.mwi.utils as mwiu

verbosity = 0 # increase to see trace
mwiu.Trace_setVerbosity("lsst.fw", verbosity)

dataDir = eups.productDir("fwData")
if not dataDir:
    raise RuntimeError("Must set up fwData to run these tests")
InputMaskedImagePath = os.path.join(dataDir, "871034p_1_MI")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def refConvolve(imVarMask, kernel, edgeBit, doNormalize):
    """Reference code to convolve a kernel with masked image data.
    
    Does NOT normalize the kernel.

    Warning: slow (especially for spatially varying kernels).
    
    Inputs:
    - imVarMask: (image, variance, mask) numpy arrays
    - kernel: lsst::fw::Core.Kernel object
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
        kImArr = imTestUtils.arrayFromImage(kernel.computeNewImage(0, 0, doNormalize)[0])
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
                kImArr = imTestUtils.arrayFromImage(kImage)
            inColEnd = inColBeg + kCols
            subImage = image[inColBeg:inColEnd, inRowBeg:inRowEnd]
            subVariance = variance[inColBeg:inColEnd, inRowBeg:inRowEnd]
            subMask = mask[inColBeg:inColEnd, inRowBeg:inRowEnd]
            retImage[retCol, retRow] = numpy.add.reduce((kImArr * subImage).flat)
            retVariance[retCol, retRow] = numpy.add.reduce((kImArr * kImArr * subVariance).flat)
            retMask[retCol, retRow] = numpy.bitwise_or.reduce((subMask).flat)

            retCol += 1
        retRow += 1
    return (retImage, retVariance, retMask)

def makeGaussianKernelVec(kCols, kRows):
    """Create a fw.VectorKernel of gaussian kernels.

    This is useful for constructing a LinearCombinationKernel.
    """
    xySigmaList = [
        (1.5, 1.5),
        (1.5, 2.5),
        (2.5, 1.5),
    ]
    kVec = fw.KernelListD()
    for xSigma, ySigma in xySigmaList:
        fPtr =  fw.Function2DPtr(fw.GaussianFunction2D(1.5, 2.5))
        basisKernelPtr = fw.KernelPtr(fw.AnalyticKernel(fPtr, kCols, kRows))
        kVec.append(basisKernelPtr)
    return kVec

def sameMaskPlaneDicts(maskedImageA, maskedImageB):
    """Return True if the mask plane dicts are the same, False otherwise.

    Handles the fact that one cannot directly compare maskPlaneDicts using ==
    """
    mpDictA = maskedImageA.getMask().getMaskPlaneDict()
    mpDictB = maskedImageB.getMask().getMaskPlaneDict()
    if mpDictA.keys() != mpDictB.keys():
        print "mpDictA.keys()  ", mpDictA.keys()
        print "mpDictB.keys()  ", mpDictB.keys()
        return False
    if mpDictA.values() != mpDictB.values():
        print "mpDictA.values()", mpDictA.values()
        print "mpDictB.values()", mpDictB.values()
        return False
    return True

class ConvolveTestCase(unittest.TestCase):
    def disabledTtestUnityConvolution(self):
        """Verify that convolution with a centered delta function reproduces the original.
        Test is disabled as long as kernel values != 0 smear the mask.
        (One could also force the mask to all 0s then the test would run.)
        """
        imCols = 45
        imRows = 55
        edgeBit = -1
        
        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(InputMaskedImagePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(50, 50, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()
        maskedImage.getMask().setMaskPlaneValues(0, 5, 7, 5)
        
        # create a delta function kernel that has 1,1 in the center
        fPtr =  fw.Function2DPtr(fw.IntegerDeltaFunction2D(0.0, 0.0))
        k = fw.AnalyticKernel(fPtr, 3, 3)
        
        cnvMaskedImage = fw.convolve(maskedImage, k, edgeBit, True)
    
        origImVarMaskArrays = imTestUtils.arraysFromMaskedImage(maskedImage)
        cnvImVarMaskArrays = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
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
        edgeBit = 7
        doNormalize = False

        fPtr =  fw.Function2DPtr(fw.GaussianFunction2D(1.5, 2.5))
        k = fw.AnalyticKernel(fPtr, kCols, kRows)
        
        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(InputMaskedImagePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(50, 50, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()
        maskedImage.getMask().setMaskPlaneValues(0, 5, 7, 5)
        
        cnvMaskedImage = fw.MaskedImageF(imCols, imRows)
        for doNormalize in (False, True):
            fw.convolve(cnvMaskedImage, maskedImage, k, edgeBit, doNormalize)
            cnvImage, cnvVariance, cnvMask = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)

            imVarMask = imTestUtils.arraysFromMaskedImage(maskedImage)
            refCnvImage, refCnvVariance, refCnvMask = \
                refConvolve(imVarMask, k, edgeBit, doNormalize)
    
            if not numpy.allclose(cnvImage, refCnvImage):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvVariance, refCnvVariance):
                self.fail("Convolved variance does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvMask, refCnvMask):
                self.fail("Convolved mask does not match reference for doNormalize=%s" % doNormalize)
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)
                
    
    def testSpatiallyInvariantConvolve(self):
        """Test convolution with a spatially invariant Gaussian function
        """
        kCols = 7
        kRows = 6
        imCols = 55
        imRows = 45
        edgeBit = 7

        fPtr =  fw.Function2DPtr(fw.GaussianFunction2D(1.5, 2.5))
        k = fw.AnalyticKernel(fPtr, kCols, kRows)
        
        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(InputMaskedImagePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(50, 50, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()
        maskedImage.getMask().setMaskPlaneValues(0, 5, 7, 5)
        
        for doNormalize in (False, True):
            cnvMaskedImage = fw.convolve(maskedImage, k, edgeBit, doNormalize)
            cnvImage, cnvVariance, cnvMask = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            imVarMask = imTestUtils.arraysFromMaskedImage(maskedImage)
            refCnvImage, refCnvVariance, refCnvMask = \
                refConvolve(imVarMask, k, edgeBit, doNormalize)
    
            if not numpy.allclose(cnvImage, refCnvImage):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvVariance, refCnvVariance):
                self.fail("Convolved variance does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvMask, refCnvMask):
                self.fail("Convolved mask does not match reference for doNormalize=%s" % doNormalize)
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)

    def testSpatiallyVaryingInPlaceConvolve(self):
        """Test in-place convolution with a spatially varying Gaussian function
        """
        kCols = 7
        kRows = 6
        imCols = 55
        imRows = 45
        edgeBit = 7

        # create spatially varying linear combination kernel
        sFuncPtr =  fw.Function2DPtr(fw.PolynomialFunction2D(1))
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, 1.0 / imCols, 0.0),
            (1.0, 0.0,  1.0 / imRows),
        )
   
        fPtr =  fw.Function2DPtr(fw.GaussianFunction2D(1.0, 1.0))
        k = fw.AnalyticKernel(fPtr, kCols, kRows, sFuncPtr, sParams)
        
        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(InputMaskedImagePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(50, 50, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()
        maskedImage.getMask().setMaskPlaneValues(0, 5, 7, 5)
        
        cnvMaskedImage = fw.MaskedImageF(imCols, imRows)
        for doNormalize in (False, True):
            fw.convolve(cnvMaskedImage, maskedImage, k, edgeBit, doNormalize)
            cnvImage, cnvVariance, cnvMask = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            imVarMask = imTestUtils.arraysFromMaskedImage(maskedImage)
            refCnvImage, refCnvVariance, refCnvMask = \
                refConvolve(imVarMask, k, edgeBit, doNormalize)
    
            if not numpy.allclose(cnvImage, refCnvImage):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvVariance, refCnvVariance):
                self.fail("Convolved variance does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvMask, refCnvMask):
                self.fail("Convolved mask does not match reference for doNormalize=%s" % doNormalize)
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)
    
    def testDeltaConvolveUnoptimized(self):
        """Test convolution with various delta function kernels,
        avoiding any optimized fw convolution code.
        """
        edgeBit = 7
        imCols = 20
        imRows = 12
        doNormalize = True

        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(InputMaskedImagePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(50, 50, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()
        maskedImage.getMask().setMaskPlaneValues(0, 5, 7, 5)
        
        for kCols in range(1, 11):
            kRows = kCols
            kNumPix = kRows * kCols
            for deltaInd in range(kNumPix):
                kerArr = numpy.zeros([kNumPix])
                kerArr[deltaInd] = 1.0
                kerArr.shape = [kCols, kRows]
                kerIm = imTestUtils.imageFromArray(kerArr)
                kernel = fw.FixedKernel(kerIm)
                
                refCnvMaskedImage = fw.convolve(maskedImage, kernel, edgeBit, doNormalize)
                refCnvImage, refCnvVariance, refCnvMask = \
                    imTestUtils.arraysFromMaskedImage(refCnvMaskedImage)
        
                imVarMask = imTestUtils.arraysFromMaskedImage(maskedImage)
                ref2CnvImage, ref2CnvVariance, ref2CnvMask = \
                   refConvolve(imVarMask, kernel, edgeBit, doNormalize)
        
                if not numpy.allclose(refCnvImage, ref2CnvImage):
                    self.fail("Image from fw.convolve does not match image from refConvolve")
                if not numpy.allclose(refCnvVariance, ref2CnvVariance):
                    self.fail("Variance from fw.convolve does not match image from refConvolve")
                if not numpy.allclose(refCnvMask, ref2CnvMask):
                    self.fail("Mask from fw.convolve does not match image from refCconvolve")

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
        doNormalize = False # must be false because convolveLinear cannot normalize

        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(InputMaskedImagePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(50, 50, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()
        maskedImage.getMask().setMaskPlaneValues(0, 5, 7, 5)

        # create spatially varying linear combination kernel
        sFuncPtr =  fw.Function2DPtr(fw.PolynomialFunction2D(1))
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, -0.5 / imCols, -0.5 / imRows),
            (0.0,  1.0 / imCols,  0.0 / imRows),
            (0.0,  0.0 / imCols,  1.0 / imRows),
        )
        
        kVec = makeGaussianKernelVec(kCols, kRows)
        lcKernel = fw.LinearCombinationKernel(kVec, sFuncPtr, sParams)

        refCnvMaskedImage = fw.convolve(maskedImage, lcKernel, edgeBit, doNormalize)
        refCnvImage, refCnvVariance, refCnvMask = \
            imTestUtils.arraysFromMaskedImage(refCnvMaskedImage)

        imVarMask = imTestUtils.arraysFromMaskedImage(maskedImage)
        ref2CnvImage, ref2CnvVariance, ref2CnvMask = \
           refConvolve(imVarMask, lcKernel, edgeBit, doNormalize)

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
            cnvImage, cnvVariance, cnvMask = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            if not numpy.allclose(cnvImage, ref2CnvImage):
                self.fail("Image from fw.convolveLinear does not match image from refConvolve in iter %d" % ii)
            if not numpy.allclose(cnvVariance, ref2CnvVariance):
                self.fail("Variance from fw.convolveLinear does not match image from refConvolve in iter %d" % ii)
            if not numpy.allclose(cnvMask, ref2CnvMask):
                self.fail("Mask from fw.convolveLinear does not match image from refConvolve in iter %d" % ii)
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)

    def testConvolveLinearNewImage(self):
        """Test variant of convolveLinear that returns a new image
        """
        kCols = 5
        kRows = 5
        edgeBit = 7
        imCols = 50
        imRows = 55
        doNormalize = False # must be false because convolveLinear cannot normalize

        fullMaskedImage = fw.MaskedImageF()
        fullMaskedImage.readFits(InputMaskedImagePath)
        
        # pick a small piece of the image to save time
        bbox = fw.BBox2i(50, 50, imCols, imRows)
        subMaskedImagePtr = fullMaskedImage.getSubImage(bbox)
        maskedImage = subMaskedImagePtr.get()
        maskedImage.this.disown()
        maskedImage.getMask().setMaskPlaneValues(0, 5, 7, 5)

        # create spatially varying linear combination kernel
        sFuncPtr =  fw.Function2DPtr(fw.PolynomialFunction2D(1))
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, -0.5 / imCols, -0.5 / imRows),
            (0.0,  1.0 / imCols,  0.0 / imRows),
            (0.0,  0.0 / imCols,  1.0 / imRows),
        )
        
        kVec = makeGaussianKernelVec(kCols, kRows)
        lcKernel = fw.LinearCombinationKernel(kVec, sFuncPtr, sParams)

        refCnvMaskedImage = fw.convolve(maskedImage, lcKernel, edgeBit, doNormalize)
        refCnvImage, refCnvVariance, refCnvMask = \
            imTestUtils.arraysFromMaskedImage(refCnvMaskedImage)

        imVarMask = imTestUtils.arraysFromMaskedImage(maskedImage)
        ref2CnvImage, ref2CnvVariance, ref2CnvMask = \
           refConvolve(imVarMask, lcKernel, edgeBit, doNormalize)

        if not numpy.allclose(refCnvImage, ref2CnvImage):
            self.fail("Image from fw.convolve does not match image from refConvolve")
        if not numpy.allclose(refCnvVariance, ref2CnvVariance):
            self.fail("Variance from fw.convolve does not match image from refConvolve")
        if not numpy.allclose(refCnvMask, ref2CnvMask):
            self.fail("Mask from fw.convolve does not match image from refCconvolve")

        # compute twice, to be sure cnvMaskedImage is properly reset
        for ii in range(2):        
            cnvMaskedImage = fw.convolveLinear(maskedImage, lcKernel, edgeBit)
            cnvImage, cnvVariance, cnvMask = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            if not numpy.allclose(cnvImage, ref2CnvImage):
                self.fail("Image from fw.convolveLinear does not match image from refConvolve in iter %d" % ii)
            if not numpy.allclose(cnvVariance, ref2CnvVariance):
                self.fail("Variance from fw.convolveLinear does not match image from refConvolve in iter %d" % ii)
            if not numpy.allclose(cnvMask, ref2CnvMask):
                self.fail("Mask from fw.convolveLinear does not match image from refConvolve in iter %d" % ii)
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(ConvolveTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    tests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

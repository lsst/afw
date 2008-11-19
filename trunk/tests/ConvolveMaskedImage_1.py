#!/usr/bin/python

"""Test lsst.afwMath.convolve

The convolve function is overloaded in two flavors:
- in-place convolve: user supplies the output image as an argument
- new-image convolve: the convolve function returns the convolved image

All tests use the new-image version unless otherwise noted.
"""
import math
import os
import re
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

try:
    Verbosity
except NameError:
    Verbosity = 0                       # increase to see trace
pexLog.Trace_setVerbosity("lsst.afw", Verbosity)

try:
    display
except NameError:
    display=False

import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests")
if False:
    InputMaskedImagePath = os.path.join(dataDir, "871034p_1_MI")
else:
    InputMaskedImagePath = os.path.join(dataDir, "small_MI")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def refConvolve(imVarMask, kernel, doNormalize, edgeBit, ignoreKernelZeroPixels=True):
    """Reference code to convolve a kernel with masked image data.
    
    Does NOT normalize the kernel.

    Warning: slow (especially for spatially varying kernels).
    
    Inputs:
    - imVarMask: (image, variance, mask) numpy arrays
    - kernel: lsst::afw::Core.Kernel object
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
    
    kCols = kernel.getWidth()
    kRows = kernel.getHeight()
    numCols = image.shape[0] + 1 - kCols
    numRows = image.shape[1] + 1 - kRows
    if numCols < 0 or numRows < 0:
        raise RuntimeError("image must be larger than kernel in both dimensions")
    colRange = range(numCols)


    kImage = afwImage.ImageD(kCols, kRows)
    isSpatiallyVarying = kernel.isSpatiallyVarying()
    if not isSpatiallyVarying:
        kernel.computeImage(kImage, doNormalize)
        kImArr = imTestUtils.arrayFromImage(kImage)

    retRow = kernel.getCtrY()
    for inRowBeg in range(numRows):
        inRowEnd = inRowBeg + kRows
        retCol = kernel.getCtrX()
        if isSpatiallyVarying:
            rowPos = afwImage.indexToPosition(retRow)
        for inColBeg in colRange:
            if isSpatiallyVarying:
                colPos = afwImage.indexToPosition(retCol)
                kernel.computeImage(kImage, doNormalize, colPos, rowPos)
                kImArr = imTestUtils.arrayFromImage(kImage)
            inColEnd = inColBeg + kCols
            subImage = image[inColBeg:inColEnd, inRowBeg:inRowEnd]
            subVariance = variance[inColBeg:inColEnd, inRowBeg:inRowEnd]
            subMask = mask[inColBeg:inColEnd, inRowBeg:inRowEnd]
            retImage[retCol, retRow] = numpy.add.reduce((kImArr * subImage).flat)
            retVariance[retCol, retRow] = numpy.add.reduce((kImArr * kImArr * subVariance).flat)
            if ignoreKernelZeroPixels:
                retMask[retCol, retRow] = numpy.bitwise_or.reduce((subMask * (kImArr != 0)).flat)
            else:
                retMask[retCol, retRow] = numpy.bitwise_or.reduce(subMask.flat)
            

            retCol += 1
        retRow += 1
    return (retImage, retVariance, retMask)

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
        kVec.append(afwMath.AnalyticKernel(kCols, kRows, kFunc))
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
    def setUp(self):
        tmp = afwImage.MaskU()          # clearMaskPlaneDict isn't static
        tmp.clearMaskPlaneDict()        # reset so tests will be deterministic

        for p in ("BAD", "SAT", "INTRP", "CR", "EDGE"):
            afwImage.MaskU_addMaskPlane(p)
            
        self.edgeBit = tmp.addMaskPlane("OUR_EDGE")
        del tmp

        if not False:
            fullImage = afwImage.MaskedImageF(InputMaskedImagePath)
            
            # pick a small piece of the image to save time
            bbox = afwImage.BBox(afwImage.PointI(0, 0), 145, 135)
            self.maskedImage = afwImage.MaskedImageF(fullImage, bbox)
        else:
            self.maskedImage = afwImage.MaskedImageF(InputMaskedImagePath)

        self.width = self.maskedImage.getWidth()
        self.height = self.maskedImage.getHeight()
        smask = afwImage.MaskU(self.maskedImage.getMask(), afwImage.BBox(afwImage.PointI(15, 17), 10, 5))
        smask.set(0x8)

        self.edgeBit = 7
        
    def tearDown(self):
        del self.maskedImage
        
    def testUnityConvolution(self):
        """Verify that convolution with a centered delta function reproduces the original.
        """
        edgeBit = -1

        # create a delta function kernel that has 1,1 in the center
        kFunc = afwMath.IntegerDeltaFunction2D(0.0, 0.0)
        k = afwMath.AnalyticKernel(3, 3, kFunc)
        
        cnvMaskedImage = afwImage.MaskedImageF(self.maskedImage.getDimensions())
        afwMath.convolve(cnvMaskedImage, self.maskedImage, k, True, edgeBit)

        origImVarMaskArrays = imTestUtils.arraysFromMaskedImage(self.maskedImage)
        cnvImVarMaskArrays = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
        for name, ind in (("image", 0), ("variance", 1), ("mask", 2)):
            if not numpy.allclose(origImVarMaskArrays[ind], cnvImVarMaskArrays[ind]):
                if display:
                    ds9.mtv(displayUtils.makeMosaic(self.maskedImage, cnvMaskedImage), frame=0)
                self.fail("Convolved %s does not match reference" % (name,))

    def testSpatiallyInvariantInPlaceConvolve(self):
        """Test convolve with a spatially invariant Gaussian function
        """
        kCols = 6
        kRows = 7
        edgeBit = self.edgeBit

        kFunc =  afwMath.GaussianFunction2D(1.5, 2.5)
        k = afwMath.AnalyticKernel(kCols, kRows, kFunc)
        
        cnvMaskedImage = afwImage.MaskedImageF(self.maskedImage.getDimensions())
        for doNormalize in (True, False):
            afwMath.convolve(cnvMaskedImage, self.maskedImage, k, doNormalize, edgeBit)
            cnvImage, cnvVariance, cnvMask = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)

            imVarMask = imTestUtils.arraysFromMaskedImage(self.maskedImage)
            refCnvImage, refCnvVariance, refCnvMask = refConvolve(imVarMask, k, doNormalize, edgeBit)

            if display:
                refMaskedImage = self.maskedImage.Factory(self.maskedImage.getDimensions())
                imTestUtils.maskedImageFromArrays(refMaskedImage, (refCnvImage, refCnvVariance, refCnvMask))
                ds9.mtv(displayUtils.makeMosaic(self.maskedImage, refMaskedImage, cnvMaskedImage), frame=0)
                if False:
                    for (x, y) in ((0, 0), (1, 0), (0, 1), (50, 50)):
                        print "Mask(%d,%d) 0x%x 0x%x" % \
                              (x, y, refMaskedImage.getMask().get(x, y), cnvMaskedImage.getMask().get(x, y))

            if not numpy.allclose(cnvImage, refCnvImage):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvVariance, refCnvVariance):
                self.fail("Convolved variance does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvMask, refCnvMask):
                self.fail("Convolved mask does not match reference for doNormalize=%s" % doNormalize)
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, self.maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)

    def testSpatiallyInvariantConvolve(self):
        """Test convolution with a spatially invariant Gaussian function
        """
        kCols = 7
        kRows = 6
        edgeBit = self.edgeBit

        kFunc =  afwMath.GaussianFunction2D(1.5, 2.5)
        k = afwMath.AnalyticKernel(kCols, kRows, kFunc)
                
        cnvMaskedImage = afwImage.MaskedImageF(self.maskedImage.getDimensions())
        for doNormalize in (False, True):
            afwMath.convolve(cnvMaskedImage, self.maskedImage, k, doNormalize, edgeBit)
            cnvImage, cnvVariance, cnvMask = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            imVarMask = imTestUtils.arraysFromMaskedImage(self.maskedImage)
            refCnvImage, refCnvVariance, refCnvMask = refConvolve(imVarMask, k, doNormalize, edgeBit)
    
            if not numpy.allclose(cnvImage, refCnvImage):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvVariance, refCnvVariance):
                self.fail("Convolved variance does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvMask, refCnvMask):
                self.fail("Convolved mask does not match reference for doNormalize=%s" % doNormalize)
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, self.maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)

    def testSpatiallyVaryingInPlaceConvolve(self):
        """Test in-place convolution with a spatially varying Gaussian function
        """
        kCols = 7
        kRows = 6
        edgeBit = self.edgeBit

        # create spatially varying linear combination kernel
        sFunc = afwMath.PolynomialFunction2D(1)
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, 1.0/self.width, 0.0),
            (1.0, 0.0, 1.0/self.height),
        )
   
        kFunc =  afwMath.GaussianFunction2D(1.0, 1.0)
        k = afwMath.AnalyticKernel(kCols, kRows, kFunc, sFunc)
        k.setSpatialParameters(sParams)
        
        cnvMaskedImage = afwImage.MaskedImageF(self.width, self.height)
        for doNormalize in (False, True):
            afwMath.convolve(cnvMaskedImage, self.maskedImage, k, doNormalize, edgeBit)
            cnvImage, cnvVariance, cnvMask = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            imVarMask = imTestUtils.arraysFromMaskedImage(self.maskedImage)
            refCnvImage, refCnvVariance, refCnvMask = refConvolve(imVarMask, k, doNormalize, edgeBit)
    
            if not numpy.allclose(cnvImage, refCnvImage):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvVariance, refCnvVariance):
                self.fail("Convolved variance does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvMask, refCnvMask):
                self.fail("Convolved mask does not match reference for doNormalize=%s" % doNormalize)
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, self.maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)

    def testSpatiallyVaryingSeparableInPlaceConvolve(self):
        """Test in-place separable convolution with a spatially varying Gaussian function
        """
        sys.stderr.write("Test convolution with SeparableKernel\n")
        kCols = 7
        kRows = 6
        edgeBit = self.edgeBit

        # create spatially varying linear combination kernel
        sFunc = afwMath.PolynomialFunction2D(1)
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, 1.0 / self.width, 0.0),
            (1.0, 0.0,  1.0 / self.height),
        )

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        gaussFunc2 = afwMath.GaussianFunction2D(1.0, 1.0)
        separableKernel = afwMath.SeparableKernel(kCols, kRows, gaussFunc1, gaussFunc1, sFunc)
        analyticKernel = afwMath.AnalyticKernel(kCols, kRows, gaussFunc2, sFunc)
        separableKernel.setSpatialParameters(sParams)
        analyticKernel.setSpatialParameters(sParams)
                
        cnvMaskedImage = afwImage.MaskedImageF(self.width, self.height)
        for doNormalize in (False, True):
            afwMath.convolve(cnvMaskedImage, self.maskedImage, separableKernel, doNormalize, edgeBit)
            cnvImage, cnvVariance, cnvMask = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            imVarMask = imTestUtils.arraysFromMaskedImage(self.maskedImage)
            refCnvImage, refCnvVariance, refCnvMask = refConvolve(imVarMask, analyticKernel, doNormalize, edgeBit)
    
            if not numpy.allclose(cnvImage, refCnvImage):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvVariance, refCnvVariance):
                self.fail("Convolved variance does not match reference for doNormalize=%s" % doNormalize)
            if not numpy.allclose(cnvMask, refCnvMask):
                self.fail("Convolved mask does not match reference for doNormalize=%s" % doNormalize)
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, self.maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)
    
    def testDeltaConvolve(self):
        """Test convolution with various delta function kernels using optimized code
        """
        sys.stderr.write("Test convolution with DeltaFunctionKernel\n")
        edgeBit = self.edgeBit
        self.width = 20
        self.height = 12
        doNormalize = True

        bbox = afwImage.BBox(afwImage.PointI(1, 1), self.width, self.height)
        maskedImage = afwImage.MaskedImageF(self.maskedImage, bbox)
        cnvMaskedImage = afwImage.MaskedImageF(maskedImage.getDimensions())
        
        for kCols in range(1, 4):
            for kRows in range(1, 4):
                for activeCol in range(kCols):
                    for activeRow in range(kRows):
                        kernel = afwMath.DeltaFunctionKernel(kCols, kRows, afwImage.PointI(activeCol, activeRow))

                        afwMath.convolve(cnvMaskedImage, maskedImage, kernel, doNormalize, edgeBit)
                        cnvImage, cnvVariance, cnvMask = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
                
                        imVarMask = imTestUtils.arraysFromMaskedImage(maskedImage)
                        refCnvImage, refCnvVariance, refCnvMask = \
                                     refConvolve(imVarMask, kernel, doNormalize, edgeBit, True)
                
                        if display and False:
                            refMaskedImage = imTestUtils.maskedImageFromArrays((refCnvImage, refCnvVariance, refCnvMask))

                            if False:
                                for (x, y) in ((0,0), (0, 11)) :
                                    print "Mask %dx%d:(%d,%d): At (%d,%d) 0x%x  0x%x" % (
                                        kCols, kRows, activeCol, activeRow,
                                        x, y,
                                        refMaskedImage.getMask().get(x,y),
                                        cnvMaskedImage.getMask().get(x,y))
                            ds9.mtv(displayUtils.makeMosaic(refMaskedImage, cnvMaskedImage), frame=0)

                        if not numpy.allclose(cnvImage, refCnvImage):
                            self.fail("Image from afwMath.convolveNew does not match image from refConvolve")
                        if not numpy.allclose(cnvVariance, refCnvVariance):
                            self.fail("Variance from afwMath.convolveNew does not match mask from refConvolve")
                        if not numpy.allclose(cnvMask, refCnvMask):
                            self.fail("Mask from afwMath.convolveNew does not match mask from refCconvolve")
        

    def testConvolveLinear(self):
        """Test convolution with a spatially varying LinearCombinationKernel
        by comparing the results of afwMath.convolveLinear to afwMath.convolveNew or refConvolve,
        depending on the value of compareToFwConvolve.
        """
        kCols = 5
        kRows = 5
        edgeBit = self.edgeBit
        doNormalize = False             # must be false because convolveLinear cannot normalize

        # create spatially varying linear combination kernel
        sFunc = afwMath.PolynomialFunction2D(1)
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, -0.5/self.width, -0.5/self.height),
            (0.0,  1.0/self.width,  0.0/self.height),
            (0.0,  0.0/self.width,  1.0/self.height),
        )
        
        kVec = makeGaussianKernelVec(kCols, kRows)
        lcKernel = afwMath.LinearCombinationKernel(kVec, sFunc)
        lcKernel.setSpatialParameters(sParams)

        cnvMaskedImage = afwImage.MaskedImageF(self.maskedImage.getDimensions())
        afwMath.convolve(cnvMaskedImage, self.maskedImage, lcKernel, doNormalize, edgeBit)
        cnvImage, cnvVariance, cnvMask = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)

        imVarMask = imTestUtils.arraysFromMaskedImage(self.maskedImage)
        refCnvImage, refCnvVariance, refCnvMask = refConvolve(imVarMask, lcKernel, doNormalize, edgeBit)

        if not numpy.allclose(cnvImage, refCnvImage):
            self.fail("Image from afwMath.convolveNew does not match image from refConvolve")
        if not numpy.allclose(cnvVariance, refCnvVariance):
            self.fail("Variance from afwMath.convolveNew does not match image from refConvolve")
        if not numpy.allclose(cnvMask, refCnvMask):
            self.fail("Mask from afwMath.convolveNew does not match image from refCconvolve")

        # compute twice, to be sure cnvMaskedImage is properly reset
        for ii in range(2):        
            afwMath.convolveLinear(cnvMaskedImage, self.maskedImage, lcKernel, edgeBit)
            cnvImage, cnvVariance, cnvMask = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            if display:
                refMaskedImage = self.maskedImage.Factory(self.maskedImage.getDimensions())
                imTestUtils.maskedImageFromArrays(refMaskedImage, (refCnvImage, refCnvVariance, refCnvMask))
                ds9.mtv(displayUtils.makeMosaic(refMaskedImage, cnvMaskedImage), frame=0)
            if not numpy.allclose(cnvImage, refCnvImage):
                self.fail("Image from afwMath.convolveLinear does not match image from refConvolve in iter %d" % ii)
            if not numpy.allclose(cnvVariance, refCnvVariance):
                self.fail("Variance from afwMath.convolveLinear does not match variance from refConvolve in iter %d" % ii)
            if not numpy.allclose(cnvMask, refCnvMask):
                self.fail("Mask from afwMath.convolveLinear does not match mask from refConvolve in iter %d" % ii)
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, self.maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)

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

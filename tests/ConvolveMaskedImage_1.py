#!/usr/bin/env python

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
pexLog.Debug("lsst.afw", Verbosity)

try:
    display
except NameError:
    display=False

import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests")

# input image contains a saturated star, a bad column, and a faint star
InputMaskedImagePath = os.path.join(dataDir, "med")
InputBBox = afwImage.BBox(afwImage.PointI(50, 500), 100, 100)
    
EdgeMaskPixel = 1 << afwImage.MaskU.getMaskPlane("EDGE")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def refConvolve(imMaskVar, kernel, doNormalize, ignoreKernelZeroPixels=True):
    """Reference code to convolve a kernel with masked image data.
    
    Does NOT normalize the kernel.

    Warning: slow (especially for spatially varying kernels).
    
    Inputs:
    - imMaskVar: (image, mask, variance) numpy arrays
    - kernel: lsst::afw::Core.Kernel object
    - doNormalize: normalize the kernel
    
    Border pixels (pixels too close to the edge to compute) are set to the standard edge pixel
    """
    image, mask, variance = imMaskVar
    
    # initialize output array to all edge pixels; non-edge pixels will get overridden below
    retImage = numpy.zeros(image.shape, dtype=image.dtype)
    retImage += numpy.nan
    retVariance = numpy.zeros(variance.shape, dtype=image.dtype)
    retVariance += numpy.inf
    retMask = numpy.zeros(mask.shape, dtype=mask.dtype)
    retMask += EdgeMaskPixel
    
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
    return (retImage, retMask, retVariance)

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

def makeDeltaFunctionKernelVec(kCols, kRows):
    """Create an afwImage.VectorKernel of delta function kernels
    """
    kVec = afwMath.KernelListD()
    for activeCol in range(kCols):
        for activeRow in range(kRows):
            kVec.append(afwMath.DeltaFunctionKernel(kCols, kRows, afwImage.PointI(activeCol, activeRow)))
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
            
        del tmp

        fullImage = afwImage.MaskedImageF(InputMaskedImagePath)
        self.maskedImage = afwImage.MaskedImageF(fullImage, InputBBox)
        self.maskedImage.writeFits("temp")

        self.width = self.maskedImage.getWidth()
        self.height = self.maskedImage.getHeight()
        smask = afwImage.MaskU(self.maskedImage.getMask(), afwImage.BBox(afwImage.PointI(15, 17), 10, 5))
        smask.set(0x8)

    def tearDown(self):
        del self.maskedImage
        
    def testUnityConvolution(self):
        """Verify that convolution with a centered delta function reproduces the original.
        """
        # create a delta function kernel that has 1,1 in the center
        kFunc = afwMath.IntegerDeltaFunction2D(0.0, 0.0)
        k = afwMath.AnalyticKernel(3, 3, kFunc)
        
        cnvMaskedImage = afwImage.MaskedImageF(self.maskedImage.getDimensions())
        afwMath.convolve(cnvMaskedImage, self.maskedImage, k, True)

        origImMaskVarArrays = imTestUtils.arraysFromMaskedImage(self.maskedImage)
        cnvImMaskVarArrays = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
        skipMaskArr = numpy.isnan(cnvImMaskVarArrays[0])
        errStr = imTestUtils.maskedImagesDiffer(origImMaskVarArrays, cnvImMaskVarArrays, skipMaskArr=skipMaskArr)
        if errStr:
            self.fail(errStr)


    def testFixedKernelConvolve(self):
        """Test convolve with a fixed kernel
        """
        kCols = 6
        kRows = 7

        kFunc =  afwMath.GaussianFunction2D(1.5, 2.5)
        analyticK = afwMath.AnalyticKernel(kCols, kRows, kFunc)
        kImg = afwImage.ImageD(kCols, kRows)
        
        cnvMaskedImage = afwImage.MaskedImageF(self.maskedImage.getDimensions())
        for doNormalize in (True, False):
            analyticK.computeImage(kImg, doNormalize)
            fixedK = afwMath.FixedKernel(kImg)

            afwMath.convolve(cnvMaskedImage, self.maskedImage, fixedK, doNormalize)
            cnvImMaskVar = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)

            imMaskVar = imTestUtils.arraysFromMaskedImage(self.maskedImage)
            refCnvImMaskVar = refConvolve(imMaskVar, fixedK, doNormalize)

            if display:
                refMaskedImage = imTestUtils.maskedImageFromArrays(refCnvImMaskVar)
                ds9.mtv(displayUtils.makeMosaic(self.maskedImage, refMaskedImage, cnvMaskedImage), frame=0)
                if False:
                    for (x, y) in ((0, 0), (1, 0), (0, 1), (50, 50)):
                        print "Mask(%d,%d) 0x%x 0x%x" % \
                              (x, y, refMaskedImage.getMask().get(x, y), cnvMaskedImage.getMask().get(x, y))

            errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
            if errStr:
                self.fail("%s (for doNormalize=%s)" % (errStr, doNormalize))
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, self.maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)

    def testSeparableConvolve(self):
        """Test convolve of a separable kernel with a spatially invariant Gaussian function
        """
        kCols = 7
        kRows = 6

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        gaussFunc2 = afwMath.GaussianFunction2D(1.0, 1.0)
        separableKernel = afwMath.SeparableKernel(kCols, kRows, gaussFunc1, gaussFunc1)
        analyticKernel = afwMath.AnalyticKernel(kCols, kRows, gaussFunc2)
                
        cnvMaskedImage = afwImage.MaskedImageF(self.maskedImage.getDimensions())
        for doNormalize in (False, True):
            afwMath.convolve(cnvMaskedImage, self.maskedImage, separableKernel, doNormalize)
            cnvImMaskVar = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)

            imMaskVar = imTestUtils.arraysFromMaskedImage(self.maskedImage)
            refCnvImMaskVar = refConvolve(imMaskVar, analyticKernel, doNormalize)

            errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
            if errStr:
                self.fail("%s (for doNormalize=%s)" % (errStr, doNormalize))
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, self.maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)

    def testSpatiallyInvariantConvolve(self):
        """Test convolution with a spatially invariant Gaussian function
        """
        kCols = 6
        kRows = 7

        kFunc =  afwMath.GaussianFunction2D(1.5, 2.5)
        k = afwMath.AnalyticKernel(kCols, kRows, kFunc)
        
        cnvMaskedImage = afwImage.MaskedImageF(self.maskedImage.getDimensions())
        for doNormalize in (True, False):
            afwMath.convolve(cnvMaskedImage, self.maskedImage, k, doNormalize)
            cnvImMaskVar = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)

            imMaskVar = imTestUtils.arraysFromMaskedImage(self.maskedImage)
            refCnvImMaskVar = refConvolve(imMaskVar, k, doNormalize)

            if display:
                refMaskedImage = imTestUtils.maskedImageFromArrays(refCnvImMaskVar)
                ds9.mtv(displayUtils.makeMosaic(self.maskedImage, refMaskedImage, cnvMaskedImage), frame=0)
                if False:
                    for (x, y) in ((0, 0), (1, 0), (0, 1), (50, 50)):
                        print "Mask(%d,%d) 0x%x 0x%x" % \
                              (x, y, refMaskedImage.getMask().get(x, y), cnvMaskedImage.getMask().get(x, y))

            errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
            if errStr:
                self.fail("%s (for doNormalize=%s)" % (errStr, doNormalize))
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, self.maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)

    def testSpatiallyVaryingConvolve(self):
        """Test in-place convolution with a spatially varying Gaussian function
        """
        kCols = 7
        kRows = 6

        # create spatial model
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
            afwMath.convolve(cnvMaskedImage, self.maskedImage, k, doNormalize)
            cnvImMaskVar = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            imMaskVar = imTestUtils.arraysFromMaskedImage(self.maskedImage)
            refCnvImMaskVar = refConvolve(imMaskVar, k, doNormalize)
    
            errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
            if errStr:
                self.fail("%s (for doNormalize=%s)" % (errStr, doNormalize))
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, self.maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)

    def testSpatiallyVaryingSeparableConvolve(self):
        """Test in-place separable convolution with a spatially varying Gaussian function
        """
        kCols = 7
        kRows = 6

        # create spatial model
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
            afwMath.convolve(cnvMaskedImage, self.maskedImage, separableKernel, doNormalize)
            cnvImMaskVar = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            imMaskVar = imTestUtils.arraysFromMaskedImage(self.maskedImage)
            refCnvImMaskVar = refConvolve(imMaskVar, analyticKernel, doNormalize)
    
            errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
            if errStr:
                self.fail("%s (for doNormalize=%s)" % (errStr, doNormalize))
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, self.maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)
    
    def testDeltaConvolve(self):
        """Test convolution with various delta function kernels using optimized code
        """
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

                        afwMath.convolve(cnvMaskedImage, maskedImage, kernel, doNormalize)
                        cnvImMaskVar = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
                
                        imMaskVar = imTestUtils.arraysFromMaskedImage(maskedImage)
                        refCnvImMaskVar = refConvolve(imMaskVar, kernel, doNormalize, True)
                
                        if display and False:
                            refMaskedImage = imTestUtils.maskedImageFromArrays(refCnvImMaskVar)

                            if False:
                                for (x, y) in ((0,0), (0, 11)) :
                                    print "Mask %dx%d:(%d,%d): At (%d,%d) 0x%x  0x%x" % (
                                        kCols, kRows, activeCol, activeRow,
                                        x, y,
                                        refMaskedImage.getMask().get(x,y),
                                        cnvMaskedImage.getMask().get(x,y))
                            ds9.mtv(displayUtils.makeMosaic(refMaskedImage, cnvMaskedImage), frame=0)

                        errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
                        if errStr:
                            self.fail(errStr)

    def testSpatiallyVaryingGaussianLinerCombination(self):
        """Test convolution with a spatially varying LinearCombinationKernel.
        """
        kCols = 5
        kRows = 5
        doNormalize = False # convolution with spatially varying LC Kernel does not yet support normalization

        # create spatial model
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

        imMaskVar = imTestUtils.arraysFromMaskedImage(self.maskedImage)
        refCnvImMaskVar = refConvolve(imMaskVar, lcKernel, doNormalize)

        # compute twice, to be sure cnvMaskedImage is properly reset
        cnvMaskedImage = afwImage.MaskedImageF(self.maskedImage.getDimensions())
        for ii in range(2):        
            afwMath.convolve(cnvMaskedImage, self.maskedImage, lcKernel, doNormalize)
            cnvImMaskVar = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
    
            if display:
                refMaskedImage = imTestUtils.maskedImageFromArrays(refCnvImMaskVar)
                ds9.mtv(displayUtils.makeMosaic(refMaskedImage, cnvMaskedImage), frame=0)

            errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
            if errStr:
                self.fail("%s (for doNormalize=%s on iter %d)" % (errStr, doNormalize, ii))
            self.assert_(sameMaskPlaneDicts(cnvMaskedImage, self.maskedImage),
                "Convolved mask dictionary does not match input for doNormalize=%s" % doNormalize)

    def testSpatiallyVaryingDeltaFunctionLinearCombination(self):
        """Test convolution with a spatially varying LinearCombinationKernel using some delta basis kernels.
        """
        kCols = 2
        kRows = 2
        doNormalize = False # convolution with spatially varying LC Kernel does not yet support normalization

        # create spatially model
        sFunc = afwMath.PolynomialFunction2D(1)
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, -0.5/self.width, -0.5/self.height),
            (0.0,  1.0/self.width,  0.0/self.height),
            (0.0,  0.0/self.width,  1.0/self.height),
            (0.5, 0.0, 0.0),
            )
        
        kVec = makeDeltaFunctionKernelVec(kCols, kRows)
        lcKernel = afwMath.LinearCombinationKernel(kVec, sFunc)
        lcKernel.setSpatialParameters(sParams)

        imMaskVar = imTestUtils.arraysFromMaskedImage(self.maskedImage)
        refCnvImMaskVar = refConvolve(imMaskVar, lcKernel, doNormalize)

        cnvMaskedImage = afwImage.MaskedImageF(self.maskedImage.getDimensions())
        # compute twice, to be sure cnvMaskedImage is properly reset
        for ii in range(2):
            pexLog.Debug("lsst.afw").debug(3, "Start convolution with delta functions")
            afwMath.convolve(cnvMaskedImage, self.maskedImage, lcKernel, doNormalize)
            pexLog.Debug("lsst.afw").debug(3, "End convolution with delta functions")
            cnvImMaskVar = imTestUtils.arraysFromMaskedImage(cnvMaskedImage)
            
            if display:
                refMaskedImage = imTestUtils.maskedImageFromArrays(refCnvImMaskVar)
                ds9.mtv(displayUtils.makeMosaic(refMaskedImage, cnvMaskedImage), frame=0)

            errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
            if errStr:
                self.fail("%s (for doNormalize=%s on iter %d)" % (errStr, doNormalize, ii))
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

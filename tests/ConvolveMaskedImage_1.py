#!/usr/bin/env python

"""Test lsst.afwMath.convolve

The convolve function is overloaded in two flavors:
- in-place convolve: user supplies the output image as an argument
- new-image convolve: the convolve function returns the convolved image
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
# the shifted BBox is for a same-sized region containing different pixels;
# this is used to initialize the convolved image, to make sure convolve fully overwrites it
ShiftedBBox = afwImage.BBox(afwImage.PointI(50, 450), 100, 100)
    
EdgeMaskPixel = 1 << afwImage.MaskU.getMaskPlane("EDGE")

# Ignore kernel pixels whose value is exactly 0 when smearing the mask plane?
# Set this to match the afw code
IgnoreKernelZeroPixels = True

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def refConvolve(imMaskVar, kernel, doNormalize, copyEdge):
    """Reference code to convolve a kernel with masked image data.
    
    Does NOT normalize the kernel.

    Warning: slow (especially for spatially varying kernels).
    
    Inputs:
    - imMaskVar: (image, mask, variance) numpy arrays
    - kernel: lsst::afw::Core.Kernel object
    - doNormalize: normalize the kernel
    - copyEdge: if True: copy edge pixels from input image to convolved image;
                if False: set edge pixels to the standard edge pixel (image=nan, var=inf, mask=EDGE)
    
    Border pixels (pixels too close to the edge to compute) are set to the standard edge pixel
    """
    image, mask, variance = imMaskVar
    
    if copyEdge:
        # copy input arrays to output arrays and set EDGE bit of mask; non-edge pixels will be overwritten below
        retImage = imMaskVar[0].copy()
        retMask = imMaskVar[1].copy()
        retMask += EdgeMaskPixel
        retVariance = imMaskVar[2].copy()
    else:
        # initialize output arrays to all edge pixels; non-edge pixels will be overwritten below
        retImage = numpy.zeros(image.shape, dtype=image.dtype)
        retImage[:,:] = numpy.nan
        retMask = numpy.zeros(mask.shape, dtype=mask.dtype)
        retMask[:,:] = EdgeMaskPixel
        retVariance = numpy.zeros(variance.shape, dtype=image.dtype)
        retVariance[:,:] = numpy.inf
    
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
            if IgnoreKernelZeroPixels:
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

        # provide a destination for the convolved data that contains junk
        # to verify that convolve overwrites all pixels;
        # make it a deep copy so we can mess with it without affecting self.inImage
        self.cnvMaskedImage = afwImage.MaskedImageF(fullImage, ShiftedBBox, True)

        self.width = self.maskedImage.getWidth()
        self.height = self.maskedImage.getHeight()
#         smask = afwImage.MaskU(self.maskedImage.getMask(), afwImage.BBox(afwImage.PointI(15, 17), 10, 5))
#         smask.set(0x8)

    def tearDown(self):
        del self.maskedImage
        
    def testUnityConvolution(self):
        """Verify that convolution with a centered delta function reproduces the original.
        """
        # create a delta function kernel that has 1,1 in the center
        kFunc = afwMath.IntegerDeltaFunction2D(0.0, 0.0)
        k = afwMath.AnalyticKernel(3, 3, kFunc)
        doNormalize = False
        copyEdge = False
        
        afwMath.convolve(self.cnvMaskedImage, self.maskedImage, k, doNormalize, copyEdge)

        origImMaskVarArrays = imTestUtils.arraysFromMaskedImage(self.maskedImage)
        cnvImMaskVarArrays = imTestUtils.arraysFromMaskedImage(self.cnvMaskedImage)
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
        
        imMaskVar = imTestUtils.arraysFromMaskedImage(self.maskedImage)
        for doNormalize in (False, True):
            for copyEdge in (False, True):
                analyticK.computeImage(kImg, doNormalize)
                fixedK = afwMath.FixedKernel(kImg)
    
                refCnvImMaskVar = refConvolve(imMaskVar, fixedK, doNormalize, copyEdge)
    
                afwMath.convolve(self.cnvMaskedImage, self.maskedImage, fixedK, doNormalize, copyEdge)
                cnvImMaskVar = imTestUtils.arraysFromMaskedImage(self.cnvMaskedImage)
    
                if display:
                    refMaskedImage = imTestUtils.maskedImageFromArrays(refCnvImMaskVar)
                    ds9.mtv(displayUtils.makeMosaic(self.maskedImage, refMaskedImage, self.cnvMaskedImage), frame=0)
                    if False:
                        for (x, y) in ((0, 0), (1, 0), (0, 1), (50, 50)):
                            print "Mask(%d,%d) 0x%x 0x%x" % \
                                  (x, y, refMaskedImage.getMask().get(x, y), self.cnvMaskedImage.getMask().get(x, y))
    
                errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))
                self.assert_(sameMaskPlaneDicts(self.cnvMaskedImage, self.maskedImage),
                    "Convolved mask dictionary does not match input for doNormalize=%s, copyEdge=%s" % \
                    (doNormalize, copyEdge))

    def testSeparableConvolve(self):
        """Test convolve of a separable kernel with a spatially invariant Gaussian function
        """
        kCols = 7
        kRows = 6

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        gaussFunc2 = afwMath.GaussianFunction2D(1.0, 1.0)
        separableKernel = afwMath.SeparableKernel(kCols, kRows, gaussFunc1, gaussFunc1)
        analyticKernel = afwMath.AnalyticKernel(kCols, kRows, gaussFunc2)
                
        imMaskVar = imTestUtils.arraysFromMaskedImage(self.maskedImage)
        for doNormalize in (False, True):
            for copyEdge in (False, True):
                refCnvImMaskVar = refConvolve(imMaskVar, analyticKernel, doNormalize, copyEdge)

                afwMath.convolve(self.cnvMaskedImage, self.maskedImage, separableKernel, doNormalize, copyEdge)
                cnvImMaskVar = imTestUtils.arraysFromMaskedImage(self.cnvMaskedImage)
    
                errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))
                self.assert_(sameMaskPlaneDicts(self.cnvMaskedImage, self.maskedImage),
                    "Convolved mask dictionary does not match input for doNormalize=%s, copyEdge=%s" % \
                    (doNormalize, copyEdge))

    def testSpatiallyInvariantConvolve(self):
        """Test convolution with a spatially invariant Gaussian function
        """
        kCols = 6
        kRows = 7

        kFunc =  afwMath.GaussianFunction2D(1.5, 2.5)
        k = afwMath.AnalyticKernel(kCols, kRows, kFunc)

        imMaskVar = imTestUtils.arraysFromMaskedImage(self.maskedImage)
        for doNormalize in (False, True):
            for copyEdge in (False, True):
                refCnvImMaskVar = refConvolve(imMaskVar, k, doNormalize, copyEdge)
    
                afwMath.convolve(self.cnvMaskedImage, self.maskedImage, k, doNormalize, copyEdge)
                cnvImMaskVar = imTestUtils.arraysFromMaskedImage(self.cnvMaskedImage)
    
                if display:
                    refMaskedImage = imTestUtils.maskedImageFromArrays(refCnvImMaskVar)
                    ds9.mtv(displayUtils.makeMosaic(self.maskedImage, refMaskedImage, self.cnvMaskedImage), frame=0)
                    if False:
                        for (x, y) in ((0, 0), (1, 0), (0, 1), (50, 50)):
                            print "Mask(%d,%d) 0x%x 0x%x" % \
                                  (x, y, refMaskedImage.getMask().get(x, y), self.cnvMaskedImage.getMask().get(x, y))
    
                errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))
                self.assert_(sameMaskPlaneDicts(self.cnvMaskedImage, self.maskedImage),
                    "Convolved mask dictionary does not match input for doNormalize=%s, copyEdge=%s" % \
                    (doNormalize, copyEdge))

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
        
        imMaskVar = imTestUtils.arraysFromMaskedImage(self.maskedImage)
        for doNormalize in (False, True):
            for copyEdge in (False, True):
                refCnvImMaskVar = refConvolve(imMaskVar, k, doNormalize, copyEdge)

                afwMath.convolve(self.cnvMaskedImage, self.maskedImage, k, doNormalize, copyEdge)
                cnvImMaskVar = imTestUtils.arraysFromMaskedImage(self.cnvMaskedImage)
        
                errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))
                self.assert_(sameMaskPlaneDicts(self.cnvMaskedImage, self.maskedImage),
                    "Convolved mask dictionary does not match input for doNormalize=%s, copyEdge=%s" % \
                    (doNormalize, copyEdge))

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
                
        imMaskVar = imTestUtils.arraysFromMaskedImage(self.maskedImage)
        for doNormalize in (False, True):
            for copyEdge in (False, True):
                refCnvImMaskVar = refConvolve(imMaskVar, analyticKernel, doNormalize, copyEdge)

                afwMath.convolve(self.cnvMaskedImage, self.maskedImage, separableKernel, doNormalize, copyEdge)
                cnvImMaskVar = imTestUtils.arraysFromMaskedImage(self.cnvMaskedImage)
        
                errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))
                self.assert_(sameMaskPlaneDicts(self.cnvMaskedImage, self.maskedImage),
                    "Convolved mask dictionary does not match input for doNormalize=%s, copyEdge=%s" % \
                    (doNormalize, copyEdge))
    
    def testDeltaConvolve(self):
        """Test convolution with various delta function kernels using optimized code
        """
        doNormalize = True
        
        imMaskVar = imTestUtils.arraysFromMaskedImage(self.maskedImage)
        for copyEdge in (False, True):
            for kCols in range(1, 4):
                for kRows in range(1, 4):
                    for activeCol in range(kCols):
                        for activeRow in range(kRows):
                            kernel = afwMath.DeltaFunctionKernel(kCols, kRows, afwImage.PointI(activeCol, activeRow))
    
                            refCnvImMaskVar = refConvolve(imMaskVar, kernel, doNormalize, copyEdge)

                            afwMath.convolve(self.cnvMaskedImage, self.maskedImage, kernel, doNormalize, copyEdge)
                            cnvImMaskVar = imTestUtils.arraysFromMaskedImage(self.cnvMaskedImage)
                    
                            if display and False:
                                refMaskedImage = imTestUtils.maskedImageFromArrays(refCnvImMaskVar)
    
                                if False:
                                    for (x, y) in ((0,0), (0, 11)) :
                                        print "Mask %dx%d:(%d,%d): At (%d,%d) 0x%x  0x%x" % (
                                            kCols, kRows, activeCol, activeRow,
                                            x, y,
                                            refMaskedImage.getMask().get(x,y),
                                            self.cnvMaskedImage.getMask().get(x,y))
                                ds9.mtv(displayUtils.makeMosaic(refMaskedImage, self.cnvMaskedImage), frame=0)
    
                            errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
                            if errStr:
                                self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))

    def testSpatiallyVaryingGaussianLinerCombination(self):
        """Test convolution with a spatially varying LinearCombinationKernel.
        """
        kCols = 5
        kRows = 5

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

        # add True once ticket #833 is resolved: support normalization of convolution with
        # spatially varying LinearCombinationKernel)
        for doNormalize in (False,): # True):
            for copyEdge in (False, True):
                refCnvImMaskVar = refConvolve(imMaskVar, lcKernel, doNormalize, copyEdge)
    
                afwMath.convolve(self.cnvMaskedImage, self.maskedImage, lcKernel, doNormalize, copyEdge)
                cnvImMaskVar = imTestUtils.arraysFromMaskedImage(self.cnvMaskedImage)
        
                if display:
                    refMaskedImage = imTestUtils.maskedImageFromArrays(refCnvImMaskVar)
                    ds9.mtv(displayUtils.makeMosaic(refMaskedImage, self.cnvMaskedImage), frame=0)
    
                errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))
                self.assert_(sameMaskPlaneDicts(self.cnvMaskedImage, self.maskedImage),
                    "Convolved mask dictionary does not match input for doNormalize=%s, copyEdge=%s" % \
                    (doNormalize, copyEdge))

    def testSpatiallyVaryingDeltaFunctionLinearCombination(self):
        """Test convolution with a spatially varying LinearCombinationKernel using some delta basis kernels.
        """
        kCols = 2
        kRows = 2

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

        # add True once ticket #833 is resolved: support normalization of convolution with
        # spatially varying LinearCombinationKernel)
        for doNormalize in (False,): # True):
            for copyEdge in (False, True):
                refCnvImMaskVar = refConvolve(imMaskVar, lcKernel, doNormalize, copyEdge)
    
                pexLog.Debug("lsst.afw").debug(3, "Start convolution with delta functions")
                afwMath.convolve(self.cnvMaskedImage, self.maskedImage, lcKernel, doNormalize, copyEdge)
                pexLog.Debug("lsst.afw").debug(3, "End convolution with delta functions")
                cnvImMaskVar = imTestUtils.arraysFromMaskedImage(self.cnvMaskedImage)
                
                if display:
                    refMaskedImage = imTestUtils.maskedImageFromArrays(refCnvImMaskVar)
                    ds9.mtv(displayUtils.makeMosaic(refMaskedImage, self.cnvMaskedImage), frame=0)
    
                errStr = imTestUtils.maskedImagesDiffer(cnvImMaskVar, refCnvImMaskVar)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))
                self.assert_(sameMaskPlaneDicts(self.cnvMaskedImage, self.maskedImage),
                    "Convolved mask dictionary does not match input for doNormalize=%s, copyEdge=%s" % \
                    (doNormalize, copyEdge))

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

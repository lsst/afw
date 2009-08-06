#!/usr/bin/env python

"""Test lsst.afwMath.convolve

The convolve function is overloaded in two flavors:
- in-place convolve: user supplies the output image as an argument
- new-image convolve: the convolve function returns the convolved image
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

try:
    Verbosity
except NameError:
    Verbosity = 0 # increase to see trace
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
InputImagePath = os.path.join(dataDir, "med_img.fits")
InputBBox = afwImage.BBox(afwImage.PointI(50, 500), 100, 100)
# the shifted BBox is for a same-sized region containing different pixels;
# this is used to initialize the convolved image, to make sure convolve fully overwrites it
ShiftedBBox = afwImage.BBox(afwImage.PointI(50, 450), 100, 100)
FullImage = afwImage.ImageF(InputImagePath)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def refConvolve(image, kernel, doNormalize, copyEdge):
    """Reference code to convolve a kernel with image data.
    
    Does NOT normalize the kernel.

    Warning: slow (especially for spatially varying kernels).
    
    Inputs:
    - image: image to be convolved (a numpy array)
    - kernel: lsst::afw::Core.Kernel object
    - doNormalize: normalize the kernel
    - copyEdge: if True: copy edge pixels from input image to convolved image;
                if False: set edge pixels to NaN
    
    Border pixels (pixels too close to the edge to compute) are copied from the input.
    """
    if copyEdge:
        # copy input image; non-edge pixels will be overwritten below
        retImage = image.copy()
    else:
        # initialize input data to nan; non-edge pixels will be overwritten below
        retImage = numpy.zeros(image.shape, dtype=image.dtype)
        retImage[:,:] = numpy.nan
    
    kCols = kernel.getWidth()
    kRows = kernel.getHeight()
    numCols = image.shape[0] + 1 - kCols
    numRows = image.shape[1] + 1 - kRows
    if numCols < 0 or numRows < 0:
        raise RuntimeError("image must be larger than kernel in both dimensions")
    colRange = range(numCols)

    isSpatiallyVarying = kernel.isSpatiallyVarying()
    kImage = afwImage.ImageD(kCols, kRows)
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
        kFunc = afwMath.GaussianFunction2D(1.5, 2.5) # XXX ? xSigma, ySigma?
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

class ConvolveTestCase(unittest.TestCase):
    def setUp(self):
        self.width, self.height = 45, 55

        self.inImage = afwImage.ImageF(FullImage, InputBBox, True)
        
        # provide a destination for the convolved data that contains junk
        # to verify that convolve overwrites all pixels;
        # make it a deep copy so we can mess with it without affecting self.inImage
        self.cnvImage = afwImage.ImageF(FullImage, ShiftedBBox, True)

    def tearDown(self):
        del self.inImage
        
    def testUnityConvolution(self):
        """Verify that convolution with a centered delta function reproduces the original.
        """
        # create a delta function kernel that has 1,1 in the center
        kFunc = afwMath.IntegerDeltaFunction2D(0.0, 0.0)
        k = afwMath.AnalyticKernel(3, 3, kFunc)
        doNormalize = False
        copyEdge = False

        afwMath.convolve(self.cnvImage, self.inImage, k, doNormalize, copyEdge)
    
        if display:
            ds9.mtv(displayUtils.makeMosaic(self.inImage, self.cnvImage))

        origImageArr = imTestUtils.arrayFromImage(self.inImage)
        cnvImageArr = imTestUtils.arrayFromImage(self.cnvImage)
        skipMaskArr = numpy.isnan(cnvImageArr)
        errStr = imTestUtils.imagesDiffer(cnvImageArr, origImageArr, skipMaskArr)
        if errStr:
            self.fail(errStr)

    def testFixedKernelConvolve(self):
        """Test convolve with a fixed kernel
        """
        kCols = 7
        kRows = 6

        kFunc =  afwMath.GaussianFunction2D(1.5, 2.5)
        analyticK = afwMath.AnalyticKernel(kCols, kRows, kFunc)
        kImg = afwImage.ImageD(kCols, kRows)

        inImageArr = imTestUtils.arrayFromImage(self.inImage)
        
        for doNormalize in (False, True):
            for copyEdge in (False, True):
                analyticK.computeImage(kImg, doNormalize)
                fixedK = afwMath.FixedKernel(kImg)

                refCnvImageArr = refConvolve(inImageArr, fixedK, doNormalize, copyEdge)
    
                afwMath.convolve(self.cnvImage, self.inImage, fixedK, doNormalize, copyEdge)
                cnvImageArr = imTestUtils.arrayFromImage(self.cnvImage)
    
                if doNormalize and display and True:    # display as two panels
                    ds9.mtv(displayUtils.makeMosaic(self.inImage, self.cnvImage))
        
                errStr = imTestUtils.imagesDiffer(cnvImageArr, refCnvImageArr)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))

    def testSpatiallyInvariantConvolve(self):
        """Test convolve with a spatially invariant Gaussian function
        """
        kCols = 7
        kRows = 6

        kFunc =  afwMath.GaussianFunction2D(1.5, 2.5)
        k = afwMath.AnalyticKernel(kCols, kRows, kFunc)
        
        inImageArr = imTestUtils.arrayFromImage(self.inImage)
        for doNormalize in (False, True):
            for copyEdge in (False, True):
                refCnvImageArr = refConvolve(inImageArr, k, doNormalize, copyEdge)

                afwMath.convolve(self.cnvImage, self.inImage, k, doNormalize, copyEdge)
    
                if doNormalize and display and True:    # display as two panels
                    ds9.mtv(displayUtils.makeMosaic(self.inImage, self.cnvImage))
    
                cnvImageArr = imTestUtils.arrayFromImage(self.cnvImage)
        
                errStr = imTestUtils.imagesDiffer(cnvImageArr, refCnvImageArr)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))

    def testSpatiallyVaryingConvolve(self):
        """Test convolve with a spatially varying Gaussian function
        """
        kCols = 7
        kRows = 6

        # create spatially model
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
                
        inImageArr = imTestUtils.arrayFromImage(self.inImage)

        for doNormalize in (False, True):
            for copyEdge in (False, True):
                refCnvImageArr= refConvolve(inImageArr, k, doNormalize, copyEdge)

                afwMath.convolve(self.cnvImage, self.inImage, k, doNormalize, copyEdge)
                cnvImageArr = imTestUtils.arrayFromImage(self.cnvImage)
        
                errStr = imTestUtils.imagesDiffer(cnvImageArr, refCnvImageArr)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))

    def testSeparableConvolve(self):
        """Test convolve of a separable kernel with a spatially invariant Gaussian function
        """
        kCols = 7
        kRows = 6

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        gaussFunc2 = afwMath.GaussianFunction2D(1.0, 1.0)
        separableKernel = afwMath.SeparableKernel(kCols, kRows, gaussFunc1, gaussFunc1)
        analyticKernel = afwMath.AnalyticKernel(kCols, kRows, gaussFunc2)
                
        inImageArr = imTestUtils.arrayFromImage(self.inImage)

        for doNormalize in (False, True):
            for copyEdge in (False, True):
                refCnvImageArr = refConvolve(inImageArr, analyticKernel, doNormalize, copyEdge)
    
                afwMath.convolve(self.cnvImage, self.inImage, separableKernel, doNormalize, copyEdge)
                cnvImageArr = imTestUtils.arrayFromImage(self.cnvImage)
    
                errStr = imTestUtils.imagesDiffer(cnvImageArr, refCnvImageArr)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))

    def testSpatiallyVaryingSeparableConvolve(self):
        """Test convolve of a separable kernel with a spatially varying Gaussian function
        """
        kCols = 7
        kRows = 6

        # create spatially model
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
                
        inImageArr = imTestUtils.arrayFromImage(self.inImage)
        for doNormalize in (False, True):
            for copyEdge in (False, True):
                refCnvImageArr = refConvolve(inImageArr, analyticKernel, doNormalize, copyEdge)

                afwMath.convolve(self.cnvImage, self.inImage, separableKernel, doNormalize, copyEdge)
                cnvImageArr = imTestUtils.arrayFromImage(self.cnvImage)
        
                errStr = imTestUtils.imagesDiffer(cnvImageArr, refCnvImageArr)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))
    
    def testDeltaConvolve(self):
        """Test convolve with various delta function kernels using optimized code
        """
        doNormalize = True
        
        inImageArr = imTestUtils.arrayFromImage(self.inImage)
        for copyEdge in (False, True):
            for kCols in range(1, 4):
                for kRows in range(1, 4):
                    for activeCol in range(kCols):
                        for activeRow in range(kRows):
                            kernel = afwMath.DeltaFunctionKernel(kCols, kRows, afwImage.PointI(activeCol, activeRow))

                            refCnvImageArr = refConvolve(inImageArr, kernel, doNormalize, copyEdge)
                            
                            afwMath.convolve(self.cnvImage, self.inImage, kernel, doNormalize, copyEdge)
                            cnvImageArr= imTestUtils.arrayFromImage(self.cnvImage)
                    
                            errStr = imTestUtils.imagesDiffer(cnvImageArr, refCnvImageArr)
                            if errStr:
                                self.fail("%s (for doNormalize=%s, copyEdge=%s, kCols=%s, kRows=%s, activeCol=%s, activeRow=%s)" % \
                                    (errStr, doNormalize, copyEdge, kCols, kRows, activeCol, activeRow))
    
    def testSpatiallyVaryingGaussianLinerCombination(self):
        """Test convolution with a spatially varying LinearCombinationKernel.
        """
        kCols = 5
        kRows = 5

        # create spatially model
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

        inImageArr = imTestUtils.arrayFromImage(self.inImage)

        for doNormalize in (False, True):
            for copyEdge in (False, True):
                refCnvImageArr = refConvolve(inImageArr, lcKernel, doNormalize, copyEdge)
                
                afwMath.convolve(self.cnvImage, self.inImage, lcKernel, doNormalize, copyEdge)
                cnvImageArr = imTestUtils.arrayFromImage(self.cnvImage)
                
                if display:
                    refImage = imTestUtils.imageFromArray(refCnvImageArr)
                    ds9.mtv(displayUtils.makeMosaic(self.inImage, self.cnvImage, refImage))
    
                errStr = imTestUtils.imagesDiffer(cnvImageArr, refCnvImageArr)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))

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

        inImageArr = imTestUtils.arrayFromImage(self.inImage)

        for doNormalize in (False, True):
            for copyEdge in (False, True):
                refCnvImageArr = refConvolve(inImageArr, lcKernel, doNormalize, copyEdge)
                
                # the debug statements show whether the delta function specialization is used for convolution;
                # they use the same verbosity as dispatch trace statements in the generic version of basicConvolve
                pexLog.Debug("lsst.afw").debug(4, "Start convolution with delta functions")
                afwMath.convolve(self.cnvImage, self.inImage, lcKernel, doNormalize, copyEdge)
                pexLog.Debug("lsst.afw").debug(4, "End convolution with delta functions")
                cnvImageArr = imTestUtils.arrayFromImage(self.cnvImage)
                
                if display:
                    refImage = imTestUtils.imageFromArray(refCnvImageArr)
                    ds9.mtv(displayUtils.makeMosaic(self.inImage, self.cnvImage, refImage))
    
                errStr = imTestUtils.imagesDiffer(cnvImageArr, refCnvImageArr)
                if errStr:
                    self.fail("%s (for doNormalize=%s, copyEdge=%s)" % (errStr, doNormalize, copyEdge))

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

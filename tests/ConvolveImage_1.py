#!/usr/bin/env python

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

try:
    Verbosity
except NameError:
    Verbosity = 0 # increase to see trace
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

class ConvolveTestCase(unittest.TestCase):
    def setUp(self):
        self.width, self.height = 45, 55

        fullImage = afwImage.ImageF(InputImagePath)
            
        # pick a small piece of the image to save time
        bbox = afwImage.BBox(afwImage.PointI(50, 50), self.width, self.height)
        self.inImage = afwImage.ImageF(fullImage, bbox)

    def tearDown(self):
        del self.inImage
        
    def testUnityConvolution(self):
        """Verify that convolution with a centered delta function reproduces the original.
        """
        # create a delta function kernel that has 1,1 in the center
        kFunc = afwMath.IntegerDeltaFunction2D(0.0, 0.0)
        k = afwMath.AnalyticKernel(3, 3, kFunc)

        cnvImage = afwImage.ImageF(self.inImage.getDimensions())
        afwMath.convolve(cnvImage, self.inImage, k, True)
    
        if display:
            ds9.mtv(displayUtils.makeMosaic(self.inImage, cnvImage))

        if False:
            origImageArr = imTestUtils.arrayFromImage(self.inImage)
            cnvImageArr = imTestUtils.arrayFromImage(cnvImage)
            for name, ind in (("image", 0), ("variance", 1)): # , ("mask", 2)):
                if not numpy.allclose(origImageArr, cnvImageArr):
                    self.fail("Convolved image does not match reference")

    def testFixedKernelConvolve(self):
        """Test convolve with a fixed kernel
        """
        kCols = 7
        kRows = 6

        kFunc =  afwMath.GaussianFunction2D(1.5, 2.5)
        analyticK = afwMath.AnalyticKernel(kCols, kRows, kFunc)
        kImg = afwImage.ImageD(kCols, kRows)
        
        for doNormalize in (False, True):
            analyticK.computeImage(kImg, doNormalize)
            fixedK = afwMath.FixedKernel(kImg)

            cnvImage = afwImage.ImageF(self.inImage.getDimensions())
            afwMath.convolve(cnvImage, self.inImage, fixedK, doNormalize)

            if doNormalize and display and True:    # display as two panels
                ds9.mtv(displayUtils.makeMosaic(self.inImage, cnvImage))

            cnvImageArr = imTestUtils.arrayFromImage(cnvImage)
            inImageArr = imTestUtils.arrayFromImage(self.inImage)
            refCnvImageArr = refConvolve(inImageArr, fixedK, doNormalize)
    
            if not numpy.allclose(cnvImageArr, refCnvImageArr):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)

    def testSpatiallyInvariantConvolve(self):
        """Test convolve with a spatially invariant Gaussian function
        """
        kCols = 7
        kRows = 6

        kFunc =  afwMath.GaussianFunction2D(1.5, 2.5)
        k = afwMath.AnalyticKernel(kCols, kRows, kFunc)
        
        for doNormalize in (False, True):
            cnvImage = afwImage.ImageF(self.inImage.getDimensions())
            afwMath.convolve(cnvImage, self.inImage, k, doNormalize)

            if doNormalize and display and True:    # display as two panels
                ds9.mtv(displayUtils.makeMosaic(self.inImage, cnvImage))

            cnvImageArr = imTestUtils.arrayFromImage(cnvImage)
            inImageArr = imTestUtils.arrayFromImage(self.inImage)
            refCnvImageArr = refConvolve(inImageArr, k, doNormalize)
    
            if not numpy.allclose(cnvImageArr, refCnvImageArr):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)

    def testSpatiallyVaryingConvolve(self):
        """Test convolve with a spatially varying Gaussian function
        """
        kCols = 7
        kRows = 6

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
                
        cnvImage = afwImage.ImageF(self.inImage.getDimensions())
        for doNormalize in (False, True):
            afwMath.convolve(cnvImage, self.inImage, k, doNormalize)
            cnvImageArr = imTestUtils.arrayFromImage(cnvImage)
    
            inImageArr = imTestUtils.arrayFromImage(self.inImage)
            refCnvImageArr= refConvolve(inImageArr, k, doNormalize)
    
            if not numpy.allclose(cnvImageArr, refCnvImageArr):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)

    def testSpatiallyVaryingSeparableConvolve(self):
        """Test convolve of a separable kernel with a spatially varying Gaussian function
        """
        kCols = 7
        kRows = 6

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
                
        cnvImage = afwImage.ImageF(self.inImage.getDimensions())
        for doNormalize in (False, True):
            afwMath.convolve(cnvImage, self.inImage, separableKernel, doNormalize)

            cnvImageArr = imTestUtils.arrayFromImage(cnvImage)
    
            inImageArr = imTestUtils.arrayFromImage(self.inImage)
            refCnvImageArr = refConvolve(inImageArr, analyticKernel, doNormalize)

            if not numpy.allclose(cnvImageArr, refCnvImageArr):
                self.fail("Convolved image does not match reference for doNormalize=%s" % doNormalize)
    
    def testDeltaConvolve(self):
        """Test convolve with various delta function kernels using optimized code
        """
        doNormalize = True
        
        for kCols in range(1, 4):
            for kRows in range(1, 4):
                for activeCol in range(kCols):
                    for activeRow in range(kRows):
                        kernel = afwMath.DeltaFunctionKernel(kCols, kRows, afwImage.PointI(activeCol, activeRow))
                        
                        refCnvImage = afwImage.ImageF(self.inImage.getDimensions())
                        afwMath.convolve(refCnvImage, self.inImage, kernel, doNormalize)
                        refCnvImageArr= imTestUtils.arrayFromImage(refCnvImage)
                
                        inImageArr = imTestUtils.arrayFromImage(self.inImage)
                        ref2CnvImageArr = refConvolve(inImageArr, kernel, doNormalize)
                
                        if not numpy.allclose(refCnvImageArr, ref2CnvImageArr):
                            print "kCols=%s, kRows=%s, refCnvImageArr=%r, ref2CnvImageArr=%r" % (kCols, kRows, refCnvImageArr, ref2CnvImageArr)
                            self.fail("Image from afwMath.convolve does not match image from refConvolve")
        

    def testConvolveLinear(self):
        """Test convolution with a spatially varying LinearCombinationKernel
        by comparing the results of afwMath.convolveLinear to afwMath.convolve or refConvolve,
        depending on the value of compareToFwConvolve.
        """
        kCols = 5
        kRows = 5
        doNormalize = False # must be false because convolveLinear cannot normalize

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

        cnvImage = afwImage.ImageF(self.inImage.getDimensions())
        afwMath.convolve(cnvImage, self.inImage, lcKernel, doNormalize)
        cnvImageArr = imTestUtils.arrayFromImage(cnvImage)

        inImageArr = imTestUtils.arrayFromImage(self.inImage)
        refCnvImageArr = refConvolve(inImageArr, lcKernel, doNormalize)
        
        if not numpy.allclose(cnvImageArr, refCnvImageArr):
            self.fail("Image from afwMath.convolve does not match image from refConvolve")

        cnvImage = afwImage.ImageF(self.inImage.getDimensions())
        # compute twice, to be sure cnvImage is properly reset
        for ii in range(2):        
            afwMath.convolveLinear(cnvImage, self.inImage, lcKernel)
            cnvImageArr = imTestUtils.arrayFromImage(cnvImage)
            
            if display:
                refImage = self.inImage.Factory(self.inImage.getDimensions())
                imTestUtils.imageFromArray(refImage, refCnvImageArr)
                ds9.mtv(displayUtils.makeMosaic(self.inImage, cnvImage, refImage))

            if not numpy.allclose(cnvImageArr, refCnvImageArr):
                self.fail("Image from afwMath.convolveLinear does not match image from refConvolve in iter %d" % ii)

    def testConvolveLinearNewImage(self):
        """Test convolveLinearNew
        """
        kCols = 5
        kRows = 5
        doNormalize = False # must be false because convolveLinear cannot normalize

        # create spatially varying linear combination kernel
        sFunc = afwMath.PolynomialFunction2D(1)
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, -0.5 / self.width, -0.5 / self.height),
            (0.0,  1.0 / self.width,  0.0 / self.height),
            (0.0,  0.0 / self.width,  1.0 / self.height),
        )
        
        kVec = makeGaussianKernelVec(kCols, kRows)
        lcKernel = afwMath.LinearCombinationKernel(kVec, sFunc)
        lcKernel.setSpatialParameters(sParams)

        refCnvImage = afwImage.ImageF(self.inImage.getDimensions())
        afwMath.convolve(refCnvImage, self.inImage, lcKernel, doNormalize)
        refCnvImageArr = imTestUtils.arrayFromImage(refCnvImage)

        inImageArr = imTestUtils.arrayFromImage(self.inImage)
        ref2CnvImageArr = refConvolve(inImageArr, lcKernel, doNormalize)

        if not numpy.allclose(refCnvImageArr, ref2CnvImageArr):
            self.fail("Image from afwMath.convolve does not match image from refConvolve")

        # compute twice, to be sure cnvImage is properly reset
        cnvImage = afwImage.ImageF(self.inImage.getDimensions())
        for ii in range(2):        
            afwMath.convolveLinear(cnvImage, self.inImage, lcKernel)
            cnvImageArr = imTestUtils.arrayFromImage(cnvImage)
    
            if not numpy.allclose(cnvImageArr, ref2CnvImageArr):
                self.fail("Image from afwMath.convolveLinearNew does not match image from refConvolve in iter %d" % ii)

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

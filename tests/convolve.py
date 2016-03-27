#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

"""Test lsst.afwMath.convolve

Tests convolution of various kernels with Images and MaskedImages.
"""
import math
import os
import os.path
import unittest
import string

import numpy

import lsst.utils
import lsst.utils.tests as utilsTests
import lsst.pex.logging as pexLog
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.math.detail as mathDetail
from kernel import makeDeltaFunctionKernelList, makeGaussianKernelList

VERBOSITY = 0   # increase to see trace; 3 will show the convolutions specializations being used

pexLog.Debug("lsst.afw", VERBOSITY)

try:
    display
except NameError:
    display = False

import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

dataDir = os.path.join(lsst.utils.getPackageDir("afwdata"), "data")

# input image contains a saturated star, a bad column, and a faint star
InputMaskedImagePath = os.path.join(dataDir, "medexp.fits")
InputBBox = afwGeom.Box2I(afwGeom.Point2I(52, 574), afwGeom.Extent2I(76, 80))
# the shifted BBox is for a same-sized region containing different pixels;
# this is used to initialize the convolved image, to make sure convolve fully overwrites it
ShiftedBBox = afwGeom.Box2I(afwGeom.Point2I(0, 460), afwGeom.Extent2I(76, 80))
FullMaskedImage = afwImage.MaskedImageF(InputMaskedImagePath)

EdgeMaskPixel = 1 << afwImage.MaskU.getMaskPlane("EDGE")
NoDataMaskPixel = afwImage.MaskU.getPlaneBitMask("NO_DATA")

# Ignore kernel pixels whose value is exactly 0 when smearing the mask plane?
# Set this to match the afw code
IgnoreKernelZeroPixels = True

NullTranslator = string.maketrans("", "")
GarbageChars = string.punctuation + string.whitespace

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def refConvolve(imMaskVar, xy0, kernel, doNormalize, doCopyEdge):
    """Reference code to convolve a kernel with a masked image.

    Warning: slow (especially for spatially varying kernels).
    
    Inputs:
    - imMaskVar: (image, mask, variance) numpy arrays
    - xy0: xy offset of imMaskVar relative to parent image
    - kernel: lsst::afw::Core.Kernel object
    - doNormalize: normalize the kernel
    - doCopyEdge: if True: copy edge pixels from input image to convolved image;
                if False: set edge pixels to the standard edge pixel (image=nan, var=inf, mask=EDGE)
    """
    # Note: the original version of this function was written when numpy/image conversions were the
    # transpose of what they are today.  Rather than transpose the logic in this function or put
    # transposes throughout the rest of the file, I have transposed only the inputs and outputs.
    #  - Jim Bosch, 3/4/2011
    image, mask, variance = (imMaskVar[0].transpose(), imMaskVar[1].transpose(), imMaskVar[2].transpose())
    
    if doCopyEdge:
        # copy input arrays to output arrays and set EDGE bit of mask; non-edge pixels are overwritten below
        retImage = image.copy()
        retMask = mask.copy()
        retMask += EdgeMaskPixel
        retVariance = variance.copy()
    else:
        # initialize output arrays to all edge pixels; non-edge pixels will be overwritten below
        retImage = numpy.zeros(image.shape, dtype=image.dtype)
        retImage[:, :] = numpy.nan
        retMask = numpy.zeros(mask.shape, dtype=mask.dtype)
        retMask[:, :] = NoDataMaskPixel
        retVariance = numpy.zeros(variance.shape, dtype=image.dtype)
        retVariance[:, :] = numpy.inf
    
    kWidth = kernel.getWidth()
    kHeight = kernel.getHeight()
    numCols = image.shape[0] + 1 - kWidth
    numRows = image.shape[1] + 1 - kHeight
    if numCols < 0 or numRows < 0:
        raise RuntimeError("image must be larger than kernel in both dimensions")
    colRange = range(numCols)


    kImage = afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight))
    isSpatiallyVarying = kernel.isSpatiallyVarying()
    if not isSpatiallyVarying:
        kernel.computeImage(kImage, doNormalize)
        kImArr = kImage.getArray().transpose()

    retRow = kernel.getCtrY()
    for inRowBeg in range(numRows):
        inRowEnd = inRowBeg + kHeight
        retCol = kernel.getCtrX()
        if isSpatiallyVarying:
            rowPos = afwImage.indexToPosition(retRow) + xy0[1]
        for inColBeg in colRange:
            if isSpatiallyVarying:
                colPos = afwImage.indexToPosition(retCol) + xy0[0]
                kernel.computeImage(kImage, doNormalize, colPos, rowPos)
                kImArr = kImage.getArray().transpose()
            inColEnd = inColBeg + kWidth
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
    return [numpy.copy(numpy.transpose(arr), order="C") for arr in (retImage, retMask, retVariance)]

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

class ConvolveTestCase(utilsTests.TestCase):
    def setUp(self):
        self.maskedImage = afwImage.MaskedImageF(FullMaskedImage, InputBBox, afwImage.LOCAL, True)
        # use a huge XY0 to make emphasize any errors related to not handling xy0 correctly.
        self.maskedImage.setXY0(300, 200)
        self.xy0 = self.maskedImage.getXY0()

        # provide destinations for the convolved MaskedImage and Image that contain junk
        # to verify that convolve overwrites all pixels;
        # make them deep copies so we can mess with them without affecting self.inImage
        self.cnvMaskedImage = afwImage.MaskedImageF(FullMaskedImage, ShiftedBBox, afwImage.LOCAL, True)
        self.cnvImage = afwImage.ImageF(FullMaskedImage.getImage(), ShiftedBBox, afwImage.LOCAL, True)

        self.width = self.maskedImage.getWidth()
        self.height = self.maskedImage.getHeight()
#         smask = afwImage.MaskU(self.maskedImage.getMask(), afwGeom.Box2I(afwGeom.Point2I(15, 17), afwGeom.Extent2I(10, 5)))
#         smask.set(0x8)

    def tearDown(self):
        del self.maskedImage
        del self.cnvMaskedImage
        del self.cnvImage

    def runBasicTest(self, kernel, convControl, refKernel=None, kernelDescr="", rtol=1.0e-05, atol=1e-08): 
        """Assert that afwMath::convolve gives the same result as reference convolution for a given kernel.
        
        Inputs:
        - kernel: convolution kernel
        - convControl: convolution control parameters (afwMath.ConvolutionControl)
        - refKernel: kernel to use for refConvolve (if None then kernel is used)
        - kernelDescr: description of kernel
        - rtol: relative tolerance (see below)
        - atol: absolute tolerance (see below)
        
        rtol and atol are positive, typically very small numbers.
        The relative difference (rtol * abs(b)) and the absolute difference "atol" are added together
        to compare against the absolute difference between "a" and "b".
        """
        if refKernel == None:
            refKernel = kernel
        # strip garbage characters (whitespace and punctuation) to make a short description for saving files
        shortKernelDescr = kernelDescr.translate(NullTranslator, GarbageChars)

        doNormalize = convControl.getDoNormalize()
        doCopyEdge = convControl.getDoCopyEdge()
        maxInterpDist = convControl.getMaxInterpolationDistance()

        imMaskVar = self.maskedImage.getArrays()
        xy0 = self.maskedImage.getXY0()
        
        refCnvImMaskVarArr = refConvolve(imMaskVar, xy0, refKernel, doNormalize, doCopyEdge)
        refMaskedImage = afwImage.makeMaskedImageFromArrays(*refCnvImMaskVarArr)

        afwMath.convolve(self.cnvImage, self.maskedImage.getImage(), kernel, convControl)
        self.assertEqual(self.cnvImage.getXY0(), self.xy0)

        afwMath.convolve(self.cnvMaskedImage, self.maskedImage, kernel, convControl)

        if display and False:
            ds9.mtv(displayUtils.Mosaic().makeMosaic([
                self.maskedImage, refMaskedImage, self.cnvMaskedImage]), frame=0)
            if False:
                for (x, y) in ((0, 0), (1, 0), (0, 1), (50, 50)):
                    print "Mask(%d,%d) 0x%x 0x%x" % (x, y, refMaskedImage.getMask().get(x, y),
                    self.cnvMaskedImage.getMask().get(x, y))

        self.assertImagesNearlyEqual(self.cnvImage, refMaskedImage.getImage(), atol=atol, rtol=rtol)
        self.assertMaskedImagesNearlyEqual(self.cnvMaskedImage, refMaskedImage, atol=atol, rtol=rtol)

        if not sameMaskPlaneDicts(self.cnvMaskedImage, self.maskedImage):
            self.cnvMaskedImage.writeFits("act%s" % (shortKernelDescr,))
            refMaskedImage.writeFits("des%s" % (shortKernelDescr,))
            self.fail("convolve(MaskedImage, kernel=%s, doNormalize=%s, doCopyEdge=%s, maxInterpDist=%s) failed:\n%s" % \
                (kernelDescr, doNormalize, doCopyEdge, maxInterpDist, "convolved mask dictionary does not match input"))

    def runStdTest(self, kernel, refKernel=None, kernelDescr="", rtol=1.0e-05, atol=1e-08,
        maxInterpDist=10):
        """Assert that afwMath::convolve gives the same result as reference convolution for a given kernel.
        
        Inputs:
        - kernel: convolution kernel
        - refKernel: kernel to use for refConvolve (if None then kernel is used)
        - kernelDescr: description of kernel
        - rtol: relative tolerance (see below)
        - atol: absolute tolerance (see below)
        - maxInterpDist: maximum allowed distance for linear interpolation during convolution
        
        rtol and atol are positive, typically very small numbers.
        The relative difference (rtol * abs(b)) and the absolute difference "atol" are added together
        to compare against the absolute difference between "a" and "b".
        """
        if VERBOSITY > 0:
            print "Test convolution with", kernelDescr
        
        convControl = afwMath.ConvolutionControl()
        convControl.setMaxInterpolationDistance(maxInterpDist)
        
        # verify dimension assertions:
        # - output image dimensions = input image dimensions
        # - input image width and height >= kernel width and height
        # Note: the assertion kernel size > 0 is tested elsewhere
        for inWidth in (kernel.getWidth() - 1, self.width-1, self.width, self.width + 1):
            for inHeight in (kernel.getHeight() - 1, self.width-1, self.width, self.width + 1):
                if (inWidth == self.width) and (inHeight == self.height):
                    continue
                inMaskedImage = afwImage.MaskedImageF(afwGeom.Extent2I(inWidth, inHeight))
                self.assertRaises(Exception, afwMath.convolve, self.cnvMaskedImage, inMaskedImage, kernel)

        for doNormalize in (True,): # (False, True):
            convControl.setDoNormalize(doNormalize)
            for doCopyEdge in (False,): # (False, True):
                convControl.setDoCopyEdge(doCopyEdge)
                self.runBasicTest(kernel, convControl=convControl, refKernel=refKernel,
                    kernelDescr=kernelDescr, rtol=rtol, atol=atol)

        # verify that basicConvolve does not write to edge pixels
        self.runBasicConvolveEdgeTest(kernel, kernelDescr)

    def runBasicConvolveEdgeTest(self, kernel, kernelDescr):
        """Verify that basicConvolve does not write to edge pixels for this kind of kernel
        """
        fullBox = afwGeom.Box2I(
            afwGeom.Point2I(0, 0),
            ShiftedBBox.getDimensions(),
        )
        goodBox = kernel.shrinkBBox(fullBox)
        cnvMaskedImage = afwImage.MaskedImageF(FullMaskedImage, ShiftedBBox, afwImage.LOCAL, True)
        cnvMaskedImageCopy = afwImage.MaskedImageF(cnvMaskedImage, fullBox, afwImage.LOCAL, True)
        cnvMaskedImageCopyViewOfGoodRegion = afwImage.MaskedImageF(cnvMaskedImageCopy, goodBox, afwImage.LOCAL, False)

        # convolve with basicConvolve, which should leave the edge pixels alone
        convControl = afwMath.ConvolutionControl()
        mathDetail.basicConvolve(cnvMaskedImage, self.maskedImage, kernel, convControl)

        # reset the good region to the original convolved image;
        # this should reset the entire convolved image to its original self
        cnvMaskedImageGoodView = afwImage.MaskedImageF(cnvMaskedImage, goodBox, afwImage.LOCAL, False)
        cnvMaskedImageGoodView[:] = cnvMaskedImageCopyViewOfGoodRegion

        # assert that these two are equal
        msg = "basicConvolve(MaskedImage, kernel=%s) wrote to edge pixels" % (kernelDescr,)
        try:
            self.assertMaskedImagesNearlyEqual(cnvMaskedImage, cnvMaskedImageCopy,
                doVariance = True, rtol=0, atol=0, msg=msg)
        except Exception:
            # write out the images, then fail
            shortKernelDescr = kernelDescr.translate(NullTranslator, GarbageChars)
            cnvMaskedImage.writeFits("actBasicConvolve%s" % (shortKernelDescr,))
            cnvMaskedImageCopy.writeFits("desBasicConvolve%s" % (shortKernelDescr,))
            raise

    def testConvolutionControl(self):
        """Test the ConvolutionControl object
        """
        convControl = afwMath.ConvolutionControl()
        self.assert_(convControl.getDoNormalize())
        for doNormalize in (False, True):
            convControl.setDoNormalize(doNormalize)
            self.assertEqual(convControl.getDoNormalize(), doNormalize)
        
        self.assert_(not convControl.getDoCopyEdge())
        for doCopyEdge in (False, True):
            convControl.setDoCopyEdge(doCopyEdge)
            self.assert_(convControl.getDoCopyEdge() == doCopyEdge)
        
        self.assertEqual(convControl.getMaxInterpolationDistance(), 10)
        for maxInterpDist in (0, 1, 2, 10, 100):
            convControl.setMaxInterpolationDistance(maxInterpDist)
            self.assertEqual(convControl.getMaxInterpolationDistance(), maxInterpDist)
        
    def testUnityConvolution(self):
        """Verify that convolution with a centered delta function reproduces the original.
        """
        # create a delta function kernel that has 1,1 in the center
        kFunc = afwMath.IntegerDeltaFunction2D(0.0, 0.0)
        kernel = afwMath.AnalyticKernel(3, 3, kFunc)
        doNormalize = False
        doCopyEdge = False

        afwMath.convolve(self.cnvImage, self.maskedImage.getImage(), kernel, doNormalize, doCopyEdge)
        
        afwMath.convolve(self.cnvMaskedImage, self.maskedImage, kernel, doNormalize, doCopyEdge)
        cnvImMaskVarArr = self.cnvMaskedImage.getArrays()
        
        skipMaskArr = numpy.array(numpy.isnan(cnvImMaskVarArr[0]), dtype=numpy.uint16)

        kernelDescr = "Centered DeltaFunctionKernel (testing unity convolution)"
        self.assertImagesNearlyEqual(self.cnvImage, self.maskedImage.getImage(), skipMask=skipMaskArr, msg=kernelDescr)
        self.assertMaskedImagesNearlyEqual(self.cnvMaskedImage, self.maskedImage, skipMask=skipMaskArr, msg=kernelDescr)

    def testFixedKernelConvolve(self):
        """Test convolve with a fixed kernel
        """
        kWidth = 6
        kHeight = 7

        kFunc =  afwMath.GaussianFunction2D(2.5, 1.5, 0.5)
        analyticKernel = afwMath.AnalyticKernel(kWidth, kHeight, kFunc)
        kernelImage = afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight))
        analyticKernel.computeImage(kernelImage, False)
        fixedKernel = afwMath.FixedKernel(kernelImage)
        
        self.runStdTest(fixedKernel, kernelDescr="Gaussian FixedKernel")

    def testSeparableConvolve(self):
        """Test convolve of a separable kernel with a spatially invariant Gaussian function
        """
        kWidth = 7
        kHeight = 6

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        gaussFunc2 = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        separableKernel = afwMath.SeparableKernel(kWidth, kHeight, gaussFunc1, gaussFunc1)
        analyticKernel = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc2)
        
        self.runStdTest(
            separableKernel,
            refKernel = analyticKernel,
            kernelDescr = "Gaussian Separable Kernel (compared to AnalyticKernel equivalent)")

    def testSpatiallyInvariantConvolve(self):
        """Test convolution with a spatially invariant Gaussian function
        """
        kWidth = 6
        kHeight = 7

        kFunc =  afwMath.GaussianFunction2D(2.5, 1.5, 0.5)
        kernel = afwMath.AnalyticKernel(kWidth, kHeight, kFunc)
        
        self.runStdTest(kernel, kernelDescr="Gaussian Analytic Kernel")

    def testSpatiallyVaryingAnalyticConvolve(self):
        """Test in-place convolution with a spatially varying AnalyticKernel
        """
        kWidth = 7
        kHeight = 6

        # create spatial model
        sFunc = afwMath.PolynomialFunction2D(1)
        
        minSigma = 1.5
        maxSigma = 1.501

        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (minSigma, (maxSigma - minSigma) / self.width, 0.0),
            (minSigma, 0.0,  (maxSigma - minSigma) / self.height),
            (0.0, 0.0, 0.0),
        )

        kFunc =  afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        kernel = afwMath.AnalyticKernel(kWidth, kHeight, kFunc, sFunc)
        kernel.setSpatialParameters(sParams)
        
        for maxInterpDist, rtol, methodStr in (
            (0,   1.0e-5, "brute force"),
            (10,  1.0e-5, "interpolation over 10 x 10 pixels"),
        ):
            self.runStdTest(
                kernel,
                kernelDescr = "Spatially Varying Gaussian Analytic Kernel using %s" % (methodStr,),
                maxInterpDist = maxInterpDist,
                rtol = rtol)

    def testSpatiallyVaryingSeparableConvolve(self):
        """Test convolution with a spatially varying SeparableKernel
        """
        kWidth = 7
        kHeight = 6

        # create spatial model
        sFunc = afwMath.PolynomialFunction2D(1)

        minSigma = 0.1
        maxSigma = 3.0
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (minSigma, (maxSigma - minSigma) / self.width, 0.0),
            (minSigma, 0.0,  (maxSigma - minSigma) / self.height),
            (0.0, 0.0, 0.0),
        )

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        gaussFunc2 = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        separableKernel = afwMath.SeparableKernel(kWidth, kHeight, gaussFunc1, gaussFunc1, sFunc)
        analyticKernel = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc2, sFunc)
        separableKernel.setSpatialParameters(sParams[0:2])
        analyticKernel.setSpatialParameters(sParams)

        self.runStdTest(separableKernel, refKernel=analyticKernel,
            kernelDescr="Spatially Varying Gaussian Separable Kernel")
    
    def testDeltaConvolve(self):
        """Test convolution with various delta function kernels using optimized code
        """
        for kWidth in range(1, 4):
            for kHeight in range(1, 4):
                for activeCol in range(kWidth):
                    for activeRow in range(kHeight):
                        kernel = afwMath.DeltaFunctionKernel(kWidth, kHeight,
                            afwGeom.Point2I(activeCol, activeRow))
                        if display and False:
                            kim = afwImage.ImageD(kWidth, kHeight); kernel.computeImage(kim, False)
                            ds9.mtv(kim, frame=1)

                        self.runStdTest(kernel, kernelDescr="Delta Function Kernel")

    def testSpatiallyVaryingGaussianLinerCombination(self):
        """Test convolution with a spatially varying LinearCombinationKernel of two Gaussian basis kernels.
        """
        kWidth = 5
        kHeight = 5

        # create spatial model
        for nBasisKernels in (3, 4):
            # at 3 the kernel will not be refactored, at 4 it will be
            sFunc = afwMath.PolynomialFunction2D(1)
            
            # spatial parameters are a list of entries, one per kernel parameter;
            # each entry is a list of spatial parameters
            sParams = (
                (1.0, -0.01/self.width, -0.01/self.height),
                (0.0,  0.01/self.width,  0.0/self.height),
                (0.0,  0.0/self.width,  0.01/self.height),
                (0.5,  0.005/self.width,  -0.005/self.height),
            )[:nBasisKernels]
            
            gaussParamsList = (
                (1.5, 1.5, 0.0),
                (2.5, 1.5, 0.0),
                (2.5, 1.5, math.pi / 2.0),
                (2.5, 2.5, 0.0),
            )[:nBasisKernels]
            basisKernelList = makeGaussianKernelList(kWidth, kHeight, gaussParamsList)
            kernel = afwMath.LinearCombinationKernel(basisKernelList, sFunc)
            kernel.setSpatialParameters(sParams)
    
            for maxInterpDist, rtol, methodStr in (
                (0,   1.0e-5, "brute force"),
                (10,  1.0e-5, "interpolation over 10 x 10 pixels"),
            ):
                self.runStdTest(
                    kernel,
                    kernelDescr = "%s with %d basis kernels convolved using %s" % \
                        ("Spatially Varying Gaussian Analytic Kernel", nBasisKernels, methodStr),
                    maxInterpDist = maxInterpDist,
                    rtol = rtol)

    def testSpatiallyVaryingDeltaFunctionLinearCombination(self):
        """Test convolution with a spatially varying LinearCombinationKernel of delta function basis kernels.
        """
        kWidth = 2
        kHeight = 2

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
        
        basisKernelList = makeDeltaFunctionKernelList(kWidth, kHeight)
        kernel = afwMath.LinearCombinationKernel(basisKernelList, sFunc)
        kernel.setSpatialParameters(sParams)

        for maxInterpDist, rtol, methodStr in (
            (0,   1.0e-5, "brute force"),
            (10,  1.0e-3, "interpolation over 10 x 10 pixels"),
        ):
            self.runStdTest(
                kernel,
                kernelDescr = "Spatially varying LinearCombinationKernel of delta function kernels using %s" %\
                    (methodStr,),
                maxInterpDist = maxInterpDist,
                rtol = rtol)

    def testZeroWidthKernel(self):
        """Convolution by a 0x0 kernel should raise an exception.
        
        The only way to produce a 0x0 kernel is to use the default constructor
        (which exists only to support persistence; it does not produce a useful kernel).
        """
        kernelList = [
            afwMath.FixedKernel(),
            afwMath.AnalyticKernel(),
            afwMath.SeparableKernel(),
#            afwMath.DeltaFunctionKernel(),  # DeltaFunctionKernel has no default constructor
            afwMath.LinearCombinationKernel(),
        ]
        convolutionControl = afwMath.ConvolutionControl()
        for kernel in kernelList:
            self.assertRaises(Exception, afwMath.convolve, self.cnvMaskedImage, self.maskedImage, kernel,
                convolutionControl)

    def testTicket873(self):
        """Demonstrate ticket 873: convolution of a MaskedImage with a spatially varying
        LinearCombinationKernel of basis kernels with low covariance gives incorrect variance.
        """
        # create spatial model
        sFunc = afwMath.PolynomialFunction2D(1)
        
        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, -0.5/self.width, -0.5/self.height),
            (0.0,  1.0/self.width,  0.0/self.height),
            (0.0,  0.0/self.width,  1.0/self.height),
        )
        
        # create three kernels with some non-overlapping pixels
        # (non-zero pixels in one kernel vs. zero pixels in other kernels);
        # note: the extreme example of this is delta function kernels, but this is less extreme
        basisKernelList = afwMath.KernelList()
        kImArr = numpy.zeros([5, 5], dtype=float)
        kImArr[1:4, 1:4] = 0.5
        kImArr[2, 2] = 1.0
        kImage = afwImage.makeImageFromArray(kImArr)
        basisKernelList.append(afwMath.FixedKernel(kImage))
        kImArr[:, :] = 0.0
        kImArr[0:2, 0:2] = 0.125
        kImArr[3:5, 3:5] = 0.125
        kImage = afwImage.makeImageFromArray(kImArr)
        basisKernelList.append(afwMath.FixedKernel(kImage))
        kImArr[:, :] = 0.0
        kImArr[0:2, 3:5] = 0.125
        kImArr[3:5, 0:2] = 0.125
        kImage = afwImage.makeImageFromArray(kImArr)
        basisKernelList.append(afwMath.FixedKernel(kImage))

        kernel = afwMath.LinearCombinationKernel(basisKernelList, sFunc)
        kernel.setSpatialParameters(sParams)

        for maxInterpDist, rtol, methodStr in (
            (0,   1.0e-5, "brute force"),
            (10,  3.0e-3, "interpolation over 10 x 10 pixels"),
        ):
            self.runStdTest(
                kernel,
                kernelDescr = \
"Spatially varying LinearCombinationKernel of basis kernels with low covariance, using %s" % (methodStr,),
                maxInterpDist = maxInterpDist,
                rtol = rtol)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ConvolveTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(doExit=False):
    """Run the tests"""
    utilsTests.run(suite(), doExit)

if __name__ == "__main__":
    run(True)

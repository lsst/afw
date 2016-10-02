#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11#"""Test lsst.afwMath.convolve
#pybind11#
#pybind11#Tests convolution of various kernels with Images and MaskedImages.
#pybind11#"""
#pybind11#import math
#pybind11#import os
#pybind11#import os.path
#pybind11#import unittest
#pybind11#import string
#pybind11#import re
#pybind11#
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.math.detail as mathDetail
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#
#pybind11#from testKernel import makeDeltaFunctionKernelList, makeGaussianKernelList
#pybind11#from lsst.log import Log
#pybind11#
#pybind11#Log.getLogger("afw.image.Mask").setLevel(Log.INFO)
#pybind11#
#pybind11#try:
#pybind11#    display
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#import lsst.afw.display.utils as displayUtils
#pybind11#
#pybind11#try:
#pybind11#    dataDir = os.path.join(lsst.utils.getPackageDir("afwdata"), "data")
#pybind11#except pexExcept.NotFoundError:
#pybind11#    dataDir = None
#pybind11#else:
#pybind11#    InputMaskedImagePath = os.path.join(dataDir, "medexp.fits")
#pybind11#    FullMaskedImage = afwImage.MaskedImageF(InputMaskedImagePath)
#pybind11#
#pybind11## input image contains a saturated star, a bad column, and a faint star
#pybind11#InputBBox = afwGeom.Box2I(afwGeom.Point2I(52, 574), afwGeom.Extent2I(76, 80))
#pybind11## the shifted BBox is for a same-sized region containing different pixels;
#pybind11## this is used to initialize the convolved image, to make sure convolve fully overwrites it
#pybind11#ShiftedBBox = afwGeom.Box2I(afwGeom.Point2I(0, 460), afwGeom.Extent2I(76, 80))
#pybind11#
#pybind11#EdgeMaskPixel = 1 << afwImage.MaskU.getMaskPlane("EDGE")
#pybind11#NoDataMaskPixel = afwImage.MaskU.getPlaneBitMask("NO_DATA")
#pybind11#
#pybind11## Ignore kernel pixels whose value is exactly 0 when smearing the mask plane?
#pybind11## Set this to match the afw code
#pybind11#IgnoreKernelZeroPixels = True
#pybind11#
#pybind11#GarbageChars = string.punctuation + string.whitespace
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#def refConvolve(imMaskVar, xy0, kernel, doNormalize, doCopyEdge):
#pybind11#    """Reference code to convolve a kernel with a masked image.
#pybind11#
#pybind11#    Warning: slow (especially for spatially varying kernels).
#pybind11#
#pybind11#    Inputs:
#pybind11#    - imMaskVar: (image, mask, variance) numpy arrays
#pybind11#    - xy0: xy offset of imMaskVar relative to parent image
#pybind11#    - kernel: lsst::afw::Core.Kernel object
#pybind11#    - doNormalize: normalize the kernel
#pybind11#    - doCopyEdge: if True: copy edge pixels from input image to convolved image;
#pybind11#                if False: set edge pixels to the standard edge pixel (image=nan, var=inf, mask=EDGE)
#pybind11#    """
#pybind11#    # Note: the original version of this function was written when numpy/image conversions were the
#pybind11#    # transpose of what they are today.  Rather than transpose the logic in this function or put
#pybind11#    # transposes throughout the rest of the file, I have transposed only the inputs and outputs.
#pybind11#    #  - Jim Bosch, 3/4/2011
#pybind11#    image, mask, variance = (imMaskVar[0].transpose(), imMaskVar[1].transpose(), imMaskVar[2].transpose())
#pybind11#
#pybind11#    if doCopyEdge:
#pybind11#        # copy input arrays to output arrays and set EDGE bit of mask; non-edge pixels are overwritten below
#pybind11#        retImage = image.copy()
#pybind11#        retMask = mask.copy()
#pybind11#        retMask += EdgeMaskPixel
#pybind11#        retVariance = variance.copy()
#pybind11#    else:
#pybind11#        # initialize output arrays to all edge pixels; non-edge pixels will be overwritten below
#pybind11#        retImage = numpy.zeros(image.shape, dtype=image.dtype)
#pybind11#        retImage[:, :] = numpy.nan
#pybind11#        retMask = numpy.zeros(mask.shape, dtype=mask.dtype)
#pybind11#        retMask[:, :] = NoDataMaskPixel
#pybind11#        retVariance = numpy.zeros(variance.shape, dtype=image.dtype)
#pybind11#        retVariance[:, :] = numpy.inf
#pybind11#
#pybind11#    kWidth = kernel.getWidth()
#pybind11#    kHeight = kernel.getHeight()
#pybind11#    numCols = image.shape[0] + 1 - kWidth
#pybind11#    numRows = image.shape[1] + 1 - kHeight
#pybind11#    if numCols < 0 or numRows < 0:
#pybind11#        raise RuntimeError("image must be larger than kernel in both dimensions")
#pybind11#    colRange = list(range(numCols))
#pybind11#
#pybind11#    kImage = afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight))
#pybind11#    isSpatiallyVarying = kernel.isSpatiallyVarying()
#pybind11#    if not isSpatiallyVarying:
#pybind11#        kernel.computeImage(kImage, doNormalize)
#pybind11#        kImArr = kImage.getArray().transpose()
#pybind11#
#pybind11#    retRow = kernel.getCtrY()
#pybind11#    for inRowBeg in range(numRows):
#pybind11#        inRowEnd = inRowBeg + kHeight
#pybind11#        retCol = kernel.getCtrX()
#pybind11#        if isSpatiallyVarying:
#pybind11#            rowPos = afwImage.indexToPosition(retRow) + xy0[1]
#pybind11#        for inColBeg in colRange:
#pybind11#            if isSpatiallyVarying:
#pybind11#                colPos = afwImage.indexToPosition(retCol) + xy0[0]
#pybind11#                kernel.computeImage(kImage, doNormalize, colPos, rowPos)
#pybind11#                kImArr = kImage.getArray().transpose()
#pybind11#            inColEnd = inColBeg + kWidth
#pybind11#            subImage = image[inColBeg:inColEnd, inRowBeg:inRowEnd]
#pybind11#            subVariance = variance[inColBeg:inColEnd, inRowBeg:inRowEnd]
#pybind11#            subMask = mask[inColBeg:inColEnd, inRowBeg:inRowEnd]
#pybind11#            retImage[retCol, retRow] = numpy.add.reduce((kImArr * subImage).flat)
#pybind11#            retVariance[retCol, retRow] = numpy.add.reduce((kImArr * kImArr * subVariance).flat)
#pybind11#            if IgnoreKernelZeroPixels:
#pybind11#                retMask[retCol, retRow] = numpy.bitwise_or.reduce((subMask * (kImArr != 0)).flat)
#pybind11#            else:
#pybind11#                retMask[retCol, retRow] = numpy.bitwise_or.reduce(subMask.flat)
#pybind11#
#pybind11#            retCol += 1
#pybind11#        retRow += 1
#pybind11#    return [numpy.copy(numpy.transpose(arr), order="C") for arr in (retImage, retMask, retVariance)]
#pybind11#
#pybind11#
#pybind11#def sameMaskPlaneDicts(maskedImageA, maskedImageB):
#pybind11#    """Return True if the mask plane dicts are the same, False otherwise.
#pybind11#
#pybind11#    Handles the fact that one cannot directly compare maskPlaneDicts using ==
#pybind11#    """
#pybind11#    mpDictA = maskedImageA.getMask().getMaskPlaneDict()
#pybind11#    mpDictB = maskedImageB.getMask().getMaskPlaneDict()
#pybind11#    if list(mpDictA.keys()) != list(mpDictB.keys()):
#pybind11#        print("mpDictA.keys()  ", mpDictA.keys())
#pybind11#        print("mpDictB.keys()  ", mpDictB.keys())
#pybind11#        return False
#pybind11#    if list(mpDictA.values()) != list(mpDictB.values()):
#pybind11#        print("mpDictA.values()", mpDictA.values())
#pybind11#        print("mpDictB.values()", mpDictB.values())
#pybind11#        return False
#pybind11#    return True
#pybind11#
#pybind11#
#pybind11#class ConvolveTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        if dataDir is not None:
#pybind11#            self.maskedImage = afwImage.MaskedImageF(FullMaskedImage, InputBBox, afwImage.LOCAL, True)
#pybind11#            # use a huge XY0 to make emphasize any errors related to not handling xy0 correctly.
#pybind11#            self.maskedImage.setXY0(300, 200)
#pybind11#            self.xy0 = self.maskedImage.getXY0()
#pybind11#
#pybind11#            # provide destinations for the convolved MaskedImage and Image that contain junk
#pybind11#            # to verify that convolve overwrites all pixels;
#pybind11#            # make them deep copies so we can mess with them without affecting self.inImage
#pybind11#            self.cnvMaskedImage = afwImage.MaskedImageF(FullMaskedImage, ShiftedBBox, afwImage.LOCAL, True)
#pybind11#            self.cnvImage = afwImage.ImageF(FullMaskedImage.getImage(), ShiftedBBox, afwImage.LOCAL, True)
#pybind11#
#pybind11#            self.width = self.maskedImage.getWidth()
#pybind11#            self.height = self.maskedImage.getHeight()
#pybind11#    #         smask = afwImage.MaskU(self.maskedImage.getMask(), afwGeom.Box2I(afwGeom.Point2I(15, 17), afwGeom.Extent2I(10, 5)))
#pybind11#    #         smask.set(0x8)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        if dataDir is not None:
#pybind11#            del self.maskedImage
#pybind11#            del self.cnvMaskedImage
#pybind11#            del self.cnvImage
#pybind11#
#pybind11#    @staticmethod
#pybind11#    def _removeGarbageChars(instring):
#pybind11#        # str.translate on python2 differs to that on python3
#pybind11#        # Performance is not critical in this helper function so use a regex
#pybind11#        print("Translating '{}' -> '{}'".format(instring, re.sub("[" + GarbageChars + "]", "", instring)))
#pybind11#        return re.sub("[" + GarbageChars + "]", "", instring)
#pybind11#
#pybind11#    def runBasicTest(self, kernel, convControl, refKernel=None, kernelDescr="", rtol=1.0e-05, atol=1e-08):
#pybind11#        """Assert that afwMath::convolve gives the same result as reference convolution for a given kernel.
#pybind11#
#pybind11#        Inputs:
#pybind11#        - kernel: convolution kernel
#pybind11#        - convControl: convolution control parameters (afwMath.ConvolutionControl)
#pybind11#        - refKernel: kernel to use for refConvolve (if None then kernel is used)
#pybind11#        - kernelDescr: description of kernel
#pybind11#        - rtol: relative tolerance (see below)
#pybind11#        - atol: absolute tolerance (see below)
#pybind11#
#pybind11#        rtol and atol are positive, typically very small numbers.
#pybind11#        The relative difference (rtol * abs(b)) and the absolute difference "atol" are added together
#pybind11#        to compare against the absolute difference between "a" and "b".
#pybind11#        """
#pybind11#        if refKernel == None:
#pybind11#            refKernel = kernel
#pybind11#        # strip garbage characters (whitespace and punctuation) to make a short description for saving files
#pybind11#        shortKernelDescr = self._removeGarbageChars(kernelDescr)
#pybind11#
#pybind11#        doNormalize = convControl.getDoNormalize()
#pybind11#        doCopyEdge = convControl.getDoCopyEdge()
#pybind11#        maxInterpDist = convControl.getMaxInterpolationDistance()
#pybind11#
#pybind11#        imMaskVar = self.maskedImage.getArrays()
#pybind11#        xy0 = self.maskedImage.getXY0()
#pybind11#
#pybind11#        refCnvImMaskVarArr = refConvolve(imMaskVar, xy0, refKernel, doNormalize, doCopyEdge)
#pybind11#        refMaskedImage = afwImage.makeMaskedImageFromArrays(*refCnvImMaskVarArr)
#pybind11#
#pybind11#        afwMath.convolve(self.cnvImage, self.maskedImage.getImage(), kernel, convControl)
#pybind11#        self.assertEqual(self.cnvImage.getXY0(), self.xy0)
#pybind11#
#pybind11#        afwMath.convolve(self.cnvMaskedImage, self.maskedImage, kernel, convControl)
#pybind11#
#pybind11#        if display and False:
#pybind11#            ds9.mtv(displayUtils.Mosaic().makeMosaic([
#pybind11#                self.maskedImage, refMaskedImage, self.cnvMaskedImage]), frame=0)
#pybind11#            if False:
#pybind11#                for (x, y) in ((0, 0), (1, 0), (0, 1), (50, 50)):
#pybind11#                    print("Mask(%d,%d) 0x%x 0x%x" % (x, y, refMaskedImage.getMask().get(x, y),
#pybind11#                                                     self.cnvMaskedImage.getMask().get(x, y)))
#pybind11#
#pybind11#        self.assertImagesNearlyEqual(self.cnvImage, refMaskedImage.getImage(), atol=atol, rtol=rtol)
#pybind11#        self.assertMaskedImagesNearlyEqual(self.cnvMaskedImage, refMaskedImage, atol=atol, rtol=rtol)
#pybind11#
#pybind11#        if not sameMaskPlaneDicts(self.cnvMaskedImage, self.maskedImage):
#pybind11#            self.cnvMaskedImage.writeFits("act%s" % (shortKernelDescr,))
#pybind11#            refMaskedImage.writeFits("des%s" % (shortKernelDescr,))
#pybind11#            self.fail("convolve(MaskedImage, kernel=%s, doNormalize=%s, "
#pybind11#                      "doCopyEdge=%s, maxInterpDist=%s) failed:\n%s" %
#pybind11#                      (kernelDescr, doNormalize, doCopyEdge, maxInterpDist,
#pybind11#                       "convolved mask dictionary does not match input"))
#pybind11#
#pybind11#    def runStdTest(self, kernel, refKernel=None, kernelDescr="", rtol=1.0e-05, atol=1e-08,
#pybind11#                   maxInterpDist=10):
#pybind11#        """Assert that afwMath::convolve gives the same result as reference convolution for a given kernel.
#pybind11#
#pybind11#        Inputs:
#pybind11#        - kernel: convolution kernel
#pybind11#        - refKernel: kernel to use for refConvolve (if None then kernel is used)
#pybind11#        - kernelDescr: description of kernel
#pybind11#        - rtol: relative tolerance (see below)
#pybind11#        - atol: absolute tolerance (see below)
#pybind11#        - maxInterpDist: maximum allowed distance for linear interpolation during convolution
#pybind11#
#pybind11#        rtol and atol are positive, typically very small numbers.
#pybind11#        The relative difference (rtol * abs(b)) and the absolute difference "atol" are added together
#pybind11#        to compare against the absolute difference between "a" and "b".
#pybind11#        """
#pybind11#        convControl = afwMath.ConvolutionControl()
#pybind11#        convControl.setMaxInterpolationDistance(maxInterpDist)
#pybind11#
#pybind11#        # verify dimension assertions:
#pybind11#        # - output image dimensions = input image dimensions
#pybind11#        # - input image width and height >= kernel width and height
#pybind11#        # Note: the assertion kernel size > 0 is tested elsewhere
#pybind11#        for inWidth in (kernel.getWidth() - 1, self.width-1, self.width, self.width + 1):
#pybind11#            for inHeight in (kernel.getHeight() - 1, self.width-1, self.width, self.width + 1):
#pybind11#                if (inWidth == self.width) and (inHeight == self.height):
#pybind11#                    continue
#pybind11#                inMaskedImage = afwImage.MaskedImageF(afwGeom.Extent2I(inWidth, inHeight))
#pybind11#                with self.assertRaises(Exception):
#pybind11#                    afwMath.convolve(self.cnvMaskedImage, inMaskedImage, kernel)
#pybind11#
#pybind11#        for doNormalize in (True,):  # (False, True):
#pybind11#            convControl.setDoNormalize(doNormalize)
#pybind11#            for doCopyEdge in (False,):  # (False, True):
#pybind11#                convControl.setDoCopyEdge(doCopyEdge)
#pybind11#                self.runBasicTest(kernel, convControl=convControl, refKernel=refKernel,
#pybind11#                                  kernelDescr=kernelDescr, rtol=rtol, atol=atol)
#pybind11#
#pybind11#        # verify that basicConvolve does not write to edge pixels
#pybind11#        self.runBasicConvolveEdgeTest(kernel, kernelDescr)
#pybind11#
#pybind11#    def runBasicConvolveEdgeTest(self, kernel, kernelDescr):
#pybind11#        """Verify that basicConvolve does not write to edge pixels for this kind of kernel
#pybind11#        """
#pybind11#        fullBox = afwGeom.Box2I(
#pybind11#            afwGeom.Point2I(0, 0),
#pybind11#            ShiftedBBox.getDimensions(),
#pybind11#        )
#pybind11#        goodBox = kernel.shrinkBBox(fullBox)
#pybind11#        cnvMaskedImage = afwImage.MaskedImageF(FullMaskedImage, ShiftedBBox, afwImage.LOCAL, True)
#pybind11#        cnvMaskedImageCopy = afwImage.MaskedImageF(cnvMaskedImage, fullBox, afwImage.LOCAL, True)
#pybind11#        cnvMaskedImageCopyViewOfGoodRegion = afwImage.MaskedImageF(
#pybind11#            cnvMaskedImageCopy, goodBox, afwImage.LOCAL, False)
#pybind11#
#pybind11#        # convolve with basicConvolve, which should leave the edge pixels alone
#pybind11#        convControl = afwMath.ConvolutionControl()
#pybind11#        mathDetail.basicConvolve(cnvMaskedImage, self.maskedImage, kernel, convControl)
#pybind11#
#pybind11#        # reset the good region to the original convolved image;
#pybind11#        # this should reset the entire convolved image to its original self
#pybind11#        cnvMaskedImageGoodView = afwImage.MaskedImageF(cnvMaskedImage, goodBox, afwImage.LOCAL, False)
#pybind11#        cnvMaskedImageGoodView[:] = cnvMaskedImageCopyViewOfGoodRegion
#pybind11#
#pybind11#        # assert that these two are equal
#pybind11#        msg = "basicConvolve(MaskedImage, kernel=%s) wrote to edge pixels" % (kernelDescr,)
#pybind11#        try:
#pybind11#            self.assertMaskedImagesNearlyEqual(cnvMaskedImage, cnvMaskedImageCopy,
#pybind11#                                               doVariance=True, rtol=0, atol=0, msg=msg)
#pybind11#        except Exception:
#pybind11#            # write out the images, then fail
#pybind11#            shortKernelDescr = self.removeGarbageChars(kernelDescr)
#pybind11#            cnvMaskedImage.writeFits("actBasicConvolve%s" % (shortKernelDescr,))
#pybind11#            cnvMaskedImageCopy.writeFits("desBasicConvolve%s" % (shortKernelDescr,))
#pybind11#            raise
#pybind11#
#pybind11#    def testConvolutionControl(self):
#pybind11#        """Test the ConvolutionControl object
#pybind11#        """
#pybind11#        convControl = afwMath.ConvolutionControl()
#pybind11#        self.assertTrue(convControl.getDoNormalize())
#pybind11#        for doNormalize in (False, True):
#pybind11#            convControl.setDoNormalize(doNormalize)
#pybind11#            self.assertEqual(convControl.getDoNormalize(), doNormalize)
#pybind11#
#pybind11#        self.assertFalse(convControl.getDoCopyEdge())
#pybind11#        for doCopyEdge in (False, True):
#pybind11#            convControl.setDoCopyEdge(doCopyEdge)
#pybind11#            self.assertEqual(convControl.getDoCopyEdge(), doCopyEdge)
#pybind11#
#pybind11#        self.assertEqual(convControl.getMaxInterpolationDistance(), 10)
#pybind11#        for maxInterpDist in (0, 1, 2, 10, 100):
#pybind11#            convControl.setMaxInterpolationDistance(maxInterpDist)
#pybind11#            self.assertEqual(convControl.getMaxInterpolationDistance(), maxInterpDist)
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testUnityConvolution(self):
#pybind11#        """Verify that convolution with a centered delta function reproduces the original.
#pybind11#        """
#pybind11#        # create a delta function kernel that has 1,1 in the center
#pybind11#        kFunc = afwMath.IntegerDeltaFunction2D(0.0, 0.0)
#pybind11#        kernel = afwMath.AnalyticKernel(3, 3, kFunc)
#pybind11#        doNormalize = False
#pybind11#        doCopyEdge = False
#pybind11#
#pybind11#        afwMath.convolve(self.cnvImage, self.maskedImage.getImage(), kernel, doNormalize, doCopyEdge)
#pybind11#
#pybind11#        afwMath.convolve(self.cnvMaskedImage, self.maskedImage, kernel, doNormalize, doCopyEdge)
#pybind11#        cnvImMaskVarArr = self.cnvMaskedImage.getArrays()
#pybind11#
#pybind11#        skipMaskArr = numpy.array(numpy.isnan(cnvImMaskVarArr[0]), dtype=numpy.uint16)
#pybind11#
#pybind11#        kernelDescr = "Centered DeltaFunctionKernel (testing unity convolution)"
#pybind11#        self.assertImagesNearlyEqual(self.cnvImage, self.maskedImage.getImage(),
#pybind11#                                     skipMask=skipMaskArr, msg=kernelDescr)
#pybind11#        self.assertMaskedImagesNearlyEqual(self.cnvMaskedImage, self.maskedImage,
#pybind11#                                           skipMask=skipMaskArr, msg=kernelDescr)
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testFixedKernelConvolve(self):
#pybind11#        """Test convolve with a fixed kernel
#pybind11#        """
#pybind11#        kWidth = 6
#pybind11#        kHeight = 7
#pybind11#
#pybind11#        kFunc = afwMath.GaussianFunction2D(2.5, 1.5, 0.5)
#pybind11#        analyticKernel = afwMath.AnalyticKernel(kWidth, kHeight, kFunc)
#pybind11#        kernelImage = afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight))
#pybind11#        analyticKernel.computeImage(kernelImage, False)
#pybind11#        fixedKernel = afwMath.FixedKernel(kernelImage)
#pybind11#
#pybind11#        self.runStdTest(fixedKernel, kernelDescr="Gaussian FixedKernel")
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testSeparableConvolve(self):
#pybind11#        """Test convolve of a separable kernel with a spatially invariant Gaussian function
#pybind11#        """
#pybind11#        kWidth = 7
#pybind11#        kHeight = 6
#pybind11#
#pybind11#        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
#pybind11#        gaussFunc2 = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        separableKernel = afwMath.SeparableKernel(kWidth, kHeight, gaussFunc1, gaussFunc1)
#pybind11#        analyticKernel = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc2)
#pybind11#
#pybind11#        self.runStdTest(
#pybind11#            separableKernel,
#pybind11#            refKernel=analyticKernel,
#pybind11#            kernelDescr="Gaussian Separable Kernel (compared to AnalyticKernel equivalent)")
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testSpatiallyInvariantConvolve(self):
#pybind11#        """Test convolution with a spatially invariant Gaussian function
#pybind11#        """
#pybind11#        kWidth = 6
#pybind11#        kHeight = 7
#pybind11#
#pybind11#        kFunc = afwMath.GaussianFunction2D(2.5, 1.5, 0.5)
#pybind11#        kernel = afwMath.AnalyticKernel(kWidth, kHeight, kFunc)
#pybind11#
#pybind11#        self.runStdTest(kernel, kernelDescr="Gaussian Analytic Kernel")
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testSpatiallyVaryingAnalyticConvolve(self):
#pybind11#        """Test in-place convolution with a spatially varying AnalyticKernel
#pybind11#        """
#pybind11#        kWidth = 7
#pybind11#        kHeight = 6
#pybind11#
#pybind11#        # create spatial model
#pybind11#        sFunc = afwMath.PolynomialFunction2D(1)
#pybind11#
#pybind11#        minSigma = 1.5
#pybind11#        maxSigma = 1.501
#pybind11#
#pybind11#        # spatial parameters are a list of entries, one per kernel parameter;
#pybind11#        # each entry is a list of spatial parameters
#pybind11#        sParams = (
#pybind11#            (minSigma, (maxSigma - minSigma) / self.width, 0.0),
#pybind11#            (minSigma, 0.0, (maxSigma - minSigma) / self.height),
#pybind11#            (0.0, 0.0, 0.0),
#pybind11#        )
#pybind11#
#pybind11#        kFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        kernel = afwMath.AnalyticKernel(kWidth, kHeight, kFunc, sFunc)
#pybind11#        kernel.setSpatialParameters(sParams)
#pybind11#
#pybind11#        for maxInterpDist, rtol, methodStr in (
#pybind11#            (0, 1.0e-5, "brute force"),
#pybind11#            (10, 1.0e-5, "interpolation over 10 x 10 pixels"),
#pybind11#        ):
#pybind11#            self.runStdTest(
#pybind11#                kernel,
#pybind11#                kernelDescr="Spatially Varying Gaussian Analytic Kernel using %s" % (methodStr,),
#pybind11#                maxInterpDist=maxInterpDist,
#pybind11#                rtol=rtol)
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testSpatiallyVaryingSeparableConvolve(self):
#pybind11#        """Test convolution with a spatially varying SeparableKernel
#pybind11#        """
#pybind11#        kWidth = 7
#pybind11#        kHeight = 6
#pybind11#
#pybind11#        # create spatial model
#pybind11#        sFunc = afwMath.PolynomialFunction2D(1)
#pybind11#
#pybind11#        minSigma = 0.1
#pybind11#        maxSigma = 3.0
#pybind11#
#pybind11#        # spatial parameters are a list of entries, one per kernel parameter;
#pybind11#        # each entry is a list of spatial parameters
#pybind11#        sParams = (
#pybind11#            (minSigma, (maxSigma - minSigma) / self.width, 0.0),
#pybind11#            (minSigma, 0.0, (maxSigma - minSigma) / self.height),
#pybind11#            (0.0, 0.0, 0.0),
#pybind11#        )
#pybind11#
#pybind11#        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
#pybind11#        gaussFunc2 = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        separableKernel = afwMath.SeparableKernel(kWidth, kHeight, gaussFunc1, gaussFunc1, sFunc)
#pybind11#        analyticKernel = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc2, sFunc)
#pybind11#        separableKernel.setSpatialParameters(sParams[0:2])
#pybind11#        analyticKernel.setSpatialParameters(sParams)
#pybind11#
#pybind11#        self.runStdTest(separableKernel, refKernel=analyticKernel,
#pybind11#                        kernelDescr="Spatially Varying Gaussian Separable Kernel")
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testDeltaConvolve(self):
#pybind11#        """Test convolution with various delta function kernels using optimized code
#pybind11#        """
#pybind11#        for kWidth in range(1, 4):
#pybind11#            for kHeight in range(1, 4):
#pybind11#                for activeCol in range(kWidth):
#pybind11#                    for activeRow in range(kHeight):
#pybind11#                        kernel = afwMath.DeltaFunctionKernel(kWidth, kHeight,
#pybind11#                                                             afwGeom.Point2I(activeCol, activeRow))
#pybind11#                        if display and False:
#pybind11#                            kim = afwImage.ImageD(kWidth, kHeight)
#pybind11#                            kernel.computeImage(kim, False)
#pybind11#                            ds9.mtv(kim, frame=1)
#pybind11#
#pybind11#                        self.runStdTest(kernel, kernelDescr="Delta Function Kernel")
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testSpatiallyVaryingGaussianLinerCombination(self):
#pybind11#        """Test convolution with a spatially varying LinearCombinationKernel of two Gaussian basis kernels.
#pybind11#        """
#pybind11#        kWidth = 5
#pybind11#        kHeight = 5
#pybind11#
#pybind11#        # create spatial model
#pybind11#        for nBasisKernels in (3, 4):
#pybind11#            # at 3 the kernel will not be refactored, at 4 it will be
#pybind11#            sFunc = afwMath.PolynomialFunction2D(1)
#pybind11#
#pybind11#            # spatial parameters are a list of entries, one per kernel parameter;
#pybind11#            # each entry is a list of spatial parameters
#pybind11#            sParams = (
#pybind11#                (1.0, -0.01/self.width, -0.01/self.height),
#pybind11#                (0.0, 0.01/self.width, 0.0/self.height),
#pybind11#                (0.0, 0.0/self.width, 0.01/self.height),
#pybind11#                (0.5, 0.005/self.width, -0.005/self.height),
#pybind11#            )[:nBasisKernels]
#pybind11#
#pybind11#            gaussParamsList = (
#pybind11#                (1.5, 1.5, 0.0),
#pybind11#                (2.5, 1.5, 0.0),
#pybind11#                (2.5, 1.5, math.pi / 2.0),
#pybind11#                (2.5, 2.5, 0.0),
#pybind11#            )[:nBasisKernels]
#pybind11#            basisKernelList = makeGaussianKernelList(kWidth, kHeight, gaussParamsList)
#pybind11#            kernel = afwMath.LinearCombinationKernel(basisKernelList, sFunc)
#pybind11#            kernel.setSpatialParameters(sParams)
#pybind11#
#pybind11#            for maxInterpDist, rtol, methodStr in (
#pybind11#                (0, 1.0e-5, "brute force"),
#pybind11#                (10, 1.0e-5, "interpolation over 10 x 10 pixels"),
#pybind11#            ):
#pybind11#                self.runStdTest(
#pybind11#                    kernel,
#pybind11#                    kernelDescr="%s with %d basis kernels convolved using %s" %
#pybind11#                    ("Spatially Varying Gaussian Analytic Kernel", nBasisKernels, methodStr),
#pybind11#                    maxInterpDist=maxInterpDist,
#pybind11#                    rtol=rtol)
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testSpatiallyVaryingDeltaFunctionLinearCombination(self):
#pybind11#        """Test convolution with a spatially varying LinearCombinationKernel of delta function basis kernels.
#pybind11#        """
#pybind11#        kWidth = 2
#pybind11#        kHeight = 2
#pybind11#
#pybind11#        # create spatially model
#pybind11#        sFunc = afwMath.PolynomialFunction2D(1)
#pybind11#
#pybind11#        # spatial parameters are a list of entries, one per kernel parameter;
#pybind11#        # each entry is a list of spatial parameters
#pybind11#        sParams = (
#pybind11#            (1.0, -0.5/self.width, -0.5/self.height),
#pybind11#            (0.0, 1.0/self.width, 0.0/self.height),
#pybind11#            (0.0, 0.0/self.width, 1.0/self.height),
#pybind11#            (0.5, 0.0, 0.0),
#pybind11#        )
#pybind11#
#pybind11#        basisKernelList = makeDeltaFunctionKernelList(kWidth, kHeight)
#pybind11#        kernel = afwMath.LinearCombinationKernel(basisKernelList, sFunc)
#pybind11#        kernel.setSpatialParameters(sParams)
#pybind11#
#pybind11#        for maxInterpDist, rtol, methodStr in (
#pybind11#            (0, 1.0e-5, "brute force"),
#pybind11#            (10, 1.0e-3, "interpolation over 10 x 10 pixels"),
#pybind11#        ):
#pybind11#            self.runStdTest(
#pybind11#                kernel,
#pybind11#                kernelDescr="Spatially varying LinearCombinationKernel of delta function kernels using %s" %
#pybind11#                (methodStr,),
#pybind11#                maxInterpDist=maxInterpDist,
#pybind11#                rtol=rtol)
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testZeroWidthKernel(self):
#pybind11#        """Convolution by a 0x0 kernel should raise an exception.
#pybind11#
#pybind11#        The only way to produce a 0x0 kernel is to use the default constructor
#pybind11#        (which exists only to support persistence; it does not produce a useful kernel).
#pybind11#        """
#pybind11#        kernelList = [
#pybind11#            afwMath.FixedKernel(),
#pybind11#            afwMath.AnalyticKernel(),
#pybind11#            afwMath.SeparableKernel(),
#pybind11#            #            afwMath.DeltaFunctionKernel(),  # DeltaFunctionKernel has no default constructor
#pybind11#            afwMath.LinearCombinationKernel(),
#pybind11#        ]
#pybind11#        convolutionControl = afwMath.ConvolutionControl()
#pybind11#        for kernel in kernelList:
#pybind11#            with self.assertRaises(Exception):
#pybind11#                afwMath.convolve(self.cnvMaskedImage, self.maskedImage, kernel, convolutionControl)
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testTicket873(self):
#pybind11#        """Demonstrate ticket 873: convolution of a MaskedImage with a spatially varying
#pybind11#        LinearCombinationKernel of basis kernels with low covariance gives incorrect variance.
#pybind11#        """
#pybind11#        # create spatial model
#pybind11#        sFunc = afwMath.PolynomialFunction2D(1)
#pybind11#
#pybind11#        # spatial parameters are a list of entries, one per kernel parameter;
#pybind11#        # each entry is a list of spatial parameters
#pybind11#        sParams = (
#pybind11#            (1.0, -0.5/self.width, -0.5/self.height),
#pybind11#            (0.0, 1.0/self.width, 0.0/self.height),
#pybind11#            (0.0, 0.0/self.width, 1.0/self.height),
#pybind11#        )
#pybind11#
#pybind11#        # create three kernels with some non-overlapping pixels
#pybind11#        # (non-zero pixels in one kernel vs. zero pixels in other kernels);
#pybind11#        # note: the extreme example of this is delta function kernels, but this is less extreme
#pybind11#        basisKernelList = afwMath.KernelList()
#pybind11#        kImArr = numpy.zeros([5, 5], dtype=float)
#pybind11#        kImArr[1:4, 1:4] = 0.5
#pybind11#        kImArr[2, 2] = 1.0
#pybind11#        kImage = afwImage.makeImageFromArray(kImArr)
#pybind11#        basisKernelList.append(afwMath.FixedKernel(kImage))
#pybind11#        kImArr[:, :] = 0.0
#pybind11#        kImArr[0:2, 0:2] = 0.125
#pybind11#        kImArr[3:5, 3:5] = 0.125
#pybind11#        kImage = afwImage.makeImageFromArray(kImArr)
#pybind11#        basisKernelList.append(afwMath.FixedKernel(kImage))
#pybind11#        kImArr[:, :] = 0.0
#pybind11#        kImArr[0:2, 3:5] = 0.125
#pybind11#        kImArr[3:5, 0:2] = 0.125
#pybind11#        kImage = afwImage.makeImageFromArray(kImArr)
#pybind11#        basisKernelList.append(afwMath.FixedKernel(kImage))
#pybind11#
#pybind11#        kernel = afwMath.LinearCombinationKernel(basisKernelList, sFunc)
#pybind11#        kernel.setSpatialParameters(sParams)
#pybind11#
#pybind11#        for maxInterpDist, rtol, methodStr in (
#pybind11#            (0, 1.0e-5, "brute force"),
#pybind11#            (10, 3.0e-3, "interpolation over 10 x 10 pixels"),
#pybind11#        ):
#pybind11#            self.runStdTest(
#pybind11#                kernel,
#pybind11#                kernelDescr="Spatially varying LinearCombinationKernel of basis "
#pybind11#                            "kernels with low covariance, using %s" % (
#pybind11#                                methodStr,),
#pybind11#                maxInterpDist=maxInterpDist,
#pybind11#                rtol=rtol)
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

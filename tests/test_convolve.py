# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Test lsst.afwMath.convolve

Tests convolution of various kernels with Images and MaskedImages.
"""
import math
import os
import os.path
import unittest
import string
import re

import numpy

import lsst.utils
import lsst.utils.tests
import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.math.detail as mathDetail
import lsst.pex.exceptions as pexExcept

from test_kernel import makeDeltaFunctionKernelList, makeGaussianKernelList
from lsst.log import Log

import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

Log.getLogger("afw.image.Mask").setLevel(Log.INFO)

try:
    display
except NameError:
    display = False

try:
    dataDir = os.path.join(lsst.utils.getPackageDir("afwdata"), "data")
except pexExcept.NotFoundError:
    dataDir = None
else:
    InputMaskedImagePath = os.path.join(dataDir, "medexp.fits")
    FullMaskedImage = afwImage.MaskedImageF(InputMaskedImagePath)

# input image contains a saturated star, a bad column, and a faint star
InputBBox = lsst.geom.Box2I(lsst.geom.Point2I(52, 574), lsst.geom.Extent2I(76, 80))
# the shifted BBox is for a same-sized region containing different pixels;
# this is used to initialize the convolved image, to make sure convolve
# fully overwrites it
ShiftedBBox = lsst.geom.Box2I(lsst.geom.Point2I(0, 460), lsst.geom.Extent2I(76, 80))

EdgeMaskPixel = 1 << afwImage.Mask.getMaskPlane("EDGE")
NoDataMaskPixel = afwImage.Mask.getPlaneBitMask("NO_DATA")

# Ignore kernel pixels whose value is exactly 0 when smearing the mask plane?
# Set this to match the afw code
IgnoreKernelZeroPixels = True

GarbageChars = string.punctuation + string.whitespace


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
    image, mask, variance = (imMaskVar[0].transpose(),
                             imMaskVar[1].transpose(),
                             imMaskVar[2].transpose())

    if doCopyEdge:
        # copy input arrays to output arrays and set EDGE bit of mask; non-edge
        # pixels are overwritten below
        retImage = image.copy()
        retMask = mask.copy()
        retMask += EdgeMaskPixel
        retVariance = variance.copy()
    else:
        # initialize output arrays to all edge pixels; non-edge pixels will be
        # overwritten below
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
        raise RuntimeError(
            "image must be larger than kernel in both dimensions")
    colRange = list(range(numCols))

    kImage = afwImage.ImageD(lsst.geom.Extent2I(kWidth, kHeight))
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
            retImage[retCol, retRow] = numpy.add.reduce(
                (kImArr * subImage).flat)
            retVariance[retCol, retRow] = numpy.add.reduce(
                (kImArr * kImArr * subVariance).flat)
            if IgnoreKernelZeroPixels:
                retMask[retCol, retRow] = numpy.bitwise_or.reduce(
                    (subMask * (kImArr != 0)).flat)
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
    if list(mpDictA.keys()) != list(mpDictB.keys()):
        print("mpDictA.keys()  ", mpDictA.keys())
        print("mpDictB.keys()  ", mpDictB.keys())
        return False
    if list(mpDictA.values()) != list(mpDictB.values()):
        print("mpDictA.values()", mpDictA.values())
        print("mpDictB.values()", mpDictB.values())
        return False
    return True


class ConvolveTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        if dataDir is not None:
            self.maskedImage = afwImage.MaskedImageF(
                FullMaskedImage, InputBBox, afwImage.LOCAL, True)
            # use a huge XY0 to make emphasize any errors related to not
            # handling xy0 correctly.
            self.maskedImage.setXY0(300, 200)
            self.xy0 = self.maskedImage.getXY0()

            # provide destinations for the convolved MaskedImage and Image that contain junk
            # to verify that convolve overwrites all pixels;
            # make them deep copies so we can mess with them without affecting
            # self.inImage
            self.cnvMaskedImage = afwImage.MaskedImageF(
                FullMaskedImage, ShiftedBBox, afwImage.LOCAL, True)
            self.cnvImage = afwImage.ImageF(
                FullMaskedImage.getImage(), ShiftedBBox, afwImage.LOCAL, True)

            self.width = self.maskedImage.getWidth()
            self.height = self.maskedImage.getHeight()

    def tearDown(self):
        if dataDir is not None:
            del self.maskedImage
            del self.cnvMaskedImage
            del self.cnvImage

    @staticmethod
    def _removeGarbageChars(instring):
        # str.translate on python2 differs to that on python3
        # Performance is not critical in this helper function so use a regex
        print("Translating '{}' -> '{}'".format(instring,
                                                re.sub("[" + GarbageChars + "]", "", instring)))
        return re.sub("[" + GarbageChars + "]", "", instring)

    def runBasicTest(self, kernel, convControl, refKernel=None,
                     kernelDescr="", rtol=1.0e-05, atol=1e-08):
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
        if refKernel is None:
            refKernel = kernel
        # strip garbage characters (whitespace and punctuation) to make a short
        # description for saving files
        shortKernelDescr = self._removeGarbageChars(kernelDescr)

        doNormalize = convControl.getDoNormalize()
        doCopyEdge = convControl.getDoCopyEdge()
        maxInterpDist = convControl.getMaxInterpolationDistance()

        imMaskVar = self.maskedImage.getArrays()
        xy0 = self.maskedImage.getXY0()

        refCnvImMaskVarArr = refConvolve(
            imMaskVar, xy0, refKernel, doNormalize, doCopyEdge)
        refMaskedImage = afwImage.makeMaskedImageFromArrays(
            *refCnvImMaskVarArr)

        afwMath.convolve(
            self.cnvImage, self.maskedImage.getImage(), kernel, convControl)
        self.assertEqual(self.cnvImage.getXY0(), self.xy0)

        afwMath.convolve(self.cnvMaskedImage,
                         self.maskedImage, kernel, convControl)

        if display and False:
            ds9.mtv(displayUtils.Mosaic().makeMosaic([
                self.maskedImage, refMaskedImage, self.cnvMaskedImage]), frame=0)
            if False:
                for (x, y) in ((0, 0), (1, 0), (0, 1), (50, 50)):
                    print("Mask(%d,%d) 0x%x 0x%x" % (x, y, refMaskedImage.getMask()[x, y, afwImage.LOCAL],
                                                     self.cnvMaskedImage.getMask()[x, y, afwImage.LOCAL]))

        self.assertImagesAlmostEqual(
            self.cnvImage, refMaskedImage.getImage(), atol=atol, rtol=rtol)
        self.assertMaskedImagesAlmostEqual(
            self.cnvMaskedImage, refMaskedImage, atol=atol, rtol=rtol)

        if not sameMaskPlaneDicts(self.cnvMaskedImage, self.maskedImage):
            self.cnvMaskedImage.writeFits("act%s" % (shortKernelDescr,))
            refMaskedImage.writeFits("des%s" % (shortKernelDescr,))
            self.fail("convolve(MaskedImage, kernel=%s, doNormalize=%s, "
                      "doCopyEdge=%s, maxInterpDist=%s) failed:\n%s" %
                      (kernelDescr, doNormalize, doCopyEdge, maxInterpDist,
                       "convolved mask dictionary does not match input"))

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
                inMaskedImage = afwImage.MaskedImageF(
                    lsst.geom.Extent2I(inWidth, inHeight))
                with self.assertRaises(Exception):
                    afwMath.convolve(self.cnvMaskedImage,
                                     inMaskedImage, kernel)

        for doNormalize in (True,):  # (False, True):
            convControl.setDoNormalize(doNormalize)
            for doCopyEdge in (False,):  # (False, True):
                convControl.setDoCopyEdge(doCopyEdge)
                self.runBasicTest(kernel, convControl=convControl, refKernel=refKernel,
                                  kernelDescr=kernelDescr, rtol=rtol, atol=atol)

        # verify that basicConvolve does not write to edge pixels
        self.runBasicConvolveEdgeTest(kernel, kernelDescr)

    def runBasicConvolveEdgeTest(self, kernel, kernelDescr):
        """Verify that basicConvolve does not write to edge pixels for this kind of kernel
        """
        fullBox = lsst.geom.Box2I(
            lsst.geom.Point2I(0, 0),
            ShiftedBBox.getDimensions(),
        )
        goodBox = kernel.shrinkBBox(fullBox)
        cnvMaskedImage = afwImage.MaskedImageF(
            FullMaskedImage, ShiftedBBox, afwImage.LOCAL, True)
        cnvMaskedImageCopy = afwImage.MaskedImageF(
            cnvMaskedImage, fullBox, afwImage.LOCAL, True)
        cnvMaskedImageCopyViewOfGoodRegion = afwImage.MaskedImageF(
            cnvMaskedImageCopy, goodBox, afwImage.LOCAL, False)

        # convolve with basicConvolve, which should leave the edge pixels alone
        convControl = afwMath.ConvolutionControl()
        mathDetail.basicConvolve(
            cnvMaskedImage, self.maskedImage, kernel, convControl)

        # reset the good region to the original convolved image;
        # this should reset the entire convolved image to its original self
        cnvMaskedImageGoodView = afwImage.MaskedImageF(
            cnvMaskedImage, goodBox, afwImage.LOCAL, False)
        cnvMaskedImageGoodView[:] = cnvMaskedImageCopyViewOfGoodRegion

        # assert that these two are equal
        msg = "basicConvolve(MaskedImage, kernel=%s) wrote to edge pixels" % (
            kernelDescr,)
        try:
            self.assertMaskedImagesAlmostEqual(cnvMaskedImage, cnvMaskedImageCopy,
                                               doVariance=True, rtol=0, atol=0, msg=msg)
        except Exception:
            # write out the images, then fail
            shortKernelDescr = self.removeGarbageChars(kernelDescr)
            cnvMaskedImage.writeFits(
                "actBasicConvolve%s" % (shortKernelDescr,))
            cnvMaskedImageCopy.writeFits(
                "desBasicConvolve%s" % (shortKernelDescr,))
            raise

    def testConvolutionControl(self):
        """Test the ConvolutionControl object
        """
        convControl = afwMath.ConvolutionControl()
        self.assertTrue(convControl.getDoNormalize())
        for doNormalize in (False, True):
            convControl.setDoNormalize(doNormalize)
            self.assertEqual(convControl.getDoNormalize(), doNormalize)

        self.assertFalse(convControl.getDoCopyEdge())
        for doCopyEdge in (False, True):
            convControl.setDoCopyEdge(doCopyEdge)
            self.assertEqual(convControl.getDoCopyEdge(), doCopyEdge)

        self.assertEqual(convControl.getMaxInterpolationDistance(), 10)
        for maxInterpDist in (0, 1, 2, 10, 100):
            convControl.setMaxInterpolationDistance(maxInterpDist)
            self.assertEqual(
                convControl.getMaxInterpolationDistance(), maxInterpDist)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testUnityConvolution(self):
        """Verify that convolution with a centered delta function reproduces the original.
        """
        # create a delta function kernel that has 1,1 in the center
        kFunc = afwMath.IntegerDeltaFunction2D(0.0, 0.0)
        kernel = afwMath.AnalyticKernel(3, 3, kFunc)
        doNormalize = False
        doCopyEdge = False

        afwMath.convolve(self.cnvImage, self.maskedImage.getImage(),
                         kernel, doNormalize, doCopyEdge)

        afwMath.convolve(self.cnvMaskedImage, self.maskedImage,
                         kernel, doNormalize, doCopyEdge)
        cnvImMaskVarArr = self.cnvMaskedImage.getArrays()

        skipMaskArr = numpy.array(numpy.isnan(
            cnvImMaskVarArr[0]), dtype=numpy.uint16)

        kernelDescr = "Centered DeltaFunctionKernel (testing unity convolution)"
        self.assertImagesAlmostEqual(self.cnvImage, self.maskedImage.getImage(),
                                     skipMask=skipMaskArr, msg=kernelDescr)
        self.assertMaskedImagesAlmostEqual(self.cnvMaskedImage, self.maskedImage,
                                           skipMask=skipMaskArr, msg=kernelDescr)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testFixedKernelConvolve(self):
        """Test convolve with a fixed kernel
        """
        kWidth = 6
        kHeight = 7

        kFunc = afwMath.GaussianFunction2D(2.5, 1.5, 0.5)
        analyticKernel = afwMath.AnalyticKernel(kWidth, kHeight, kFunc)
        kernelImage = afwImage.ImageD(lsst.geom.Extent2I(kWidth, kHeight))
        analyticKernel.computeImage(kernelImage, False)
        fixedKernel = afwMath.FixedKernel(kernelImage)

        self.runStdTest(fixedKernel, kernelDescr="Gaussian FixedKernel")

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testSeparableConvolve(self):
        """Test convolve of a separable kernel with a spatially invariant Gaussian function
        """
        kWidth = 7
        kHeight = 6

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        gaussFunc2 = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        separableKernel = afwMath.SeparableKernel(
            kWidth, kHeight, gaussFunc1, gaussFunc1)
        analyticKernel = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc2)

        self.runStdTest(
            separableKernel,
            refKernel=analyticKernel,
            kernelDescr="Gaussian Separable Kernel (compared to AnalyticKernel equivalent)")

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testSpatiallyInvariantConvolve(self):
        """Test convolution with a spatially invariant Gaussian function
        """
        kWidth = 6
        kHeight = 7

        kFunc = afwMath.GaussianFunction2D(2.5, 1.5, 0.5)
        kernel = afwMath.AnalyticKernel(kWidth, kHeight, kFunc)

        self.runStdTest(kernel, kernelDescr="Gaussian Analytic Kernel")

    @unittest.skipIf(dataDir is None, "afwdata not setup")
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
            (minSigma, 0.0, (maxSigma - minSigma) / self.height),
            (0.0, 0.0, 0.0),
        )

        kFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        kernel = afwMath.AnalyticKernel(kWidth, kHeight, kFunc, sFunc)
        kernel.setSpatialParameters(sParams)

        for maxInterpDist, rtol, methodStr in (
            (0, 1.0e-5, "brute force"),
            (10, 1.0e-5, "interpolation over 10 x 10 pixels"),
        ):
            self.runStdTest(
                kernel,
                kernelDescr="Spatially Varying Gaussian Analytic Kernel using %s" % (
                    methodStr,),
                maxInterpDist=maxInterpDist,
                rtol=rtol)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
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
            (minSigma, 0.0, (maxSigma - minSigma) / self.height),
            (0.0, 0.0, 0.0),
        )

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        gaussFunc2 = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        separableKernel = afwMath.SeparableKernel(
            kWidth, kHeight, gaussFunc1, gaussFunc1, sFunc)
        analyticKernel = afwMath.AnalyticKernel(
            kWidth, kHeight, gaussFunc2, sFunc)
        separableKernel.setSpatialParameters(sParams[0:2])
        analyticKernel.setSpatialParameters(sParams)

        self.runStdTest(separableKernel, refKernel=analyticKernel,
                        kernelDescr="Spatially Varying Gaussian Separable Kernel")

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testDeltaConvolve(self):
        """Test convolution with various delta function kernels using optimized code
        """
        for kWidth in range(1, 4):
            for kHeight in range(1, 4):
                for activeCol in range(kWidth):
                    for activeRow in range(kHeight):
                        kernel = afwMath.DeltaFunctionKernel(
                            kWidth, kHeight,
                            lsst.geom.Point2I(activeCol, activeRow))
                        if display and False:
                            kim = afwImage.ImageD(kWidth, kHeight)
                            kernel.computeImage(kim, False)
                            ds9.mtv(kim, frame=1)

                        self.runStdTest(
                            kernel, kernelDescr="Delta Function Kernel")

    @unittest.skipIf(dataDir is None, "afwdata not setup")
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
                (0.0, 0.01/self.width, 0.0/self.height),
                (0.0, 0.0/self.width, 0.01/self.height),
                (0.5, 0.005/self.width, -0.005/self.height),
            )[:nBasisKernels]

            gaussParamsList = (
                (1.5, 1.5, 0.0),
                (2.5, 1.5, 0.0),
                (2.5, 1.5, math.pi / 2.0),
                (2.5, 2.5, 0.0),
            )[:nBasisKernels]
            basisKernelList = makeGaussianKernelList(
                kWidth, kHeight, gaussParamsList)
            kernel = afwMath.LinearCombinationKernel(basisKernelList, sFunc)
            kernel.setSpatialParameters(sParams)

            for maxInterpDist, rtol, methodStr in (
                (0, 1.0e-5, "brute force"),
                (10, 1.0e-5, "interpolation over 10 x 10 pixels"),
            ):
                self.runStdTest(
                    kernel,
                    kernelDescr="%s with %d basis kernels convolved using %s" %
                    ("Spatially Varying Gaussian Analytic Kernel",
                     nBasisKernels, methodStr),
                    maxInterpDist=maxInterpDist,
                    rtol=rtol)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
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
            (0.0, 1.0/self.width, 0.0/self.height),
            (0.0, 0.0/self.width, 1.0/self.height),
            (0.5, 0.0, 0.0),
        )

        basisKernelList = makeDeltaFunctionKernelList(kWidth, kHeight)
        kernel = afwMath.LinearCombinationKernel(basisKernelList, sFunc)
        kernel.setSpatialParameters(sParams)

        for maxInterpDist, rtol, methodStr in (
            (0, 1.0e-5, "brute force"),
            (10, 1.0e-3, "interpolation over 10 x 10 pixels"),
        ):
            self.runStdTest(
                kernel,
                kernelDescr="Spatially varying LinearCombinationKernel of delta function kernels using %s" %
                (methodStr,),
                maxInterpDist=maxInterpDist,
                rtol=rtol)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testZeroWidthKernel(self):
        """Convolution by a 0x0 kernel should raise an exception.

        The only way to produce a 0x0 kernel is to use the default constructor
        (which exists only to support persistence; it does not produce a useful kernel).
        """
        kernelList = [
            afwMath.FixedKernel(),
            afwMath.AnalyticKernel(),
            afwMath.SeparableKernel(),
            # afwMath.DeltaFunctionKernel(),  # DeltaFunctionKernel has no
            # default constructor
            afwMath.LinearCombinationKernel(),
        ]
        convolutionControl = afwMath.ConvolutionControl()
        for kernel in kernelList:
            with self.assertRaises(Exception):
                afwMath.convolve(self.cnvMaskedImage,
                                 self.maskedImage, kernel, convolutionControl)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
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
            (0.0, 1.0/self.width, 0.0/self.height),
            (0.0, 0.0/self.width, 1.0/self.height),
        )

        # create three kernels with some non-overlapping pixels
        # (non-zero pixels in one kernel vs. zero pixels in other kernels);
        # note: the extreme example of this is delta function kernels, but this
        # is less extreme
        basisKernelList = []
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
            (0, 1.0e-5, "brute force"),
            (10, 3.0e-3, "interpolation over 10 x 10 pixels"),
        ):
            self.runStdTest(
                kernel,
                kernelDescr="Spatially varying LinearCombinationKernel of basis "
                            "kernels with low covariance, using %s" % (
                                methodStr,),
                maxInterpDist=maxInterpDist,
                rtol=rtol)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

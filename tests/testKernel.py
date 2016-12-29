#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
from __future__ import absolute_import, division, print_function
import math
import re
import unittest

from builtins import range
import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions as pexExcept
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath


def makeGaussianKernelList(kWidth, kHeight, gaussParamsList):
    """Create a list of gaussian kernels.

    This is useful for constructing a LinearCombinationKernel.

    Inputs:
    - kWidth, kHeight: width and height of kernel
    - gaussParamsList: a list of parameters for GaussianFunction2D (each a 3-tuple of floats)
    """
    kVec = []
    for majorSigma, minorSigma, angle in gaussParamsList:
        kFunc = afwMath.GaussianFunction2D(majorSigma, minorSigma, angle)
        kVec.append(afwMath.AnalyticKernel(kWidth, kHeight, kFunc))
    return kVec


def makeDeltaFunctionKernelList(kWidth, kHeight):
    """Create a list of delta function kernels

    This is useful for constructing a LinearCombinationKernel.
    """
    kVec = []
    for activeCol in range(kWidth):
        for activeRow in range(kHeight):
            kVec.append(afwMath.DeltaFunctionKernel(kWidth, kHeight, afwGeom.Point2I(activeCol, activeRow)))
    return kVec


class KernelTestCase(lsst.utils.tests.TestCase):
    """A test case for Kernels"""

    def testAnalyticKernel(self):
        """Test AnalyticKernel using a Gaussian function
        """
        kWidth = 5
        kHeight = 8

        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        kernel = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc)
        self.basicTests(kernel, 3, dimMustMatch=False)
        fArr = np.zeros(shape=[kernel.getWidth(), kernel.getHeight()], dtype=float)
        for xsigma in (0.1, 1.0, 3.0):
            for ysigma in (0.1, 1.0, 3.0):
                gaussFunc.setParameters((xsigma, ysigma, 0.0))
                # compute array of function values and normalize
                for row in range(kernel.getHeight()):
                    y = row - kernel.getCtrY()
                    for col in range(kernel.getWidth()):
                        x = col - kernel.getCtrX()
                        fArr[col, row] = gaussFunc(x, y)
                fArr /= fArr.sum()

                kernel.setKernelParameters((xsigma, ysigma, 0.0))
                kImage = afwImage.ImageD(kernel.getDimensions())
                kernel.computeImage(kImage, True)

                kArr = kImage.getArray().transpose()
                if not np.allclose(fArr, kArr):
                    self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" %
                              (kernel.__class__.__name__, kArr, fArr, xsigma, ysigma))

        kernel.setKernelParameters((0.5, 1.1, 0.3))
        kernelClone = kernel.clone()
        errStr = self.compareKernels(kernel, kernelClone)
        if errStr:
            self.fail(errStr)

        kernel.setKernelParameters((1.5, 0.2, 0.7))
        errStr = self.compareKernels(kernel, kernelClone)
        if not errStr:
            self.fail("Clone was modified by changing original's kernel parameters")

        self.verifyCache(kernel, hasCache=False)

    def verifyCache(self, kernel, hasCache=False):
        """Verify the kernel cache

        @param kernel: kernel to test
        @param hasCache: set True if this kind of kernel supports a cache, False otherwise
        """
        for cacheSize in (0, 100, 2000):
            kernel.computeCache(cacheSize)
            self.assertEqual(kernel.getCacheSize(), cacheSize if hasCache else 0)
            kernelCopy = kernel.clone()
            self.assertEqual(kernelCopy.getCacheSize(), kernel.getCacheSize())

    def testShrinkGrowBBox(self):
        """Test Kernel methods shrinkBBox and growBBox
        """
        boxStart = afwGeom.Point2I(3, -3)
        for kWidth in (1, 2, 6):
            for kHeight in (1, 2, 5):
                for deltaWidth in (-1, 0, 1, 20):
                    fullWidth = kWidth + deltaWidth
                    for deltaHeight in (-1, 0, 1, 20):
                        fullHeight = kHeight + deltaHeight
                        kernel = afwMath.DeltaFunctionKernel(kWidth, kHeight, afwGeom.Point2I(0, 0))
                        fullBBox = afwGeom.Box2I(boxStart, afwGeom.Extent2I(fullWidth, fullHeight))
                        if (fullWidth < kWidth) or (fullHeight < kHeight):
                            self.assertRaises(Exception, kernel.shrinkBBox, fullBBox)
                            continue

                        shrunkBBox = kernel.shrinkBBox(fullBBox)
                        self.assertEqual(shrunkBBox.getWidth(), fullWidth + 1 - kWidth)
                        self.assertEqual(shrunkBBox.getHeight(), fullHeight + 1 - kHeight)
                        self.assertEqual(shrunkBBox.getMinX(), boxStart[0] + kernel.getCtrX())
                        self.assertEqual(shrunkBBox.getMinY(), boxStart[1] + kernel.getCtrY())
                        newFullBBox = kernel.growBBox(shrunkBBox)
                        self.assertEqual(newFullBBox, fullBBox, "growBBox(shrinkBBox(x)) != x")

    def testDeltaFunctionKernel(self):
        """Test DeltaFunctionKernel
        """
        for kWidth in range(1, 4):
            for kHeight in range(1, 4):
                for activeCol in range(kWidth):
                    for activeRow in range(kHeight):
                        kernel = afwMath.DeltaFunctionKernel(kWidth, kHeight,
                                                             afwGeom.Point2I(activeCol, activeRow))
                        kImage = afwImage.ImageD(kernel.getDimensions())
                        kSum = kernel.computeImage(kImage, False)
                        self.assertEqual(kSum, 1.0)
                        kArr = kImage.getArray().transpose()
                        self.assertEqual(kArr[activeCol, activeRow], 1.0)
                        kArr[activeCol, activeRow] = 0.0
                        self.assertEqual(kArr.sum(), 0.0)

                        errStr = self.compareKernels(kernel, kernel.clone())
                        if errStr:
                            self.fail(errStr)

                self.assertRaises(pexExcept.InvalidParameterError,
                                  afwMath.DeltaFunctionKernel, 0, kHeight, afwGeom.Point2I(kWidth, kHeight))
                self.assertRaises(pexExcept.InvalidParameterError,
                                  afwMath.DeltaFunctionKernel, kWidth, 0, afwGeom.Point2I(kWidth, kHeight))

        kernel = afwMath.DeltaFunctionKernel(5, 6, afwGeom.Point2I(1, 1))
        self.basicTests(kernel, 0)

        self.verifyCache(kernel, hasCache=False)

    def testFixedKernel(self):
        """Test FixedKernel using a ramp function
        """
        kWidth = 5
        kHeight = 6

        inArr = np.arange(kWidth * kHeight, dtype=float)
        inArr.shape = [kWidth, kHeight]

        inImage = afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight))
        for row in range(inImage.getHeight()):
            for col in range(inImage.getWidth()):
                inImage.set(col, row, inArr[col, row])

        kernel = afwMath.FixedKernel(inImage)
        self.basicTests(kernel, 0)
        outImage = afwImage.ImageD(kernel.getDimensions())
        kernel.computeImage(outImage, False)

        outArr = outImage.getArray().transpose()
        if not np.allclose(inArr, outArr):
            self.fail("%s = %s != %s (not normalized)" %
                      (kernel.__class__.__name__, inArr, outArr))

        normInArr = inArr / inArr.sum()
        normOutImage = afwImage.ImageD(kernel.getDimensions())
        kernel.computeImage(normOutImage, True)
        normOutArr = normOutImage.getArray().transpose()
        if not np.allclose(normOutArr, normInArr):
            self.fail("%s = %s != %s (normalized)" %
                      (kernel.__class__.__name__, normInArr, normOutArr))

        errStr = self.compareKernels(kernel, kernel.clone())
        if errStr:
            self.fail(errStr)

        self.verifyCache(kernel, hasCache=False)

    def testLinearCombinationKernelDelta(self):
        """Test LinearCombinationKernel using a set of delta basis functions
        """
        kWidth = 3
        kHeight = 2

        # create list of kernels
        basisKernelList = makeDeltaFunctionKernelList(kWidth, kHeight)
        basisImArrList = []
        for basisKernel in basisKernelList:
            basisImage = afwImage.ImageD(basisKernel.getDimensions())
            basisKernel.computeImage(basisImage, True)
            basisImArrList.append(basisImage.getArray())

        kParams = [0.0]*len(basisKernelList)
        kernel = afwMath.LinearCombinationKernel(basisKernelList, kParams)
        self.assertTrue(kernel.isDeltaFunctionBasis())
        self.basicTests(kernel, len(kParams))
        for ii in range(len(basisKernelList)):
            kParams = [0.0]*len(basisKernelList)
            kParams[ii] = 1.0
            kernel.setKernelParameters(kParams)
            kIm = afwImage.ImageD(kernel.getDimensions())
            kernel.computeImage(kIm, True)
            kImArr = kIm.getArray()
            if not np.allclose(kImArr, basisImArrList[ii]):
                self.fail("%s = %s != %s for the %s'th basis kernel" %
                          (kernel.__class__.__name__, kImArr, basisImArrList[ii], ii))

        kernelClone = kernel.clone()
        errStr = self.compareKernels(kernel, kernelClone)
        if errStr:
            self.fail(errStr)

        self.verifyCache(kernel, hasCache=False)

    def testComputeImageRaise(self):
        """Test Kernel.computeImage raises OverflowException iff doNormalize True and kernel sum exactly 0
        """
        kWidth = 4
        kHeight = 3

        polyFunc1 = afwMath.PolynomialFunction1D(0)
        polyFunc2 = afwMath.PolynomialFunction2D(0)
        analKernel = afwMath.AnalyticKernel(kWidth, kHeight, polyFunc2)
        kImage = afwImage.ImageD(analKernel.getDimensions())
        for coeff in (-1, -1e-99, 0, 1e99, 1):
            analKernel.setKernelParameters([coeff])

            analKernel.computeImage(kImage, False)
            fixedKernel = afwMath.FixedKernel(kImage)

            sepKernel = afwMath.SeparableKernel(kWidth, kHeight, polyFunc1, polyFunc1)
            sepKernel.setKernelParameters([coeff, coeff])

            kernelList = []
            kernelList.append(analKernel)
            lcKernel = afwMath.LinearCombinationKernel(kernelList, [1])
            self.assertFalse(lcKernel.isDeltaFunctionBasis())

            doRaise = (coeff == 0)
            self.basicTestComputeImageRaise(analKernel, doRaise, "AnalyticKernel")
            self.basicTestComputeImageRaise(fixedKernel, doRaise, "FixedKernel")
            self.basicTestComputeImageRaise(sepKernel, doRaise, "SeparableKernel")
            self.basicTestComputeImageRaise(lcKernel, doRaise, "LinearCombinationKernel")

        lcKernel.setKernelParameters([0])
        self.basicTestComputeImageRaise(lcKernel, True, "LinearCombinationKernel")

    def testLinearCombinationKernelAnalytic(self):
        """Test LinearCombinationKernel using analytic basis kernels.

        The basis kernels are mutable so that we can verify that the
        LinearCombinationKernel has private copies of the basis kernels.
        """
        kWidth = 5
        kHeight = 8

        # create list of kernels
        basisImArrList = []
        basisKernelList = []
        for basisKernelParams in [(1.2, 0.3, 1.570796), (1.0, 0.2, 0.0)]:
            basisKernelFunction = afwMath.GaussianFunction2D(*basisKernelParams)
            basisKernel = afwMath.AnalyticKernel(kWidth, kHeight, basisKernelFunction)
            basisImage = afwImage.ImageD(basisKernel.getDimensions())
            basisKernel.computeImage(basisImage, True)
            basisImArrList.append(basisImage.getArray())
            basisKernelList.append(basisKernel)

        kParams = [0.0]*len(basisKernelList)
        kernel = afwMath.LinearCombinationKernel(basisKernelList, kParams)
        self.assertTrue(not kernel.isDeltaFunctionBasis())
        self.basicTests(kernel, len(kParams))

        # make sure the linear combination kernel has private copies of its basis kernels
        # by altering the local basis kernels and making sure the new images do NOT match
        modBasisImArrList = []
        for basisKernel in basisKernelList:
            basisKernel.setKernelParameters((0.4, 0.5, 0.6))
            modBasisImage = afwImage.ImageD(basisKernel.getDimensions())
            basisKernel.computeImage(modBasisImage, True)
            modBasisImArrList.append(modBasisImage.getArray())

        for ii in range(len(basisKernelList)):
            kParams = [0.0]*len(basisKernelList)
            kParams[ii] = 1.0
            kernel.setKernelParameters(kParams)
            kIm = afwImage.ImageD(kernel.getDimensions())
            kernel.computeImage(kIm, True)
            kImArr = kIm.getArray()
            if not np.allclose(kImArr, basisImArrList[ii]):
                self.fail("%s = %s != %s for the %s'th basis kernel" %
                          (kernel.__class__.__name__, kImArr, basisImArrList[ii], ii))
            if np.allclose(kImArr, modBasisImArrList[ii]):
                self.fail("%s = %s == %s for *modified* %s'th basis kernel" %
                          (kernel.__class__.__name__, kImArr, modBasisImArrList[ii], ii))

        kernelClone = kernel.clone()
        errStr = self.compareKernels(kernel, kernelClone)
        if errStr:
            self.fail(errStr)

        self.verifyCache(kernel, hasCache=False)

    def testSeparableKernel(self):
        """Test SeparableKernel using a Gaussian function
        """
        kWidth = 5
        kHeight = 8

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        kernel = afwMath.SeparableKernel(kWidth, kHeight, gaussFunc1, gaussFunc1)
        self.basicTests(kernel, 2)
        fArr = np.zeros(shape=[kernel.getWidth(), kernel.getHeight()], dtype=float)
        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        for xsigma in (0.1, 1.0, 3.0):
            gaussFunc1.setParameters((xsigma,))
            for ysigma in (0.1, 1.0, 3.0):
                gaussFunc.setParameters((xsigma, ysigma, 0.0))
                # compute array of function values and normalize
                for row in range(kernel.getHeight()):
                    y = row - kernel.getCtrY()
                    for col in range(kernel.getWidth()):
                        x = col - kernel.getCtrX()
                        fArr[col, row] = gaussFunc(x, y)
                fArr /= fArr.sum()

                kernel.setKernelParameters((xsigma, ysigma))
                kImage = afwImage.ImageD(kernel.getDimensions())
                kernel.computeImage(kImage, True)
                kArr = kImage.getArray().transpose()
                if not np.allclose(fArr, kArr):
                    self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" %
                              (kernel.__class__.__name__, kArr, fArr, xsigma, ysigma))
        kernelClone = kernel.clone()
        errStr = self.compareKernels(kernel, kernelClone)
        if errStr:
            self.fail(errStr)

        kernel.setKernelParameters((1.2, 0.6))
        errStr = self.compareKernels(kernel, kernelClone)
        if not errStr:
            self.fail("Clone was modified by changing original's kernel parameters")

        self.verifyCache(kernel, hasCache=True)

    def testMakeBadKernels(self):
        """Attempt to make various invalid kernels; make sure the constructor shows an exception
        """
        kWidth = 4
        kHeight = 3

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        gaussFunc2 = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        spFunc = afwMath.PolynomialFunction2D(1)
        kernelList = []
        kernelList.append(afwMath.FixedKernel(afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight), 0.1)))
        kernelList.append(afwMath.FixedKernel(afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight), 0.2)))

        for numKernelParams in (2, 4):
            spFuncList = []
            for ii in range(numKernelParams):
                spFuncList.append(spFunc.clone())
            try:
                afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc2, spFuncList)
                self.fail("Should have failed with wrong # of spatial functions")
            except pexExcept.Exception:
                pass

        for numKernelParams in (1, 3):
            spFuncList = []
            for ii in range(numKernelParams):
                spFuncList.append(spFunc.clone())
            try:
                afwMath.LinearCombinationKernel(kernelList, spFuncList)
                self.fail("Should have failed with wrong # of spatial functions")
            except pexExcept.Exception:
                pass
            kParamList = [0.2]*numKernelParams
            try:
                afwMath.LinearCombinationKernel(kernelList, kParamList)
                self.fail("Should have failed with wrong # of kernel parameters")
            except pexExcept.Exception:
                pass
            try:
                afwMath.SeparableKernel(kWidth, kHeight, gaussFunc1, gaussFunc1, spFuncList)
                self.fail("Should have failed with wrong # of spatial functions")
            except pexExcept.Exception:
                pass

        for pointX in range(-1, kWidth+2):
            for pointY in range(-1, kHeight+2):
                if (0 <= pointX < kWidth) and (0 <= pointY < kHeight):
                    continue
                try:
                    afwMath.DeltaFunctionKernel(kWidth, kHeight, afwGeom.Point2I(pointX, pointY))
                    self.fail("Should have failed with point not on kernel")
                except pexExcept.Exception:
                    pass

    def testSVAnalyticKernel(self):
        """Test spatially varying AnalyticKernel using a Gaussian function

        Just tests cloning.
        """
        kWidth = 5
        kHeight = 8

        # spatial model
        spFunc = afwMath.PolynomialFunction2D(1)

        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.5, 0.5, 0.5),
        )

        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        kernel = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc, spFunc)
        kernel.setSpatialParameters(sParams)

        kernelClone = kernel.clone()
        errStr = self.compareKernels(kernel, kernelClone)
        if errStr:
            self.fail(errStr)

        newSParams = (
            (0.1, 0.2, 0.5),
            (0.1, 0.5, 0.2),
            (0.2, 0.3, 0.3),
        )
        kernel.setSpatialParameters(newSParams)
        errStr = self.compareKernels(kernel, kernelClone)
        if not errStr:
            self.fail("Clone was modified by changing original's spatial parameters")

        #
        # check that we can construct a FixedKernel from a LinearCombinationKernel
        #
        x, y = 100, 200
        kernel2 = afwMath.FixedKernel(kernel, afwGeom.PointD(x, y))

        self.assertTrue(re.search("AnalyticKernel", kernel.toString()))
        self.assertFalse(kernel2.isSpatiallyVarying())

        self.assertTrue(re.search("FixedKernel", kernel2.toString()))
        self.assertTrue(kernel.isSpatiallyVarying())

        kim = afwImage.ImageD(kernel.getDimensions())
        kernel.computeImage(kim, True, x, y)

        kim2 = afwImage.ImageD(kernel2.getDimensions())
        kernel2.computeImage(kim2, True)

        self.assertTrue(np.allclose(kim.getArray(), kim2.getArray()))

    def testSVLinearCombinationKernelFixed(self):
        """Test a spatially varying LinearCombinationKernel whose bases are FixedKernels"""
        kWidth = 3
        kHeight = 2

        # create image arrays for the basis kernels
        basisImArrList = []
        imArr = np.zeros((kWidth, kHeight), dtype=float)
        imArr += 0.1
        imArr[kWidth//2, :] = 0.9
        basisImArrList.append(imArr)
        imArr = np.zeros((kWidth, kHeight), dtype=float)
        imArr += 0.2
        imArr[:, kHeight//2] = 0.8
        basisImArrList.append(imArr)

        # create a list of basis kernels from the images
        basisKernelList = []
        for basisImArr in basisImArrList:
            basisImage = afwImage.makeImageFromArray(basisImArr.transpose().copy())
            kernel = afwMath.FixedKernel(basisImage)
            basisKernelList.append(kernel)

        # create spatially varying linear combination kernel
        spFunc = afwMath.PolynomialFunction2D(1)

        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )

        kernel = afwMath.LinearCombinationKernel(basisKernelList, spFunc)
        self.assertFalse(kernel.isDeltaFunctionBasis())
        self.basicTests(kernel, 2, nSpatialParams=3)
        kernel.setSpatialParameters(sParams)
        kImage = afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight))
        for colPos, rowPos, coeff0, coeff1 in [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5, 0.5),
        ]:
            kernel.computeImage(kImage, False, colPos, rowPos)
            kImArr = kImage.getArray().transpose()
            refKImArr = (basisImArrList[0] * coeff0) + (basisImArrList[1] * coeff1)
            if not np.allclose(kImArr, refKImArr):
                self.fail("%s = %s != %s at colPos=%s, rowPos=%s" %
                          (kernel.__class__.__name__, kImArr, refKImArr, colPos, rowPos))

        sParams = (
            (0.1, 1.0, 0.0),
            (0.1, 0.0, 1.0),
        )
        kernel.setSpatialParameters(sParams)
        kernelClone = kernel.clone()
        errStr = self.compareKernels(kernel, kernelClone)
        if errStr:
            self.fail(errStr)

        newSParams = (
            (0.1, 0.2, 0.5),
            (0.1, 0.5, 0.2),
        )
        kernel.setSpatialParameters(newSParams)
        errStr = self.compareKernels(kernel, kernelClone)
        if not errStr:
            self.fail("Clone was modified by changing original's spatial parameters")

    def testSVSeparableKernel(self):
        """Test spatially varying SeparableKernel using a Gaussian function

        Just tests cloning.
        """
        kWidth = 5
        kHeight = 8

        # spatial model
        spFunc = afwMath.PolynomialFunction2D(1)

        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
        )

        gaussFunc = afwMath.GaussianFunction1D(1.0)
        kernel = afwMath.SeparableKernel(kWidth, kHeight, gaussFunc, gaussFunc, spFunc)
        kernel.setSpatialParameters(sParams)

        kernelClone = kernel.clone()
        errStr = self.compareKernels(kernel, kernelClone)
        if errStr:
            self.fail(errStr)

        newSParams = (
            (0.1, 0.2, 0.5),
            (0.1, 0.5, 0.2),
        )
        kernel.setSpatialParameters(newSParams)
        errStr = self.compareKernels(kernel, kernelClone)
        if not errStr:
            self.fail("Clone was modified by changing original's spatial parameters")

    def testSetCtr(self):
        """Test setCtrCol/Row"""
        kWidth = 3
        kHeight = 4

        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        kernel = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc)
        for xCtr in range(kWidth):
            kernel.setCtrX(xCtr)
            for yCtr in range(kHeight):
                kernel.setCtrY(yCtr)
                self.assertEqual(kernel.getCtrX(), xCtr)
                self.assertEqual(kernel.getCtrY(), yCtr)

    def testZeroSizeKernel(self):
        """Creating a kernel with width or height < 1 should raise an exception.

        Note: this ignores the default constructors, which produce kernels with height = width = 0.
        The default constructors are only intended to support persistence, not to produce useful kernels.
        """
        gaussFunc2D = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        gaussFunc1D = afwMath.GaussianFunction1D(1.0)
        zeroPoint = afwGeom.Point2I(0, 0)
        for kWidth in (-1, 0, 1):
            for kHeight in (-1, 0, 1):
                if (kHeight > 0) and (kWidth > 0):
                    continue
                if (kHeight >= 0) and (kWidth >= 0):
                    # don't try to create an image with negative dimensions
                    blankImage = afwImage.ImageF(afwGeom.Extent2I(kWidth, kHeight))
                    self.assertRaises(Exception, afwMath.FixedKernel, blankImage)
                self.assertRaises(Exception, afwMath.AnalyticKernel, kWidth, kHeight, gaussFunc2D)
                self.assertRaises(Exception, afwMath.SeparableKernel,
                                  kWidth, kHeight, gaussFunc1D, gaussFunc1D)
                self.assertRaises(Exception, afwMath.DeltaFunctionKernel, kWidth, kHeight, zeroPoint)

    def testRefactorDeltaLinearCombinationKernel(self):
        """Test LinearCombinationKernel.refactor with delta function basis kernels
        """
        kWidth = 4
        kHeight = 3

        for spOrder in (0, 1, 2):
            spFunc = afwMath.PolynomialFunction2D(spOrder)
            numSpParams = spFunc.getNParameters()

            basisKernelList = makeDeltaFunctionKernelList(kWidth, kHeight)
            kernel = afwMath.LinearCombinationKernel(basisKernelList, spFunc)

            numBasisKernels = kernel.getNKernelParameters()
            maxVal = 1.01 + ((numSpParams - 1) * 0.1)
            sParamList = [np.arange(kInd + 1.0, kInd + maxVal, 0.1) for kInd in range(numBasisKernels)]
            kernel.setSpatialParameters(sParamList)

            refKernel = kernel.refactor()
            self.assertTrue(refKernel)
            errStr = self.compareKernels(kernel, refKernel, compareParams=False)
            if errStr:
                self.fail("failed with %s for spOrder=%s (numSpCoeff=%s)" % (errStr, spOrder, numSpParams))

    def testRefactorGaussianLinearCombinationKernel(self):
        """Test LinearCombinationKernel.refactor with Gaussian basis kernels
        """
        kWidth = 4
        kHeight = 3

        for spOrder in (0, 1, 2):
            spFunc = afwMath.PolynomialFunction2D(spOrder)
            numSpParams = spFunc.getNParameters()

            gaussParamsList = [
                (1.5, 1.5, 0.0),
                (2.5, 1.5, 0.0),
                (2.5, 1.5, math.pi / 2.0),
            ]
            gaussBasisKernelList = makeGaussianKernelList(kWidth, kHeight, gaussParamsList)
            kernel = afwMath.LinearCombinationKernel(gaussBasisKernelList, spFunc)

            numBasisKernels = kernel.getNKernelParameters()
            maxVal = 1.01 + ((numSpParams - 1) * 0.1)
            sParamList = [np.arange(kInd + 1.0, kInd + maxVal, 0.1) for kInd in range(numBasisKernels)]
            kernel.setSpatialParameters(sParamList)

            refKernel = kernel.refactor()
            self.assertTrue(refKernel)
            errStr = self.compareKernels(kernel, refKernel, compareParams=False)
            if errStr:
                self.fail("failed with %s for spOrder=%s; numSpCoeff=%s" % (errStr, spOrder, numSpParams))

    def basicTests(self, kernel, nKernelParams, nSpatialParams=0, dimMustMatch=True):
        """Basic tests of a kernel"""
        self.assertEqual(kernel.getNSpatialParameters(), nSpatialParams)
        self.assertEqual(kernel.getNKernelParameters(), nKernelParams)
        if nSpatialParams == 0:
            self.assertFalse(kernel.isSpatiallyVarying())
            for ii in range(nKernelParams+5):
                self.assertRaises(pexExcept.InvalidParameterError,
                                  kernel.getSpatialFunction, ii)
        else:
            self.assertTrue(kernel.isSpatiallyVarying())
            for ii in range(nKernelParams):
                kernel.getSpatialFunction(ii)
            for ii in range(nKernelParams, nKernelParams+5):
                self.assertRaises(pexExcept.InvalidParameterError,
                                  kernel.getSpatialFunction, ii)

        # test a range of numbers of parameters, including both valid and invalid sized tuples.
        for nsp in range(nSpatialParams + 2):
            spatialParamsForOneKernel = [1.0,]*nsp
            for nkp in range(nKernelParams + 2):
                spatialParams = [spatialParamsForOneKernel,]*nkp
                if ((nkp == nKernelParams) and ((nsp == nSpatialParams) or (nkp == 0))):
                    kernel.setSpatialParameters(spatialParams)
                    if nsp == 0:
                        # A non-spatially varying kernel returns an empty tuple, even though
                        # it can only be set with a tuple of empty tuples, one per kernel parameter.
                        self.assertEqual(kernel.getSpatialParameters(), [])
                    else:
                        # a spatially varying kernel should return exactly what we set it to be.
                        self.assertEqual(kernel.getSpatialParameters(), spatialParams)
                else:
                    with self.assertRaises(pexExcept.InvalidParameterError):
                        kernel.setSpatialParameters(spatialParams)

        kernelDim = kernel.getDimensions()
        kernelCtr = kernel.getCtr()
        for dx in (-1, 0, 1):
            xDim = kernelDim.getX() + dx
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                yDim = kernelDim.getY() + dy
                image = afwImage.ImageD(xDim, yDim)
                if (dx == dy == 0) or not dimMustMatch:
                    ksum = kernel.computeImage(image, True)
                    self.assertAlmostEqual(ksum, 1.0)
                    llBorder = ((image.getDimensions() - kernelDim) / 2).truncate()
                    predCtr = afwGeom.Point2I(llBorder + kernelCtr)
                    self.assertEqual(kernel.getCtr(), predCtr)
                else:
                    self.assertRaises(Exception, kernel.computeImage, image, True)

    def basicTestComputeImageRaise(self, kernel, doRaise, kernelDescr=""):
        """Test that computeImage either does or does not raise an exception, as appropriate
        """
        kImage = afwImage.ImageD(kernel.getDimensions())
        try:
            kernel.computeImage(kImage, True)
            if doRaise:
                self.fail(kernelDescr + ".computeImage should have raised an exception")
        except pexExcept.Exception:
            if not doRaise:
                self.fail(kernelDescr + ".computeImage should not have raised an exception")

    def compareKernels(self, kernel1, kernel2, compareParams=True, newCtr1=(0, 0)):
        """Compare two kernels; return None if they match, else return a string kernelDescribing a difference.

        kernel1: one kernel to test
        kernel2: the other kernel to test
        compareParams: compare spatial parameters and kernel parameters if they exist
        newCtr: if not None then set the center of kernel1 and see if it changes the center of kernel2
        """
        retStrs = []
        if kernel1.getDimensions() != kernel2.getDimensions():
            retStrs.append("dimensions differ: %s != %s" % (kernel1.getDimensions(), kernel2.getDimensions()))
        ctr1 = kernel1.getCtrX(), kernel1.getCtrY()
        ctr2 = kernel2.getCtrX(), kernel2.getCtrY()
        if ctr1 != ctr2:
            retStrs.append("centers differ: %s != %s" % (ctr1, ctr2))
        if kernel1.isSpatiallyVarying() != kernel2.isSpatiallyVarying():
            retStrs.append("isSpatiallyVarying differs: %s != %s" %
                           (kernel1.isSpatiallyVarying(), kernel2.isSpatiallyVarying()))

        if compareParams:
            if kernel1.getSpatialParameters() != kernel2.getSpatialParameters():
                retStrs.append("spatial parameters differ: %s != %s" %
                               (kernel1.getSpatialParameters(), kernel2.getSpatialParameters()))
            if kernel1.getNSpatialParameters() != kernel2.getNSpatialParameters():
                retStrs.append("# spatial parameters differs: %s != %s" %
                               (kernel1.getNSpatialParameters(), kernel2.getNSpatialParameters()))
            if not kernel1.isSpatiallyVarying() and hasattr(kernel1, "getKernelParameters"):
                if kernel1.getKernelParameters() != kernel2.getKernelParameters():
                    retStrs.append("kernel parameters differs: %s != %s" %
                                   (kernel1.getKernelParameters(), kernel2.getKernelParameters()))
        if retStrs:
            return "; ".join(retStrs)

        im1 = afwImage.ImageD(kernel1.getDimensions())
        im2 = afwImage.ImageD(kernel2.getDimensions())
        if kernel1.isSpatiallyVarying():
            posList = [(0, 0), (200, 0), (0, 200), (200, 200)]
        else:
            posList = [(0, 0)]

        for doNormalize in (False, True):
            for pos in posList:
                kernel1.computeImage(im1, doNormalize, pos[0], pos[1])
                kernel2.computeImage(im2, doNormalize, pos[0], pos[1])
                im1Arr = im1.getArray()
                im2Arr = im2.getArray()
                if not np.allclose(im1Arr, im2Arr):
                    print("im1Arr =", im1Arr)
                    print("im2Arr =", im2Arr)
                    return "kernel images do not match at %s with doNormalize=%s" % (pos, doNormalize)

        if newCtr1 is not None:
            kernel1.setCtrX(newCtr1[0])
            kernel1.setCtrY(newCtr1[1])
            newCtr2 = kernel2.getCtrX(), kernel2.getCtrY()
            if ctr2 != newCtr2:
                return "changing center of kernel1 to %s changed the center of kernel2 from %s to %s" % \
                    (newCtr1, ctr2, newCtr2)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

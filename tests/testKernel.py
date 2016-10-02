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
#pybind11#import math
#pybind11#import re
#pybind11#import unittest
#pybind11#
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#
#pybind11#
#pybind11#def makeGaussianKernelList(kWidth, kHeight, gaussParamsList):
#pybind11#    """Create a list of gaussian kernels.
#pybind11#
#pybind11#    This is useful for constructing a LinearCombinationKernel.
#pybind11#
#pybind11#    Inputs:
#pybind11#    - kWidth, kHeight: width and height of kernel
#pybind11#    - gaussParamsList: a list of parameters for GaussianFunction2D (each a 3-tuple of floats)
#pybind11#    """
#pybind11#    kVec = afwMath.KernelList()
#pybind11#    for majorSigma, minorSigma, angle in gaussParamsList:
#pybind11#        kFunc = afwMath.GaussianFunction2D(majorSigma, minorSigma, angle)
#pybind11#        kVec.append(afwMath.AnalyticKernel(kWidth, kHeight, kFunc))
#pybind11#    return kVec
#pybind11#
#pybind11#
#pybind11#def makeDeltaFunctionKernelList(kWidth, kHeight):
#pybind11#    """Create a list of delta function kernels
#pybind11#
#pybind11#    This is useful for constructing a LinearCombinationKernel.
#pybind11#    """
#pybind11#    kVec = afwMath.KernelList()
#pybind11#    for activeCol in range(kWidth):
#pybind11#        for activeRow in range(kHeight):
#pybind11#            kVec.append(afwMath.DeltaFunctionKernel(kWidth, kHeight, afwGeom.Point2I(activeCol, activeRow)))
#pybind11#    return kVec
#pybind11#
#pybind11#
#pybind11#class KernelTestCase(lsst.utils.tests.TestCase):
#pybind11#    """A test case for Kernels"""
#pybind11#
#pybind11#    def testAnalyticKernel(self):
#pybind11#        """Test AnalyticKernel using a Gaussian function
#pybind11#        """
#pybind11#        kWidth = 5
#pybind11#        kHeight = 8
#pybind11#
#pybind11#        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        kernel = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc)
#pybind11#        self.basicTests(kernel, 3, dimMustMatch=False)
#pybind11#        fArr = numpy.zeros(shape=[kernel.getWidth(), kernel.getHeight()], dtype=float)
#pybind11#        for xsigma in (0.1, 1.0, 3.0):
#pybind11#            for ysigma in (0.1, 1.0, 3.0):
#pybind11#                gaussFunc.setParameters((xsigma, ysigma, 0.0))
#pybind11#                # compute array of function values and normalize
#pybind11#                for row in range(kernel.getHeight()):
#pybind11#                    y = row - kernel.getCtrY()
#pybind11#                    for col in range(kernel.getWidth()):
#pybind11#                        x = col - kernel.getCtrX()
#pybind11#                        fArr[col, row] = gaussFunc(x, y)
#pybind11#                fArr /= fArr.sum()
#pybind11#
#pybind11#                kernel.setKernelParameters((xsigma, ysigma, 0.0))
#pybind11#                kImage = afwImage.ImageD(kernel.getDimensions())
#pybind11#                kernel.computeImage(kImage, True)
#pybind11#
#pybind11#                kArr = kImage.getArray().transpose()
#pybind11#                if not numpy.allclose(fArr, kArr):
#pybind11#                    self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" %
#pybind11#                              (kernel.__class__.__name__, kArr, fArr, xsigma, ysigma))
#pybind11#
#pybind11#        kernel.setKernelParameters((0.5, 1.1, 0.3))
#pybind11#        kernelClone = kernel.clone()
#pybind11#        errStr = self.compareKernels(kernel, kernelClone)
#pybind11#        if errStr:
#pybind11#            self.fail(errStr)
#pybind11#
#pybind11#        kernel.setKernelParameters((1.5, 0.2, 0.7))
#pybind11#        errStr = self.compareKernels(kernel, kernelClone)
#pybind11#        if not errStr:
#pybind11#            self.fail("Clone was modified by changing original's kernel parameters")
#pybind11#
#pybind11#        self.verifyCache(kernel, hasCache=False)
#pybind11#
#pybind11#    def verifyCache(self, kernel, hasCache=False):
#pybind11#        """Verify the kernel cache
#pybind11#
#pybind11#        @param kernel: kernel to test
#pybind11#        @param hasCache: set True if this kind of kernel supports a cache, False otherwise
#pybind11#        """
#pybind11#        for cacheSize in (0, 100, 2000):
#pybind11#            kernel.computeCache(cacheSize)
#pybind11#            self.assertEqual(kernel.getCacheSize(), cacheSize if hasCache else 0)
#pybind11#            kernelCopy = kernel.clone()
#pybind11#            self.assertEqual(kernelCopy.getCacheSize(), kernel.getCacheSize())
#pybind11#
#pybind11#    def testShrinkGrowBBox(self):
#pybind11#        """Test Kernel methods shrinkBBox and growBBox
#pybind11#        """
#pybind11#        boxStart = afwGeom.Point2I(3, -3)
#pybind11#        for kWidth in (1, 2, 6):
#pybind11#            for kHeight in (1, 2, 5):
#pybind11#                for deltaWidth in (-1, 0, 1, 20):
#pybind11#                    fullWidth = kWidth + deltaWidth
#pybind11#                    for deltaHeight in (-1, 0, 1, 20):
#pybind11#                        fullHeight = kHeight + deltaHeight
#pybind11#                        kernel = afwMath.DeltaFunctionKernel(kWidth, kHeight, afwGeom.Point2I(0, 0))
#pybind11#                        fullBBox = afwGeom.Box2I(boxStart, afwGeom.Extent2I(fullWidth, fullHeight))
#pybind11#                        if (fullWidth < kWidth) or (fullHeight < kHeight):
#pybind11#                            self.assertRaises(Exception, kernel.shrinkBBox, fullBBox)
#pybind11#                            continue
#pybind11#
#pybind11#                        shrunkBBox = kernel.shrinkBBox(fullBBox)
#pybind11#                        self.assertEqual(shrunkBBox.getWidth(), fullWidth + 1 - kWidth)
#pybind11#                        self.assertEqual(shrunkBBox.getHeight(), fullHeight + 1 - kHeight)
#pybind11#                        self.assertEqual(shrunkBBox.getMinX(), boxStart[0] + kernel.getCtrX())
#pybind11#                        self.assertEqual(shrunkBBox.getMinY(), boxStart[1] + kernel.getCtrY())
#pybind11#                        newFullBBox = kernel.growBBox(shrunkBBox)
#pybind11#                        self.assertEqual(newFullBBox, fullBBox, "growBBox(shrinkBBox(x)) != x")
#pybind11#
#pybind11#    def testDeltaFunctionKernel(self):
#pybind11#        """Test DeltaFunctionKernel
#pybind11#        """
#pybind11#        for kWidth in range(1, 4):
#pybind11#            for kHeight in range(1, 4):
#pybind11#                for activeCol in range(kWidth):
#pybind11#                    for activeRow in range(kHeight):
#pybind11#                        kernel = afwMath.DeltaFunctionKernel(kWidth, kHeight,
#pybind11#                                                             afwGeom.Point2I(activeCol, activeRow))
#pybind11#                        kImage = afwImage.ImageD(kernel.getDimensions())
#pybind11#                        kSum = kernel.computeImage(kImage, False)
#pybind11#                        self.assertEqual(kSum, 1.0)
#pybind11#                        kArr = kImage.getArray().transpose()
#pybind11#                        self.assertEqual(kArr[activeCol, activeRow], 1.0)
#pybind11#                        kArr[activeCol, activeRow] = 0.0
#pybind11#                        self.assertEqual(kArr.sum(), 0.0)
#pybind11#
#pybind11#                        errStr = self.compareKernels(kernel, kernel.clone())
#pybind11#                        if errStr:
#pybind11#                            self.fail(errStr)
#pybind11#
#pybind11#                self.assertRaises(pexExcept.InvalidParameterError,
#pybind11#                                  afwMath.DeltaFunctionKernel, 0, kHeight, afwGeom.Point2I(kWidth, kHeight))
#pybind11#                self.assertRaises(pexExcept.InvalidParameterError,
#pybind11#                                  afwMath.DeltaFunctionKernel, kWidth, 0, afwGeom.Point2I(kWidth, kHeight))
#pybind11#
#pybind11#        kernel = afwMath.DeltaFunctionKernel(5, 6, afwGeom.Point2I(1, 1))
#pybind11#        self.basicTests(kernel, 0)
#pybind11#
#pybind11#        self.verifyCache(kernel, hasCache=False)
#pybind11#
#pybind11#    def testFixedKernel(self):
#pybind11#        """Test FixedKernel using a ramp function
#pybind11#        """
#pybind11#        kWidth = 5
#pybind11#        kHeight = 6
#pybind11#
#pybind11#        inArr = numpy.arange(kWidth * kHeight, dtype=float)
#pybind11#        inArr.shape = [kWidth, kHeight]
#pybind11#
#pybind11#        inImage = afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight))
#pybind11#        for row in range(inImage.getHeight()):
#pybind11#            for col in range(inImage.getWidth()):
#pybind11#                inImage.set(col, row, inArr[col, row])
#pybind11#
#pybind11#        kernel = afwMath.FixedKernel(inImage)
#pybind11#        self.basicTests(kernel, 0)
#pybind11#        outImage = afwImage.ImageD(kernel.getDimensions())
#pybind11#        kernel.computeImage(outImage, False)
#pybind11#
#pybind11#        outArr = outImage.getArray().transpose()
#pybind11#        if not numpy.allclose(inArr, outArr):
#pybind11#            self.fail("%s = %s != %s (not normalized)" %
#pybind11#                      (kernel.__class__.__name__, inArr, outArr))
#pybind11#
#pybind11#        normInArr = inArr / inArr.sum()
#pybind11#        normOutImage = afwImage.ImageD(kernel.getDimensions())
#pybind11#        kernel.computeImage(normOutImage, True)
#pybind11#        normOutArr = normOutImage.getArray().transpose()
#pybind11#        if not numpy.allclose(normOutArr, normInArr):
#pybind11#            self.fail("%s = %s != %s (normalized)" %
#pybind11#                      (kernel.__class__.__name__, normInArr, normOutArr))
#pybind11#
#pybind11#        errStr = self.compareKernels(kernel, kernel.clone())
#pybind11#        if errStr:
#pybind11#            self.fail(errStr)
#pybind11#
#pybind11#        self.verifyCache(kernel, hasCache=False)
#pybind11#
#pybind11#    def testLinearCombinationKernelDelta(self):
#pybind11#        """Test LinearCombinationKernel using a set of delta basis functions
#pybind11#        """
#pybind11#        kWidth = 3
#pybind11#        kHeight = 2
#pybind11#
#pybind11#        # create list of kernels
#pybind11#        basisKernelList = makeDeltaFunctionKernelList(kWidth, kHeight)
#pybind11#        basisImArrList = []
#pybind11#        for basisKernel in basisKernelList:
#pybind11#            basisImage = afwImage.ImageD(basisKernel.getDimensions())
#pybind11#            basisKernel.computeImage(basisImage, True)
#pybind11#            basisImArrList.append(basisImage.getArray())
#pybind11#
#pybind11#        kParams = [0.0]*len(basisKernelList)
#pybind11#        kernel = afwMath.LinearCombinationKernel(basisKernelList, kParams)
#pybind11#        self.assertTrue(kernel.isDeltaFunctionBasis())
#pybind11#        self.basicTests(kernel, len(kParams))
#pybind11#        for ii in range(len(basisKernelList)):
#pybind11#            kParams = [0.0]*len(basisKernelList)
#pybind11#            kParams[ii] = 1.0
#pybind11#            kernel.setKernelParameters(kParams)
#pybind11#            kIm = afwImage.ImageD(kernel.getDimensions())
#pybind11#            kernel.computeImage(kIm, True)
#pybind11#            kImArr = kIm.getArray()
#pybind11#            if not numpy.allclose(kImArr, basisImArrList[ii]):
#pybind11#                self.fail("%s = %s != %s for the %s'th basis kernel" %
#pybind11#                          (kernel.__class__.__name__, kImArr, basisImArrList[ii], ii))
#pybind11#
#pybind11#        kernelClone = kernel.clone()
#pybind11#        errStr = self.compareKernels(kernel, kernelClone)
#pybind11#        if errStr:
#pybind11#            self.fail(errStr)
#pybind11#
#pybind11#        self.verifyCache(kernel, hasCache=False)
#pybind11#
#pybind11#    def testComputeImageRaise(self):
#pybind11#        """Test Kernel.computeImage raises OverflowException iff doNormalize True and kernel sum exactly 0
#pybind11#        """
#pybind11#        kWidth = 4
#pybind11#        kHeight = 3
#pybind11#
#pybind11#        polyFunc1 = afwMath.PolynomialFunction1D(0)
#pybind11#        polyFunc2 = afwMath.PolynomialFunction2D(0)
#pybind11#        analKernel = afwMath.AnalyticKernel(kWidth, kHeight, polyFunc2)
#pybind11#        kImage = afwImage.ImageD(analKernel.getDimensions())
#pybind11#        for coeff in (-1, -1e-99, 0, 1e99, 1):
#pybind11#            analKernel.setKernelParameters([coeff])
#pybind11#
#pybind11#            analKernel.computeImage(kImage, False)
#pybind11#            fixedKernel = afwMath.FixedKernel(kImage)
#pybind11#
#pybind11#            sepKernel = afwMath.SeparableKernel(kWidth, kHeight, polyFunc1, polyFunc1)
#pybind11#            sepKernel.setKernelParameters([coeff, coeff])
#pybind11#
#pybind11#            kernelList = afwMath.KernelList()
#pybind11#            kernelList.append(analKernel)
#pybind11#            lcKernel = afwMath.LinearCombinationKernel(kernelList, [1])
#pybind11#            self.assertFalse(lcKernel.isDeltaFunctionBasis())
#pybind11#
#pybind11#            doRaise = (coeff == 0)
#pybind11#            self.basicTestComputeImageRaise(analKernel, doRaise, "AnalyticKernel")
#pybind11#            self.basicTestComputeImageRaise(fixedKernel, doRaise, "FixedKernel")
#pybind11#            self.basicTestComputeImageRaise(sepKernel, doRaise, "SeparableKernel")
#pybind11#            self.basicTestComputeImageRaise(lcKernel, doRaise, "LinearCombinationKernel")
#pybind11#
#pybind11#        lcKernel.setKernelParameters([0])
#pybind11#        self.basicTestComputeImageRaise(lcKernel, True, "LinearCombinationKernel")
#pybind11#
#pybind11#    def testLinearCombinationKernelAnalytic(self):
#pybind11#        """Test LinearCombinationKernel using analytic basis kernels.
#pybind11#
#pybind11#        The basis kernels are mutable so that we can verify that the
#pybind11#        LinearCombinationKernel has private copies of the basis kernels.
#pybind11#        """
#pybind11#        kWidth = 5
#pybind11#        kHeight = 8
#pybind11#
#pybind11#        # create list of kernels
#pybind11#        basisImArrList = []
#pybind11#        basisKernelList = afwMath.KernelList()
#pybind11#        for basisKernelParams in [(1.2, 0.3, 1.570796), (1.0, 0.2, 0.0)]:
#pybind11#            basisKernelFunction = afwMath.GaussianFunction2D(*basisKernelParams)
#pybind11#            basisKernel = afwMath.AnalyticKernel(kWidth, kHeight, basisKernelFunction)
#pybind11#            basisImage = afwImage.ImageD(basisKernel.getDimensions())
#pybind11#            basisKernel.computeImage(basisImage, True)
#pybind11#            basisImArrList.append(basisImage.getArray())
#pybind11#            basisKernelList.append(basisKernel)
#pybind11#
#pybind11#        kParams = [0.0]*len(basisKernelList)
#pybind11#        kernel = afwMath.LinearCombinationKernel(basisKernelList, kParams)
#pybind11#        self.assertTrue(not kernel.isDeltaFunctionBasis())
#pybind11#        self.basicTests(kernel, len(kParams))
#pybind11#
#pybind11#        # make sure the linear combination kernel has private copies of its basis kernels
#pybind11#        # by altering the local basis kernels and making sure the new images do NOT match
#pybind11#        modBasisImArrList = []
#pybind11#        for basisKernel in basisKernelList:
#pybind11#            basisKernel.setKernelParameters((0.4, 0.5, 0.6))
#pybind11#            modBasisImage = afwImage.ImageD(basisKernel.getDimensions())
#pybind11#            basisKernel.computeImage(modBasisImage, True)
#pybind11#            modBasisImArrList.append(modBasisImage.getArray())
#pybind11#
#pybind11#        for ii in range(len(basisKernelList)):
#pybind11#            kParams = [0.0]*len(basisKernelList)
#pybind11#            kParams[ii] = 1.0
#pybind11#            kernel.setKernelParameters(kParams)
#pybind11#            kIm = afwImage.ImageD(kernel.getDimensions())
#pybind11#            kernel.computeImage(kIm, True)
#pybind11#            kImArr = kIm.getArray()
#pybind11#            if not numpy.allclose(kImArr, basisImArrList[ii]):
#pybind11#                self.fail("%s = %s != %s for the %s'th basis kernel" %
#pybind11#                          (kernel.__class__.__name__, kImArr, basisImArrList[ii], ii))
#pybind11#            if numpy.allclose(kImArr, modBasisImArrList[ii]):
#pybind11#                self.fail("%s = %s == %s for *modified* %s'th basis kernel" %
#pybind11#                          (kernel.__class__.__name__, kImArr, modBasisImArrList[ii], ii))
#pybind11#
#pybind11#        kernelClone = kernel.clone()
#pybind11#        errStr = self.compareKernels(kernel, kernelClone)
#pybind11#        if errStr:
#pybind11#            self.fail(errStr)
#pybind11#
#pybind11#        self.verifyCache(kernel, hasCache=False)
#pybind11#
#pybind11#    def testSeparableKernel(self):
#pybind11#        """Test SeparableKernel using a Gaussian function
#pybind11#        """
#pybind11#        kWidth = 5
#pybind11#        kHeight = 8
#pybind11#
#pybind11#        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
#pybind11#        kernel = afwMath.SeparableKernel(kWidth, kHeight, gaussFunc1, gaussFunc1)
#pybind11#        self.basicTests(kernel, 2)
#pybind11#        fArr = numpy.zeros(shape=[kernel.getWidth(), kernel.getHeight()], dtype=float)
#pybind11#        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        for xsigma in (0.1, 1.0, 3.0):
#pybind11#            gaussFunc1.setParameters((xsigma,))
#pybind11#            for ysigma in (0.1, 1.0, 3.0):
#pybind11#                gaussFunc.setParameters((xsigma, ysigma, 0.0))
#pybind11#                # compute array of function values and normalize
#pybind11#                for row in range(kernel.getHeight()):
#pybind11#                    y = row - kernel.getCtrY()
#pybind11#                    for col in range(kernel.getWidth()):
#pybind11#                        x = col - kernel.getCtrX()
#pybind11#                        fArr[col, row] = gaussFunc(x, y)
#pybind11#                fArr /= fArr.sum()
#pybind11#
#pybind11#                kernel.setKernelParameters((xsigma, ysigma))
#pybind11#                kImage = afwImage.ImageD(kernel.getDimensions())
#pybind11#                kernel.computeImage(kImage, True)
#pybind11#                kArr = kImage.getArray().transpose()
#pybind11#                if not numpy.allclose(fArr, kArr):
#pybind11#                    self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" %
#pybind11#                              (kernel.__class__.__name__, kArr, fArr, xsigma, ysigma))
#pybind11#        kernelClone = kernel.clone()
#pybind11#        errStr = self.compareKernels(kernel, kernelClone)
#pybind11#        if errStr:
#pybind11#            self.fail(errStr)
#pybind11#
#pybind11#        kernel.setKernelParameters((1.2, 0.6))
#pybind11#        errStr = self.compareKernels(kernel, kernelClone)
#pybind11#        if not errStr:
#pybind11#            self.fail("Clone was modified by changing original's kernel parameters")
#pybind11#
#pybind11#        self.verifyCache(kernel, hasCache=True)
#pybind11#
#pybind11#    def testMakeBadKernels(self):
#pybind11#        """Attempt to make various invalid kernels; make sure the constructor shows an exception
#pybind11#        """
#pybind11#        kWidth = 4
#pybind11#        kHeight = 3
#pybind11#
#pybind11#        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
#pybind11#        gaussFunc2 = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        spFunc = afwMath.PolynomialFunction2D(1)
#pybind11#        kernelList = afwMath.KernelList()
#pybind11#        kernelList.append(afwMath.FixedKernel(afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight), 0.1)))
#pybind11#        kernelList.append(afwMath.FixedKernel(afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight), 0.2)))
#pybind11#
#pybind11#        for numKernelParams in (2, 4):
#pybind11#            spFuncList = afwMath.Function2DList()
#pybind11#            for ii in range(numKernelParams):
#pybind11#                spFuncList.append(spFunc.clone())
#pybind11#            try:
#pybind11#                afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc2, spFuncList)
#pybind11#                self.fail("Should have failed with wrong # of spatial functions")
#pybind11#            except pexExcept.Exception:
#pybind11#                pass
#pybind11#
#pybind11#        for numKernelParams in (1, 3):
#pybind11#            spFuncList = afwMath.Function2DList()
#pybind11#            for ii in range(numKernelParams):
#pybind11#                spFuncList.append(spFunc.clone())
#pybind11#            try:
#pybind11#                afwMath.LinearCombinationKernel(kernelList, spFuncList)
#pybind11#                self.fail("Should have failed with wrong # of spatial functions")
#pybind11#            except pexExcept.Exception:
#pybind11#                pass
#pybind11#            kParamList = [0.2]*numKernelParams
#pybind11#            try:
#pybind11#                afwMath.LinearCombinationKernel(kernelList, kParamList)
#pybind11#                self.fail("Should have failed with wrong # of kernel parameters")
#pybind11#            except pexExcept.Exception:
#pybind11#                pass
#pybind11#            try:
#pybind11#                afwMath.SeparableKernel(kWidth, kHeight, gaussFunc1, gaussFunc1, spFuncList)
#pybind11#                self.fail("Should have failed with wrong # of spatial functions")
#pybind11#            except pexExcept.Exception:
#pybind11#                pass
#pybind11#
#pybind11#        for pointX in range(-1, kWidth+2):
#pybind11#            for pointY in range(-1, kHeight+2):
#pybind11#                if (0 <= pointX < kWidth) and (0 <= pointY < kHeight):
#pybind11#                    continue
#pybind11#                try:
#pybind11#                    afwMath.DeltaFunctionKernel(kWidth, kHeight, afwGeom.Point2I(pointX, pointY))
#pybind11#                    self.fail("Should have failed with point not on kernel")
#pybind11#                except pexExcept.Exception:
#pybind11#                    pass
#pybind11#
#pybind11#    def testSVAnalyticKernel(self):
#pybind11#        """Test spatially varying AnalyticKernel using a Gaussian function
#pybind11#
#pybind11#        Just tests cloning.
#pybind11#        """
#pybind11#        kWidth = 5
#pybind11#        kHeight = 8
#pybind11#
#pybind11#        # spatial model
#pybind11#        spFunc = afwMath.PolynomialFunction2D(1)
#pybind11#
#pybind11#        # spatial parameters are a list of entries, one per kernel parameter;
#pybind11#        # each entry is a list of spatial parameters
#pybind11#        sParams = (
#pybind11#            (1.0, 1.0, 0.0),
#pybind11#            (1.0, 0.0, 1.0),
#pybind11#            (0.5, 0.5, 0.5),
#pybind11#        )
#pybind11#
#pybind11#        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        kernel = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc, spFunc)
#pybind11#        kernel.setSpatialParameters(sParams)
#pybind11#
#pybind11#        kernelClone = kernel.clone()
#pybind11#        errStr = self.compareKernels(kernel, kernelClone)
#pybind11#        if errStr:
#pybind11#            self.fail(errStr)
#pybind11#
#pybind11#        newSParams = (
#pybind11#            (0.1, 0.2, 0.5),
#pybind11#            (0.1, 0.5, 0.2),
#pybind11#            (0.2, 0.3, 0.3),
#pybind11#        )
#pybind11#        kernel.setSpatialParameters(newSParams)
#pybind11#        errStr = self.compareKernels(kernel, kernelClone)
#pybind11#        if not errStr:
#pybind11#            self.fail("Clone was modified by changing original's spatial parameters")
#pybind11#
#pybind11#        #
#pybind11#        # check that we can construct a FixedKernel from a LinearCombinationKernel
#pybind11#        #
#pybind11#        x, y = 100, 200
#pybind11#        kernel2 = afwMath.FixedKernel(kernel, afwGeom.PointD(x, y))
#pybind11#
#pybind11#        self.assertTrue(re.search("AnalyticKernel", kernel.toString()))
#pybind11#        self.assertFalse(kernel2.isSpatiallyVarying())
#pybind11#
#pybind11#        self.assertTrue(re.search("FixedKernel", kernel2.toString()))
#pybind11#        self.assertTrue(kernel.isSpatiallyVarying())
#pybind11#
#pybind11#        kim = afwImage.ImageD(kernel.getDimensions())
#pybind11#        kernel.computeImage(kim, True, x, y)
#pybind11#
#pybind11#        kim2 = afwImage.ImageD(kernel2.getDimensions())
#pybind11#        kernel2.computeImage(kim2, True)
#pybind11#
#pybind11#        self.assertTrue(numpy.allclose(kim.getArray(), kim2.getArray()))
#pybind11#
#pybind11#    def testSVLinearCombinationKernelFixed(self):
#pybind11#        """Test a spatially varying LinearCombinationKernel whose bases are FixedKernels"""
#pybind11#        kWidth = 3
#pybind11#        kHeight = 2
#pybind11#
#pybind11#        # create image arrays for the basis kernels
#pybind11#        basisImArrList = []
#pybind11#        imArr = numpy.zeros((kWidth, kHeight), dtype=float)
#pybind11#        imArr += 0.1
#pybind11#        imArr[kWidth//2, :] = 0.9
#pybind11#        basisImArrList.append(imArr)
#pybind11#        imArr = numpy.zeros((kWidth, kHeight), dtype=float)
#pybind11#        imArr += 0.2
#pybind11#        imArr[:, kHeight//2] = 0.8
#pybind11#        basisImArrList.append(imArr)
#pybind11#
#pybind11#        # create a list of basis kernels from the images
#pybind11#        basisKernelList = afwMath.KernelList()
#pybind11#        for basisImArr in basisImArrList:
#pybind11#            basisImage = afwImage.makeImageFromArray(basisImArr.transpose().copy())
#pybind11#            kernel = afwMath.FixedKernel(basisImage)
#pybind11#            basisKernelList.append(kernel)
#pybind11#
#pybind11#        # create spatially varying linear combination kernel
#pybind11#        spFunc = afwMath.PolynomialFunction2D(1)
#pybind11#
#pybind11#        # spatial parameters are a list of entries, one per kernel parameter;
#pybind11#        # each entry is a list of spatial parameters
#pybind11#        sParams = (
#pybind11#            (0.0, 1.0, 0.0),
#pybind11#            (0.0, 0.0, 1.0),
#pybind11#        )
#pybind11#
#pybind11#        kernel = afwMath.LinearCombinationKernel(basisKernelList, spFunc)
#pybind11#        self.assertFalse(kernel.isDeltaFunctionBasis())
#pybind11#        self.basicTests(kernel, 2, nSpatialParams=3)
#pybind11#        kernel.setSpatialParameters(sParams)
#pybind11#        kImage = afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight))
#pybind11#        for colPos, rowPos, coeff0, coeff1 in [
#pybind11#            (0.0, 0.0, 0.0, 0.0),
#pybind11#            (1.0, 0.0, 1.0, 0.0),
#pybind11#            (0.0, 1.0, 0.0, 1.0),
#pybind11#            (1.0, 1.0, 1.0, 1.0),
#pybind11#            (0.5, 0.5, 0.5, 0.5),
#pybind11#        ]:
#pybind11#            kernel.computeImage(kImage, False, colPos, rowPos)
#pybind11#            kImArr = kImage.getArray().transpose()
#pybind11#            refKImArr = (basisImArrList[0] * coeff0) + (basisImArrList[1] * coeff1)
#pybind11#            if not numpy.allclose(kImArr, refKImArr):
#pybind11#                self.fail("%s = %s != %s at colPos=%s, rowPos=%s" %
#pybind11#                          (kernel.__class__.__name__, kImArr, refKImArr, colPos, rowPos))
#pybind11#
#pybind11#        sParams = (
#pybind11#            (0.1, 1.0, 0.0),
#pybind11#            (0.1, 0.0, 1.0),
#pybind11#        )
#pybind11#        kernel.setSpatialParameters(sParams)
#pybind11#        kernelClone = kernel.clone()
#pybind11#        errStr = self.compareKernels(kernel, kernelClone)
#pybind11#        if errStr:
#pybind11#            self.fail(errStr)
#pybind11#
#pybind11#        newSParams = (
#pybind11#            (0.1, 0.2, 0.5),
#pybind11#            (0.1, 0.5, 0.2),
#pybind11#        )
#pybind11#        kernel.setSpatialParameters(newSParams)
#pybind11#        errStr = self.compareKernels(kernel, kernelClone)
#pybind11#        if not errStr:
#pybind11#            self.fail("Clone was modified by changing original's spatial parameters")
#pybind11#
#pybind11#    def testSVSeparableKernel(self):
#pybind11#        """Test spatially varying SeparableKernel using a Gaussian function
#pybind11#
#pybind11#        Just tests cloning.
#pybind11#        """
#pybind11#        kWidth = 5
#pybind11#        kHeight = 8
#pybind11#
#pybind11#        # spatial model
#pybind11#        spFunc = afwMath.PolynomialFunction2D(1)
#pybind11#
#pybind11#        # spatial parameters are a list of entries, one per kernel parameter;
#pybind11#        # each entry is a list of spatial parameters
#pybind11#        sParams = (
#pybind11#            (1.0, 1.0, 0.0),
#pybind11#            (1.0, 0.0, 1.0),
#pybind11#        )
#pybind11#
#pybind11#        gaussFunc = afwMath.GaussianFunction1D(1.0)
#pybind11#        kernel = afwMath.SeparableKernel(kWidth, kHeight, gaussFunc, gaussFunc, spFunc)
#pybind11#        kernel.setSpatialParameters(sParams)
#pybind11#
#pybind11#        kernelClone = kernel.clone()
#pybind11#        errStr = self.compareKernels(kernel, kernelClone)
#pybind11#        if errStr:
#pybind11#            self.fail(errStr)
#pybind11#
#pybind11#        newSParams = (
#pybind11#            (0.1, 0.2, 0.5),
#pybind11#            (0.1, 0.5, 0.2),
#pybind11#        )
#pybind11#        kernel.setSpatialParameters(newSParams)
#pybind11#        errStr = self.compareKernels(kernel, kernelClone)
#pybind11#        if not errStr:
#pybind11#            self.fail("Clone was modified by changing original's spatial parameters")
#pybind11#
#pybind11#    def testSetCtr(self):
#pybind11#        """Test setCtrCol/Row"""
#pybind11#        kWidth = 3
#pybind11#        kHeight = 4
#pybind11#
#pybind11#        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        kernel = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc)
#pybind11#        for xCtr in range(kWidth):
#pybind11#            kernel.setCtrX(xCtr)
#pybind11#            for yCtr in range(kHeight):
#pybind11#                kernel.setCtrY(yCtr)
#pybind11#                self.assertEqual(kernel.getCtrX(), xCtr)
#pybind11#                self.assertEqual(kernel.getCtrY(), yCtr)
#pybind11#
#pybind11#    def testZeroSizeKernel(self):
#pybind11#        """Creating a kernel with width or height < 1 should raise an exception.
#pybind11#
#pybind11#        Note: this ignores the default constructors, which produce kernels with height = width = 0.
#pybind11#        The default constructors are only intended to support persistence, not to produce useful kernels.
#pybind11#        """
#pybind11#        gaussFunc2D = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        gaussFunc1D = afwMath.GaussianFunction1D(1.0)
#pybind11#        zeroPoint = afwGeom.Point2I(0, 0)
#pybind11#        for kWidth in (-1, 0, 1):
#pybind11#            for kHeight in (-1, 0, 1):
#pybind11#                if (kHeight > 0) and (kWidth > 0):
#pybind11#                    continue
#pybind11#                if (kHeight >= 0) and (kWidth >= 0):
#pybind11#                    # don't try to create an image with negative dimensions
#pybind11#                    blankImage = afwImage.ImageF(afwGeom.Extent2I(kWidth, kHeight))
#pybind11#                    self.assertRaises(Exception, afwMath.FixedKernel, blankImage)
#pybind11#                self.assertRaises(Exception, afwMath.AnalyticKernel, kWidth, kHeight, gaussFunc2D)
#pybind11#                self.assertRaises(Exception, afwMath.SeparableKernel,
#pybind11#                                  kWidth, kHeight, gaussFunc1D, gaussFunc1D)
#pybind11#                self.assertRaises(Exception, afwMath.DeltaFunctionKernel, kWidth, kHeight, zeroPoint)
#pybind11#
#pybind11#    def testRefactorDeltaLinearCombinationKernel(self):
#pybind11#        """Test LinearCombinationKernel.refactor with delta function basis kernels
#pybind11#        """
#pybind11#        kWidth = 4
#pybind11#        kHeight = 3
#pybind11#
#pybind11#        for spOrder in (0, 1, 2):
#pybind11#            spFunc = afwMath.PolynomialFunction2D(spOrder)
#pybind11#            numSpParams = spFunc.getNParameters()
#pybind11#
#pybind11#            basisKernelList = makeDeltaFunctionKernelList(kWidth, kHeight)
#pybind11#            kernel = afwMath.LinearCombinationKernel(basisKernelList, spFunc)
#pybind11#
#pybind11#            numBasisKernels = kernel.getNKernelParameters()
#pybind11#            maxVal = 1.01 + ((numSpParams - 1) * 0.1)
#pybind11#            sParamList = [numpy.arange(kInd + 1.0, kInd + maxVal, 0.1) for kInd in range(numBasisKernels)]
#pybind11#            kernel.setSpatialParameters(sParamList)
#pybind11#
#pybind11#            refKernel = kernel.refactor()
#pybind11#            self.assertTrue(refKernel)
#pybind11#            errStr = self.compareKernels(kernel, refKernel, compareParams=False)
#pybind11#            if errStr:
#pybind11#                self.fail("failed with %s for spOrder=%s (numSpCoeff=%s)" % (errStr, spOrder, numSpParams))
#pybind11#
#pybind11#    def testRefactorGaussianLinearCombinationKernel(self):
#pybind11#        """Test LinearCombinationKernel.refactor with Gaussian basis kernels
#pybind11#        """
#pybind11#        kWidth = 4
#pybind11#        kHeight = 3
#pybind11#
#pybind11#        for spOrder in (0, 1, 2):
#pybind11#            spFunc = afwMath.PolynomialFunction2D(spOrder)
#pybind11#            numSpParams = spFunc.getNParameters()
#pybind11#
#pybind11#            gaussParamsList = [
#pybind11#                (1.5, 1.5, 0.0),
#pybind11#                (2.5, 1.5, 0.0),
#pybind11#                (2.5, 1.5, math.pi / 2.0),
#pybind11#            ]
#pybind11#            gaussBasisKernelList = makeGaussianKernelList(kWidth, kHeight, gaussParamsList)
#pybind11#            kernel = afwMath.LinearCombinationKernel(gaussBasisKernelList, spFunc)
#pybind11#
#pybind11#            numBasisKernels = kernel.getNKernelParameters()
#pybind11#            maxVal = 1.01 + ((numSpParams - 1) * 0.1)
#pybind11#            sParamList = [numpy.arange(kInd + 1.0, kInd + maxVal, 0.1) for kInd in range(numBasisKernels)]
#pybind11#            kernel.setSpatialParameters(sParamList)
#pybind11#
#pybind11#            refKernel = kernel.refactor()
#pybind11#            self.assertTrue(refKernel)
#pybind11#            errStr = self.compareKernels(kernel, refKernel, compareParams=False)
#pybind11#            if errStr:
#pybind11#                self.fail("failed with %s for spOrder=%s; numSpCoeff=%s" % (errStr, spOrder, numSpParams))
#pybind11#
#pybind11#    def basicTests(self, kernel, nKernelParams, nSpatialParams=0, dimMustMatch=True):
#pybind11#        """Basic tests of a kernel"""
#pybind11#        self.assertEqual(kernel.getNSpatialParameters(), nSpatialParams)
#pybind11#        self.assertEqual(kernel.getNKernelParameters(), nKernelParams)
#pybind11#        if nSpatialParams == 0:
#pybind11#            self.assertFalse(kernel.isSpatiallyVarying())
#pybind11#            for ii in range(nKernelParams+5):
#pybind11#                self.assertRaises(pexExcept.InvalidParameterError,
#pybind11#                                  kernel.getSpatialFunction, ii)
#pybind11#        else:
#pybind11#            self.assertTrue(kernel.isSpatiallyVarying())
#pybind11#            for ii in range(nKernelParams):
#pybind11#                kernel.getSpatialFunction(ii)
#pybind11#            for ii in range(nKernelParams, nKernelParams+5):
#pybind11#                self.assertRaises(pexExcept.InvalidParameterError,
#pybind11#                                  kernel.getSpatialFunction, ii)
#pybind11#
#pybind11#        # test a range of numbers of parameters, including both valid and invalid sized tuples.
#pybind11#        for nsp in range(nSpatialParams + 2):
#pybind11#            spatialParamsForOneKernel = (1.0,)*nsp
#pybind11#            for nkp in range(nKernelParams + 2):
#pybind11#                spatialParams = (spatialParamsForOneKernel,)*nkp
#pybind11#                if ((nkp == nKernelParams) and ((nsp == nSpatialParams) or (nkp == 0))):
#pybind11#                    kernel.setSpatialParameters(spatialParams)
#pybind11#                    if nsp == 0:
#pybind11#                        # A non-spatially varying kernel returns an empty tuple, even though
#pybind11#                        # it can only be set with a tuple of empty tuples, one per kernel parameter.
#pybind11#                        self.assertEqual(kernel.getSpatialParameters(), ())
#pybind11#                    else:
#pybind11#                        # a spatially varying kernel should return exactly what we set it to be.
#pybind11#                        self.assertEqual(kernel.getSpatialParameters(), spatialParams)
#pybind11#                else:
#pybind11#                    with self.assertRaises(pexExcept.InvalidParameterError):
#pybind11#                        kernel.setSpatialParameters(spatialParams)
#pybind11#
#pybind11#        kernelDim = kernel.getDimensions()
#pybind11#        kernelCtr = kernel.getCtr()
#pybind11#        for dx in (-1, 0, 1):
#pybind11#            xDim = kernelDim.getX() + dx
#pybind11#            for dy in (-1, 0, 1):
#pybind11#                if dx == dy == 0:
#pybind11#                    continue
#pybind11#                yDim = kernelDim.getY() + dy
#pybind11#                image = afwImage.ImageD(xDim, yDim)
#pybind11#                if (dx == dy == 0) or not dimMustMatch:
#pybind11#                    ksum = kernel.computeImage(image, True)
#pybind11#                    self.assertAlmostEqual(ksum, 1.0)
#pybind11#                    llBorder = ((image.getDimensions() - kernelDim) / 2).truncate()
#pybind11#                    predCtr = afwGeom.Point2I(llBorder + kernelCtr)
#pybind11#                    self.assertEqual(kernel.getCtr(), predCtr)
#pybind11#                else:
#pybind11#                    self.assertRaises(Exception, kernel.computeImage, image, True)
#pybind11#
#pybind11#    def basicTestComputeImageRaise(self, kernel, doRaise, kernelDescr=""):
#pybind11#        """Test that computeImage either does or does not raise an exception, as appropriate
#pybind11#        """
#pybind11#        kImage = afwImage.ImageD(kernel.getDimensions())
#pybind11#        try:
#pybind11#            kernel.computeImage(kImage, True)
#pybind11#            if doRaise:
#pybind11#                self.fail(kernelDescr + ".computeImage should have raised an exception")
#pybind11#        except pexExcept.Exception:
#pybind11#            if not doRaise:
#pybind11#                self.fail(kernelDescr + ".computeImage should not have raised an exception")
#pybind11#
#pybind11#    def compareKernels(self, kernel1, kernel2, compareParams=True, newCtr1=(0, 0)):
#pybind11#        """Compare two kernels; return None if they match, else return a string kernelDescribing a difference.
#pybind11#
#pybind11#        kernel1: one kernel to test
#pybind11#        kernel2: the other kernel to test
#pybind11#        compareParams: compare spatial parameters and kernel parameters if they exist
#pybind11#        newCtr: if not None then set the center of kernel1 and see if it changes the center of kernel2
#pybind11#        """
#pybind11#        retStrs = []
#pybind11#        if kernel1.getDimensions() != kernel2.getDimensions():
#pybind11#            retStrs.append("dimensions differ: %s != %s" % (kernel1.getDimensions(), kernel2.getDimensions()))
#pybind11#        ctr1 = kernel1.getCtrX(), kernel1.getCtrY()
#pybind11#        ctr2 = kernel2.getCtrX(), kernel2.getCtrY()
#pybind11#        if ctr1 != ctr2:
#pybind11#            retStrs.append("centers differ: %s != %s" % (ctr1, ctr2))
#pybind11#        if kernel1.isSpatiallyVarying() != kernel2.isSpatiallyVarying():
#pybind11#            retStrs.append("isSpatiallyVarying differs: %s != %s" %
#pybind11#                           (kernel1.isSpatiallyVarying(), kernel2.isSpatiallyVarying()))
#pybind11#
#pybind11#        if compareParams:
#pybind11#            if kernel1.getSpatialParameters() != kernel2.getSpatialParameters():
#pybind11#                retStrs.append("spatial parameters differ: %s != %s" %
#pybind11#                               (kernel1.getSpatialParameters(), kernel2.getSpatialParameters()))
#pybind11#            if kernel1.getNSpatialParameters() != kernel2.getNSpatialParameters():
#pybind11#                retStrs.append("# spatial parameters differs: %s != %s" %
#pybind11#                               (kernel1.getNSpatialParameters(), kernel2.getNSpatialParameters()))
#pybind11#            if not kernel1.isSpatiallyVarying() and hasattr(kernel1, "getKernelParameters"):
#pybind11#                if kernel1.getKernelParameters() != kernel2.getKernelParameters():
#pybind11#                    retStrs.append("kernel parameters differs: %s != %s" %
#pybind11#                                   (kernel1.getKernelParameters(), kernel2.getKernelParameters()))
#pybind11#        if retStrs:
#pybind11#            return "; ".join(retStrs)
#pybind11#
#pybind11#        im1 = afwImage.ImageD(kernel1.getDimensions())
#pybind11#        im2 = afwImage.ImageD(kernel2.getDimensions())
#pybind11#        if kernel1.isSpatiallyVarying():
#pybind11#            posList = [(0, 0), (200, 0), (0, 200), (200, 200)]
#pybind11#        else:
#pybind11#            posList = [(0, 0)]
#pybind11#
#pybind11#        for doNormalize in (False, True):
#pybind11#            for pos in posList:
#pybind11#                kernel1.computeImage(im1, doNormalize, pos[0], pos[1])
#pybind11#                kernel2.computeImage(im2, doNormalize, pos[0], pos[1])
#pybind11#                im1Arr = im1.getArray()
#pybind11#                im2Arr = im2.getArray()
#pybind11#                if not numpy.allclose(im1Arr, im2Arr):
#pybind11#                    print("im1Arr =", im1Arr)
#pybind11#                    print("im2Arr =", im2Arr)
#pybind11#                    return "kernel images do not match at %s with doNormalize=%s" % (pos, doNormalize)
#pybind11#
#pybind11#        if newCtr1 is not None:
#pybind11#            kernel1.setCtrX(newCtr1[0])
#pybind11#            kernel1.setCtrY(newCtr1[1])
#pybind11#            newCtr2 = kernel2.getCtrX(), kernel2.getCtrY()
#pybind11#            if ctr2 != newCtr2:
#pybind11#                return "changing center of kernel1 to %s changed the center of kernel2 from %s to %s" % \
#pybind11#                    (newCtr1, ctr2, newCtr2)
#pybind11#
#pybind11#    def testCast(self):
#pybind11#        instances = []
#pybind11#        kVec = makeGaussianKernelList(9, 9, [(2.0, 2.0, 0.0)])
#pybind11#        kParams = [0.0]*len(kVec)
#pybind11#        instances.append(afwMath.LinearCombinationKernel(kVec, kParams))
#pybind11#        instances.append(afwMath.AnalyticKernel(7, 7, afwMath.GaussianFunction2D(2.0, 2.0, 0.0)))
#pybind11#        instances.append(afwMath.DeltaFunctionKernel(5, 5, afwGeom.Point2I(1, 1)))
#pybind11#        instances.append(afwMath.FixedKernel(afwImage.ImageD(afwGeom.Extent2I(7, 7))))
#pybind11#        instances.append(afwMath.SeparableKernel(3, 3, afwMath.PolynomialFunction1D(0),
#pybind11#                                                 afwMath.PolynomialFunction1D(0)))
#pybind11#        for instance in instances:
#pybind11#            Class = type(instance)
#pybind11#            base = instance.clone()
#pybind11#            self.assertEqual(type(base), afwMath.Kernel)
#pybind11#            derived = Class.cast(base)
#pybind11#            self.assertEqual(type(derived), Class)
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

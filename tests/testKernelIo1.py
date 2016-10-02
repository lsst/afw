#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
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
#pybind11#
#pybind11#import unittest
#pybind11#
#pybind11#import numpy
#pybind11#import os
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.policy as pexPolicy
#pybind11#import lsst.daf.base as dafBase
#pybind11#import lsst.daf.persistence as dafPersist
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#from lsst.log import Log
#pybind11#
#pybind11## Change the level to Log.DEBUG to see debug messages
#pybind11#Log.getLogger("afw.math.KernelFormatter").setLevel(Log.INFO)
#pybind11#
#pybind11#
#pybind11#testPath = os.path.abspath(os.path.dirname(__file__))
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class KernelIOTestCase(unittest.TestCase):
#pybind11#    """A test case for Kernel I/O"""
#pybind11#
#pybind11#    def kernelCheck(self, k1, k2):
#pybind11#        self.assertEqual(k1.getWidth(), k2.getWidth())
#pybind11#        self.assertEqual(k1.getHeight(), k2.getHeight())
#pybind11#        self.assertEqual(k1.getCtrX(), k2.getCtrX())
#pybind11#        self.assertEqual(k1.getCtrY(), k2.getCtrY())
#pybind11#        self.assertEqual(k1.getNKernelParameters(), k2.getNKernelParameters())
#pybind11#        self.assertEqual(k1.getNSpatialParameters(), k2.getNSpatialParameters())
#pybind11#        self.assertEqual(k1.getKernelParameters(), k2.getKernelParameters())
#pybind11#        self.assertEqual(k1.getSpatialParameters(), k2.getSpatialParameters())
#pybind11#        self.assertEqual(k1.isSpatiallyVarying(), k2.isSpatiallyVarying())
#pybind11#        self.assertEqual(k1.toString(), k2.toString())
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
#pybind11#        k = afwMath.FixedKernel(inImage)
#pybind11#
#pybind11#        pol = pexPolicy.Policy()
#pybind11#        additionalData = dafBase.PropertySet()
#pybind11#        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data", "kernel1.boost"))
#pybind11#        persistence = dafPersist.Persistence.getPersistence(pol)
#pybind11#
#pybind11#        storageList = dafPersist.StorageList()
#pybind11#        storage = persistence.getPersistStorage("XmlStorage", loc)
#pybind11#        storageList.append(storage)
#pybind11#        persistence.persist(k, storageList, additionalData)
#pybind11#
#pybind11#        storageList2 = dafPersist.StorageList()
#pybind11#        storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
#pybind11#        storageList2.append(storage2)
#pybind11#        x = persistence.unsafeRetrieve("FixedKernel", storageList2, additionalData)
#pybind11#        k2 = afwMath.FixedKernel.swigConvert(x)
#pybind11#
#pybind11#        self.kernelCheck(k, k2)
#pybind11#
#pybind11#        outImage = afwImage.ImageD(k2.getDimensions())
#pybind11#        k2.computeImage(outImage, False)
#pybind11#
#pybind11#        outArr = outImage.getArray().transpose()
#pybind11#        if not numpy.allclose(inArr, outArr):
#pybind11#            self.fail("%s = %s != %s (not normalized)" %
#pybind11#                      (k2.__class__.__name__, inArr, outArr))
#pybind11#        normInArr = inArr / inArr.sum()
#pybind11#        normOutImage = afwImage.ImageD(k2.getDimensions())
#pybind11#        k2.computeImage(normOutImage, True)
#pybind11#        normOutArr = normOutImage.getArray().transpose()
#pybind11#        if not numpy.allclose(normOutArr, normInArr):
#pybind11#            self.fail("%s = %s != %s (normalized)" %
#pybind11#                      (k2.__class__.__name__, normInArr, normOutArr))
#pybind11#
#pybind11#    def testAnalyticKernel(self):
#pybind11#        """Test AnalyticKernel using a Gaussian function
#pybind11#        """
#pybind11#        kWidth = 5
#pybind11#        kHeight = 8
#pybind11#
#pybind11#        pol = pexPolicy.Policy()
#pybind11#        additionalData = dafBase.PropertySet()
#pybind11#        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data", "kernel2.boost"))
#pybind11#        persistence = dafPersist.Persistence.getPersistence(pol)
#pybind11#
#pybind11#        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        k = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc)
#pybind11#        fArr = numpy.zeros(shape=[k.getWidth(), k.getHeight()], dtype=float)
#pybind11#        for xsigma in (0.1, 1.0, 3.0):
#pybind11#            for ysigma in (0.1, 1.0, 3.0):
#pybind11#                for angle in (0.0, 0.4, 1.1):
#pybind11#                    gaussFunc.setParameters((xsigma, ysigma, angle))
#pybind11#                    # compute array of function values and normalize
#pybind11#                    for row in range(k.getHeight()):
#pybind11#                        y = row - k.getCtrY()
#pybind11#                        for col in range(k.getWidth()):
#pybind11#                            x = col - k.getCtrX()
#pybind11#                            fArr[col, row] = gaussFunc(x, y)
#pybind11#                    fArr /= fArr.sum()
#pybind11#
#pybind11#                    k.setKernelParameters((xsigma, ysigma, angle))
#pybind11#
#pybind11#                    storageList = dafPersist.StorageList()
#pybind11#                    storage = persistence.getPersistStorage("XmlStorage", loc)
#pybind11#                    storageList.append(storage)
#pybind11#                    persistence.persist(k, storageList, additionalData)
#pybind11#
#pybind11#                    storageList2 = dafPersist.StorageList()
#pybind11#                    storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
#pybind11#                    storageList2.append(storage2)
#pybind11#                    x = persistence.unsafeRetrieve("AnalyticKernel",
#pybind11#                                                   storageList2, additionalData)
#pybind11#                    k2 = afwMath.AnalyticKernel.swigConvert(x)
#pybind11#
#pybind11#                    self.kernelCheck(k, k2)
#pybind11#
#pybind11#                    kImage = afwImage.ImageD(k2.getDimensions())
#pybind11#                    k2.computeImage(kImage, True)
#pybind11#                    kArr = kImage.getArray().transpose()
#pybind11#                    if not numpy.allclose(fArr, kArr):
#pybind11#                        self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" %
#pybind11#                                  (k2.__class__.__name__, kArr, fArr, xsigma, ysigma))
#pybind11#
#pybind11#    def testDeltaFunctionKernel(self):
#pybind11#        """Test DeltaFunctionKernel
#pybind11#        """
#pybind11#        pol = pexPolicy.Policy()
#pybind11#        additionalData = dafBase.PropertySet()
#pybind11#        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data", "kernel3.boost"))
#pybind11#        persistence = dafPersist.Persistence.getPersistence(pol)
#pybind11#
#pybind11#        for kWidth in range(1, 4):
#pybind11#            for kHeight in range(1, 4):
#pybind11#                for activeCol in range(kWidth):
#pybind11#                    for activeRow in range(kHeight):
#pybind11#                        kernel = afwMath.DeltaFunctionKernel(kWidth, kHeight,
#pybind11#                                                             afwGeom.Point2I(activeCol, activeRow))
#pybind11#
#pybind11#                        storageList = dafPersist.StorageList()
#pybind11#                        storage = persistence.getPersistStorage("XmlStorage", loc)
#pybind11#                        storageList.append(storage)
#pybind11#                        persistence.persist(kernel, storageList, additionalData)
#pybind11#
#pybind11#                        storageList2 = dafPersist.StorageList()
#pybind11#                        storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
#pybind11#                        storageList2.append(storage2)
#pybind11#                        x = persistence.unsafeRetrieve("DeltaFunctionKernel",
#pybind11#                                                       storageList2, additionalData)
#pybind11#                        k2 = afwMath.DeltaFunctionKernel.swigConvert(x)
#pybind11#
#pybind11#                        self.kernelCheck(kernel, k2)
#pybind11#                        self.assertEqual(kernel.getPixel(), k2.getPixel())
#pybind11#
#pybind11#                        kImage = afwImage.ImageD(k2.getDimensions())
#pybind11#                        kSum = k2.computeImage(kImage, False)
#pybind11#                        self.assertEqual(kSum, 1.0)
#pybind11#                        kArr = kImage.getArray().transpose()
#pybind11#                        self.assertEqual(kArr[activeCol, activeRow], 1.0)
#pybind11#                        kArr[activeCol, activeRow] = 0.0
#pybind11#                        self.assertEqual(kArr.sum(), 0.0)
#pybind11#
#pybind11#    def testSeparableKernel(self):
#pybind11#        """Test SeparableKernel using a Gaussian function
#pybind11#        """
#pybind11#        kWidth = 5
#pybind11#        kHeight = 8
#pybind11#
#pybind11#        pol = pexPolicy.Policy()
#pybind11#        additionalData = dafBase.PropertySet()
#pybind11#        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data", "kernel4.boost"))
#pybind11#        persistence = dafPersist.Persistence.getPersistence(pol)
#pybind11#
#pybind11#        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
#pybind11#        k = afwMath.SeparableKernel(kWidth, kHeight, gaussFunc1, gaussFunc1)
#pybind11#        fArr = numpy.zeros(shape=[k.getWidth(), k.getHeight()], dtype=float)
#pybind11#        numpy.zeros(shape=[k.getWidth(), k.getHeight()], dtype=float)
#pybind11#        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        for xsigma in (0.1, 1.0, 3.0):
#pybind11#            gaussFunc1.setParameters((xsigma,))
#pybind11#            for ysigma in (0.1, 1.0, 3.0):
#pybind11#                gaussFunc.setParameters((xsigma, ysigma, 0.0))
#pybind11#                # compute array of function values and normalize
#pybind11#                for row in range(k.getHeight()):
#pybind11#                    y = row - k.getCtrY()
#pybind11#                    for col in range(k.getWidth()):
#pybind11#                        x = col - k.getCtrX()
#pybind11#                        fArr[col, row] = gaussFunc(x, y)
#pybind11#                fArr /= fArr.sum()
#pybind11#
#pybind11#                k.setKernelParameters((xsigma, ysigma))
#pybind11#
#pybind11#                storageList = dafPersist.StorageList()
#pybind11#                storage = persistence.getPersistStorage("XmlStorage", loc)
#pybind11#                storageList.append(storage)
#pybind11#                persistence.persist(k, storageList, additionalData)
#pybind11#
#pybind11#                storageList2 = dafPersist.StorageList()
#pybind11#                storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
#pybind11#                storageList2.append(storage2)
#pybind11#                x = persistence.unsafeRetrieve("SeparableKernel",
#pybind11#                                               storageList2, additionalData)
#pybind11#                k2 = afwMath.SeparableKernel.swigConvert(x)
#pybind11#
#pybind11#                self.kernelCheck(k, k2)
#pybind11#
#pybind11#                kImage = afwImage.ImageD(k2.getDimensions())
#pybind11#                k2.computeImage(kImage, True)
#pybind11#                kArr = kImage.getArray().transpose()
#pybind11#                if not numpy.allclose(fArr, kArr):
#pybind11#                    self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" %
#pybind11#                              (k2.__class__.__name__, kArr, fArr, xsigma, ysigma))
#pybind11#
#pybind11#    def testLinearCombinationKernel(self):
#pybind11#        """Test LinearCombinationKernel using a set of delta basis functions
#pybind11#        """
#pybind11#        kWidth = 3
#pybind11#        kHeight = 2
#pybind11#
#pybind11#        pol = pexPolicy.Policy()
#pybind11#        additionalData = dafBase.PropertySet()
#pybind11#        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data","kernel5.boost"))
#pybind11#        persistence = dafPersist.Persistence.getPersistence(pol)
#pybind11#
#pybind11#        # create list of kernels
#pybind11#        basisImArrList = []
#pybind11#        kVec = afwMath.KernelList()
#pybind11#        for row in range(kHeight):
#pybind11#            for col in range(kWidth):
#pybind11#                kernel = afwMath.DeltaFunctionKernel(kWidth, kHeight, afwGeom.Point2I(col, row))
#pybind11#                basisImage = afwImage.ImageD(kernel.getDimensions())
#pybind11#                kernel.computeImage(basisImage, True)
#pybind11#                basisImArrList.append(basisImage.getArray().transpose().copy())
#pybind11#                kVec.append(kernel)
#pybind11#
#pybind11#        kParams = [0.0]*len(kVec)
#pybind11#        k = afwMath.LinearCombinationKernel(kVec, kParams)
#pybind11#        for ii in range(len(kVec)):
#pybind11#            kParams = [0.0]*len(kVec)
#pybind11#            kParams[ii] = 1.0
#pybind11#            k.setKernelParameters(kParams)
#pybind11#
#pybind11#            storageList = dafPersist.StorageList()
#pybind11#            storage = persistence.getPersistStorage("XmlStorage", loc)
#pybind11#            storageList.append(storage)
#pybind11#            persistence.persist(k, storageList, additionalData)
#pybind11#
#pybind11#            storageList2 = dafPersist.StorageList()
#pybind11#            storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
#pybind11#            storageList2.append(storage2)
#pybind11#            x = persistence.unsafeRetrieve("LinearCombinationKernel",
#pybind11#                                           storageList2, additionalData)
#pybind11#            k2 = afwMath.LinearCombinationKernel.swigConvert(x)
#pybind11#
#pybind11#            self.kernelCheck(k, k2)
#pybind11#
#pybind11#            kIm = afwImage.ImageD(k2.getDimensions())
#pybind11#            k2.computeImage(kIm, True)
#pybind11#            kImArr = kIm.getArray().transpose()
#pybind11#            if not numpy.allclose(kImArr, basisImArrList[ii]):
#pybind11#                self.fail("%s = %s != %s for the %s'th basis kernel" %
#pybind11#                          (k2.__class__.__name__, kImArr, basisImArrList[ii], ii))
#pybind11#
#pybind11#    def testSVLinearCombinationKernel(self):
#pybind11#        """Test a spatially varying LinearCombinationKernel
#pybind11#        """
#pybind11#        kWidth = 3
#pybind11#        kHeight = 2
#pybind11#
#pybind11#        pol = pexPolicy.Policy()
#pybind11#        additionalData = dafBase.PropertySet()
#pybind11#        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data", "kernel6.boost"))
#pybind11#        persistence = dafPersist.Persistence.getPersistence(pol)
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
#pybind11#        kVec = afwMath.KernelList()
#pybind11#        for basisImArr in basisImArrList:
#pybind11#            basisImage = afwImage.makeImageFromArray(basisImArr.transpose().copy())
#pybind11#            kernel = afwMath.FixedKernel(basisImage)
#pybind11#            kVec.append(kernel)
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
#pybind11#        k = afwMath.LinearCombinationKernel(kVec, spFunc)
#pybind11#        k.setSpatialParameters(sParams)
#pybind11#
#pybind11#        storageList = dafPersist.StorageList()
#pybind11#        storage = persistence.getPersistStorage("XmlStorage", loc)
#pybind11#        storageList.append(storage)
#pybind11#        persistence.persist(k, storageList, additionalData)
#pybind11#
#pybind11#        storageList2 = dafPersist.StorageList()
#pybind11#        storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
#pybind11#        storageList2.append(storage2)
#pybind11#        x = persistence.unsafeRetrieve("LinearCombinationKernel",
#pybind11#                                       storageList2, additionalData)
#pybind11#        k2 = afwMath.LinearCombinationKernel.swigConvert(x)
#pybind11#
#pybind11#        self.kernelCheck(k, k2)
#pybind11#
#pybind11#        kImage = afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight))
#pybind11#        for colPos, rowPos, coeff0, coeff1 in [
#pybind11#            (0.0, 0.0, 0.0, 0.0),
#pybind11#            (1.0, 0.0, 1.0, 0.0),
#pybind11#            (0.0, 1.0, 0.0, 1.0),
#pybind11#            (1.0, 1.0, 1.0, 1.0),
#pybind11#            (0.5, 0.5, 0.5, 0.5),
#pybind11#        ]:
#pybind11#            k2.computeImage(kImage, False, colPos, rowPos)
#pybind11#            kImArr = kImage.getArray().transpose()
#pybind11#            refKImArr = (basisImArrList[0] * coeff0) + (basisImArrList[1] * coeff1)
#pybind11#            if not numpy.allclose(kImArr, refKImArr):
#pybind11#                self.fail("%s = %s != %s at colPos=%s, rowPos=%s" %
#pybind11#                          (k2.__class__.__name__, kImArr, refKImArr, colPos, rowPos))
#pybind11#
#pybind11#    def testSetCtr(self):
#pybind11#        """Test setCtrCol/Row"""
#pybind11#        kWidth = 3
#pybind11#        kHeight = 4
#pybind11#
#pybind11#        pol = pexPolicy.Policy()
#pybind11#        additionalData = dafBase.PropertySet()
#pybind11#        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data", "kernel7.boost"))
#pybind11#        persistence = dafPersist.Persistence.getPersistence(pol)
#pybind11#
#pybind11#        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        k = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc)
#pybind11#        for xCtr in range(kWidth):
#pybind11#            k.setCtrX(xCtr)
#pybind11#            for yCtr in range(kHeight):
#pybind11#                k.setCtrY(yCtr)
#pybind11#
#pybind11#                storageList = dafPersist.StorageList()
#pybind11#                storage = persistence.getPersistStorage("XmlStorage", loc)
#pybind11#                storageList.append(storage)
#pybind11#                persistence.persist(k, storageList, additionalData)
#pybind11#
#pybind11#                storageList2 = dafPersist.StorageList()
#pybind11#                storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
#pybind11#                storageList2.append(storage2)
#pybind11#                x = persistence.unsafeRetrieve("AnalyticKernel",
#pybind11#                                               storageList2, additionalData)
#pybind11#                k2 = afwMath.AnalyticKernel.swigConvert(x)
#pybind11#
#pybind11#                self.kernelCheck(k, k2)
#pybind11#
#pybind11#                self.assertEqual(k2.getCtrX(), xCtr)
#pybind11#                self.assertEqual(k2.getCtrY(), yCtr)
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

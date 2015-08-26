#!/usr/bin/env python
from __future__ import absolute_import, division
from builtins import range

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


import unittest

import numpy
import os

import lsst.utils.tests
import lsst.pex.policy as pexPolicy
import lsst.daf.base as dafBase
import lsst.daf.persistence as dafPersist
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom


testPath = os.path.abspath(os.path.dirname(__file__))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class KernelIOTestCase(unittest.TestCase):
    """A test case for Kernel I/O"""

    def kernelCheck(self, k1, k2):
        self.assertEqual(k1.getWidth(), k2.getWidth())
        self.assertEqual(k1.getHeight(), k2.getHeight())
        self.assertEqual(k1.getCtrX(), k2.getCtrX())
        self.assertEqual(k1.getCtrY(), k2.getCtrY())
        self.assertEqual(k1.getNKernelParameters(), k2.getNKernelParameters())
        self.assertEqual(k1.getNSpatialParameters(), k2.getNSpatialParameters())
        self.assertEqual(k1.getKernelParameters(), k2.getKernelParameters())
        self.assertEqual(k1.getSpatialParameters(), k2.getSpatialParameters())
        self.assertEqual(k1.isSpatiallyVarying(), k2.isSpatiallyVarying())
        self.assertEqual(k1.toString(), k2.toString())

    def testFixedKernel(self):
        """Test FixedKernel using a ramp function
        """
        kWidth = 5
        kHeight = 6

        inArr = numpy.arange(kWidth * kHeight, dtype=float)
        inArr.shape = [kWidth, kHeight]

        inImage = afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight))
        for row in range(inImage.getHeight()):
            for col in range(inImage.getWidth()):
                inImage.set(col, row, inArr[col, row])

        k = afwMath.FixedKernel(inImage)

        pol = pexPolicy.Policy()
        additionalData = dafBase.PropertySet()
        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data", "kernel1.boost"))
        persistence = dafPersist.Persistence.getPersistence(pol)

        storageList = dafPersist.StorageList()
        storage = persistence.getPersistStorage("XmlStorage", loc)
        storageList.append(storage)
        persistence.persist(k, storageList, additionalData)

        storageList2 = dafPersist.StorageList()
        storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
        storageList2.append(storage2)
        x = persistence.unsafeRetrieve("FixedKernel", storageList2, additionalData)
        k2 = afwMath.FixedKernel.swigConvert(x)

        self.kernelCheck(k, k2)

        outImage = afwImage.ImageD(k2.getDimensions())
        k2.computeImage(outImage, False)

        outArr = outImage.getArray().transpose()
        if not numpy.allclose(inArr, outArr):
            self.fail("%s = %s != %s (not normalized)" %
                      (k2.__class__.__name__, inArr, outArr))
        normInArr = inArr / inArr.sum()
        normOutImage = afwImage.ImageD(k2.getDimensions())
        k2.computeImage(normOutImage, True)
        normOutArr = normOutImage.getArray().transpose()
        if not numpy.allclose(normOutArr, normInArr):
            self.fail("%s = %s != %s (normalized)" %
                      (k2.__class__.__name__, normInArr, normOutArr))

    def testAnalyticKernel(self):
        """Test AnalyticKernel using a Gaussian function
        """
        kWidth = 5
        kHeight = 8

        pol = pexPolicy.Policy()
        additionalData = dafBase.PropertySet()
        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data", "kernel2.boost"))
        persistence = dafPersist.Persistence.getPersistence(pol)

        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        k = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc)
        fArr = numpy.zeros(shape=[k.getWidth(), k.getHeight()], dtype=float)
        for xsigma in (0.1, 1.0, 3.0):
            for ysigma in (0.1, 1.0, 3.0):
                for angle in (0.0, 0.4, 1.1):
                    gaussFunc.setParameters((xsigma, ysigma, angle))
                    # compute array of function values and normalize
                    for row in range(k.getHeight()):
                        y = row - k.getCtrY()
                        for col in range(k.getWidth()):
                            x = col - k.getCtrX()
                            fArr[col, row] = gaussFunc(x, y)
                    fArr /= fArr.sum()

                    k.setKernelParameters((xsigma, ysigma, angle))

                    storageList = dafPersist.StorageList()
                    storage = persistence.getPersistStorage("XmlStorage", loc)
                    storageList.append(storage)
                    persistence.persist(k, storageList, additionalData)

                    storageList2 = dafPersist.StorageList()
                    storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
                    storageList2.append(storage2)
                    x = persistence.unsafeRetrieve("AnalyticKernel",
                                                   storageList2, additionalData)
                    k2 = afwMath.AnalyticKernel.swigConvert(x)

                    self.kernelCheck(k, k2)

                    kImage = afwImage.ImageD(k2.getDimensions())
                    k2.computeImage(kImage, True)
                    kArr = kImage.getArray().transpose()
                    if not numpy.allclose(fArr, kArr):
                        self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" %
                                  (k2.__class__.__name__, kArr, fArr, xsigma, ysigma))

    def testDeltaFunctionKernel(self):
        """Test DeltaFunctionKernel
        """
        pol = pexPolicy.Policy()
        additionalData = dafBase.PropertySet()
        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data", "kernel3.boost"))
        persistence = dafPersist.Persistence.getPersistence(pol)

        for kWidth in range(1, 4):
            for kHeight in range(1, 4):
                for activeCol in range(kWidth):
                    for activeRow in range(kHeight):
                        kernel = afwMath.DeltaFunctionKernel(kWidth, kHeight,
                                                             afwGeom.Point2I(activeCol, activeRow))

                        storageList = dafPersist.StorageList()
                        storage = persistence.getPersistStorage("XmlStorage", loc)
                        storageList.append(storage)
                        persistence.persist(kernel, storageList, additionalData)

                        storageList2 = dafPersist.StorageList()
                        storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
                        storageList2.append(storage2)
                        x = persistence.unsafeRetrieve("DeltaFunctionKernel",
                                                       storageList2, additionalData)
                        k2 = afwMath.DeltaFunctionKernel.swigConvert(x)

                        self.kernelCheck(kernel, k2)
                        self.assertEqual(kernel.getPixel(), k2.getPixel())

                        kImage = afwImage.ImageD(k2.getDimensions())
                        kSum = k2.computeImage(kImage, False)
                        self.assertEqual(kSum, 1.0)
                        kArr = kImage.getArray().transpose()
                        self.assertEqual(kArr[activeCol, activeRow], 1.0)
                        kArr[activeCol, activeRow] = 0.0
                        self.assertEqual(kArr.sum(), 0.0)

    def testSeparableKernel(self):
        """Test SeparableKernel using a Gaussian function
        """
        kWidth = 5
        kHeight = 8

        pol = pexPolicy.Policy()
        additionalData = dafBase.PropertySet()
        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data", "kernel4.boost"))
        persistence = dafPersist.Persistence.getPersistence(pol)

        gaussFunc1 = afwMath.GaussianFunction1D(1.0)
        k = afwMath.SeparableKernel(kWidth, kHeight, gaussFunc1, gaussFunc1)
        fArr = numpy.zeros(shape=[k.getWidth(), k.getHeight()], dtype=float)
        numpy.zeros(shape=[k.getWidth(), k.getHeight()], dtype=float)
        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        for xsigma in (0.1, 1.0, 3.0):
            gaussFunc1.setParameters((xsigma,))
            for ysigma in (0.1, 1.0, 3.0):
                gaussFunc.setParameters((xsigma, ysigma, 0.0))
                # compute array of function values and normalize
                for row in range(k.getHeight()):
                    y = row - k.getCtrY()
                    for col in range(k.getWidth()):
                        x = col - k.getCtrX()
                        fArr[col, row] = gaussFunc(x, y)
                fArr /= fArr.sum()

                k.setKernelParameters((xsigma, ysigma))

                storageList = dafPersist.StorageList()
                storage = persistence.getPersistStorage("XmlStorage", loc)
                storageList.append(storage)
                persistence.persist(k, storageList, additionalData)

                storageList2 = dafPersist.StorageList()
                storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
                storageList2.append(storage2)
                x = persistence.unsafeRetrieve("SeparableKernel",
                                               storageList2, additionalData)
                k2 = afwMath.SeparableKernel.swigConvert(x)

                self.kernelCheck(k, k2)

                kImage = afwImage.ImageD(k2.getDimensions())
                k2.computeImage(kImage, True)
                kArr = kImage.getArray().transpose()
                if not numpy.allclose(fArr, kArr):
                    self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" %
                              (k2.__class__.__name__, kArr, fArr, xsigma, ysigma))

    def testLinearCombinationKernel(self):
        """Test LinearCombinationKernel using a set of delta basis functions
        """
        kWidth = 3
        kHeight = 2

        pol = pexPolicy.Policy()
        additionalData = dafBase.PropertySet()
        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data","kernel5.boost"))
        persistence = dafPersist.Persistence.getPersistence(pol)

        # create list of kernels
        basisImArrList = []
        kVec = afwMath.KernelList()
        for row in range(kHeight):
            for col in range(kWidth):
                kernel = afwMath.DeltaFunctionKernel(kWidth, kHeight, afwGeom.Point2I(col, row))
                basisImage = afwImage.ImageD(kernel.getDimensions())
                kernel.computeImage(basisImage, True)
                basisImArrList.append(basisImage.getArray().transpose().copy())
                kVec.append(kernel)

        kParams = [0.0]*len(kVec)
        k = afwMath.LinearCombinationKernel(kVec, kParams)
        for ii in range(len(kVec)):
            kParams = [0.0]*len(kVec)
            kParams[ii] = 1.0
            k.setKernelParameters(kParams)

            storageList = dafPersist.StorageList()
            storage = persistence.getPersistStorage("XmlStorage", loc)
            storageList.append(storage)
            persistence.persist(k, storageList, additionalData)

            storageList2 = dafPersist.StorageList()
            storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
            storageList2.append(storage2)
            x = persistence.unsafeRetrieve("LinearCombinationKernel",
                                           storageList2, additionalData)
            k2 = afwMath.LinearCombinationKernel.swigConvert(x)

            self.kernelCheck(k, k2)

            kIm = afwImage.ImageD(k2.getDimensions())
            k2.computeImage(kIm, True)
            kImArr = kIm.getArray().transpose()
            if not numpy.allclose(kImArr, basisImArrList[ii]):
                self.fail("%s = %s != %s for the %s'th basis kernel" %
                          (k2.__class__.__name__, kImArr, basisImArrList[ii], ii))

    def testSVLinearCombinationKernel(self):
        """Test a spatially varying LinearCombinationKernel
        """
        kWidth = 3
        kHeight = 2

        pol = pexPolicy.Policy()
        additionalData = dafBase.PropertySet()
        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data", "kernel6.boost"))
        persistence = dafPersist.Persistence.getPersistence(pol)

        # create image arrays for the basis kernels
        basisImArrList = []
        imArr = numpy.zeros((kWidth, kHeight), dtype=float)
        imArr += 0.1
        imArr[kWidth//2, :] = 0.9
        basisImArrList.append(imArr)
        imArr = numpy.zeros((kWidth, kHeight), dtype=float)
        imArr += 0.2
        imArr[:, kHeight//2] = 0.8
        basisImArrList.append(imArr)

        # create a list of basis kernels from the images
        kVec = afwMath.KernelList()
        for basisImArr in basisImArrList:
            basisImage = afwImage.makeImageFromArray(basisImArr.transpose().copy())
            kernel = afwMath.FixedKernel(basisImage)
            kVec.append(kernel)

        # create spatially varying linear combination kernel
        spFunc = afwMath.PolynomialFunction2D(1)

        # spatial parameters are a list of entries, one per kernel parameter;
        # each entry is a list of spatial parameters
        sParams = (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )

        k = afwMath.LinearCombinationKernel(kVec, spFunc)
        k.setSpatialParameters(sParams)

        storageList = dafPersist.StorageList()
        storage = persistence.getPersistStorage("XmlStorage", loc)
        storageList.append(storage)
        persistence.persist(k, storageList, additionalData)

        storageList2 = dafPersist.StorageList()
        storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
        storageList2.append(storage2)
        x = persistence.unsafeRetrieve("LinearCombinationKernel",
                                       storageList2, additionalData)
        k2 = afwMath.LinearCombinationKernel.swigConvert(x)

        self.kernelCheck(k, k2)

        kImage = afwImage.ImageD(afwGeom.Extent2I(kWidth, kHeight))
        for colPos, rowPos, coeff0, coeff1 in [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5, 0.5),
        ]:
            k2.computeImage(kImage, False, colPos, rowPos)
            kImArr = kImage.getArray().transpose()
            refKImArr = (basisImArrList[0] * coeff0) + (basisImArrList[1] * coeff1)
            if not numpy.allclose(kImArr, refKImArr):
                self.fail("%s = %s != %s at colPos=%s, rowPos=%s" %
                          (k2.__class__.__name__, kImArr, refKImArr, colPos, rowPos))

    def testSetCtr(self):
        """Test setCtrCol/Row"""
        kWidth = 3
        kHeight = 4

        pol = pexPolicy.Policy()
        additionalData = dafBase.PropertySet()
        loc = dafPersist.LogicalLocation(os.path.join(testPath, "data", "kernel7.boost"))
        persistence = dafPersist.Persistence.getPersistence(pol)

        gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
        k = afwMath.AnalyticKernel(kWidth, kHeight, gaussFunc)
        for xCtr in range(kWidth):
            k.setCtrX(xCtr)
            for yCtr in range(kHeight):
                k.setCtrY(yCtr)

                storageList = dafPersist.StorageList()
                storage = persistence.getPersistStorage("XmlStorage", loc)
                storageList.append(storage)
                persistence.persist(k, storageList, additionalData)

                storageList2 = dafPersist.StorageList()
                storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
                storageList2.append(storage2)
                x = persistence.unsafeRetrieve("AnalyticKernel",
                                               storageList2, additionalData)
                k2 = afwMath.AnalyticKernel.swigConvert(x)

                self.kernelCheck(k, k2)

                self.assertEqual(k2.getCtrX(), xCtr)
                self.assertEqual(k2.getCtrY(), yCtr)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass

def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

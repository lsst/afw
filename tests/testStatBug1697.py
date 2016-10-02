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
#pybind11#
#pybind11#import unittest
#pybind11#
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.image as afwImage
#pybind11#
#pybind11#
#pybind11#class weightedStatsBugTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def reportBadPixels(self, maskedImage, badPixelMask):
#pybind11#        """Report the number of bad pixels in each plane of a masked image
#pybind11#
#pybind11#        Reports:
#pybind11#        - the number of non-finite values in the image plane
#pybind11#        - the number of bad pixels in the mask plane
#pybind11#        - the number of non-finite values in the variance plane
#pybind11#        """
#pybind11#        arrayList = maskedImage.getArrays()
#pybind11#        nBadImg = numpy.logical_not(numpy.isfinite(arrayList[0])).sum()
#pybind11#        nBadMsk = numpy.sum(numpy.bitwise_and(arrayList[1], badPixelMask) > 0)
#pybind11#        nBadVar = numpy.logical_not(numpy.isfinite(arrayList[2])).sum()
#pybind11#        print("%d bad image pixels, %d bad mask pixels, %d bad variance pixels" % (nBadImg, nBadMsk, nBadVar))
#pybind11#        self.assertEqual(nBadImg, 0)
#pybind11#        self.assertEqual(nBadMsk, 0)
#pybind11#        self.assertEqual(nBadVar, 0)
#pybind11#
#pybind11#    def testWeightedStats(self):
#pybind11#        """Test that bug from #1697 (weighted stats returning NaN) stays fixed."""
#pybind11#
#pybind11#        rand = afwMath.Random()
#pybind11#        mu = 10000
#pybind11#
#pybind11#        afwImage.MaskU.getPlaneBitMask("EDGE")
#pybind11#
#pybind11#        badPixelMask = afwImage.MaskU.getPlaneBitMask("EDGE")
#pybind11#        statsCtrl = afwMath.StatisticsControl()
#pybind11#        statsCtrl.setNumSigmaClip(3.0)
#pybind11#        statsCtrl.setNumIter(2)
#pybind11#        statsCtrl.setAndMask(badPixelMask)
#pybind11#
#pybind11#        for weight in (300.0, 10.0, 1.0):
#pybind11#            print("Testing with weight=%0.1f" % (weight,))
#pybind11#            maskedImageList = afwImage.vectorMaskedImageF()  # [] is rejected by afwMath.statisticsStack
#pybind11#            weightList = []
#pybind11#
#pybind11#            nx, ny = 256, 256
#pybind11#            for i in range(3):
#pybind11#                print("Processing ", i)
#pybind11#                maskedImage = afwImage.MaskedImageF(nx, ny)
#pybind11#                maskedImageList.append(maskedImage)
#pybind11#
#pybind11#                afwMath.randomPoissonImage(maskedImage.getImage(), rand, mu)
#pybind11#                maskedImage.getVariance().set(mu)
#pybind11#                weightList.append(weight)
#pybind11#
#pybind11#            self.reportBadPixels(maskedImage, badPixelMask)
#pybind11#
#pybind11#            print("Stack: ", end=' ')
#pybind11#            coaddMaskedImage = afwMath.statisticsStack(
#pybind11#                maskedImageList, afwMath.MEANCLIP, statsCtrl, weightList)
#pybind11#            self.reportBadPixels(coaddMaskedImage, badPixelMask)
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

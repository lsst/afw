#!/usr/bin/env python
from __future__ import absolute_import, division
from __future__ import print_function
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

import lsst.utils.tests
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage


class weightedStatsBugTestCase(unittest.TestCase):

    def reportBadPixels(self, maskedImage, badPixelMask):
        """Report the number of bad pixels in each plane of a masked image

        Reports:
        - the number of non-finite values in the image plane
        - the number of bad pixels in the mask plane
        - the number of non-finite values in the variance plane
        """
        arrayList = maskedImage.getArrays()
        nBadImg = numpy.logical_not(numpy.isfinite(arrayList[0])).sum()
        nBadMsk = numpy.sum(numpy.bitwise_and(arrayList[1], badPixelMask) > 0)
        nBadVar = numpy.logical_not(numpy.isfinite(arrayList[2])).sum()
        print("%d bad image pixels, %d bad mask pixels, %d bad variance pixels" % (nBadImg, nBadMsk, nBadVar))
        self.assertEqual(nBadImg, 0)
        self.assertEqual(nBadMsk, 0)
        self.assertEqual(nBadVar, 0)

    def testWeightedStats(self):
        """Test that bug from #1697 (weighted stats returning NaN) stays fixed."""

        rand = afwMath.Random()
        mu = 10000

        afwImage.MaskU.getPlaneBitMask("EDGE")

        badPixelMask = afwImage.MaskU.getPlaneBitMask("EDGE")
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(3.0)
        statsCtrl.setNumIter(2)
        statsCtrl.setAndMask(badPixelMask)

        for weight in (300.0, 10.0, 1.0):
            print("Testing with weight=%0.1f" % (weight,))
            maskedImageList = afwImage.vectorMaskedImageF()  # [] is rejected by afwMath.statisticsStack
            weightList = []

            nx, ny = 256, 256
            for i in range(3):
                print("Processing ", i)
                maskedImage = afwImage.MaskedImageF(nx, ny)
                maskedImageList.append(maskedImage)

                afwMath.randomPoissonImage(maskedImage.getImage(), rand, mu)
                maskedImage.getVariance().set(mu)
                weightList.append(weight)

            self.reportBadPixels(maskedImage, badPixelMask)

            print("Stack: ", end=' ')
            coaddMaskedImage = afwMath.statisticsStack(
                maskedImageList, afwMath.MEANCLIP, statsCtrl, weightList)
            self.reportBadPixels(coaddMaskedImage, badPixelMask)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass

def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
#!/usr/bin/env python2
from __future__ import absolute_import, division

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

import lsst.utils.tests as utilsTests
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.image.testUtils as imTestUtils

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ScaledPlus(unittest.TestCase):
    def setUp(self):
        self.random = afwMath.Random()
        self.imWidth = 200
        self.imHeight = 200
        self.maskedImage0 = afwImage.MaskedImageF(afwGeom.Extent2I(self.imWidth, self.imHeight))
        afwMath.randomUniformImage(self.maskedImage0.getImage(), self.random)
        afwMath.randomUniformImage(self.maskedImage0.getVariance(), self.random)
#        afwMath.randomUniformImage(self.maskedImage0.getMask(), self.random)
        self.maskedImage1 = afwImage.MaskedImageF(afwGeom.Extent2I(self.imWidth, self.imHeight))
        afwMath.randomUniformImage(self.maskedImage1.getImage(), self.random)
        afwMath.randomUniformImage(self.maskedImage1.getVariance(), self.random)
#        afwMath.randomUniformImage(self.maskedImage1.getMask(), self.random)

    def tearDown(self):
        self.random = None
        self.maskedImage0 = None
        self.maskedImage1 = None

    def runScaledAddTest(self, coeff0, coeff1):
        """Run one test of scaledPlus
        
        Inputs:
        - coeff0: coefficient of image 0
        - coeff1: coefficient of image 1
        """
        im0ArrSet = self.maskedImage0.getArrays()
        im1ArrSet = self.maskedImage1.getArrays()
        
        desMaskedImage = afwImage.MaskedImageF(self.maskedImage0.getDimensions())
        desMaskedImage <<= self.maskedImage0
        desMaskedImage *= coeff0
        desMaskedImage.scaledPlus(coeff1, self.maskedImage1)
        desImArrSet = desMaskedImage.getArrays()
        
        actMaskedImage = afwImage.MaskedImageF(afwGeom.Extent2I(self.imWidth, self.imHeight))
        afwMath.randomUniformImage(actMaskedImage.getImage(), self.random)
        afwMath.randomUniformImage(actMaskedImage.getVariance(), self.random)

        afwMath.scaledPlus(actMaskedImage, coeff0, self.maskedImage0, coeff1, self.maskedImage1)
        actImArrSet = actMaskedImage.getArrays()
        
        actImage = afwImage.ImageF(afwGeom.Extent2I(self.imWidth, self.imHeight))
        afwMath.randomUniformImage(actImage, self.random)
        afwMath.scaledPlus(actImage, coeff0, self.maskedImage0.getImage(), coeff1, self.maskedImage1.getImage())
        actImArr = actImage.getArray()
        
        errStr = imTestUtils.imagesDiffer(actImArr, desImArrSet[0])
        if errStr:
            self.fail("scaledPlus failed in images; coeff0=%s, coeff1=%s:\n%s" % (coeff0, coeff1, errStr,))
        errStr = imTestUtils.maskedImagesDiffer(actImArrSet, desImArrSet)
        if errStr:
            self.fail("scaledPlus failed on masked images; coeff0=%s, coeff1=%s:\n%s" %
                (coeff0, coeff1, errStr,))

    def testScaledPlus(self):
        for coeff0 in (0.0, -0.1e-5, 0.1e-5, 1.0e3):
            for coeff1 in (0.0, 0.1e-5, -0.1e-5, 1.0e3):
                self.runScaledAddTest(coeff0, coeff1)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ScaledPlus)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(doExit=False):
    """Run the tests"""
    utilsTests.run(suite(), doExit)

if __name__ == "__main__":
    run(True)

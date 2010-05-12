#!/usr/bin/env python
import math
import unittest

import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.pex.logging as pexLog
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.image.testUtils as imTestUtils

VERBOSITY = 0 # increase to see trace

pexLog.Debug("lsst.afw", VERBOSITY)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ImageMath(unittest.TestCase):
    def setUp(self):
        self.random = afwMath.Random()
        self.imWidth = 200
        self.imHeight = 200
        self.maskedImage0 = afwImage.MaskedImageF(self.imWidth, self.imHeight)
        afwMath.randomUniformImage(self.maskedImage0.getImage(), self.random)
        afwMath.randomUniformImage(self.maskedImage0.getVariance(), self.random)
#        afwMath.randomUniformImage(self.maskedImage0.getMask(), self.random)
        self.maskedImage1 = afwImage.MaskedImageF(self.imWidth, self.imHeight)
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
        im0ArrSet = imTestUtils.arraysFromMaskedImage(self.maskedImage0)
        im1ArrSet = imTestUtils.arraysFromMaskedImage(self.maskedImage1)
        
        desMaskedImage = afwImage.MaskedImageF(self.maskedImage0.getDimensions())
        desMaskedImage <<= self.maskedImage0
        desMaskedImage *= coeff0
        desMaskedImage.scaledPlus(coeff1, self.maskedImage1)
        desImArrSet = imTestUtils.arraysFromMaskedImage(desMaskedImage)
        
        actMaskedImage = afwImage.MaskedImageF(self.imWidth, self.imHeight)
        afwMath.randomUniformImage(actMaskedImage.getImage(), self.random)
        afwMath.randomUniformImage(actMaskedImage.getVariance(), self.random)

        afwMath.scaledPlus(actMaskedImage, coeff0, self.maskedImage0, coeff1, self.maskedImage1)
        actImArrSet = imTestUtils.arraysFromMaskedImage(actMaskedImage)
        
        actImage = afwImage.ImageF(self.imWidth, self.imHeight)
        afwMath.randomUniformImage(actImage, self.random)
        afwMath.scaledPlus(actImage, coeff0, self.maskedImage0.getImage(), coeff1, self.maskedImage1.getImage())
        actImArr = imTestUtils.arrayFromImage(actImage)
        
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
    suites += unittest.makeSuite(ImageMath)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(doExit=False):
    """Run the tests"""
    utilsTests.run(suite(), doExit)

if __name__ == "__main__":
    run(True)

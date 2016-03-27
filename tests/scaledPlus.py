#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.logging as pexLog
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath

VERBOSITY = 0 # increase to see trace

pexLog.Debug("lsst.afw", VERBOSITY)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ScaledPlus(utilsTests.TestCase):
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
        desMaskedImage = afwImage.MaskedImageF(self.maskedImage0.getDimensions())
        desMaskedImage[:] = self.maskedImage0
        desMaskedImage *= coeff0
        desMaskedImage.scaledPlus(coeff1, self.maskedImage1)
        
        actMaskedImage = afwImage.MaskedImageF(afwGeom.Extent2I(self.imWidth, self.imHeight))
        afwMath.randomUniformImage(actMaskedImage.getImage(), self.random)
        afwMath.randomUniformImage(actMaskedImage.getVariance(), self.random)

        afwMath.scaledPlus(actMaskedImage, coeff0, self.maskedImage0, coeff1, self.maskedImage1)
        
        actImage = afwImage.ImageF(afwGeom.Extent2I(self.imWidth, self.imHeight))
        afwMath.randomUniformImage(actImage, self.random)
        afwMath.scaledPlus(actImage, coeff0, self.maskedImage0.getImage(), coeff1, self.maskedImage1.getImage())
        
        msg = "scaledPlus failed for images; coeff0=%s, coeff1=%s" % (coeff0, coeff1)
        self.assertImagesNearlyEqual(actImage, desMaskedImage.getImage(), msg=msg)
        msg = "scaledPlus failed for masked images; coeff0=%s, coeff1=%s" % (coeff0, coeff1)
        self.assertMaskedImagesNearlyEqual(actMaskedImage, desMaskedImage, msg=msg)

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

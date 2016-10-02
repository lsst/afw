#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
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
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#from lsst.log import Log
#pybind11#
#pybind11#Log.getLogger("afw.image.Mask").setLevel(Log.INFO)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class ScaledPlus(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.random = afwMath.Random()
#pybind11#        self.imWidth = 200
#pybind11#        self.imHeight = 200
#pybind11#        self.maskedImage0 = afwImage.MaskedImageF(afwGeom.Extent2I(self.imWidth, self.imHeight))
#pybind11#        afwMath.randomUniformImage(self.maskedImage0.getImage(), self.random)
#pybind11#        afwMath.randomUniformImage(self.maskedImage0.getVariance(), self.random)
#pybind11##        afwMath.randomUniformImage(self.maskedImage0.getMask(), self.random)
#pybind11#        self.maskedImage1 = afwImage.MaskedImageF(afwGeom.Extent2I(self.imWidth, self.imHeight))
#pybind11#        afwMath.randomUniformImage(self.maskedImage1.getImage(), self.random)
#pybind11#        afwMath.randomUniformImage(self.maskedImage1.getVariance(), self.random)
#pybind11##        afwMath.randomUniformImage(self.maskedImage1.getMask(), self.random)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        self.random = None
#pybind11#        self.maskedImage0 = None
#pybind11#        self.maskedImage1 = None
#pybind11#
#pybind11#    def runScaledAddTest(self, coeff0, coeff1):
#pybind11#        """Run one test of scaledPlus
#pybind11#
#pybind11#        Inputs:
#pybind11#        - coeff0: coefficient of image 0
#pybind11#        - coeff1: coefficient of image 1
#pybind11#        """
#pybind11#        desMaskedImage = afwImage.MaskedImageF(self.maskedImage0.getDimensions())
#pybind11#        desMaskedImage[:] = self.maskedImage0
#pybind11#        desMaskedImage *= coeff0
#pybind11#        desMaskedImage.scaledPlus(coeff1, self.maskedImage1)
#pybind11#
#pybind11#        actMaskedImage = afwImage.MaskedImageF(afwGeom.Extent2I(self.imWidth, self.imHeight))
#pybind11#        afwMath.randomUniformImage(actMaskedImage.getImage(), self.random)
#pybind11#        afwMath.randomUniformImage(actMaskedImage.getVariance(), self.random)
#pybind11#
#pybind11#        afwMath.scaledPlus(actMaskedImage, coeff0, self.maskedImage0, coeff1, self.maskedImage1)
#pybind11#
#pybind11#        actImage = afwImage.ImageF(afwGeom.Extent2I(self.imWidth, self.imHeight))
#pybind11#        afwMath.randomUniformImage(actImage, self.random)
#pybind11#        afwMath.scaledPlus(actImage, coeff0, self.maskedImage0.getImage(),
#pybind11#                           coeff1, self.maskedImage1.getImage())
#pybind11#
#pybind11#        msg = "scaledPlus failed for images; coeff0=%s, coeff1=%s" % (coeff0, coeff1)
#pybind11#        self.assertImagesNearlyEqual(actImage, desMaskedImage.getImage(), msg=msg)
#pybind11#        msg = "scaledPlus failed for masked images; coeff0=%s, coeff1=%s" % (coeff0, coeff1)
#pybind11#        self.assertMaskedImagesNearlyEqual(actMaskedImage, desMaskedImage, msg=msg)
#pybind11#
#pybind11#    def testScaledPlus(self):
#pybind11#        for coeff0 in (0.0, -0.1e-5, 0.1e-5, 1.0e3):
#pybind11#            for coeff1 in (0.0, 0.1e-5, -0.1e-5, 1.0e3):
#pybind11#                self.runScaledAddTest(coeff0, coeff1)
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

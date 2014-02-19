#!/usr/bin/env python2
from __future__ import absolute_import, division
#
# LSST Data Management System
# Copyright 2014 LSST Corporation.
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
"""
Tests for lsst.afw.geom.XYTransform and xyTransformRegistry
"""
import unittest
import math

import lsst.utils.tests
from lsst.pex.exceptions import LsstCppException
from lsst.afw.geom import Extent2D, Point2D, xyTransformRegistry, \
    IdentityXYTransform, AffineXYTransform, RadialXYTransform

class XYTransformTestCase(unittest.TestCase):
    def fromIter(self):
        for x in (-1.1, 0, 2.2):
            for y in (3.1, 0, 2.1):
                yield Point2D(x, y)

    def checkForwardReverse(self, transform):
        for fromPoint in self.fromIter():
            roundTripPoint = transform.reverseTransform(transform.forwardTransform(fromPoint))
            for i in range(2):
                self.assertAlmostEqual(fromPoint[i], roundTripPoint[i])

    def testIdentity(self):
        """Test identity = IdentityXYTransform
        """
        identClass = xyTransformRegistry["identity"]
        ident = identClass(identClass.ConfigClass())
        self.assertEquals(type(ident), IdentityXYTransform)
        self.checkForwardReverse(ident)
        for fromPoint in self.fromIter():
            toPoint = ident.forwardTransform(fromPoint)
            for i in range(2):
                self.assertAlmostEqual(fromPoint[i], toPoint[i])

    def testDefaultAffine(self):
        """Test affine = AffineXYTransform with default coeffs (identity transform)
        """
        affineClass = xyTransformRegistry["affine"]
        affine = affineClass(affineClass.ConfigClass())
        self.assertEquals(type(affine), AffineXYTransform)
        self.checkForwardReverse(affine)
        for fromPoint in self.fromIter():
            toPoint = affine.forwardTransform(fromPoint)
            for i in range(2):
                self.assertAlmostEqual(fromPoint[i], toPoint[i])

    def testTranslateAffine(self):
        """Test affine = AffineXYTransform with just translation coefficients
        """
        affineClass = xyTransformRegistry["affine"]
        affineConfig = affineClass.ConfigClass()
        affineConfig.translation = (1.2, -3.4)
        affine = affineClass(affineConfig)
        for fromPoint in self.fromIter():
            toPoint = affine.forwardTransform(fromPoint)
            predToPoint = fromPoint + Extent2D(*affineConfig.translation)
            for i in range(2):
                self.assertAlmostEqual(toPoint[i], predToPoint[i])

    def testLinearAffine(self):
        """Test affine = AffineXYTransform with just linear coefficients
        """
        affineClass = xyTransformRegistry["affine"]
        affineConfig = affineClass.ConfigClass()
        rotAng = 0.25 # radians
        xScale = 1.2
        yScale = 0.8
        affineConfig.linear = (
             math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
        )
        affine = affineClass(affineConfig)
        for fromPoint in self.fromIter():
            toPoint = affine.forwardTransform(fromPoint)
            predToPoint = Point2D(
                affineConfig.linear[0] * fromPoint[0] + affineConfig.linear[1] * fromPoint[1],
                affineConfig.linear[2] * fromPoint[0] + affineConfig.linear[3] * fromPoint[1],
            )
            for i in range(2):
                self.assertAlmostEqual(toPoint[i], predToPoint[i])

    def testFullAffine(self):
        """Test affine = AffineXYTransform with just linear coefficients
        """
        affineClass = xyTransformRegistry["affine"]
        affineConfig = affineClass.ConfigClass()
        affineConfig.translation = (-2.1, 3.4)
        rotAng = 0.832 # radians
        xScale = 3.7
        yScale = 45.3
        affineConfig.linear = (
             math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
        )
        affine = affineClass(affineConfig)
        for fromPoint in self.fromIter():
            toPoint = affine.forwardTransform(fromPoint)
            predToPoint = Point2D(
                affineConfig.linear[0] * fromPoint[0] + affineConfig.linear[1] * fromPoint[1],
                affineConfig.linear[2] * fromPoint[0] + affineConfig.linear[3] * fromPoint[1],
            )
            predToPoint = predToPoint + Extent2D(*affineConfig.translation)
            for i in range(2):
                self.assertAlmostEqual(toPoint[i], predToPoint[i])

    def testRadial(self):
        """Test radial = RadialXYTransform
        """
        radialClass = xyTransformRegistry["radial"]
        radialConfig = radialClass.ConfigClass()
        radialConfig.coeffs = (0, 1.05, 0.1)
        radial = radialClass(radialConfig)
        self.assertEquals(type(radial), RadialXYTransform)
        self.checkForwardReverse(radial)
        for fromPoint in self.fromIter():
            fromRadius = math.hypot(fromPoint[0], fromPoint[1])
            fromAngle = math.atan2(fromPoint[1], fromPoint[0])
            predToRadius = fromRadius * (radialConfig.coeffs[2] * fromRadius + radialConfig.coeffs[1])
            predToPoint = Point2D(predToRadius * math.cos(fromAngle), predToRadius * math.sin(fromAngle))
            toPoint = radial.forwardTransform(fromPoint)
            for i in range(2):
                self.assertAlmostEqual(toPoint[i], predToPoint[i])

    def testBadRadial(self):
        """Test radial with invalid coefficients
        """
        for badCoeffs in (
            (0.1,),     # len(coeffs) must be > 1
            (0.1, 1.0), # coeffs[0] must be zero
            (0.0, 0.0), # coeffs[1] must be nonzero
            (0.0, 0.0, 0.1), # coeffs[1] must be nonzero
        ):
            self.assertRaises(LsstCppException, RadialXYTransform, badCoeffs)

            radialClass = xyTransformRegistry["radial"]
            radialConfig = radialClass.ConfigClass()
            radialConfig.coeffs = badCoeffs
            self.assertRaises(Exception, radialConfig.validate)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(XYTransformTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

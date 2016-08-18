#!/usr/bin/env python
from __future__ import absolute_import, division
from builtins import range
from builtins import object
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
import itertools
import math
import unittest

import lsst.utils.tests
import lsst.pex.exceptions
from lsst.afw.geom import Extent2D, Point2D, xyTransformRegistry, OneXYTransformConfig, \
    IdentityXYTransform, AffineXYTransform, RadialXYTransform


class RefMultiAffineTransform(object):

    def __init__(self, affineTransformList):
        self.affineTransformList = affineTransformList

    def __call__(self, point):
        for tr in self.affineTransformList:
            point = tr(point)
        return point


class RefMultiXYTransform(object):

    def __init__(self, transformList):
        self.transformList = transformList

    def forwardTransform(self, point):
        for tr in self.transformList:
            point = tr.forwardTransform(point)
        return point

    def reverseTransform(self, point):
        for tr in reversed(self.transformList):
            point = tr.reverseTransform(point)
        return point

    def linearizeForwardTransform(self, point):
        affineTransformList = [tr.linearizeForwardTransform(point) for tr in self.transformList]
        return RefMultiAffineTransform(affineTransformList)

    def linearizeReverseTransform(self, point):
        affineTransformList = [tr.linearizeReverseTransform(point) for tr in reversed(self.transformList)]
        return RefMultiAffineTransform(affineTransformList)


class XYTransformTestCase(unittest.TestCase):

    def fromIter(self):
        for x in (-1.1, 0, 2.2):
            for y in (3.1, 0, 2.1):
                yield Point2D(x, y)

    def checkBasics(self, transform):
        """Check round trip and linearization of transform
        """
        for fromPoint in self.fromIter():
            toPoint = transform.forwardTransform(fromPoint)
            roundTripPoint = transform.reverseTransform(toPoint)
            for i in range(2):
                self.assertAlmostEqual(fromPoint[i], roundTripPoint[i])

            for deltaFrom in (
                Extent2D(0),
                Extent2D(0.1, -0.1),
                Extent2D(-0.15, 0.1),
            ):
                tweakedFromPoint = fromPoint + deltaFrom
                tweakedToPoint = transform.forwardTransform(tweakedFromPoint)
                linToPoint = transform.linearizeForwardTransform(fromPoint)(tweakedFromPoint)
                linRoundTripPoint = transform.linearizeReverseTransform(toPoint)(tweakedToPoint)
                for i in range(2):
                    self.assertAlmostEqual(tweakedToPoint[i], linToPoint[i], places=2)
                    self.assertAlmostEqual(tweakedFromPoint[i], linRoundTripPoint[i], places=2)

    def checkConfig(self, tClass, tConfig, filePath):
        """Check round trip of config
        """
        tConfig.save(filePath)
        loadConfig = tConfig.__class__()
        loadConfig.load(filePath)
        transform = tClass(loadConfig)
        self.checkBasics(transform)

    def testIdentity(self):
        """Test identity = IdentityXYTransform
        """
        identClass = xyTransformRegistry["identity"]
        with lsst.utils.tests.getTempFilePath(".py") as filePath:
            self.checkConfig(identClass, identClass.ConfigClass(), filePath)
            ident = identClass(identClass.ConfigClass())
            self.assertEqual(type(ident), IdentityXYTransform)
            self.checkBasics(ident)
            for fromPoint in self.fromIter():
                toPoint = ident.forwardTransform(fromPoint)
                for i in range(2):
                    self.assertAlmostEqual(fromPoint[i], toPoint[i])

    def testInverted(self):
        """Test inverted = InvertedXYTransform
        """
        invertedClass = xyTransformRegistry["inverted"]
        invertedConfig = invertedClass.ConfigClass()
        affineClass = xyTransformRegistry["affine"]
        invertedConfig.transform.retarget(affineClass)
        affineConfig = invertedConfig.transform
        affineConfig.translation = (1.2, -3.4)
        with lsst.utils.tests.getTempFilePath(".py") as filePath:
            self.checkConfig(invertedClass, invertedConfig, filePath)
            inverted = invertedClass(invertedConfig)
            self.checkBasics(inverted)
            for fromPoint in self.fromIter():
                toPoint = inverted.forwardTransform(fromPoint)
                predToPoint = fromPoint - Extent2D(*invertedConfig.transform.translation)
                for i in range(2):
                    self.assertAlmostEqual(toPoint[i], predToPoint[i])

    def testDefaultAffine(self):
        """Test affine = AffineXYTransform with default coeffs (identity transform)
        """
        affineClass = xyTransformRegistry["affine"]
        affineConfig = affineClass.ConfigClass()
        with lsst.utils.tests.getTempFilePath(".py") as filePath:
            self.checkConfig(affineClass, affineConfig, filePath)
            affine = affineClass(affineConfig)
            self.assertEqual(type(affine), AffineXYTransform)
            self.checkBasics(affine)
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
        with lsst.utils.tests.getTempFilePath(".py") as filePath:
            self.checkConfig(affineClass, affineConfig, filePath)
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
        rotAng = 0.25  # radians
        xScale = 1.2
        yScale = 0.8
        affineConfig.linear = (
            math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
        )
        with lsst.utils.tests.getTempFilePath(".py") as filePath:
            self.checkConfig(affineClass, affineConfig, filePath)
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
        rotAng = 0.832  # radians
        xScale = 3.7
        yScale = 45.3
        affineConfig.linear = (
            math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
        )
        with lsst.utils.tests.getTempFilePath(".py") as filePath:
            self.checkConfig(affineClass, affineConfig, filePath)
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
        with lsst.utils.tests.getTempFilePath(".py") as filePath:
            self.checkConfig(radialClass, radialConfig, filePath)
            radial = radialClass(radialConfig)
            self.assertEqual(type(radial), RadialXYTransform)
            self.assertEqual(len(radial.getCoeffs()), len(radialConfig.coeffs))
            for coeff, predCoeff in zip(radial.getCoeffs(), radialConfig.coeffs):
                self.assertAlmostEqual(coeff, predCoeff)
            self.checkBasics(radial)
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
            (0.1, 1.0),  # coeffs[0] must be zero
            (0.0, 0.0),  # coeffs[1] must be nonzero
            (0.0, 0.0, 0.1),  # coeffs[1] must be nonzero
        ):
            self.assertRaises(lsst.pex.exceptions.Exception, RadialXYTransform, badCoeffs)

            radialClass = xyTransformRegistry["radial"]
            radialConfig = radialClass.ConfigClass()
            radialConfig.coeffs = badCoeffs
            self.assertRaises(Exception, radialConfig.validate)

    def testMulti(self):
        """Test multi = MultiXYTransform
        """
        affineClass = xyTransformRegistry["affine"]
        wrapper0 = OneXYTransformConfig()
        wrapper0.transform.retarget(affineClass)
        affineConfig0 = wrapper0.transform
        affineConfig0.translation = (-2.1, 3.4)
        rotAng = 0.832  # radians
        xScale = 3.7
        yScale = 45.3
        affineConfig0.linear = (
            math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
        )

        wrapper1 = OneXYTransformConfig()
        wrapper1.transform.retarget(affineClass)
        affineConfig1 = wrapper1.transform
        affineConfig1.translation = (26.5, -35.1)
        rotAng = -0.25  # radians
        xScale = 1.45
        yScale = 0.9
        affineConfig1.linear = (
            math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
        )

        multiClass = xyTransformRegistry["multi"]
        multiConfig = multiClass.ConfigClass()
        multiConfig.transformDict = {
            0: wrapper0,
            1: wrapper1,
        }
        with lsst.utils.tests.getTempFilePath(".py") as filePath:
            self.checkConfig(multiClass, multiConfig, filePath)
            multiXYTransform = multiClass(multiConfig)

            affine0 = affineClass(affineConfig0)
            affine1 = affineClass(affineConfig1)
            transformList = (affine0, affine1)
            refMultiXYTransform = RefMultiXYTransform(transformList)

            self.checkBasics(refMultiXYTransform)

            for fromPoint in self.fromIter():
                toPoint = multiXYTransform.forwardTransform(fromPoint)
                predToPoint = refMultiXYTransform.forwardTransform(fromPoint)
                for i in range(2):
                    self.assertAlmostEqual(toPoint[i], predToPoint[i])


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

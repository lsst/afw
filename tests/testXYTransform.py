#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import range
#pybind11#from builtins import object
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2014 LSST Corporation.
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
#pybind11#"""
#pybind11#Tests for lsst.afw.geom.XYTransform and xyTransformRegistry
#pybind11#"""
#pybind11#import itertools
#pybind11#import math
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#from lsst.afw.geom import Extent2D, Point2D, xyTransformRegistry, OneXYTransformConfig, \
#pybind11#    IdentityXYTransform, AffineXYTransform, RadialXYTransform
#pybind11#
#pybind11#
#pybind11#class RefMultiAffineTransform(object):
#pybind11#
#pybind11#    def __init__(self, affineTransformList):
#pybind11#        self.affineTransformList = affineTransformList
#pybind11#
#pybind11#    def __call__(self, point):
#pybind11#        for tr in self.affineTransformList:
#pybind11#            point = tr(point)
#pybind11#        return point
#pybind11#
#pybind11#
#pybind11#class RefMultiXYTransform(object):
#pybind11#
#pybind11#    def __init__(self, transformList):
#pybind11#        self.transformList = transformList
#pybind11#
#pybind11#    def forwardTransform(self, point):
#pybind11#        for tr in self.transformList:
#pybind11#            point = tr.forwardTransform(point)
#pybind11#        return point
#pybind11#
#pybind11#    def reverseTransform(self, point):
#pybind11#        for tr in reversed(self.transformList):
#pybind11#            point = tr.reverseTransform(point)
#pybind11#        return point
#pybind11#
#pybind11#    def linearizeForwardTransform(self, point):
#pybind11#        affineTransformList = [tr.linearizeForwardTransform(point) for tr in self.transformList]
#pybind11#        return RefMultiAffineTransform(affineTransformList)
#pybind11#
#pybind11#    def linearizeReverseTransform(self, point):
#pybind11#        affineTransformList = [tr.linearizeReverseTransform(point) for tr in reversed(self.transformList)]
#pybind11#        return RefMultiAffineTransform(affineTransformList)
#pybind11#
#pybind11#
#pybind11#class XYTransformTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def fromIter(self):
#pybind11#        for x in (-1.1, 0, 2.2):
#pybind11#            for y in (3.1, 0, 2.1):
#pybind11#                yield Point2D(x, y)
#pybind11#
#pybind11#    def checkBasics(self, transform):
#pybind11#        """Check round trip and linearization of transform
#pybind11#        """
#pybind11#        for fromPoint in self.fromIter():
#pybind11#            toPoint = transform.forwardTransform(fromPoint)
#pybind11#            roundTripPoint = transform.reverseTransform(toPoint)
#pybind11#            for i in range(2):
#pybind11#                self.assertAlmostEqual(fromPoint[i], roundTripPoint[i])
#pybind11#
#pybind11#            for deltaFrom in (
#pybind11#                Extent2D(0),
#pybind11#                Extent2D(0.1, -0.1),
#pybind11#                Extent2D(-0.15, 0.1),
#pybind11#            ):
#pybind11#                tweakedFromPoint = fromPoint + deltaFrom
#pybind11#                tweakedToPoint = transform.forwardTransform(tweakedFromPoint)
#pybind11#                linToPoint = transform.linearizeForwardTransform(fromPoint)(tweakedFromPoint)
#pybind11#                linRoundTripPoint = transform.linearizeReverseTransform(toPoint)(tweakedToPoint)
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEqual(tweakedToPoint[i], linToPoint[i], places=2)
#pybind11#                    self.assertAlmostEqual(tweakedFromPoint[i], linRoundTripPoint[i], places=2)
#pybind11#
#pybind11#    def checkConfig(self, tClass, tConfig, filePath):
#pybind11#        """Check round trip of config
#pybind11#        """
#pybind11#        tConfig.save(filePath)
#pybind11#        loadConfig = tConfig.__class__()
#pybind11#        loadConfig.load(filePath)
#pybind11#        transform = tClass(loadConfig)
#pybind11#        self.checkBasics(transform)
#pybind11#
#pybind11#    def testIdentity(self):
#pybind11#        """Test identity = IdentityXYTransform
#pybind11#        """
#pybind11#        identClass = xyTransformRegistry["identity"]
#pybind11#        with lsst.utils.tests.getTempFilePath(".py") as filePath:
#pybind11#            self.checkConfig(identClass, identClass.ConfigClass(), filePath)
#pybind11#            ident = identClass(identClass.ConfigClass())
#pybind11#            self.assertEqual(type(ident), IdentityXYTransform)
#pybind11#            self.checkBasics(ident)
#pybind11#            for fromPoint in self.fromIter():
#pybind11#                toPoint = ident.forwardTransform(fromPoint)
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEqual(fromPoint[i], toPoint[i])
#pybind11#
#pybind11#    def testInverted(self):
#pybind11#        """Test inverted = InvertedXYTransform
#pybind11#        """
#pybind11#        invertedClass = xyTransformRegistry["inverted"]
#pybind11#        invertedConfig = invertedClass.ConfigClass()
#pybind11#        affineClass = xyTransformRegistry["affine"]
#pybind11#        invertedConfig.transform.retarget(affineClass)
#pybind11#        affineConfig = invertedConfig.transform
#pybind11#        affineConfig.translation = (1.2, -3.4)
#pybind11#        with lsst.utils.tests.getTempFilePath(".py") as filePath:
#pybind11#            self.checkConfig(invertedClass, invertedConfig, filePath)
#pybind11#            inverted = invertedClass(invertedConfig)
#pybind11#            self.checkBasics(inverted)
#pybind11#            for fromPoint in self.fromIter():
#pybind11#                toPoint = inverted.forwardTransform(fromPoint)
#pybind11#                predToPoint = fromPoint - Extent2D(*invertedConfig.transform.translation)
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEqual(toPoint[i], predToPoint[i])
#pybind11#
#pybind11#    def testDefaultAffine(self):
#pybind11#        """Test affine = AffineXYTransform with default coeffs (identity transform)
#pybind11#        """
#pybind11#        affineClass = xyTransformRegistry["affine"]
#pybind11#        affineConfig = affineClass.ConfigClass()
#pybind11#        with lsst.utils.tests.getTempFilePath(".py") as filePath:
#pybind11#            self.checkConfig(affineClass, affineConfig, filePath)
#pybind11#            affine = affineClass(affineConfig)
#pybind11#            self.assertEqual(type(affine), AffineXYTransform)
#pybind11#            self.checkBasics(affine)
#pybind11#            for fromPoint in self.fromIter():
#pybind11#                toPoint = affine.forwardTransform(fromPoint)
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEqual(fromPoint[i], toPoint[i])
#pybind11#
#pybind11#    def testTranslateAffine(self):
#pybind11#        """Test affine = AffineXYTransform with just translation coefficients
#pybind11#        """
#pybind11#        affineClass = xyTransformRegistry["affine"]
#pybind11#        affineConfig = affineClass.ConfigClass()
#pybind11#        affineConfig.translation = (1.2, -3.4)
#pybind11#        with lsst.utils.tests.getTempFilePath(".py") as filePath:
#pybind11#            self.checkConfig(affineClass, affineConfig, filePath)
#pybind11#            affine = affineClass(affineConfig)
#pybind11#            for fromPoint in self.fromIter():
#pybind11#                toPoint = affine.forwardTransform(fromPoint)
#pybind11#                predToPoint = fromPoint + Extent2D(*affineConfig.translation)
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEqual(toPoint[i], predToPoint[i])
#pybind11#
#pybind11#    def testLinearAffine(self):
#pybind11#        """Test affine = AffineXYTransform with just linear coefficients
#pybind11#        """
#pybind11#        affineClass = xyTransformRegistry["affine"]
#pybind11#        affineConfig = affineClass.ConfigClass()
#pybind11#        rotAng = 0.25  # radians
#pybind11#        xScale = 1.2
#pybind11#        yScale = 0.8
#pybind11#        affineConfig.linear = (
#pybind11#            math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
#pybind11#            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
#pybind11#        )
#pybind11#        with lsst.utils.tests.getTempFilePath(".py") as filePath:
#pybind11#            self.checkConfig(affineClass, affineConfig, filePath)
#pybind11#            affine = affineClass(affineConfig)
#pybind11#            for fromPoint in self.fromIter():
#pybind11#                toPoint = affine.forwardTransform(fromPoint)
#pybind11#                predToPoint = Point2D(
#pybind11#                    affineConfig.linear[0] * fromPoint[0] + affineConfig.linear[1] * fromPoint[1],
#pybind11#                    affineConfig.linear[2] * fromPoint[0] + affineConfig.linear[3] * fromPoint[1],
#pybind11#                )
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEqual(toPoint[i], predToPoint[i])
#pybind11#
#pybind11#    def testFullAffine(self):
#pybind11#        """Test affine = AffineXYTransform with just linear coefficients
#pybind11#        """
#pybind11#        affineClass = xyTransformRegistry["affine"]
#pybind11#        affineConfig = affineClass.ConfigClass()
#pybind11#        affineConfig.translation = (-2.1, 3.4)
#pybind11#        rotAng = 0.832  # radians
#pybind11#        xScale = 3.7
#pybind11#        yScale = 45.3
#pybind11#        affineConfig.linear = (
#pybind11#            math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
#pybind11#            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
#pybind11#        )
#pybind11#        with lsst.utils.tests.getTempFilePath(".py") as filePath:
#pybind11#            self.checkConfig(affineClass, affineConfig, filePath)
#pybind11#            affine = affineClass(affineConfig)
#pybind11#            for fromPoint in self.fromIter():
#pybind11#                toPoint = affine.forwardTransform(fromPoint)
#pybind11#                predToPoint = Point2D(
#pybind11#                    affineConfig.linear[0] * fromPoint[0] + affineConfig.linear[1] * fromPoint[1],
#pybind11#                    affineConfig.linear[2] * fromPoint[0] + affineConfig.linear[3] * fromPoint[1],
#pybind11#                )
#pybind11#                predToPoint = predToPoint + Extent2D(*affineConfig.translation)
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEqual(toPoint[i], predToPoint[i])
#pybind11#
#pybind11#    def testRadial(self):
#pybind11#        """Test radial = RadialXYTransform
#pybind11#        """
#pybind11#        radialClass = xyTransformRegistry["radial"]
#pybind11#        radialConfig = radialClass.ConfigClass()
#pybind11#        radialConfig.coeffs = [0, 1.05, 0.1]
#pybind11#        with lsst.utils.tests.getTempFilePath(".py") as filePath:
#pybind11#            self.checkConfig(radialClass, radialConfig, filePath)
#pybind11#            radial = radialClass(radialConfig)
#pybind11#            self.assertEqual(type(radial), RadialXYTransform)
#pybind11#            self.assertEqual(len(radial.getCoeffs()), len(radialConfig.coeffs))
#pybind11#            for coeff, predCoeff in zip(radial.getCoeffs(), radialConfig.coeffs):
#pybind11#                self.assertAlmostEqual(coeff, predCoeff)
#pybind11#            self.checkBasics(radial)
#pybind11#            for fromPoint in self.fromIter():
#pybind11#                fromRadius = math.hypot(fromPoint[0], fromPoint[1])
#pybind11#                fromAngle = math.atan2(fromPoint[1], fromPoint[0])
#pybind11#                predToRadius = fromRadius * (radialConfig.coeffs[2] * fromRadius + radialConfig.coeffs[1])
#pybind11#                predToPoint = Point2D(predToRadius * math.cos(fromAngle), predToRadius * math.sin(fromAngle))
#pybind11#                toPoint = radial.forwardTransform(fromPoint)
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEqual(toPoint[i], predToPoint[i])
#pybind11#
#pybind11#    def testBadRadial(self):
#pybind11#        """Test radial with invalid coefficients
#pybind11#        """
#pybind11#        for badCoeffs in (
#pybind11#            [0.1,],     # len(coeffs) must be > 1
#pybind11#            [0.1, 1.0],  # coeffs[0] must be zero
#pybind11#            [0.0, 0.0],  # coeffs[1] must be nonzero
#pybind11#            [0.0, 0.0, 0.1],  # coeffs[1] must be nonzero
#pybind11#        ):
#pybind11#            with self.assertRaises(lsst.pex.exceptions.Exception):
#pybind11#                RadialXYTransform(badCoeffs)
#pybind11#
#pybind11#            radialClass = xyTransformRegistry["radial"]
#pybind11#            radialConfig = radialClass.ConfigClass()
#pybind11#            radialConfig.coeffs = badCoeffs
#pybind11#            with self.assertRaises(Exception):
#pybind11#                radialConfig.validate()
#pybind11#
#pybind11#    def testMulti(self):
#pybind11#        """Test multi = MultiXYTransform
#pybind11#        """
#pybind11#        affineClass = xyTransformRegistry["affine"]
#pybind11#        wrapper0 = OneXYTransformConfig()
#pybind11#        wrapper0.transform.retarget(affineClass)
#pybind11#        affineConfig0 = wrapper0.transform
#pybind11#        affineConfig0.translation = (-2.1, 3.4)
#pybind11#        rotAng = 0.832  # radians
#pybind11#        xScale = 3.7
#pybind11#        yScale = 45.3
#pybind11#        affineConfig0.linear = (
#pybind11#            math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
#pybind11#            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
#pybind11#        )
#pybind11#
#pybind11#        wrapper1 = OneXYTransformConfig()
#pybind11#        wrapper1.transform.retarget(affineClass)
#pybind11#        affineConfig1 = wrapper1.transform
#pybind11#        affineConfig1.translation = (26.5, -35.1)
#pybind11#        rotAng = -0.25  # radians
#pybind11#        xScale = 1.45
#pybind11#        yScale = 0.9
#pybind11#        affineConfig1.linear = (
#pybind11#            math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
#pybind11#            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
#pybind11#        )
#pybind11#
#pybind11#        multiClass = xyTransformRegistry["multi"]
#pybind11#        multiConfig = multiClass.ConfigClass()
#pybind11#        multiConfig.transformDict = {
#pybind11#            0: wrapper0,
#pybind11#            1: wrapper1,
#pybind11#        }
#pybind11#        with lsst.utils.tests.getTempFilePath(".py") as filePath:
#pybind11#            self.checkConfig(multiClass, multiConfig, filePath)
#pybind11#            multiXYTransform = multiClass(multiConfig)
#pybind11#
#pybind11#            affine0 = affineClass(affineConfig0)
#pybind11#            affine1 = affineClass(affineConfig1)
#pybind11#            transformList = (affine0, affine1)
#pybind11#            refMultiXYTransform = RefMultiXYTransform(transformList)
#pybind11#
#pybind11#            self.checkBasics(refMultiXYTransform)
#pybind11#
#pybind11#            for fromPoint in self.fromIter():
#pybind11#                toPoint = multiXYTransform.forwardTransform(fromPoint)
#pybind11#                predToPoint = refMultiXYTransform.forwardTransform(fromPoint)
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEqual(toPoint[i], predToPoint[i])
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

#
# LSST Data Management System
# Copyright 2017 LSST Corporation.
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

"""Tests for custom Transforms and their factories
"""

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from astshim.test import makeForwardPolyMap

import lsst.afw.geom as afwGeom
from lsst.afw.geom.testUtils import TransformTestBaseClass
import lsst.pex.exceptions as pexExcept
import lsst.utils.tests


class TransformFactoryTestSuite(TransformTestBaseClass):

    def setUp(self):
        TransformTestBaseClass.setUp(self)
        self.endpointPrefixes = tuple(
            x for x in self.endpointPrefixes if x != "SpherePoint")

    def point2DList(self):
        for x in (-1.1, 0, 2.2):
            for y in (3.1, 0, 2.1):
                yield afwGeom.Point2D(x, y)

    def testLinearize(self):
        for transform, invertible in (
            (afwGeom.TransformPoint2ToPoint2(makeForwardPolyMap(2, 2)), False),
            (afwGeom.makeIdentityTransform(), True),
            (afwGeom.makeTransform(afwGeom.AffineTransform(np.array([[3.0, -2.0], [2.0, -1.0]]))), True),
            (afwGeom.makeRadialTransform([0.0, 8.0e-05, 0.0, -4.5e-12]), True),
        ):
            self.checkLinearize(transform, invertible)

    def checkLinearize(self, transform, invertible):
        """Test whether a specific transform is correctly linearized.

        Parameters
        ----------
        transform: `lsst.afw.geom.Transform`
            the transform whose linearization will be tested. Should not be
            strongly curved within ~1 unit of the origin, or the test may rule
            the approximation isn't good enough.
        invertible: `bool`
            whether `transform` is invertible. The test will verify that the
            linearized form is invertible iff `transform` is. If `transform`
            is invertible, the test will also verify that the inverse of the
            linearization approximates the inverse of `transform`.
        """
        fromEndpoint = transform.fromEndpoint
        toEndpoint = transform.toEndpoint
        nIn = fromEndpoint.nAxes
        nOut = toEndpoint.nAxes
        msg = "TransformClass={}, nIn={}, nOut={}".format(type(transform).__name__, nIn, nOut)

        rawLinPoint = self.makeRawPointData(nIn)
        linPoint = fromEndpoint.pointFromData(rawLinPoint)
        affine = afwGeom.linearizeTransform(transform, linPoint)
        self.assertIsInstance(affine, afwGeom.AffineTransform)

        # Does affine match exact transform at linPoint?
        outPoint = transform.applyForward(linPoint)
        outPointLinearized = affine(linPoint)
        assert_allclose(toEndpoint.dataFromPoint(outPoint),
                        toEndpoint.dataFromPoint(outPointLinearized),
                        err_msg=msg)
        jacobian = transform.getJacobian(linPoint)
        jacobianLinearized = affine.getLinear().getMatrix()
        assert_allclose(jacobian, jacobianLinearized)

        # Is affine a local approximation around linPoint?
        for deltaFrom in (
            np.zeros(nIn),
            np.full(nIn, 0.1),
            np.array([0.1, -0.15, 0.20, -0.05, 0.0, -0.1][0:nIn])
        ):
            tweakedInPoint = fromEndpoint.pointFromData(
                rawLinPoint + deltaFrom)
            tweakedOutPoint = transform.applyForward(tweakedInPoint)
            tweakedOutPointLinearized = affine(tweakedInPoint)
            assert_allclose(
                toEndpoint.dataFromPoint(tweakedOutPoint),
                toEndpoint.dataFromPoint(tweakedOutPointLinearized),
                atol=1e-3,
                err_msg=msg)

        # Is affine invertible?
        # AST lets all-zero MatrixMaps be invertible though inverse
        # ill-defined; exclude this case
        if invertible:
            rng = np.random.RandomState(42)
            nDelta = 100
            inverse = affine.invert()
            deltaFrom = rng.normal(0.0, 10.0, (nIn, nDelta))
            for i in range(nDelta):
                pointMsg = "{}, point={}".format(msg, tweakedInPoint)
                tweakedInPoint = fromEndpoint.pointFromData(
                    rawLinPoint + deltaFrom[:, i])
                tweakedOutPoint = affine(tweakedInPoint)

                roundTrip = inverse(tweakedOutPoint)
                assert_allclose(
                    roundTrip, tweakedInPoint,
                    err_msg=pointMsg)
                assert_allclose(
                    inverse.getLinear().getMatrix(),
                    np.linalg.inv(jacobian),
                    err_msg=pointMsg)
        else:
            # TODO: replace with correct type after fixing DM-11248
            with self.assertRaises(Exception):
                affine.invert()

        # Can't test exceptions without reliable way to make invalid transform

    def checkGenericTransform(self, tFactory, tConfig, transformCheck=None,
                              **kwargs):
        """Check Transform by building it from a factory.
        """
        self.checkConfig(tFactory, tConfig, transformCheck, **kwargs)

        with lsst.utils.tests.getTempFilePath(".py") as filePath:
            tConfig.save(filePath)
            loadConfig = tConfig.__class__()
            loadConfig.load(filePath)
            self.checkConfig(tFactory, loadConfig, transformCheck, **kwargs)

    def checkConfig(self, tFactory, tConfig, transformCheck, **kwargs):
        """Check Transform built from a particular config
        """
        transform = tFactory(tConfig)
        self.checkRoundTrip(transform, **kwargs)
        if transformCheck is not None:
            transformCheck(transform)

    def checkRoundTrip(self, transform, **kwargs):
        """Check round trip of transform
        """
        for fromPoint in self.point2DList():
            toPoint = transform.applyForward(fromPoint)
            roundTripPoint = transform.applyInverse(toPoint)
            # Don't let NaNs pass the test!
            assert_allclose(roundTripPoint, fromPoint, atol=1e-14, **kwargs)

    def testIdentity(self):
        """Test identity transform.
        """
        identFactory = afwGeom.transformRegistry["identity"]
        identConfig = identFactory.ConfigClass()
        self.checkGenericTransform(identFactory, identConfig,
                                   self.checkIdentity)

    def checkIdentity(self, transform):
        for fromPoint in self.point2DList():
            toPoint = transform.applyForward(fromPoint)
            self.assertPairsAlmostEqual(fromPoint, toPoint)

    def testDefaultAffine(self):
        """Test affine = affine Transform with default coeffs (identity transform)
        """
        affineFactory = afwGeom.transformRegistry["affine"]
        affineConfig = affineFactory.ConfigClass()
        self.checkGenericTransform(affineFactory, affineConfig,
                                   self.checkIdentity)

    def testTranslateAffine(self):
        """Test affine = affine Transform with just translation coefficients
        """
        affineFactory = afwGeom.transformRegistry["affine"]
        affineConfig = affineFactory.ConfigClass()
        affineConfig.translation = (1.2, -3.4)

        def check(transform):
            self.checkTranslateAffine(
                transform,
                afwGeom.Extent2D(*affineConfig.translation))
        self.checkGenericTransform(affineFactory, affineConfig, check)

    def checkTranslateAffine(self, transform, offset):
        for fromPoint in self.point2DList():
            toPoint = transform.applyForward(fromPoint)
            predToPoint = fromPoint + offset
            self.assertPairsAlmostEqual(toPoint, predToPoint)

    def testLinearAffine(self):
        """Test affine = affine Transform with just linear coefficients
        """
        affineFactory = afwGeom.transformRegistry["affine"]
        affineConfig = affineFactory.ConfigClass()
        rotAng = 0.25  # radians
        xScale = 1.2
        yScale = 0.8
        affineConfig.linear = (
            math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
        )

        def check(transform):
            self.checkLinearAffine(transform, affineConfig.linear)
        self.checkGenericTransform(affineFactory, affineConfig, check)

    def checkLinearAffine(self, transform, matrix):
        for fromPoint in self.point2DList():
            toPoint = transform.applyForward(fromPoint)
            predToPoint = afwGeom.Point2D(
                matrix[0] * fromPoint[0] +
                matrix[1] * fromPoint[1],
                matrix[2] * fromPoint[0] +
                matrix[3] * fromPoint[1],
            )
            self.assertPairsAlmostEqual(toPoint, predToPoint)

    def testFullAffine(self):
        """Test affine = affine Transform with arbitrary coefficients
        """
        affineFactory = afwGeom.transformRegistry["affine"]
        affineConfig = affineFactory.ConfigClass()
        affineConfig.translation = (-2.1, 3.4)
        rotAng = 0.832  # radians
        xScale = 3.7
        yScale = 45.3
        affineConfig.linear = (
            math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
        )

        def check(transform):
            self.checkFullAffine(
                transform,
                afwGeom.Extent2D(*affineConfig.translation),
                affineConfig.linear)
        self.checkGenericTransform(affineFactory, affineConfig, check)

    def checkFullAffine(self, transform, offset, matrix):
            for fromPoint in self.point2DList():
                toPoint = transform.applyForward(fromPoint)
                predToPoint = afwGeom.Point2D(
                    matrix[0] * fromPoint[0] +
                    matrix[1] * fromPoint[1],
                    matrix[2] * fromPoint[0] +
                    matrix[3] * fromPoint[1],
                )
                predToPoint = predToPoint + offset
                self.assertPairsAlmostEqual(toPoint, predToPoint)

    def testRadial(self):
        """Test radial = radial Transform
        """
        radialFactory = afwGeom.transformRegistry["radial"]
        radialConfig = radialFactory.ConfigClass()
        radialConfig.coeffs = (0.0, 8.5165e-05, 0.0, -4.5014e-12)

        def check(transform):
            self.checkRadial(transform, radialConfig.coeffs)
        self.checkGenericTransform(radialFactory, radialConfig, check)

        invertibleCoeffs = (0.0, 1.0, 0.05)
        inverseCoeffs = (0.0, 1.0, -0.05, 0.005, -0.000625, 0.0000875,
                         -1.3125e-5, 2.0625e-6, -3.3515625e-7, 5.5859375e-8,
                         -9.49609375e-9, 1.640234375e-9, -2.870410156e-10)
        transform = afwGeom.makeRadialTransform(invertibleCoeffs,
                                                inverseCoeffs)
        self.checkRadialInvertible(transform, invertibleCoeffs)

    def checkRadial(self, transform, coeffs):
        if len(coeffs) < 4:
            coeffs = tuple(coeffs + (0.0,) * (4 - len(coeffs)))
        for fromPoint in self.point2DList():
            fromRadius = math.hypot(fromPoint[0], fromPoint[1])
            fromAngle = math.atan2(fromPoint[1], fromPoint[0])
            predToRadius = fromRadius * \
                (coeffs[3] * fromRadius**2 + coeffs[2] * fromRadius + coeffs[1])
            if predToRadius > 0:
                predToPoint = afwGeom.Point2D(
                    predToRadius * math.cos(fromAngle),
                    predToRadius * math.sin(fromAngle))
            else:
                predToPoint = afwGeom.Point2D()
            toPoint = transform.applyForward(fromPoint)
            # Don't let NaNs pass the test!
            assert_allclose(toPoint, predToPoint, atol=1e-14)

    def checkRadialInvertible(self, transform, coeffs):
        self.checkRadial(transform, coeffs)
        self.checkRoundTrip(transform, rtol=0.01)

    def testBadRadial(self):
        """Test radial with invalid coefficients
        """
        for badCoeffs in (
            (0.0,),     # len(coeffs) must be > 1
            (0.1, 1.0),  # coeffs[0] must be zero
            (0.0, 0.0),  # coeffs[1] must be nonzero
            (0.0, 0.0, 0.1),  # coeffs[1] must be nonzero
        ):
            with self.assertRaises(pexExcept.InvalidParameterError):
                afwGeom.makeRadialTransform(badCoeffs)

            radialFactory = afwGeom.transformRegistry["radial"]
            radialConfig = radialFactory.ConfigClass()
            radialConfig.coeffs = badCoeffs
            with self.assertRaises(Exception):
                radialConfig.validate()

    def testInverted(self):
        """Test radial = radial Transform
        """
        affineFactory = afwGeom.transformRegistry["affine"]
        wrapper = afwGeom.OneTransformConfig()
        wrapper.transform.retarget(affineFactory)
        affineConfig = wrapper.transform
        affineConfig.translation = (-2.1, 3.4)
        rotAng = 0.832  # radians
        xScale = 3.7
        yScale = 45.3
        affineConfig.linear = (
            math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
        )

        inverseFactory = afwGeom.transformRegistry["inverted"]
        inverseConfig = inverseFactory.ConfigClass()
        inverseConfig.transform = affineConfig

        def check(transform):
            self.checkInverted(transform, affineConfig.apply())
        self.checkGenericTransform(inverseFactory, inverseConfig, check)

    def checkInverted(self, transform, original):
        for fromPoint in self.point2DList():
            toPoint = transform.applyForward(fromPoint)
            predToPoint = original.applyInverse(fromPoint)
            self.assertPairsAlmostEqual(toPoint, predToPoint)
            roundTrip = transform.applyInverse(toPoint)
            predRoundTrip = original.applyForward(toPoint)
            self.assertPairsAlmostEqual(roundTrip, predRoundTrip)

    def testMulti(self):
        """Test multi transform
        """
        affineFactory = afwGeom.transformRegistry["affine"]
        wrapper0 = afwGeom.OneTransformConfig()
        wrapper0.transform.retarget(affineFactory)
        affineConfig0 = wrapper0.transform
        affineConfig0.translation = (-2.1, 3.4)
        rotAng = 0.832  # radians
        xScale = 3.7
        yScale = 45.3
        affineConfig0.linear = (
            math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
        )

        wrapper1 = afwGeom.OneTransformConfig()
        wrapper1.transform.retarget(affineFactory)
        affineConfig1 = wrapper1.transform
        affineConfig1.translation = (26.5, -35.1)
        rotAng = -0.25  # radians
        xScale = 1.45
        yScale = 0.9
        affineConfig1.linear = (
            math.cos(rotAng) * xScale, math.sin(rotAng) * yScale,
            -math.sin(rotAng) * xScale, math.cos(rotAng) * yScale,
        )

        multiFactory = afwGeom.transformRegistry["multi"]
        multiConfig = multiFactory.ConfigClass()
        multiConfig.transformDict = {
            0: wrapper0,
            1: wrapper1,
        }

        def check(transform):
            self.checkMulti(transform,
                            [c.apply() for c in
                             [affineConfig0, affineConfig1]])
        self.checkGenericTransform(multiFactory, multiConfig, check)

    def checkMulti(self, multiTransform, transformList):
        for fromPoint in self.point2DList():
            toPoint = multiTransform.applyForward(fromPoint)
            predToPoint = fromPoint
            for transform in transformList:
                predToPoint = transform.applyForward(predToPoint)
            self.assertPairsAlmostEqual(toPoint, predToPoint)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

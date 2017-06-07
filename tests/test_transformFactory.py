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

from __future__ import absolute_import, division, print_function
import itertools
import unittest

import numpy as np
from numpy.testing import assert_allclose

from astshim.test import makeForwardPolyMap

import lsst.utils.tests
import lsst.afw.geom as afwGeom
from lsst.afw.geom.testUtils import TransformTestBaseClass


class transformFactoryTestSuite(TransformTestBaseClass):

    def setUp(self):
        TransformTestBaseClass.setUp(self)
        self.endpointPrefixes = tuple(
            x for x in self.endpointPrefixes if x != "SpherePoint")

    def testLinearize(self):
        for fromName in self.endpointPrefixes:
            for toName in self.endpointPrefixes:
                self.checkLinearization(fromName, toName)

    def checkLinearization(self, fromName, toName):
        transformClassName = "Transform{}To{}".format(fromName, toName)
        TransformClass = getattr(afwGeom, transformClassName)
        baseMsg = "TransformClass={}".format(TransformClass.__name__)

        for nIn, nOut in itertools.product(self.goodNAxes[fromName],
                                           self.goodNAxes[toName]):
            msg = "{}, nIn={}, nOut={}".format(baseMsg, nIn, nOut)
            polyMap = makeForwardPolyMap(nIn, nOut)
            transform = TransformClass(polyMap)
            fromEndpoint = transform.fromEndpoint
            toEndpoint = transform.toEndpoint

            rawLinPoint = self.makeRawPointData(nIn)
            linPoint = fromEndpoint.pointFromData(rawLinPoint)
            linearized = afwGeom.linearizeTransform(transform, linPoint)

            # Does linearized match exact transform at linPoint?
            outPoint = transform.applyForward(linPoint)
            outPointLinearized = linearized.applyForward(linPoint)
            assert_allclose(toEndpoint.dataFromPoint(outPoint),
                            toEndpoint.dataFromPoint(outPointLinearized),
                            err_msg=msg)
            # First derivative will be tested in next section

            # Is linearized linear?
            # Test that jacobian always has the same value, and also matches
            # Jacobian of original transform at linPoint
            jacobian = transform.getJacobian(linPoint)
            rng = np.random.RandomState(42)
            nDelta = 100
            deltaFrom = rng.normal(0.0, 10.0, (nIn, nDelta))
            for i in range(nDelta):
                tweakedInPoint = fromEndpoint.pointFromData(
                    rawLinPoint + deltaFrom[:, i])
                assert_allclose(jacobian,
                                linearized.getJacobian(tweakedInPoint),
                                err_msg="{}, point={}".format(
                                    msg, tweakedInPoint))

            # Is linearized a local approximation around linPoint?
            for deltaFrom in (
                np.zeros(nIn),
                np.full(nIn, 0.1),
                np.array([0.1, -0.15, 0.20, -0.05, 0.0, -0.1][0:nIn])
            ):
                tweakedInPoint = fromEndpoint.pointFromData(
                    rawLinPoint + deltaFrom)
                tweakedOutPoint = transform.applyForward(tweakedInPoint)
                tweakedOutPointLinearized = linearized.applyForward(
                    tweakedInPoint)
                assert_allclose(
                    toEndpoint.dataFromPoint(tweakedOutPoint),
                    toEndpoint.dataFromPoint(tweakedOutPointLinearized),
                    atol=1e-3,
                    err_msg=msg)

        # Can't test exceptions without reliable way to make invalid transform


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

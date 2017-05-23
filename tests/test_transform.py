"""
LSST Data Management System
See COPYRIGHT file at the top of the source tree.

This product includes software developed by the
LSST Project (http://www.lsst.org/).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the LSST License Statement and
the GNU General Public License along with this program. If not,
see <http://www.lsstcorp.org/LegalNotices/>.
"""
from __future__ import absolute_import, division, print_function
import unittest

from numpy.testing import assert_allclose
from astshim.test import makeForwardPolyMap

import lsst.utils.tests
import lsst.afw.geom as afwGeom
from lsst.afw.geom.testUtils import TransformTestBaseClass


class TransformTestCase(TransformTestBaseClass):

    def testTransforms(self):
        for fromName in self.endpointPrefixes:
            for toName in self.endpointPrefixes:
                self.checkTransformFromMapping(fromName, toName)
                self.checkTransformFromFrameSet(fromName, toName)
                self.checkGetInverse(fromName, toName)
                self.checkGetJacobian(fromName, toName)
                for midName in self.endpointPrefixes:
                    self.checkOf(fromName, midName, toName)

    def testFrameSetIndependence(self):
        """Test that the FrameSet returned by getFrameSet is independent of the contained FrameSet
        """
        baseFrame = self.makeGoodFrame("Generic", 2)
        currFrame = self.makeGoodFrame("Generic", 2)
        initialFrameSet = self.makeFrameSet(baseFrame, currFrame)
        initialIdent = "Initial Ident"
        initialFrameSet.ident = initialIdent
        transform = afwGeom.TransformGenericToGeneric(initialFrameSet)
        extractedFrameSet = transform.getFrameSet()
        extractedFrameSet.ident = "Extracted Ident"
        self.assertEqual(initialIdent, transform.getFrameSet().ident)

    def testOfChaining(self):
        """Test that the order of chaining Transform.of does not matter

        Test that C.of(B.of(A)) gives the same transformation as (C.of(B)).of(A)
        Internal details may differ (e.g. frame indices if the frames in the
        contained FrameSet), but the mathematical result of the two transforms
        should be the same.
        """
        transform1 = afwGeom.TransformGenericToGeneric(
            makeForwardPolyMap(2, 3))
        transform2 = afwGeom.TransformGenericToGeneric(
            makeForwardPolyMap(3, 4))
        transform3 = afwGeom.TransformGenericToGeneric(
            makeForwardPolyMap(4, 1))

        merged1 = transform3.of(transform2).of(transform1)
        merged2 = transform3.of(transform2.of(transform1))

        fromEndpoint = transform1.fromEndpoint
        toEndpoint = transform3.toEndpoint

        inPoint = fromEndpoint.pointFromData(self.makeRawPointData(2))
        assert_allclose(toEndpoint.dataFromPoint(merged1.tranForward(inPoint)),
                        toEndpoint.dataFromPoint(merged2.tranForward(inPoint)))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

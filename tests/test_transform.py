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
import astshim as ast
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
                    self.checkThen(fromName, midName, toName)

    def testMappingIndependence(self):
        """Test that the mapping returned by getMapping is independent of the contained mapping
        """
        initialMapping = ast.ZoomMap(2, 1.5)
        initialIdent = "Initial Ident"
        initialMapping.ident = initialIdent
        transform = afwGeom.TransformGenericToGeneric(initialMapping)
        extractedMapping = transform.getMapping()
        extractedMapping.ident = "Extracted Ident"
        self.assertEqual(initialIdent, transform.getMapping().ident)

    def testThen(self):
        """Test that Transform.then behaves as expected
        """
        transform1 = afwGeom.TransformGenericToGeneric(
            makeForwardPolyMap(2, 3))
        transform2 = afwGeom.TransformGenericToGeneric(
            makeForwardPolyMap(3, 4))

        for simplify in (False, True):
            merged = transform1.then(transform2, simplify = simplify)

            inPoint = self.makeRawPointData(2)
            assert_allclose(merged.applyForward(inPoint),
                            transform2.applyForward(transform1.applyForward(inPoint)))

    def testThenChaining(self):
        """Test that the order of chaining Transform.then does not matter

        Test that A.then(B.then(C)) gives the same transformation as
        (A.then(B)).then(C)
        Internal details may differ, but the mathematical result of the two
        transforms should be the same.
        """
        transform1 = afwGeom.TransformGenericToGeneric(
            makeForwardPolyMap(2, 3))
        transform2 = afwGeom.TransformGenericToGeneric(
            makeForwardPolyMap(3, 4))
        transform3 = afwGeom.TransformGenericToGeneric(
            makeForwardPolyMap(4, 1))

        merged1 = transform1.then(transform2.then(transform3))
        merged2 = transform1.then(transform2).then(transform3)

        inPoint = self.makeRawPointData(2)
        assert_allclose(merged1.applyForward(inPoint),
                        merged2.applyForward(inPoint))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

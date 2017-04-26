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
        self.checkOfChaining()

    def testFrameSetIndependence(self):
        """Test that the FrameSet returned by getFrameSet is independent of the contained FrameSet
        """
        baseFrame = self.makeGoodFrame("Generic", 2)
        currFrame = self.makeGoodFrame("Generic", 2)
        initialFrameSet = self.makeFrameSet(baseFrame, currFrame)
        initialIdent = "Initial Ident"
        initialFrameSet.setIdent(initialIdent)
        transform = afwGeom.TransformGenericToGeneric(initialFrameSet)
        extractedFrameSet = transform.getFrameSet()
        extractedFrameSet.setIdent("Extracted Ident")
        self.assertEqual(initialIdent, transform.getFrameSet().getIdent())


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

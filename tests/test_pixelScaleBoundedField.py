# This file is part of afw.

# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest

import numpy as np

import lsst.utils.tests
import lsst.geom
import lsst.afw.geom
from lsst.afw.math import PixelScaleBoundedField


class PixelScaleBoundedFieldTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        # Trivial WCS:
        crpix = lsst.geom.Point2D(100, 100)
        crval = lsst.geom.SpherePoint(0, 45, lsst.geom.degrees)
        cdMatrix = lsst.afw.geom.makeCdMatrix(1*lsst.geom.arcseconds, 0*lsst.geom.degrees)
        self.skyWcs = lsst.afw.geom.makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix)

        self.bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(200, 200))
        self.points = [crpix, ]
        # ensure that the points are all Point2D
        self.points.extend(lsst.geom.Box2D(self.bbox).getCorners())

        self.boundedField = PixelScaleBoundedField(self.bbox, self.skyWcs)

    def testConstruct(self):
        self.assertEqual(self.boundedField.getBBox(), self.bbox)
        self.assertEqual(self.boundedField.getSkyWcs(), self.skyWcs)
        self.assertEqual(self.boundedField.getInverseScale(),
                         1.0 / self.skyWcs.getPixelScale().asDegrees()**2)

    def _computeExpected(self, points):
        """Return an array with the expected result of evaluate(point)."""
        expect = np.zeros(len(points), dtype=float)
        for i, point in enumerate(points):
            scale = self.skyWcs.getPixelScale(lsst.geom.Point2D(point)).asDegrees()
            expect[i] = scale*scale / (self.skyWcs.getPixelScale().asDegrees())**2
        return expect

    def testEvaluate(self):
        expect = self._computeExpected(self.points)
        result = np.zeros(len(self.points), dtype=float)
        for i, point in enumerate(self.points):
            result[i] = self.boundedField.evaluate(point)
        self.assertFloatsAlmostEqual(expect, result)

    def testEvaluateArray(self):
        xx = np.linspace(self.bbox.getMinX(), self.bbox.getMaxX())
        yy = np.linspace(self.bbox.getMinY(), self.bbox.getMaxY())
        xv, yv = np.meshgrid(xx, yy)
        points = list(zip(xv.flatten(), yv.flatten()))
        expect = self._computeExpected(points)
        result = self.boundedField.evaluate(xv.flatten(), yv.flatten())
        self.assertFloatsAlmostEqual(expect, result)

    def testEquality(self):
        # not equal to a different type of BoundedField
        other = lsst.afw.math.ChebyshevBoundedField(self.bbox, np.random.random((2, 2)))
        self.assertNotEqual(self.boundedField, other)

        # equal to something created with the same parameters.
        other = PixelScaleBoundedField(self.bbox, self.skyWcs)
        self.assertEqual(self.boundedField, other)

        # not equal to something with different bbox.
        newBox = lsst.geom.Box2I(self.bbox)
        newBox.grow(10)
        other = PixelScaleBoundedField(newBox, self.skyWcs)
        self.assertNotEqual(self.boundedField, other)

        # not equal to something with different wcs
        crpix = self.skyWcs.getPixelOrigin()
        crpix.scale(10)
        newWcs = lsst.afw.geom.makeSkyWcs(crpix=crpix,
                                          crval=self.skyWcs.getSkyOrigin(),
                                          cdMatrix=self.skyWcs.getCdMatrix())
        other = PixelScaleBoundedField(self.bbox, newWcs)
        self.assertNotEqual(self.boundedField, other)

    def testString(self):
        # NOTE: Nothing else currently to test in the str() output.
        self.assertIn("PixelScaleBoundedField(", str(self.boundedField))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

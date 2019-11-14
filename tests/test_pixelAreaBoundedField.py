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
from lsst.afw.math import PixelAreaBoundedField


class PixelAreaBoundedFieldTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        # Trivial WCS:
        crpix = lsst.geom.Point2D(100, 100)
        crval = lsst.geom.SpherePoint(0, 45, lsst.geom.degrees)
        cdMatrix = lsst.afw.geom.makeCdMatrix(0.25*lsst.geom.arcseconds, 60*lsst.geom.degrees)
        self.skyWcs = lsst.afw.geom.makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix)
        self.bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(200, 200))
        self.points = [crpix]
        self.points.extend(lsst.geom.Box2D(self.bbox).getCorners())
        self.boundedField = PixelAreaBoundedField(self.bbox, self.skyWcs, unit=lsst.geom.arcseconds)

    def testBBox(self):
        self.assertEqual(self.boundedField.getBBox(), self.bbox)

    def _computeExpected(self, points):
        """Return an array with the expected result of evaluate(point)."""
        expect = np.zeros(len(points), dtype=float)
        for i, point in enumerate(points):
            scale = self.skyWcs.getPixelScale(lsst.geom.Point2D(point)).asArcseconds()
            expect[i] = scale**2
        return expect

    def testEvaluate(self):
        """Test regular evaluation (including evaluation after multiplication
        by a scalar).
        """
        expect = self._computeExpected(self.points)
        result = np.zeros(len(self.points), dtype=float)
        result2 = np.zeros(len(self.points), dtype=float)
        product = self.boundedField * 2.5
        for i, point in enumerate(self.points):
            result[i] = self.boundedField.evaluate(point)
            result2[i] = product.evaluate(point)
        self.assertFloatsAlmostEqual(expect, result)
        self.assertFloatsAlmostEqual(expect*2.5, result2)

    def testEvaluateArray(self):
        """Test vectorized evaluation (including vectorized evaluation
        after multiplication by a scalar).
        """
        xx = np.linspace(self.bbox.getMinX(), self.bbox.getMaxX())
        yy = np.linspace(self.bbox.getMinY(), self.bbox.getMaxY())
        xv, yv = np.meshgrid(xx, yy)
        points = list(zip(xv.flatten(), yv.flatten()))
        expect = self._computeExpected(points)
        result = self.boundedField.evaluate(xv.flatten(), yv.flatten())
        self.assertFloatsAlmostEqual(expect, result, rtol=1E-15)
        product = self.boundedField * 2.5
        result2 = product.evaluate(xv.flatten(), yv.flatten())
        self.assertFloatsAlmostEqual(expect*2.5, result2)

    def testEquality(self):
        """Test the implementation of operator== / __eq__.
        """
        # not equal to a different type of BoundedField
        other = lsst.afw.math.ChebyshevBoundedField(self.bbox, np.random.random((2, 2)))
        self.assertNotEqual(self.boundedField, other)

        # equal to something created with the same parameters.
        other = PixelAreaBoundedField(self.bbox, self.skyWcs, unit=lsst.geom.arcseconds)
        self.assertEqual(self.boundedField, other)

        # not equal to something with different bbox.
        newBox = lsst.geom.Box2I(self.bbox)
        newBox.grow(10)
        other = PixelAreaBoundedField(newBox, self.skyWcs, unit=lsst.geom.arcseconds)
        self.assertNotEqual(self.boundedField, other)

        # not equal to something with different units.
        other = PixelAreaBoundedField(self.bbox, self.skyWcs, unit=lsst.geom.radians)
        self.assertNotEqual(self.boundedField, other)

        # not equal to something with different wcs
        crpix = self.skyWcs.getPixelOrigin()
        crpix.scale(10)
        newWcs = lsst.afw.geom.makeSkyWcs(crpix=crpix,
                                          crval=self.skyWcs.getSkyOrigin(),
                                          cdMatrix=self.skyWcs.getCdMatrix())
        other = PixelAreaBoundedField(self.bbox, newWcs)
        self.assertNotEqual(self.boundedField, other)

    def testPersistence(self):
        """Test that we can round-trip a PixelAreaBoundedField through
        persistence.
        """
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            self.boundedField.writeFits(filename)
            out = lsst.afw.math.PixelAreaBoundedField.readFits(filename)
            self.assertEqual(self.boundedField, out)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

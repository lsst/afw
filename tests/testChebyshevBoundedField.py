#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
Tests for math.ChebyshevBoundedField

Run with:
   ./testChebyshevBoundedField.py
or
   python
   >>> import testSchema; testSchema.run()
"""
from __future__ import absolute_import, division, print_function
from builtins import range

import os
import unittest

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.geom
import lsst.afw.image
import lsst.afw.math

try:
    type(display)
except NameError:
    display = False

CHEBYSHEV_T = [
    lambda x: x**0,
    lambda x: x,
    lambda x: 2*x**2 - 1,
    lambda x: (4*x**2 - 3)*x,
    lambda x: (8*x**2 - 8)*x**2 + 1,
    lambda x: ((16*x**2 - 20)*x**2 + 5)*x,
]


class ChebyshevBoundedFieldTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        np.random.seed(5)
        self.bbox = lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(-5, -5), lsst.afw.geom.Point2I(5, 5))
        self.x1d = np.linspace(self.bbox.getBeginX(), self.bbox.getEndX())
        self.y1d = np.linspace(self.bbox.getBeginY(), self.bbox.getEndY())
        self.x2d, self.y2d = np.meshgrid(self.x1d, self.y1d)
        self.xFlat = np.ravel(self.x2d)
        self.yFlat = np.ravel(self.y2d)
        self.cases = []
        for orderX in range(0, 5):
            for orderY in range(0, 5):
                indexX, indexY = np.meshgrid(np.arange(orderX+1, dtype=int),
                                             np.arange(orderY+1, dtype=int))
                for triangular in (True, False):
                    ctrl = lsst.afw.math.ChebyshevBoundedFieldControl()
                    ctrl.orderX = orderX
                    ctrl.orderY = orderY
                    ctrl.triangular = triangular
                    coefficients = np.random.randn(orderY+1, orderX+1)
                    if triangular:
                        coefficients[indexX + indexY > max(orderX, orderY)] = 0.0
                    self.cases.append((ctrl, coefficients))

    def tearDown(self):
        del self.bbox

    def testEvaluate(self):
        """Test the single-point evaluate method against explicitly-defined 1-d Chebyshevs
        (at the top of this file).
        """
        factor = 12.345
        boxD = lsst.afw.geom.Box2D(self.bbox)
        # sx, sy: transform from self.bbox range to [-1, -1]
        sx = 2.0 / boxD.getWidth()
        sy = 2.0 / boxD.getHeight()
        nPoints = 50
        for ctrl, coefficients in self.cases:
            field = lsst.afw.math.ChebyshevBoundedField(self.bbox, coefficients)
            x = np.random.rand(nPoints)*boxD.getWidth() + boxD.getMinX()
            y = np.random.rand(nPoints)*boxD.getHeight() + boxD.getMinY()
            z1 = field.evaluate(x, y)
            tx = np.array([CHEBYSHEV_T[i](sx*x) for i in range(coefficients.shape[1])])
            ty = np.array([CHEBYSHEV_T[i](sy*y) for i in range(coefficients.shape[0])])
            self.assertEqual(tx.shape, (coefficients.shape[1], x.size))
            self.assertEqual(ty.shape, (coefficients.shape[0], y.size))
            z2 = np.array([np.dot(ty[:, i], np.dot(coefficients, tx[:, i])) for i in range(nPoints)])
            self.assertFloatsAlmostEqual(z1, z2, rtol=1E-12)

            scaled = field*factor
            self.assertFloatsAlmostEqual(scaled.evaluate(x, y), factor*z2, rtol=factor*1E-13)
            self.assertFloatsEqual(scaled.getCoefficients(), factor*field.getCoefficients())

    def _testIntegrateBox(self, bbox, coeffs, expect):
        field = lsst.afw.math.ChebyshevBoundedField(bbox, coeffs)
        self.assertFloatsAlmostEqual(field.integrate(), expect, rtol=1E-14)

    def testIntegrateTrivialBox(self):
        """Test integrating over a "trivial" [-1,1] box.

        NOTE: a "trivial" BBox can't be constructed exactly, given that Box2I
        is inclusive, but the [0,1] box has the same area (because it is
        actually (-0.5, 1.5) when converted to a Box2D), and the translation
        doesn't affect the integral.
        """
        bbox = lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(0, 0), lsst.afw.geom.Point2I(1, 1))

        # 0th order polynomial
        coeffs = np.array([[5.0]])
        self._testIntegrateBox(bbox, coeffs, 4.0*coeffs[0, 0])

        # 1st order polynomial: odd orders drop out of integral
        coeffs = np.array([[5.0, 2.0], [3.0, 4.0]])
        self._testIntegrateBox(bbox, coeffs, 4.0*coeffs[0, 0])

        # 2nd order polynomial in x, 0th in y
        coeffs = np.array([[5.0, 0.0, 7.0]])
        self._testIntegrateBox(bbox, coeffs, 4.0*coeffs[0, 0] - (4.0/3.0)*coeffs[0, 2])

        # 2nd order polynomial in y, 0th in x
        coeffs = np.zeros((3, 3))
        coeffs[0, 0] = 5.0
        coeffs[2, 0] = 7.0
        self._testIntegrateBox(bbox, coeffs, 4.0*coeffs[0, 0] - (4.0/3.0)*coeffs[2, 0])

        # 2nd order polynomial in x and y, no cross-term
        coeffs = np.zeros((3, 3))
        coeffs[0, 0] = 5.0
        coeffs[1, 0] = 7.0
        coeffs[0, 2] = 3.0
        self._testIntegrateBox(bbox, coeffs, 4.0*coeffs[0, 0] -
                               (4.0/3.0)*coeffs[2, 0] - (4.0/3.0)*coeffs[0, 2])

    def testIntegrateBox(self):
        """Test integrating over an "interesting" box.

        The values of these integrals were checked in Mathematica. The code
        block below can be pasted into Mathematica to re-do those calculations.

        ::

            f[x_, y_, n_, m_] := \!\(
                \*UnderoverscriptBox[\(\[Sum]\), \(i = 0\), \(n\)]\(
                \*UnderoverscriptBox[\(\[Sum]\), \(j = 0\), \(m\)]
                \*SubscriptBox[\(a\), \(i, j\)]*ChebyshevT[i, x]*ChebyshevT[j, y]\)\)
            integrate2dBox[n_, m_, x0_, x1_, y0_, y1_] := \!\(
                \*SubsuperscriptBox[\(\[Integral]\), \(y0\), \(y1\)]\(
                \*SubsuperscriptBox[\(\[Integral]\), \(x0\), \(x1\)]f[
                \*FractionBox[\(2  x - x0 - x1\), \(x1 - x0\)],
                \*FractionBox[\(2  y - y0 - y1\), \(y1 - y0\)], n,
                     m] \[DifferentialD]x \[DifferentialD]y\)\)
            integrate2dBox[0, 0, -2.5, 5.5, -3.5, 7.5]
            integrate2dBox[1, 0, -2.5, 5.5, -3.5, 7.5]
            integrate2dBox[0, 1, -2.5, 5.5, -3.5, 7.5]
            integrate2dBox[1, 1, -2.5, 5.5, -3.5, 7.5]
            integrate2dBox[1, 2, -2.5, 5.5, -3.5, 7.5]
            integrate2dBox[2, 2, -2.5, 5.5, -3.5, 7.5]
        """
        bbox = lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(-2, -3), lsst.afw.geom.Point2I(5, 7))

        # 0th order polynomial
        coeffs = np.array([[5.0]])
        self._testIntegrateBox(bbox, coeffs, 88.0*coeffs[0, 0])

        # 1st order polynomial: odd orders drop out of integral
        coeffs = np.array([[5.0, 2.0], [3.0, 4.0]])
        self._testIntegrateBox(bbox, coeffs, 88.0*coeffs[0, 0])

        # 2nd order polynomial in x, 0th in y
        coeffs = np.array([[5.0, 0.0, 7.0]])
        self._testIntegrateBox(bbox, coeffs, 88.0*coeffs[0, 0] - (88.0/3.0)*coeffs[0, 2])

        # 2nd order polynomial in y, 0th in x
        coeffs = np.zeros((3, 3))
        coeffs[0, 0] = 5.0
        coeffs[2, 0] = 7.0
        self._testIntegrateBox(bbox, coeffs, 88.0*coeffs[0, 0] - (88.0/3.0)*coeffs[2, 0])

        # 2nd order polynomial in x,y
        coeffs = np.zeros((3, 3))
        coeffs[2, 2] = 11.0
        self._testIntegrateBox(bbox, coeffs, (88.0/9.0)*coeffs[2, 2])

    def testMean(self):
        """
        The mean of the nth 1d Chebyshev (a_n*T_n(x)) on [-1,1] is
            0 for odd n
            a_n / (1-n^2) for even n
        Similarly, the mean of the (n,m)th 2d Chebyshev is the appropriate
        product of the above.
        """
        bbox = lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(-2, -3), lsst.afw.geom.Point2I(5, 7))

        coeffs = np.array([[5.0]])
        field = lsst.afw.math.ChebyshevBoundedField(bbox, coeffs)
        self.assertEqual(field.mean(), coeffs[0, 0])

        coeffs = np.array([[5.0, 0.0, 3.0]])
        field = lsst.afw.math.ChebyshevBoundedField(bbox, coeffs)
        self.assertEqual(field.mean(), coeffs[0, 0] - coeffs[0, 2]/3.0)

        # 2nd order polynomial in x,y
        coeffs = np.zeros((3, 3))
        coeffs[0, 0] = 7.0
        coeffs[1, 0] = 31.0
        coeffs[0, 2] = 13.0
        coeffs[2, 2] = 11.0
        field = lsst.afw.math.ChebyshevBoundedField(bbox, coeffs)
        self.assertFloatsAlmostEqual(field.mean(), coeffs[0, 0] - coeffs[0, 2]/3.0 + coeffs[2, 2]/9.0)

    def testImageFit(self):
        """Test that we can fit an image produced by a ChebyshevBoundedField and
        get the same coefficients back.
        """
        for ctrl, coefficients in self.cases:
            inField = lsst.afw.math.ChebyshevBoundedField(self.bbox, coefficients)
            for Image in (lsst.afw.image.ImageF, lsst.afw.image.ImageD):
                image = Image(self.bbox)
                inField.fillImage(image)
                outField = lsst.afw.math.ChebyshevBoundedField.fit(image, ctrl)
                self.assertFloatsAlmostEqual(outField.getCoefficients(), coefficients, rtol=1E-6, atol=1E-7)

    def testArrayFit(self):
        """Test that we can fit 1-d arrays produced by a ChebyshevBoundedField and
        get the same coefficients back.
        """
        for ctrl, coefficients in self.cases:
            inField = lsst.afw.math.ChebyshevBoundedField(self.bbox, coefficients)
            for Image in (lsst.afw.image.ImageF, lsst.afw.image.ImageD):
                array = inField.evaluate(self.xFlat, self.yFlat)
                outField1 = lsst.afw.math.ChebyshevBoundedField.fit(self.bbox, self.xFlat, self.yFlat,
                                                                    array, ctrl)
                self.assertFloatsAlmostEqual(outField1.getCoefficients(), coefficients, rtol=1E-6, atol=1E-7)
                weights = (1.0 + np.random.randn(array.size)**2)
                # Should get same results with different weights, since we still have no noise
                # and a model that can exactly reproduce the data.
                outField2 = lsst.afw.math.ChebyshevBoundedField.fit(self.bbox, self.xFlat, self.yFlat,
                                                                    array, weights, ctrl)
                self.assertFloatsAlmostEqual(outField2.getCoefficients(), coefficients, rtol=1E-7, atol=1E-7)

    def testPersistence(self):
        """Test that we can fit 1-d arrays produced by a ChebyshevBoundedField and
        get the same coefficients back.
        """
        filename = "testChebyshevBoundedField.fits"
        boxD = lsst.afw.geom.Box2D(self.bbox)
        nPoints = 50
        for ctrl, coefficients in self.cases:
            inField = lsst.afw.math.ChebyshevBoundedField(self.bbox, coefficients)
            inField.writeFits(filename)
            outField = lsst.afw.math.ChebyshevBoundedField.readFits(filename)
            self.assertEqual(inField.getBBox(), outField.getBBox())
            self.assertFloatsAlmostEqual(inField.getCoefficients(), outField.getCoefficients())
            x = np.random.rand(nPoints)*boxD.getWidth() + boxD.getMinX()
            y = np.random.rand(nPoints)*boxD.getHeight() + boxD.getMinY()
            z1 = inField.evaluate(x, y)
            z2 = inField.evaluate(x, y)
            self.assertFloatsAlmostEqual(z1, z2, rtol=1E-13)
        os.remove(filename)

    def testTruncate(self):
        """Test that truncate() works as expected
        """
        for ctrl, coefficients in self.cases:
            field1 = lsst.afw.math.ChebyshevBoundedField(self.bbox, coefficients)
            field2 = field1.truncate(ctrl)
            self.assertFloatsAlmostEqual(field1.getCoefficients(), field2.getCoefficients())
            self.assertEqual(field1.getBBox(), field2.getBBox())
            config3 = lsst.afw.math.ChebyshevBoundedField.ConfigClass()
            config3.readControl(ctrl)
            if ctrl.orderX > 0:
                config3.orderX -= 1
            if ctrl.orderY > 0:
                config3.orderY -= 1
            field3 = field1.truncate(config3.makeControl())
            for i in range(config3.orderY + 1):
                for j in range(config3.orderX + 1):
                    if config3.triangular and i + j > max(config3.orderX, config3.orderY):
                        self.assertEqual(field3.getCoefficients()[i, j], 0.0)
                    else:
                        self.assertEqual(field3.getCoefficients()[i, j], field1.getCoefficients()[i, j])


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

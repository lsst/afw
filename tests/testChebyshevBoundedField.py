#!/usr/bin/env python
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
from __future__ import division
from builtins import range

import os
import unittest

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.geom
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

            scaled = lsst.afw.math.ChebyshevBoundedField.cast(field*factor)
            self.assertFloatsAlmostEqual(scaled.evaluate(x, y), factor*z2, rtol=factor*1E-13)
            self.assertFloatsEqual(scaled.getCoefficients(), factor*field.getCoefficients())

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

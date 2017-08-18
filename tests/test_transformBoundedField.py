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

"""
Tests for math.TransformBoundedField

Run with:
   ./testTransformBoundedField.py
or
   python
   >>> import testSchema; testSchema.run()
"""
from __future__ import absolute_import, division, print_function

import os
import unittest

import astshim
import numpy as np
from numpy.testing import assert_allclose

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.geom
import lsst.afw.image
from lsst.afw.math import TransformBoundedField

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


class TransformBoundedFieldTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.longMessage = True

        # an arbitrary bounding box (not that this kind of field cares)
        self.bbox = lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(-3, 4),
                                        lsst.afw.geom.Extent2I(5, 30))

        # a list of points contained in the bbox
        self.pointList = lsst.afw.geom.Box2D(self.bbox).getCorners()
        self.pointList.append(lsst.afw.geom.Box2D(self.bbox).getCenter())
        self.xList = np.array([p[0] for p in self.pointList])
        self.yList = np.array([p[1] for p in self.pointList])

        # a simple polynomial mapping
        coeff_f = np.array([
            [1.5, 1, 0, 0],
            [-0.5, 1, 1, 0],
            [1.0, 1, 0, 1],
        ])
        polyMap = astshim.PolyMap(coeff_f, 1)
        self.transform = lsst.afw.geom.TransformPoint2ToGeneric(polyMap)
        self.boundedField = TransformBoundedField(self.bbox, self.transform)

    def tearDown(self):
        del self.transform

    def testEvaluate(self):
        """Test the various overloads of `evaluate`
        """
        for point in self.pointList:
            # applylForward returns a vector with one entry per axis
            # and in this case there is just one axis
            predRes = self.transform.applyForward(point)[0]

            res = self.boundedField.evaluate(point)
            self.assertFloatsAlmostEqual(res, predRes)

            x, y = point
            res2 = self.boundedField.evaluate(x, y)
            self.assertFloatsAlmostEqual(res2, predRes)

        resArr = self.boundedField.evaluate(self.xList, self.yList)
        # applylForward returns an array with one row of values per axis
        # and in this case there is just one axis
        predResArr = self.transform.applyForward(self.pointList)[0]
        assert_allclose(resArr, predResArr)

    def testMultiplyOperator(self):
        """Test operator*
        """
        maxVal = np.max(np.abs(self.transform.applyForward(self.pointList)[0]))
        for multFactor in (-9e99, -1.5e-7, 3.6e-7, 1.5, 9.23e99):
            atol = abs(maxVal * multFactor * 1e-15)
            predResult = self.transform.applyForward(self.pointList)[0] * multFactor

            scaledField1 = self.boundedField * multFactor
            assert_allclose(scaledField1.evaluate(self.xList, self.yList), predResult, atol = atol)

            scaledField2 = multFactor * self.boundedField
            assert_allclose(scaledField2.evaluate(self.xList, self.yList), predResult, atol = atol)

    def testBBox(self):
        """The BBox should have no effect on the kind of transform being tested

        Use an empty bbox as an extreme test of this
        """
        self.assertEqual(self.boundedField.getBBox(), self.bbox)

        emptyBBox = lsst.afw.geom.Box2I()
        noBBoxField = TransformBoundedField(emptyBBox, self.transform)
        self.assertEqual(noBBoxField.getBBox(), emptyBBox)

        resArr = self.boundedField.evaluate(self.xList, self.yList)
        resArrNoBBox = noBBoxField.evaluate(self.xList, self.yList)
        assert_allclose(resArr, resArrNoBBox)

    def testPersistenceAndEquality(self):
        """Test persistence using writeFits and readFits

        Also test operator==
        """
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            self.boundedField.writeFits(filename)
            readField = TransformBoundedField.readFits(filename)

        self.assertTrue(self.boundedField == readField)
        self.assertFalse(self.boundedField != readField)
        self.assertEqual(self.boundedField, readField)

        resArr = self.boundedField.evaluate(self.xList, self.yList)
        readResArr = readField.evaluate(self.xList, self.yList)
        assert_allclose(resArr, readResArr)
        self.assertEqual(readField.getBBox(), self.bbox)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

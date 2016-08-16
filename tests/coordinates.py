#!/usr/bin/env python
from __future__ import absolute_import, division
from builtins import zip
from builtins import range

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
Tests for geom.Point, geom.Extent, geom.CoordinateExpr

Run with:
   ./Coordinates.py
or
   python
   >>> import coordinates; coordinates.run()
"""

import unittest
import numpy
import math
import operator

import lsst.utils.tests as utilsTests
import lsst.afw.geom as geom

numpy.random.seed(1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class CoordinateTestCase(unittest.TestCase):

    def assertClose(self, a, b, msg=None):
        if not numpy.allclose(a, b):
            return self.assertEqual(a, b, msg=msg)
        else:
            return self.assert_(True, msg=msg)

    def testAccessors(self):
        for dtype, cls, rnd in self.classes:
            vector1 = rnd()
            p = cls(*vector1)
            self.assertEqual(p.__class__, cls)
            self.assertEqual(tuple(p), tuple(vector1))
            self.assertEqual(tuple(p.clone()), tuple(p))
            self.assert_(p.clone() is not p)
            vector2 = rnd()
            for n in range(cls.dimensions):
                p[n] = vector2[n]
            self.assertEqual(tuple(p), tuple(vector2))

    def testComparison(self):
        for dtype, cls, rnd in self.classes:
            CoordinateExpr = geom.CoordinateExpr[cls.dimensions]
            vector1 = rnd()
            vector2 = rnd()
            p1 = cls(*vector1)
            p2 = cls(*vector2)

            self.assertEqual(p1 == p2, all(p1.eq(p2)))
            self.assertEqual(p1 != p2, any(p1.ne(p2)))
            self.assertNotEqual(p1, None)  # should not throw
            self.assertNotEqual(p1, tuple(p1))  # should not throw

            self.assertEqual(tuple(p1.eq(p2)), tuple([v1 == v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(tuple(p1.ne(p2)), tuple([v1 != v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(tuple(p1.lt(p2)), tuple([v1 < v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(tuple(p1.le(p2)), tuple([v1 <= v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(tuple(p1.gt(p2)), tuple([v1 > v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(tuple(p1.ge(p2)), tuple([v1 >= v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(type(p1.eq(p2)), CoordinateExpr)
            self.assertEqual(type(p1.ne(p2)), CoordinateExpr)
            self.assertEqual(type(p1.lt(p2)), CoordinateExpr)
            self.assertEqual(type(p1.le(p2)), CoordinateExpr)
            self.assertEqual(type(p1.gt(p2)), CoordinateExpr)
            self.assertEqual(type(p1.ge(p2)), CoordinateExpr)
            scalar = dtype(rnd()[0])
            self.assertEqual(tuple(p1.eq(scalar)), tuple([v1 == scalar for v1 in vector1]))
            self.assertEqual(tuple(p1.ne(scalar)), tuple([v1 != scalar for v1 in vector1]))
            self.assertEqual(tuple(p1.lt(scalar)), tuple([v1 < scalar for v1 in vector1]))
            self.assertEqual(tuple(p1.le(scalar)), tuple([v1 <= scalar for v1 in vector1]))
            self.assertEqual(tuple(p1.gt(scalar)), tuple([v1 > scalar for v1 in vector1]))
            self.assertEqual(tuple(p1.ge(scalar)), tuple([v1 >= scalar for v1 in vector1]))
            self.assertEqual(type(p1.eq(scalar)), CoordinateExpr)
            self.assertEqual(type(p1.ne(scalar)), CoordinateExpr)
            self.assertEqual(type(p1.lt(scalar)), CoordinateExpr)
            self.assertEqual(type(p1.le(scalar)), CoordinateExpr)
            self.assertEqual(type(p1.gt(scalar)), CoordinateExpr)
            self.assertEqual(type(p1.ge(scalar)), CoordinateExpr)


class PointTestCase(CoordinateTestCase):
    """A test case for Point"""

    def setUp(self):
        self.classes = [
            (float, geom.Point2D, lambda: [float(x) for x in numpy.random.randn(2)]),
            (int, geom.Point2I, lambda: [int(x) for x in numpy.random.randint(-5, 5, 2)]),
            (float, geom.Point3D, lambda: [float(x) for x in numpy.random.randn(3)]),
            (int, geom.Point3I, lambda: [int(x) for x in numpy.random.randint(-5, 5, 3)]),
        ]

    def testSpanIteration(self):
        span = geom.Span(4, 3, 8)
        points = list(span)
        self.assertEqual(len(span), len(points))
        self.assertEqual(points, [geom.Point2I(x, 4) for x in range(3, 9)])

    def testConstructors(self):
        # test 2-d
        e1 = geom.Point2I(1, 2)
        e2 = geom.Point2I(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Point2D(1.2, 3.4)
        e2 = geom.Point2D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Point2I(1, 3)
        e2 = geom.Point2D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        # test 3-d
        e1 = geom.Point3I(1, 2, 3)
        e2 = geom.Point3I(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Point3D(1.2, 3.4, 5.6)
        e2 = geom.Point3D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Point3I(1, 2, 3)
        e2 = geom.Point3D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        # test rounding to integral coordinates
        e1 = geom.Point2D(1.2, 3.4)
        e2 = geom.Point2I(e1)
        self.assertClose(tuple([math.floor(v + 0.5) for v in e1]), tuple(e2))

        e1 = geom.Point3D(1.2, 3.4, 5.6)
        e2 = geom.Point3I(e1)
        self.assertClose(tuple([math.floor(v + 0.5) for v in e1]), tuple(e2))


class ExtentTestCase(CoordinateTestCase):
    """A test case for Extent"""

    def setUp(self):
        self.classes = [
            (float, geom.Extent2D, lambda: [float(x) for x in numpy.random.randn(2)]),
            (int, geom.Extent2I, lambda: [int(x) for x in numpy.random.randint(-5, 5, 2)]),
            (float, geom.Extent3D, lambda: [float(x) for x in numpy.random.randn(3)]),
            (int, geom.Extent3I, lambda: [int(x) for x in numpy.random.randint(-5, 5, 3)]),
        ]

    def testRounding(self):
        e1 = geom.Extent2D(1.2, -3.4)
        self.assertEqual(e1.floor(), geom.Extent2I(1, -4))
        self.assertEqual(e1.ceil(), geom.Extent2I(2, -3))
        self.assertEqual(e1.truncate(), geom.Extent2I(1, -3))

    def testConstructors(self):
        # test extent from extent 2-d
        e1 = geom.Extent2I(1, 2)
        e2 = geom.Extent2I(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Extent2D(1.2, 3.4)
        e2 = geom.Extent2D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Extent2I(1, 2)
        e2 = geom.Extent2D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        # test extent from extent 3-d
        e1 = geom.Extent3I(1, 2, 3)
        e2 = geom.Extent3I(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Extent3D(1.2, 3.4, 5.6)
        e2 = geom.Extent3D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Extent3I(1, 2, 3)
        e2 = geom.Extent3D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        # test extent from point 2-d
        e1 = geom.Point2I(1, 2)
        e2 = geom.Extent2I(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Point2D(1.2, 3.4)
        e2 = geom.Extent2D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Point2I(1, 2)
        e2 = geom.Extent2D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        # test extent from point 3-d
        e1 = geom.Point3I(1, 2, 3)
        e2 = geom.Extent3I(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Point3D(1.2, 3.4, 5.6)
        e2 = geom.Extent3D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Point3I(1, 2, 3)
        e2 = geom.Extent3D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        # test invalid constructors
        try:
            e1 = geom.Extent2D(1.2, 3.4)
            e2 = geom.Extent2I(e1)
        except:
            pass
        else:
            self.fail("Should not allow conversion Extent2D to Extent2I")
        try:
            e1 = geom.Extent3D(1.2, 3.4, 5.6)
            e2 = geom.Extent3I(e1)
        except:
            pass
        else:
            self.fail("Should not allow conversion Extent3D to Extent3I")

        try:
            e1 = geom.Point2D(1.2, 3.4)
            e2 = geom.Extent2I(e1)
        except:
            pass
        else:
            self.fail("Should not allow conversion Point2D to Extent 2I")
        try:
            e1 = geom.Point3D(1.2, 3.4, 5.6)
            e2 = geom.Extent3I(e1)
        except:
            pass
        else:
            self.fail("Should not allow conversion Point3D to Extent3I")


class OperatorTestCase(utilsTests.TestCase):

    @staticmethod
    def makeRandom(cls):
        """Make a random Point, Extent, int, or float of the given type."""
        if cls is int:
            v = 0
            while v == 0:
                v = int(numpy.random.randn()*10)
        elif cls is float:
            v = float(numpy.random.randn()*10)
        else:
            v = cls()
            t = type(v[0])
            for i in range(len(v)):
                while v[i] == 0:
                    v[i] = t(numpy.random.randn()*10)
        return v

    def checkOperator(self, op, lhs, rhs, expected, inPlace=False):
        """Check that the type and result of applying operator 'op' to types 'lhs' and 'rhs'
        yield a result of type 'expected', and that the computed value is correct.  If
        'expected' is an Exception subclass, instead check that attempting to apply the
        operator raises that exception.
        """
        v1 = self.makeRandom(lhs)
        v2 = self.makeRandom(rhs)
        if issubclass(expected, Exception):
            self.assertRaises(expected, op, v1, v2)
        else:
            check = op(numpy.array(v1), numpy.array(v2))
            result = op(v1, v2)
            if type(result) != expected:
                self.fail("%s(%s, %s): expected %s, got %s" %
                          (op.__name__, lhs.__name__, rhs.__name__,
                           expected.__name__, type(result).__name__))
            if not numpy.allclose(result, check):
                self.fail("%s(%s, %s): expected %s, got %s" %
                          (op.__name__, lhs.__name__, rhs.__name__, tuple(check), tuple(result)))
            if inPlace and result is not v1:
                self.fail("%s(%s, %s): result is not self" % (op.__name__, lhs.__name__, rhs.__name__))

    def testPointAsExtent(self):
        for n in (2, 3):
            for t in (int, float):
                p = self.makeRandom(geom.Point[t, n])
                e = p.asExtent()
                self.assertEqual(type(e), geom.Extent[t, n])
                self.assertClose(numpy.array(p), numpy.array(e), rtol=0.0, atol=0.0)

    def testExtentAsPoint(self):
        for n in (2, 3):
            for t in (int, float):
                e = self.makeRandom(geom.Extent[t, n])
                p = e.asPoint()
                self.assertEqual(type(p), geom.Point[t, n])
                self.assertClose(numpy.array(p), numpy.array(e), rtol=0.0, atol=0.0)

    def testUnaryOperators(self):
        for n in (2, 3):
            for t in (int, float):
                e1 = self.makeRandom(geom.Extent[t, n])
                e2 = +e1
                self.assertEqual(type(e1), type(e2))
                self.assertClose(numpy.array(e1), numpy.array(e2), rtol=0.0, atol=0.0)
                e3 = -e1
                self.assertEqual(type(e1), type(e3))
                self.assertClose(numpy.array(e3), -numpy.array(e1), rtol=0.0, atol=0.0)

    def testBinaryOperators(self):
        for n in (2, 3):
            pD = geom.Point[float, n]
            pI = geom.Point[int, n]
            eD = geom.Extent[float, n]
            eI = geom.Extent[int, n]
            # Addition
            self.checkOperator(operator.add, pD, pD, TypeError)
            self.checkOperator(operator.add, pD, pI, TypeError)
            self.checkOperator(operator.add, pD, eD, pD)
            self.checkOperator(operator.add, pD, eI, pD)
            self.checkOperator(operator.add, pI, pD, TypeError)
            self.checkOperator(operator.add, pI, pI, TypeError)
            self.checkOperator(operator.add, pI, eD, pD)
            self.checkOperator(operator.add, pI, eI, pI)
            self.checkOperator(operator.add, eD, pD, pD)
            self.checkOperator(operator.add, eD, pI, pD)
            self.checkOperator(operator.add, eD, eI, eD)
            self.checkOperator(operator.add, eD, eD, eD)
            self.checkOperator(operator.add, eI, pD, pD)
            self.checkOperator(operator.add, eI, pI, pI)
            self.checkOperator(operator.add, eI, eD, eD)
            self.checkOperator(operator.add, eI, eI, eI)
            # Subtraction
            self.checkOperator(operator.sub, pD, pD, eD)
            self.checkOperator(operator.sub, pD, pI, eD)
            self.checkOperator(operator.sub, pD, eD, pD)
            self.checkOperator(operator.sub, pD, eI, pD)
            self.checkOperator(operator.sub, pI, pD, eD)
            self.checkOperator(operator.sub, pI, pI, eI)
            self.checkOperator(operator.sub, pI, eD, pD)
            self.checkOperator(operator.sub, pI, eI, pI)
            self.checkOperator(operator.sub, eD, pD, TypeError)
            self.checkOperator(operator.sub, eD, pI, TypeError)
            self.checkOperator(operator.sub, eD, eD, eD)
            self.checkOperator(operator.sub, eD, eI, eD)
            self.checkOperator(operator.sub, eI, pD, TypeError)
            self.checkOperator(operator.sub, eI, pI, TypeError)
            self.checkOperator(operator.sub, eI, eD, eD)
            self.checkOperator(operator.sub, eI, eI, eI)
            # Multiplication
            self.checkOperator(operator.mul, eD, int, eD)
            self.checkOperator(operator.mul, eD, float, eD)
            self.checkOperator(operator.mul, eI, int, eI)
            self.checkOperator(operator.mul, eI, float, eD)
            self.checkOperator(operator.mul, int, eD, eD)
            self.checkOperator(operator.mul, float, eD, eD)
            self.checkOperator(operator.mul, int, eI, eI)
            self.checkOperator(operator.mul, float, eI, eD)
            # Old-Style Division (note that operator.div doesn't obey the future statement; it just calls
            # __div__ directly.
            if hasattr(operator, "div"):
                self.checkOperator(operator.div, eD, int, eD)
                self.checkOperator(operator.div, eD, float, eD)
                self.checkOperator(operator.div, eI, int, eI)
                self.checkOperator(operator.div, eI, float, eD)
            # New-Style Division
            self.checkOperator(operator.truediv, eD, int, eD)
            self.checkOperator(operator.truediv, eD, float, eD)
            self.checkOperator(operator.truediv, eI, int, eD)
            self.checkOperator(operator.truediv, eI, float, eD)
            # Floor Division
            self.checkOperator(operator.floordiv, eD, int, TypeError)
            self.checkOperator(operator.floordiv, eD, float, TypeError)
            self.checkOperator(operator.floordiv, eI, int, eI)
            self.checkOperator(operator.floordiv, eI, float, TypeError)

    def testInPlaceOperators(self):
        # Note: I have no idea why Swig throws NotImplementedError sometimes for in-place operators
        # that don't match rather than TypeError (which is what it throws for regular binary operators,
        # and what it should be throwing consistently here, if the Python built-ins are any indication).
        # However, I've determined that it's not worth my time to fix it, as the only approach
        # I could think of was to use %feature("shadow"), which I tried, and Swig simply ignored it
        # (the code I put in those blocks never appeared in the .py file).
        for n in (2, 3):
            pD = geom.Point[float, n]
            pI = geom.Point[int, n]
            eD = geom.Extent[float, n]
            eI = geom.Extent[int, n]
            # Addition
            self.checkOperator(operator.iadd, pD, pD, NotImplementedError)
            self.checkOperator(operator.iadd, pD, pI, NotImplementedError)
            self.checkOperator(operator.iadd, pD, eD, pD, inPlace=True)
            self.checkOperator(operator.iadd, pD, eI, pD, inPlace=True)
            self.checkOperator(operator.iadd, pI, pD, TypeError)
            self.checkOperator(operator.iadd, pI, pI, TypeError)
            self.checkOperator(operator.iadd, pI, eD, TypeError)
            self.checkOperator(operator.iadd, pI, eI, pI, inPlace=True)
            self.checkOperator(operator.iadd, eD, pD, NotImplementedError)
            self.checkOperator(operator.iadd, eD, pI, NotImplementedError)
            self.checkOperator(operator.iadd, eD, eI, eD, inPlace=True)
            self.checkOperator(operator.iadd, eD, eD, eD, inPlace=True)
            self.checkOperator(operator.iadd, eI, pD, TypeError)
            self.checkOperator(operator.iadd, eI, pI, TypeError)
            self.checkOperator(operator.iadd, eI, eD, TypeError)
            self.checkOperator(operator.iadd, eI, eI, eI, inPlace=True)
            # Subtraction
            self.checkOperator(operator.isub, pD, pD, NotImplementedError)
            self.checkOperator(operator.isub, pD, pI, NotImplementedError)
            self.checkOperator(operator.isub, pD, eD, pD, inPlace=True)
            self.checkOperator(operator.isub, pD, eI, pD, inPlace=True)
            self.checkOperator(operator.isub, pI, pD, TypeError)
            self.checkOperator(operator.isub, pI, pI, TypeError)
            self.checkOperator(operator.isub, pI, eD, TypeError)
            self.checkOperator(operator.isub, pI, eI, pI, inPlace=True)
            self.checkOperator(operator.isub, eD, pD, NotImplementedError)
            self.checkOperator(operator.isub, eD, pI, NotImplementedError)
            self.checkOperator(operator.isub, eD, eD, eD, inPlace=True)
            self.checkOperator(operator.isub, eD, eI, eD, inPlace=True)
            self.checkOperator(operator.isub, eI, pD, TypeError)
            self.checkOperator(operator.isub, eI, pI, TypeError)
            self.checkOperator(operator.isub, eI, eD, TypeError)
            self.checkOperator(operator.isub, eI, eI, eI, inPlace=True)
            # Multiplication
            self.checkOperator(operator.imul, eD, int, eD, inPlace=True)
            self.checkOperator(operator.imul, eD, float, eD, inPlace=True)
            self.checkOperator(operator.imul, eI, int, eI, inPlace=True)
            self.checkOperator(operator.imul, eI, float, TypeError)
            # Old-Style Division (note that operator.div doesn't obey the future statement; it just calls
            # __div__ directly).
            if hasattr(operator, "idiv"):
                self.checkOperator(operator.idiv, eD, int, eD, inPlace=True)
                self.checkOperator(operator.idiv, eD, float, eD, inPlace=True)
                self.checkOperator(operator.idiv, eI, int, eI, inPlace=True)
                self.checkOperator(operator.idiv, eI, float, TypeError)
            # New-Style Division
            self.checkOperator(operator.itruediv, eD, int, eD, inPlace=True)
            self.checkOperator(operator.itruediv, eD, float, eD, inPlace=True)
            self.checkOperator(operator.itruediv, eI, int, TypeError)
            self.checkOperator(operator.itruediv, eI, float, TypeError)
            # Floor Division
            self.checkOperator(operator.floordiv, eD, int, TypeError)
            self.checkOperator(operator.floordiv, eD, float, TypeError)
            self.checkOperator(operator.floordiv, eI, int, eI)
            self.checkOperator(operator.floordiv, eI, float, TypeError)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(PointTestCase)
    suites += unittest.makeSuite(ExtentTestCase)
    suites += unittest.makeSuite(OperatorTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

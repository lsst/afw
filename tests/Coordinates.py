#!/usr/bin/env python
"""
Tests for geom.Point, geom.Extent, geom.CoordinateExpr

Run with:
   ./Coordinates.py
or
   python
   >>> import Coordinates; Coordinates.run()
"""

import os
import pdb  # we may want to say pdb.set_trace()
import sys
import unittest
import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.geom as geom

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class CoordinateTestCase(unittest.TestCase):
    
    def assertClose(self,a,b):
        if not numpy.allclose(a,b):
            return self.assertEqual(a,b)
        else:
            return self.assert_(True)

    def testAccessors(self):
        for dtype, cls, rnd, ctor in self.classes:
            vector1 = rnd()
            p = ctor(*vector1)
            self.assertEqual(p.__class__, cls)
            self.assertEqual(tuple(p), tuple(vector1))
            self.assertEqual(tuple(p.clone()), tuple(p))
            self.assert_(p.clone() is not p)
            vector2 = rnd()
            for n in range(cls.dimensions):
                p[n] = vector2[n]
            self.assertEqual(tuple(p), tuple(vector2))

    def testComparison(self):
        for dtype, cls, rnd, ctor in self.classes:
            CoordinateExpr = geom.CoordinateExpr[cls.dimensions]
            vector1 = rnd()
            vector2 = rnd()
            p1 = ctor(*vector1)
            p2 = ctor(*vector2)
            self.assertEqual(tuple(p1.eq(p2)), tuple(vector1 == vector2))
            self.assertEqual(tuple(p1.ne(p2)), tuple(vector1 != vector2))
            self.assertEqual(tuple(p1.lt(p2)), tuple(vector1 < vector2))
            self.assertEqual(tuple(p1.le(p2)), tuple(vector1 <= vector2))
            self.assertEqual(tuple(p1.gt(p2)), tuple(vector1 > vector2))
            self.assertEqual(tuple(p1.ge(p2)), tuple(vector1 >= vector2))
            self.assertEqual(type(p1.eq(p2)), CoordinateExpr)
            self.assertEqual(type(p1.ne(p2)), CoordinateExpr)
            self.assertEqual(type(p1.lt(p2)), CoordinateExpr)
            self.assertEqual(type(p1.le(p2)), CoordinateExpr)
            self.assertEqual(type(p1.gt(p2)), CoordinateExpr)
            self.assertEqual(type(p1.ge(p2)), CoordinateExpr)
            scalar = rnd()[0]
            self.assertEqual(tuple(p1.eq(scalar)), tuple(vector1 == scalar))
            self.assertEqual(tuple(p1.ne(scalar)), tuple(vector1 != scalar))
            self.assertEqual(tuple(p1.lt(scalar)), tuple(vector1 < scalar))
            self.assertEqual(tuple(p1.le(scalar)), tuple(vector1 <= scalar))
            self.assertEqual(tuple(p1.gt(scalar)), tuple(vector1 > scalar))
            self.assertEqual(tuple(p1.ge(scalar)), tuple(vector1 >= scalar))
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
            (float, geom.Point2D, lambda: numpy.random.randn(2), geom.Point2D.make),
            (int, geom.Point2I, lambda: numpy.random.randint(-5,5,2), geom.Point2I.make),
            (float, geom.Point3D, lambda: numpy.random.randn(3), geom.Point3D.make),
            (int, geom.Point3I, lambda: numpy.random.randint(-5,5,3), geom.Point3I.make),
            ]

    def testArithmetic(self):
        for dtype, cls, rnd, ctor in self.classes:
            Extent = geom.Extent[dtype,cls.dimensions]
            vector1 = rnd()
            vector2 = rnd()
            p1 = ctor(*vector1)
            p2 = ctor(*vector2)
            self.assertClose(tuple(p1-p2), tuple(vector1-vector2))
            self.assertEqual(type(p1-p2), Extent)
            self.assertClose(tuple(p1+Extent(p2)), tuple(vector1+vector2))
            self.assertEqual(type(p1+Extent(p2)), cls)
            self.assertClose(tuple(p1-Extent(p2)), tuple(vector1-vector2))
            self.assertEqual(type(p1-Extent(p2)), cls)
            p1 += Extent(p2)
            vector1 += vector2
            self.assertEqual(tuple(p1), tuple(vector1))
            p1 -= Extent(p2)
            vector1 -= vector2
            self.assertClose(tuple(p1), tuple(vector1))
            p1.shift(Extent(p2))
            vector1 += vector2
            self.assertClose(tuple(p1), tuple(vector1))

class ExtentTestCase(CoordinateTestCase):
    """A test case for Extent"""

    def setUp(self):
        self.classes = [
            (float, geom.Extent2D, lambda: numpy.random.randn(2), geom.Extent2D.make),
            (int, geom.Extent2I, lambda: numpy.random.randint(-5,5,2), geom.Extent2I.make),
            (float, geom.Extent3D, lambda: numpy.random.randn(3), geom.Extent3D.make),
            (int, geom.Extent3I, lambda: numpy.random.randint(-5,5,3), geom.Extent3I.make),
            ]

    def testArithmetic(self):
        for dtype, cls, rnd, ctor in self.classes:
            Point = geom.Point[dtype,cls.dimensions]
            vector1 = rnd()
            vector2 = rnd()
            p1 = ctor(*vector1)
            p2 = ctor(*vector2)
            self.assertClose(tuple(p1+Point(p2)), tuple(vector1+vector2))
            self.assertEqual(type(p1+Point(p2)), Point)
            self.assertClose(tuple(p1+p2), tuple(vector1+vector2))
            self.assertEqual(type(p1+p2), cls)
            self.assertClose(tuple(p1-p2), tuple(vector1-vector2))
            self.assertEqual(type(p1-p2), cls)
            self.assertClose(tuple(+p1), tuple(+vector1))
            self.assertEqual(type(+p1), cls)
            self.assertClose(tuple(-p1), tuple(-vector1))
            self.assertEqual(type(-p1), cls)
            p1 += p2
            vector1 += vector2
            self.assertClose(tuple(p1), tuple(vector1))
            p1 -= p2
            vector1 -= vector2
            self.assertClose(tuple(p1), tuple(vector1))
            scalar = 2
            # Python handles integer division differently from C++ for negative numbers
            vector1 = numpy.abs(vector1) 
            p1 = ctor(*vector1)
            self.assertClose(tuple(p1*scalar), tuple(vector1*scalar))
            self.assertEqual(type(p1*scalar), cls)
            self.assertClose(tuple(p1/scalar), tuple(vector1/scalar))
            self.assertEqual(type(p1/scalar), cls)
            p1 *= scalar
            vector1 *= scalar
            self.assertClose(tuple(p1), tuple(vector1))
            p1 /= scalar
            vector1 /= scalar
            self.assertClose(tuple(p1), tuple(vector1))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(PointTestCase)
    suites += unittest.makeSuite(ExtentTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

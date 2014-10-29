#!/usr/bin/env python2
from __future__ import absolute_import, division

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

import lsst.utils.tests as utilsTests
import lsst.afw.geom as geom

numpy.random.seed(1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class CoordinateTestCase(unittest.TestCase):
    
    def assertClose(self, a, b):
        if not numpy.allclose(a, b):
            return self.assertEqual(a, b)
        else:
            return self.assert_(True)

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
            self.assertNotEqual(p1, None) # should not throw
            self.assertNotEqual(p1, tuple(p1)) # should not throw

            self.assertEqual(tuple(p1.eq(p2)), tuple([v1 == v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(tuple(p1.ne(p2)), tuple([v1 != v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(tuple(p1.lt(p2)), tuple([v1 <  v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(tuple(p1.le(p2)), tuple([v1 <= v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(tuple(p1.gt(p2)), tuple([v1 >  v2 for v1, v2 in zip(vector1, vector2)]))
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
            self.assertEqual(tuple(p1.lt(scalar)), tuple([v1 <  scalar for v1 in vector1]))
            self.assertEqual(tuple(p1.le(scalar)), tuple([v1 <= scalar for v1 in vector1]))
            self.assertEqual(tuple(p1.gt(scalar)), tuple([v1 >  scalar for v1 in vector1]))
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

    def testArithmetic(self):
        for dtype, cls, rnd in self.classes:
            Extent = geom.Extent[dtype, cls.dimensions]
            vector1 = rnd()
            vector2 = rnd()
            p1 = cls(*vector1)
            p2 = cls(*vector2)
            self.assertClose(tuple(p1-p2), tuple([v1 - v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(type(p1-p2), Extent)
            self.assertClose(tuple(p1+Extent(p2)), tuple([v1 + v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(type(p1+Extent(p2)), cls)
            self.assertClose(tuple(p1-Extent(p2)), tuple([v1 - v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(type(p1-Extent(p2)), cls)
            p1 += Extent(p2)
            vector1 = [v1 + v2 for v1, v2 in zip(vector1, vector2)]
            self.assertEqual(tuple(p1), tuple(vector1))
            p1 -= Extent(p2)
            vector1 = [v1 - v2 for v1, v2 in zip(vector1, vector2)]
            self.assertClose(tuple(p1), tuple(vector1))
            p1.shift(Extent(p2))
            vector1 = [v1 + v2 for v1, v2 in zip(vector1, vector2)]
            self.assertClose(tuple(p1), tuple(vector1))

    def testSpanIteration(self):
        span = geom.Span(4, 3, 8)
        points = list(span)
        self.assertEqual(len(span), len(points))
        self.assertEqual(points, [geom.Point2I(x, 4) for x in xrange(3, 9)])

    def testConstructors(self):
        #test 2-d
        e1 = geom.Point2I(1, 2)
        e2 = geom.Point2I(e1)
        self.assertClose(tuple(e1), tuple(e2))
        
        e1 = geom.Point2D(1.2, 3.4)
        e2 = geom.Point2D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        
        e1 = geom.Point2I(1, 3)
        e2 = geom.Point2D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        #test 3-d
        e1 = geom.Point3I(1, 2, 3)
        e2 = geom.Point3I(e1)
        self.assertClose(tuple(e1), tuple(e2))
        
        e1 = geom.Point3D(1.2, 3.4, 5.6)
        e2 = geom.Point3D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Point3I(1, 2, 3)
        e2 = geom.Point3D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        #test rounding to integral coordinates
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

    def testArithmetic(self):
        for dtype, cls, rnd in self.classes:
            Point = geom.Point[dtype, cls.dimensions]
            vector1 = rnd()
            vector2 = rnd()
            p1 = cls(*vector1)
            p2 = cls(*vector2)
            self.assertClose(tuple(p1+Point(p2)), tuple([v1 + v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(type(p1+Point(p2)), Point)
            self.assertClose(tuple(p1+p2), tuple([v1 + v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(type(p1+p2), cls)
            self.assertClose(tuple(p1-p2), tuple([v1 - v2 for v1, v2 in zip(vector1, vector2)]))
            self.assertEqual(type(p1-p2), cls)
            self.assertClose(tuple(+p1), tuple(vector1))
            self.assertEqual(type(+p1), cls)
            self.assertClose(tuple(-p1), tuple([-v1 for v1 in vector1]))
            self.assertEqual(type(-p1), cls)
            p1 += p2
            vector1 = [v1 + v2 for v1, v2 in zip(vector1, vector2)]
            self.assertClose(tuple(p1), tuple(vector1))
            p1 -= p2
            vector1 = [v1 - v2 for v1, v2 in zip(vector1, vector2)]
            self.assertClose(tuple(p1), tuple(vector1))
            scalar = 2
            # Python handles integer division differently from C++ for negative numbers
            vector1 = [abs(x) for x in vector1]
            p1 = cls(*vector1)
            self.assertClose(tuple(p1*scalar), tuple([v1*scalar for v1 in vector1]))
            self.assertEqual(type(p1*scalar), cls)
            if type(p1[0]) == int:
                desDivTuple = tuple(v1//scalar for v1 in vector1)
            else:
                desDivTuple = tuple(v1/scalar for v1 in vector1)
            self.assertClose(tuple(p1/scalar), desDivTuple)
            self.assertEqual(type(p1/scalar), cls)
            p1 *= scalar
            vector1 = [v1*scalar for v1 in vector1]
            self.assertClose(tuple(p1), tuple(vector1))
            p1 /= scalar
            vector1 = [v1/scalar for v1 in vector1]
            self.assertClose(tuple(p1), tuple(vector1))

    def testConstructors(self):
        #test extent from extent 2-d
        e1 = geom.Extent2I(1, 2)
        e2 = geom.Extent2I(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Extent2D(1.2, 3.4)
        e2 = geom.Extent2D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Extent2I(1,2)
        e2 = geom.Extent2D(e1)
        self.assertClose(tuple(e1), tuple(e2))
        
        #test extent from extent 3-d
        e1 = geom.Extent3I(1, 2, 3)
        e2 = geom.Extent3I(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Extent3D(1.2, 3.4, 5.6)
        e2 = geom.Extent3D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Extent3I(1,2,3)
        e2 = geom.Extent3D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        #test extent from point 2-d
        e1 = geom.Point2I(1, 2)
        e2 = geom.Extent2I(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Point2D(1.2, 3.4)
        e2 = geom.Extent2D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Point2I(1,2)
        e2 = geom.Extent2D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        #test extent from point 3-d
        e1 = geom.Point3I(1, 2, 3)
        e2 = geom.Extent3I(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Point3D(1.2, 3.4, 5.6)
        e2 = geom.Extent3D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        e1 = geom.Point3I(1,2,3)
        e2 = geom.Extent3D(e1)
        self.assertClose(tuple(e1), tuple(e2))

        #test invalid constructors
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


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(PointTestCase)
    suites += unittest.makeSuite(ExtentTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

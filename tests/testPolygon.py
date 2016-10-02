#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import zip
#pybind11#from builtins import range
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008-2014 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11#import numpy
#pybind11#import pickle
#pybind11#import unittest
#pybind11#import os
#pybind11#import lsst.utils.tests
#pybind11#
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.coord as afwCoord
#pybind11#
#pybind11#from lsst.afw.geom.polygon import Polygon, SinglePolygonException
#pybind11#
#pybind11#DEBUG = False
#pybind11#
#pybind11#
#pybind11#def circle(radius, num, x0=0.0, y0=0.0):
#pybind11#    """Generate points on a circle
#pybind11#
#pybind11#    @param radius: radius of circle
#pybind11#    @param num: number of points
#pybind11#    @param x0,y0: Offset in x,y
#pybind11#    @return x,y coordinates as numpy array
#pybind11#    """
#pybind11#    theta = numpy.linspace(0, 2*numpy.pi, num=num, endpoint=False)
#pybind11#    x = radius*numpy.cos(theta) + x0
#pybind11#    y = radius*numpy.sin(theta) + y0
#pybind11#    return numpy.array([x, y]).transpose()
#pybind11#
#pybind11#
#pybind11#class PolygonTest(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.x0 = 0.0
#pybind11#        self.y0 = 0.0
#pybind11#
#pybind11#    def polygon(self, num, radius=1.0, x0=None, y0=None):
#pybind11#        """Generate a polygon
#pybind11#
#pybind11#        @param num: Number of points
#pybind11#        @param radius: Radius of polygon
#pybind11#        @param x0,y0: Offset of center
#pybind11#        @return polygon
#pybind11#        """
#pybind11#        if x0 is None:
#pybind11#            x0 = self.x0
#pybind11#        if y0 is None:
#pybind11#            y0 = self.y0
#pybind11#        points = circle(radius, num, x0=x0, y0=y0)
#pybind11#        return Polygon([afwGeom.Point2D(x, y) for x, y in reversed(points)])
#pybind11#
#pybind11#    def square(self, size=1.0, x0=0, y0=0):
#pybind11#        """Generate a square
#pybind11#
#pybind11#        @param size: Half-length of the sides
#pybind11#        @param x0,y0: Offset of center
#pybind11#        """
#pybind11#        return Polygon([afwGeom.Point2D(size*x + x0, size*y + y0) for
#pybind11#                        x, y in ((-1, -1), (-1, 1), (1, 1), (1, -1))])
#pybind11#
#pybind11#    def testGetters(self):
#pybind11#        """Test Polygon getters"""
#pybind11#        for num in range(3, 30):
#pybind11#            poly = self.polygon(num)
#pybind11#            self.assertEqual(poly, poly)
#pybind11#            self.assertNotEqual(poly, self.square(1.0, 2.0, 3.0))
#pybind11#            self.assertEqual(poly.getNumEdges(), num)
#pybind11#            self.assertEqual(len(poly.getVertices()), num + 1)  # One extra for the closing point
#pybind11#            self.assertEqual(len(poly.getEdges()), num)
#pybind11#            perimeter = 0.0
#pybind11#            for p1, p2 in poly.getEdges():
#pybind11#                perimeter += numpy.hypot(p1.getX() - p2.getX(), p1.getY() - p2.getY())
#pybind11#            self.assertAlmostEqual(poly.calculatePerimeter(), perimeter)
#pybind11#
#pybind11#            self.assertEqual(pickle.loads(pickle.dumps(poly)), poly)
#pybind11#
#pybind11#        size = 3.0
#pybind11#        poly = self.square(size=size)
#pybind11#        self.assertEqual(poly.calculateArea(), (2*size)**2)
#pybind11#        self.assertEqual(poly.calculatePerimeter(), 2*size*4)
#pybind11#        edges = poly.getEdges()
#pybind11#        self.assertEqual(len(edges), 4)
#pybind11#        perimeter = 0.0
#pybind11#        for p1, p2 in edges:
#pybind11#            self.assertEqual(abs(p1.getX()), size)
#pybind11#            self.assertEqual(abs(p1.getY()), size)
#pybind11#            self.assertEqual(abs(p2.getX()), size)
#pybind11#            self.assertEqual(abs(p2.getY()), size)
#pybind11#            self.assertNotEqual(p1, p2)
#pybind11#
#pybind11#    def testFromBox(self):
#pybind11#        size = 1.0
#pybind11#        poly1 = self.square(size=size)
#pybind11#        box = afwGeom.Box2D(afwGeom.Point2D(-1.0, -1.0), afwGeom.Point2D(1.0, 1.0))
#pybind11#        poly2 = Polygon(box)
#pybind11#        self.assertEqual(poly1, poly2)
#pybind11#
#pybind11#    def testBBox(self):
#pybind11#        """Test Polygon.getBBox"""
#pybind11#        size = 3.0
#pybind11#        poly = self.square(size=size)
#pybind11#        box = poly.getBBox()
#pybind11#        self.assertEqual(box.getMinX(), -size)
#pybind11#        self.assertEqual(box.getMinY(), -size)
#pybind11#        self.assertEqual(box.getMaxX(), size)
#pybind11#        self.assertEqual(box.getMaxY(), size)
#pybind11#
#pybind11#    def testCenter(self):
#pybind11#        """Test Polygon.calculateCenter"""
#pybind11#        for num in range(3, 30):
#pybind11#            poly = self.polygon(num)
#pybind11#            center = poly.calculateCenter()
#pybind11#            self.assertAlmostEqual(center.getX(), self.x0)
#pybind11#            self.assertAlmostEqual(center.getY(), self.y0)
#pybind11#
#pybind11#    def testContains(self):
#pybind11#        """Test Polygon.contains"""
#pybind11#        radius = 1.0
#pybind11#        for num in range(3, 30):
#pybind11#            poly = self.polygon(num, radius=radius)
#pybind11#            self.assertTrue(poly.contains(afwGeom.Point2D(self.x0, self.y0)))
#pybind11#            self.assertFalse(poly.contains(afwGeom.Point2D(self.x0 + radius, self.y0 + radius)))
#pybind11#
#pybind11#    def testOverlaps(self):
#pybind11#        """Test Polygon.overlaps"""
#pybind11#        radius = 1.0
#pybind11#        for num in range(3, 30):
#pybind11#            poly1 = self.polygon(num, radius=radius)
#pybind11#            poly2 = self.polygon(num, radius=radius, x0=radius, y0=radius)
#pybind11#            poly3 = self.polygon(num, radius=2*radius)
#pybind11#            poly4 = self.polygon(num, radius=radius, x0=3*radius, y0=3*radius)
#pybind11#            self.assertTrue(poly1.overlaps(poly2))
#pybind11#            self.assertTrue(poly2.overlaps(poly1))
#pybind11#            self.assertTrue(poly1.overlaps(poly3))
#pybind11#            self.assertTrue(poly3.overlaps(poly1))
#pybind11#            self.assertFalse(poly1.overlaps(poly4))
#pybind11#            self.assertFalse(poly4.overlaps(poly1))
#pybind11#
#pybind11#    def testIntersection(self):
#pybind11#        """Test Polygon.intersection"""
#pybind11#        poly1 = self.square(2.0, -1.0, -1.0)
#pybind11#        poly2 = self.square(2.0, +1.0, +1.0)
#pybind11#        poly3 = self.square(1.0, 0.0, 0.0)
#pybind11#        poly4 = self.square(1.0, +5.0, +5.0)
#pybind11#
#pybind11#        # intersectionSingle: assumes there's a single intersection (convex polygons)
#pybind11#        self.assertEqual(poly1.intersectionSingle(poly2), poly3)
#pybind11#        self.assertEqual(poly2.intersectionSingle(poly1), poly3)
#pybind11#        self.assertRaises(SinglePolygonException, poly1.intersectionSingle, poly4)
#pybind11#        self.assertRaises(SinglePolygonException, poly4.intersectionSingle, poly1)
#pybind11#
#pybind11#        # intersection: no assumptions
#pybind11#        polyList1 = poly1.intersection(poly2)
#pybind11#        polyList2 = poly2.intersection(poly1)
#pybind11#        self.assertEqual(polyList1, polyList2)
#pybind11#        self.assertEqual(len(polyList1), 1)
#pybind11#        self.assertEqual(polyList1[0], poly3)
#pybind11#        polyList3 = poly1.intersection(poly4)
#pybind11#        polyList4 = poly4.intersection(poly1)
#pybind11#        self.assertEqual(polyList3, polyList4)
#pybind11#        self.assertEqual(len(polyList3), 0)
#pybind11#
#pybind11#    def testUnion(self):
#pybind11#        """Test Polygon.union"""
#pybind11#        poly1 = self.square(2.0, -1.0, -1.0)
#pybind11#        poly2 = self.square(2.0, +1.0, +1.0)
#pybind11#        poly3 = Polygon([afwGeom.Point2D(x, y) for x, y in
#pybind11#                         ((-3.0, -3.0), (-3.0, +1.0), (-1.0, +1.0), (-1.0, +3.0),
#pybind11#                          (+3.0, +3.0), (+3.0, -1.0), (+1.0, -1.0), (+1.0, -3.0))])
#pybind11#        poly4 = self.square(1.0, +5.0, +5.0)
#pybind11#
#pybind11#        # unionSingle: assumes there's a single union (intersecting polygons)
#pybind11#        self.assertEqual(poly1.unionSingle(poly2), poly3)
#pybind11#        self.assertEqual(poly2.unionSingle(poly1), poly3)
#pybind11#        self.assertRaises(SinglePolygonException, poly1.unionSingle, poly4)
#pybind11#        self.assertRaises(SinglePolygonException, poly4.unionSingle, poly1)
#pybind11#
#pybind11#        # union: no assumptions
#pybind11#        polyList1 = poly1.union(poly2)
#pybind11#        polyList2 = poly2.union(poly1)
#pybind11#        self.assertEqual(polyList1, polyList2)
#pybind11#        self.assertEqual(len(polyList1), 1)
#pybind11#        self.assertEqual(polyList1[0], poly3)
#pybind11#        polyList3 = poly1.union(poly4)
#pybind11#        polyList4 = poly4.union(poly1)
#pybind11#        self.assertEqual(len(polyList3), 2)
#pybind11#        self.assertEqual(len(polyList3), len(polyList4))
#pybind11#        self.assertTrue((polyList3[0] == polyList4[0] and polyList3[1] == polyList4[1]) or
#pybind11#                        (polyList3[0] == polyList4[1] and polyList3[1] == polyList4[0]))
#pybind11#        self.assertTrue((polyList3[0] == poly1 and polyList3[1] == poly4) or
#pybind11#                        (polyList3[0] == poly4 and polyList3[1] == poly1))
#pybind11#
#pybind11#    def testSymDifference(self):
#pybind11#        """Test Polygon.symDifference"""
#pybind11#        poly1 = self.square(2.0, -1.0, -1.0)
#pybind11#        poly2 = self.square(2.0, +1.0, +1.0)
#pybind11#
#pybind11#        poly3 = Polygon([afwGeom.Point2D(x, y) for x, y in
#pybind11#                         ((-3.0, -3.0), (-3.0, +1.0), (-1.0, +1.0), (-1.0, -1.0), (+1.0, -1.0), (1.0, -3.0))])
#pybind11#        poly4 = Polygon([afwGeom.Point2D(x, y) for x, y in
#pybind11#                         ((-1.0, +1.0), (-1.0, +3.0), (+3.0, +3.0), (+3.0, -1.0), (+1.0, -1.0), (1.0, +1.0))])
#pybind11#
#pybind11#        diff1 = poly1.symDifference(poly2)
#pybind11#        diff2 = poly2.symDifference(poly1)
#pybind11#
#pybind11#        self.assertEqual(len(diff1), 2)
#pybind11#        self.assertEqual(len(diff2), 2)
#pybind11#        self.assertTrue((diff1[0] == diff2[0] and diff1[1] == diff2[1]) or
#pybind11#                        (diff1[1] == diff2[0] and diff1[0] == diff2[1]))
#pybind11#        self.assertTrue((diff1[0] == poly3 and diff1[1] == poly4) or
#pybind11#                        (diff1[1] == poly3 and diff1[0] == poly4))
#pybind11#
#pybind11#    def testConvexHull(self):
#pybind11#        """Test Polygon.convexHull"""
#pybind11#        poly1 = self.square(2.0, -1.0, -1.0)
#pybind11#        poly2 = self.square(2.0, +1.0, +1.0)
#pybind11#        poly = poly1.unionSingle(poly2)
#pybind11#        expected = Polygon([afwGeom.Point2D(x, y) for x, y in
#pybind11#                            ((-3.0, -3.0), (-3.0, +1.0), (-1.0, +3.0),
#pybind11#                             (+3.0, +3.0), (+3.0, -1.0), (+1.0, -3.0))])
#pybind11#        self.assertEqual(poly.convexHull(), expected)
#pybind11#
#pybind11#    def testImage(self):
#pybind11#        """Test Polygon.createImage"""
#pybind11#        for i, num in enumerate(range(3, 30)):
#pybind11#            poly = self.polygon(num, 25, 75, 75)
#pybind11#            box = afwGeom.Box2I(afwGeom.Point2I(15, 15), afwGeom.Extent2I(115, 115))
#pybind11#            image = poly.createImage(box)
#pybind11#            if DEBUG:
#pybind11#                import lsst.afw.display.ds9 as ds9
#pybind11#                ds9.mtv(image, frame=i+1, title="Polygon nside=%d" % num)
#pybind11#                for p1, p2 in poly.getEdges():
#pybind11#                    ds9.line((p1, p2), frame=i+1)
#pybind11#            self.assertAlmostEqual(image.getArray().sum()/poly.calculateArea(), 1.0, 6)
#pybind11#
#pybind11#    def testTransform(self):
#pybind11#        """Test constructor for Polygon involving transforms"""
#pybind11#        box = afwGeom.Box2D(afwGeom.Point2D(0.0, 0.0), afwGeom.Point2D(123.4, 567.8))
#pybind11#        poly1 = Polygon(box)
#pybind11#        scale = (0.2*afwGeom.arcseconds).asDegrees()
#pybind11#        wcs = afwImage.makeWcs(afwCoord.Coord(0.0*afwGeom.degrees, 0.0*afwGeom.degrees),
#pybind11#                               afwGeom.Point2D(0.0, 0.0), scale, 0.0, 0.0, scale)
#pybind11#        transform = afwImage.XYTransformFromWcsPair(wcs, wcs)
#pybind11#        poly2 = Polygon(box, transform)
#pybind11#
#pybind11#        # We lose some very small precision in the XYTransformFromWcsPair
#pybind11#        # so we can't compare the polygons directly.
#pybind11#        self.assertEqual(poly1.getNumEdges(), poly2.getNumEdges())
#pybind11#        for p1, p2 in zip(poly1.getVertices(), poly2.getVertices()):
#pybind11#            self.assertAlmostEqual(p1.getX(), p2.getX())
#pybind11#            self.assertAlmostEqual(p1.getY(), p2.getY())
#pybind11#
#pybind11#        transform = afwGeom.AffineTransform.makeScaling(1.0)
#pybind11#        poly3 = Polygon(box, transform)
#pybind11#        self.assertEqual(poly1, poly3)
#pybind11#
#pybind11#    def testIteration(self):
#pybind11#        """Test iteration over polygon"""
#pybind11#        for num in range(3, 30):
#pybind11#            poly = self.polygon(num)
#pybind11#            self.assertEqual(len(poly), num)
#pybind11#            points1 = [p for p in poly]
#pybind11#            points2 = poly.getVertices()
#pybind11#            self.assertEqual(points2[0], points2[-1])  # Closed representation
#pybind11#            for p1, p2 in zip(points1, points2):
#pybind11#                self.assertEqual(p1, p2)
#pybind11#            for i, p1 in enumerate(points1):
#pybind11#                self.assertEqual(poly[i], p1)
#pybind11#
#pybind11#    def testSubSample(self):
#pybind11#        """Test Polygon.subSample"""
#pybind11#        for num in range(3, 30):
#pybind11#            poly = self.polygon(num)
#pybind11#            sub = poly.subSample(2)
#pybind11#
#pybind11#            if DEBUG:
#pybind11#                import matplotlib.pyplot as plt
#pybind11#                axes = poly.plot(c='b')
#pybind11#                sub.plot(axes, c='r')
#pybind11#                plt.show()
#pybind11#
#pybind11#            self.assertEqual(len(sub), 2*num)
#pybind11#            self.assertAlmostEqual(sub.calculateArea(), poly.calculateArea())
#pybind11#            self.assertAlmostEqual(sub.calculatePerimeter(), poly.calculatePerimeter())
#pybind11#            polyCenter = poly.calculateCenter()
#pybind11#            subCenter = sub.calculateCenter()
#pybind11#            self.assertAlmostEqual(polyCenter[0], subCenter[0])
#pybind11#            self.assertAlmostEqual(polyCenter[1], subCenter[1])
#pybind11#            for i in range(num):
#pybind11#                self.assertEqual(sub[2*i], poly[i])
#pybind11#
#pybind11#            sub = poly.subSample(0.1)
#pybind11#            self.assertAlmostEqual(sub.calculateArea(), poly.calculateArea())
#pybind11#            self.assertAlmostEqual(sub.calculatePerimeter(), poly.calculatePerimeter())
#pybind11#            polyCenter = poly.calculateCenter()
#pybind11#            subCenter = sub.calculateCenter()
#pybind11#            self.assertAlmostEqual(polyCenter[0], subCenter[0])
#pybind11#            self.assertAlmostEqual(polyCenter[1], subCenter[1])
#pybind11#
#pybind11#    def testTransform2(self):
#pybind11#        scale = 2.0
#pybind11#        shift = afwGeom.Extent2D(3.0, 4.0)
#pybind11#        transform = afwGeom.AffineTransform.makeTranslation(shift)*afwGeom.AffineTransform.makeScaling(scale)
#pybind11#        for num in range(3, 30):
#pybind11#            small = self.polygon(num, 1.0, 0.0, 0.0)
#pybind11#            large = small.transform(transform)
#pybind11#            expect = self.polygon(num, scale, shift[0], shift[1])
#pybind11#            self.assertEqual(large, expect)
#pybind11#
#pybind11#            if DEBUG:
#pybind11#                import matplotlib.pyplot as plt
#pybind11#                axes = small.plot(c='k')
#pybind11#                large.plot(axes, c='b')
#pybind11#                plt.show()
#pybind11#
#pybind11#    def testReadWrite(self):
#pybind11#        """Test that polygons can be read and written to fits files"""
#pybind11#        for num in range(3, 30):
#pybind11#            poly = self.polygon(num)
#pybind11#            filename = 'polygon.fits'
#pybind11#            poly.writeFits(filename)
#pybind11#            poly2 = Polygon.readFits(filename)
#pybind11#            self.assertEqual(poly, poly2)
#pybind11#            os.remove(filename)
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

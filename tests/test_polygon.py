#
# LSST Data Management System
# Copyright 2008-2014 LSST Corporation.
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

from __future__ import absolute_import, division, print_function
import pickle
import unittest
import os

from builtins import zip
from builtins import range
import numpy as np

import lsst.utils.tests
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.coord as afwCoord
from lsst.afw.geom.polygon import Polygon, SinglePolygonException

DEBUG = False


def circle(radius, num, x0=0.0, y0=0.0):
    """Generate points on a circle

    @param radius: radius of circle
    @param num: number of points
    @param x0,y0: Offset in x,y
    @return x,y coordinates as numpy array
    """
    theta = np.linspace(0, 2*np.pi, num=num, endpoint=False)
    x = radius*np.cos(theta) + x0
    y = radius*np.sin(theta) + y0
    return np.array([x, y]).transpose()


class PolygonTest(lsst.utils.tests.TestCase):

    def setUp(self):
        self.x0 = 0.0
        self.y0 = 0.0

    def polygon(self, num, radius=1.0, x0=None, y0=None):
        """Generate a polygon

        @param num: Number of points
        @param radius: Radius of polygon
        @param x0,y0: Offset of center
        @return polygon
        """
        if x0 is None:
            x0 = self.x0
        if y0 is None:
            y0 = self.y0
        points = circle(radius, num, x0=x0, y0=y0)
        return Polygon([afwGeom.Point2D(x, y) for x, y in reversed(points)])

    def square(self, size=1.0, x0=0, y0=0):
        """Generate a square

        @param size: Half-length of the sides
        @param x0,y0: Offset of center
        """
        return Polygon([afwGeom.Point2D(size*x + x0, size*y + y0) for
                        x, y in ((-1, -1), (-1, 1), (1, 1), (1, -1))])

    def testGetters(self):
        """Test Polygon getters"""
        for num in range(3, 30):
            poly = self.polygon(num)
            self.assertEqual(poly, poly)
            self.assertNotEqual(poly, self.square(1.0, 2.0, 3.0))
            self.assertEqual(poly.getNumEdges(), num)
            # One extra for the closing point
            self.assertEqual(len(poly.getVertices()), num + 1)
            self.assertEqual(len(poly.getEdges()), num)
            perimeter = 0.0
            for p1, p2 in poly.getEdges():
                perimeter += np.hypot(p1.getX() - p2.getX(),
                                      p1.getY() - p2.getY())
            self.assertAlmostEqual(poly.calculatePerimeter(), perimeter)

        size = 3.0
        poly = self.square(size=size)
        self.assertEqual(poly.calculateArea(), (2*size)**2)
        self.assertEqual(poly.calculatePerimeter(), 2*size*4)
        edges = poly.getEdges()
        self.assertEqual(len(edges), 4)
        perimeter = 0.0
        for p1, p2 in edges:
            self.assertEqual(abs(p1.getX()), size)
            self.assertEqual(abs(p1.getY()), size)
            self.assertEqual(abs(p2.getX()), size)
            self.assertEqual(abs(p2.getY()), size)
            self.assertNotEqual(p1, p2)

    def testPickle(self):
        for num in range(3, 30):
            poly = self.polygon(num)
            self.assertEqual(pickle.loads(pickle.dumps(poly)), poly)

    def testFromBox(self):
        size = 1.0
        poly1 = self.square(size=size)
        box = afwGeom.Box2D(afwGeom.Point2D(-1.0, -1.0),
                            afwGeom.Point2D(1.0, 1.0))
        poly2 = Polygon(box)
        self.assertEqual(poly1, poly2)

    def testBBox(self):
        """Test Polygon.getBBox"""
        size = 3.0
        poly = self.square(size=size)
        box = poly.getBBox()
        self.assertEqual(box.getMinX(), -size)
        self.assertEqual(box.getMinY(), -size)
        self.assertEqual(box.getMaxX(), size)
        self.assertEqual(box.getMaxY(), size)

    def testCenter(self):
        """Test Polygon.calculateCenter"""
        for num in range(3, 30):
            poly = self.polygon(num)
            center = poly.calculateCenter()
            self.assertAlmostEqual(center.getX(), self.x0)
            self.assertAlmostEqual(center.getY(), self.y0)

    def testContains(self):
        """Test Polygon.contains"""
        radius = 1.0
        for num in range(3, 30):
            poly = self.polygon(num, radius=radius)
            self.assertTrue(poly.contains(afwGeom.Point2D(self.x0, self.y0)))
            self.assertFalse(poly.contains(
                afwGeom.Point2D(self.x0 + radius, self.y0 + radius)))

    def testOverlaps(self):
        """Test Polygon.overlaps"""
        radius = 1.0
        for num in range(3, 30):
            poly1 = self.polygon(num, radius=radius)
            poly2 = self.polygon(num, radius=radius, x0=radius, y0=radius)
            poly3 = self.polygon(num, radius=2*radius)
            poly4 = self.polygon(num, radius=radius, x0=3*radius, y0=3*radius)
            self.assertTrue(poly1.overlaps(poly2))
            self.assertTrue(poly2.overlaps(poly1))
            self.assertTrue(poly1.overlaps(poly3))
            self.assertTrue(poly3.overlaps(poly1))
            self.assertFalse(poly1.overlaps(poly4))
            self.assertFalse(poly4.overlaps(poly1))

    def testIntersection(self):
        """Test Polygon.intersection"""
        poly1 = self.square(2.0, -1.0, -1.0)
        poly2 = self.square(2.0, +1.0, +1.0)
        poly3 = self.square(1.0, 0.0, 0.0)
        poly4 = self.square(1.0, +5.0, +5.0)

        # intersectionSingle: assumes there's a single intersection (convex
        # polygons)
        self.assertEqual(poly1.intersectionSingle(poly2), poly3)
        self.assertEqual(poly2.intersectionSingle(poly1), poly3)
        self.assertRaises(SinglePolygonException,
                          poly1.intersectionSingle, poly4)
        self.assertRaises(SinglePolygonException,
                          poly4.intersectionSingle, poly1)

        # intersection: no assumptions
        polyList1 = poly1.intersection(poly2)
        polyList2 = poly2.intersection(poly1)
        self.assertEqual(polyList1, polyList2)
        self.assertEqual(len(polyList1), 1)
        self.assertEqual(polyList1[0], poly3)
        polyList3 = poly1.intersection(poly4)
        polyList4 = poly4.intersection(poly1)
        self.assertEqual(polyList3, polyList4)
        self.assertEqual(len(polyList3), 0)

    def testUnion(self):
        """Test Polygon.union"""
        poly1 = self.square(2.0, -1.0, -1.0)
        poly2 = self.square(2.0, +1.0, +1.0)
        poly3 = Polygon([afwGeom.Point2D(x, y) for x, y in
                         ((-3.0, -3.0), (-3.0, +1.0), (-1.0, +1.0), (-1.0, +3.0),
                          (+3.0, +3.0), (+3.0, -1.0), (+1.0, -1.0), (+1.0, -3.0))])
        poly4 = self.square(1.0, +5.0, +5.0)

        # unionSingle: assumes there's a single union (intersecting polygons)
        self.assertEqual(poly1.unionSingle(poly2), poly3)
        self.assertEqual(poly2.unionSingle(poly1), poly3)
        self.assertRaises(SinglePolygonException, poly1.unionSingle, poly4)
        self.assertRaises(SinglePolygonException, poly4.unionSingle, poly1)

        # union: no assumptions
        polyList1 = poly1.union(poly2)
        polyList2 = poly2.union(poly1)
        self.assertEqual(polyList1, polyList2)
        self.assertEqual(len(polyList1), 1)
        self.assertEqual(polyList1[0], poly3)
        polyList3 = poly1.union(poly4)
        polyList4 = poly4.union(poly1)
        self.assertEqual(len(polyList3), 2)
        self.assertEqual(len(polyList3), len(polyList4))
        self.assertTrue((polyList3[0] == polyList4[0] and polyList3[1] == polyList4[1]) or
                        (polyList3[0] == polyList4[1] and polyList3[1] == polyList4[0]))
        self.assertTrue((polyList3[0] == poly1 and polyList3[1] == poly4) or
                        (polyList3[0] == poly4 and polyList3[1] == poly1))

    def testSymDifference(self):
        """Test Polygon.symDifference"""
        poly1 = self.square(2.0, -1.0, -1.0)
        poly2 = self.square(2.0, +1.0, +1.0)

        poly3 = Polygon([afwGeom.Point2D(x, y) for x, y in
                         ((-3.0, -3.0), (-3.0, +1.0), (-1.0, +1.0), (-1.0, -1.0), (+1.0, -1.0), (1.0, -3.0))])
        poly4 = Polygon([afwGeom.Point2D(x, y) for x, y in
                         ((-1.0, +1.0), (-1.0, +3.0), (+3.0, +3.0), (+3.0, -1.0), (+1.0, -1.0), (1.0, +1.0))])

        diff1 = poly1.symDifference(poly2)
        diff2 = poly2.symDifference(poly1)

        self.assertEqual(len(diff1), 2)
        self.assertEqual(len(diff2), 2)
        self.assertTrue((diff1[0] == diff2[0] and diff1[1] == diff2[1]) or
                        (diff1[1] == diff2[0] and diff1[0] == diff2[1]))
        self.assertTrue((diff1[0] == poly3 and diff1[1] == poly4) or
                        (diff1[1] == poly3 and diff1[0] == poly4))

    def testConvexHull(self):
        """Test Polygon.convexHull"""
        poly1 = self.square(2.0, -1.0, -1.0)
        poly2 = self.square(2.0, +1.0, +1.0)
        poly = poly1.unionSingle(poly2)
        expected = Polygon([afwGeom.Point2D(x, y) for x, y in
                            ((-3.0, -3.0), (-3.0, +1.0), (-1.0, +3.0),
                             (+3.0, +3.0), (+3.0, -1.0), (+1.0, -3.0))])
        self.assertEqual(poly.convexHull(), expected)

    def testImage(self):
        """Test Polygon.createImage"""
        for i, num in enumerate(range(3, 30)):
            poly = self.polygon(num, 25, 75, 75)
            box = afwGeom.Box2I(afwGeom.Point2I(15, 15),
                                afwGeom.Extent2I(115, 115))
            image = poly.createImage(box)
            if DEBUG:
                import lsst.afw.display.ds9 as ds9
                ds9.mtv(image, frame=i+1, title="Polygon nside=%d" % num)
                for p1, p2 in poly.getEdges():
                    ds9.line((p1, p2), frame=i+1)
            self.assertAlmostEqual(
                image.getArray().sum()/poly.calculateArea(), 1.0, 6)

    def testTransform(self):
        """Test constructor for Polygon involving transforms"""
        box = afwGeom.Box2D(afwGeom.Point2D(0.0, 0.0),
                            afwGeom.Point2D(123.4, 567.8))
        poly1 = Polygon(box)
        scale = (0.2*afwGeom.arcseconds).asDegrees()
        wcs = afwImage.makeWcs(afwCoord.Coord(0.0*afwGeom.degrees, 0.0*afwGeom.degrees),
                               afwGeom.Point2D(0.0, 0.0), scale, 0.0, 0.0, scale)
        transform = afwImage.XYTransformFromWcsPair(wcs, wcs)
        poly2 = Polygon(box, transform)

        # We lose some very small precision in the XYTransformFromWcsPair
        # so we can't compare the polygons directly.
        self.assertEqual(poly1.getNumEdges(), poly2.getNumEdges())
        for p1, p2 in zip(poly1.getVertices(), poly2.getVertices()):
            self.assertAlmostEqual(p1.getX(), p2.getX())
            self.assertAlmostEqual(p1.getY(), p2.getY())

        transform = afwGeom.AffineTransform.makeScaling(1.0)
        poly3 = Polygon(box, transform)
        self.assertEqual(poly1, poly3)

    def testIteration(self):
        """Test iteration over polygon"""
        for num in range(3, 30):
            poly = self.polygon(num)
            self.assertEqual(len(poly), num)
            points1 = [p for p in poly]
            points2 = poly.getVertices()
            self.assertEqual(len(points1), num + 1)
            self.assertEqual(len(points2), num + 1)
            self.assertEqual(points2[0], points2[-1])  # Closed representation
            for p1, p2 in zip(points1, points2):
                self.assertEqual(p1, p2)
            for i, p1 in enumerate(points1):
                self.assertEqual(poly[i], p1)

    def testSubSample(self):
        """Test Polygon.subSample"""
        for num in range(3, 30):
            poly = self.polygon(num)
            sub = poly.subSample(2)

            if DEBUG:
                import matplotlib.pyplot as plt
                axes = poly.plot(c='b')
                sub.plot(axes, c='r')
                plt.show()

            self.assertEqual(len(sub), 2*num)
            self.assertAlmostEqual(sub.calculateArea(), poly.calculateArea())
            self.assertAlmostEqual(
                sub.calculatePerimeter(), poly.calculatePerimeter())
            polyCenter = poly.calculateCenter()
            subCenter = sub.calculateCenter()
            self.assertAlmostEqual(polyCenter[0], subCenter[0])
            self.assertAlmostEqual(polyCenter[1], subCenter[1])
            for i in range(num):
                self.assertEqual(sub[2*i], poly[i])

            sub = poly.subSample(0.1)
            self.assertAlmostEqual(sub.calculateArea(), poly.calculateArea())
            self.assertAlmostEqual(
                sub.calculatePerimeter(), poly.calculatePerimeter())
            polyCenter = poly.calculateCenter()
            subCenter = sub.calculateCenter()
            self.assertAlmostEqual(polyCenter[0], subCenter[0])
            self.assertAlmostEqual(polyCenter[1], subCenter[1])

    def testTransform2(self):
        scale = 2.0
        shift = afwGeom.Extent2D(3.0, 4.0)
        transform = afwGeom.AffineTransform.makeTranslation(shift) * \
            afwGeom.AffineTransform.makeScaling(scale)
        for num in range(3, 30):
            small = self.polygon(num, 1.0, 0.0, 0.0)
            large = small.transform(transform)
            expect = self.polygon(num, scale, shift[0], shift[1])
            self.assertEqual(large, expect)

            if DEBUG:
                import matplotlib.pyplot as plt
                axes = small.plot(c='k')
                large.plot(axes, c='b')
                plt.show()

    def testReadWrite(self):
        """Test that polygons can be read and written to fits files"""
        for num in range(3, 30):
            poly = self.polygon(num)
            filename = 'polygon.fits'
            poly.writeFits(filename)
            poly2 = Polygon.readFits(filename)
            self.assertEqual(poly, poly2)
            os.remove(filename)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

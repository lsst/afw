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
from __future__ import absolute_import, division, print_function
import unittest

from builtins import range
import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.geom as geom


class Box2ITestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        np.random.seed(1)

    def testEmpty(self):
        box = geom.Box2I()
        self.assertTrue(box.isEmpty())
        self.assertEqual(box.getWidth(), 0)
        self.assertEqual(box.getHeight(), 0)
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                point = geom.Point2I(x, y)
                self.assertFalse(box.contains(point))
                box.include(point)
                self.assertTrue(box.contains(point))
                box = geom.Box2I()
        box.grow(3)
        self.assertTrue(box.isEmpty())

    def testConstruction(self):
        for n in range(10):
            xmin, xmax, ymin, ymax = [int(i) for i in np.random.randint(low=-5, high=5, size=4)]
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            pmin = geom.Point2I(xmin, ymin)
            pmax = geom.Point2I(xmax, ymax)
            # min/max constructor
            box = geom.Box2I(pmin, pmax)
            self.assertEqual(box.getMin(), pmin)
            self.assertEqual(box.getMax(), pmax)
            box = geom.Box2I(pmax, pmin)
            self.assertEqual(box.getMin(), pmin)
            self.assertEqual(box.getMax(), pmax)
            box = geom.Box2I(pmin, pmax, False)
            self.assertEqual(box.getMin(), pmin)
            self.assertEqual(box.getMax(), pmax)
            box = geom.Box2I(pmax, pmin, False)
            self.assertTrue(box.isEmpty() or pmax == pmin)
            self.assertEqual(box, geom.Box2I(box))
            # min/dim constructor
            dim = geom.Extent2I(1) + pmax - pmin
            if any(dim.eq(0)):
                box = geom.Box2I(pmin, dim)
                self.assertTrue(box.isEmpty())
                box = geom.Box2I(pmin, dim, False)
                self.assertTrue(box.isEmpty())
            else:
                box = geom.Box2I(pmin, dim)
                self.assertEqual(box.getMin(), pmin)
                self.assertEqual(box.getDimensions(), dim)
                box = geom.Box2I(pmin, dim, False)
                self.assertEqual(box.getMin(), pmin)
                self.assertEqual(box.getDimensions(), dim)
                dim = -dim
                box = geom.Box2I(pmin, dim)
                self.assertEqual(box.getMin(), pmin + dim + geom.Extent2I(1))
                self.assertEqual(box.getDimensions(),
                                 geom.Extent2I(abs(dim.getX()), abs(dim.getY())))

    def testOverflowDetection(self):
        try:
            box = geom.Box2I(geom.Point2I(2147483645, 149), geom.Extent2I(8, 8))
        except lsst.pex.exceptions.OverflowError:
            pass
        else:
            # On some platforms, sizeof(int) may be > 4, so this test doesn't overflow.
            # In that case, we just verify that there was in fact no overflow.
            # It's hard to construct a more platform-independent test because Python doesn't
            # provide an easy way to get sizeof(int); note that sys.maxint is usually sizeof(long).
            self.assertLess(box.getWidth(), 0)

    def testSwap(self):
        x00, y00, x01, y01 = (0, 1, 2, 3)
        x10, y10, x11, y11 = (4, 5, 6, 7)

        box0 = geom.Box2I(geom.PointI(x00, y00), geom.PointI(x01, y01))
        box1 = geom.Box2I(geom.PointI(x10, y10), geom.PointI(x11, y11))
        box0.swap(box1)

        self.assertEqual(box0.getMinX(), x10)
        self.assertEqual(box0.getMinY(), y10)
        self.assertEqual(box0.getMaxX(), x11)
        self.assertEqual(box0.getMaxY(), y11)

        self.assertEqual(box1.getMinX(), x00)
        self.assertEqual(box1.getMinY(), y00)
        self.assertEqual(box1.getMaxX(), x01)
        self.assertEqual(box1.getMaxY(), y01)

    def testConversion(self):
        for n in range(10):
            xmin, xmax, ymin, ymax = np.random.uniform(low=-10, high=10, size=4)
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            fpMin = geom.Point2D(xmin, ymin)
            fpMax = geom.Point2D(xmax, ymax)
            if any((fpMax-fpMin).lt(3)):
                continue  # avoid empty boxes
            fpBox = geom.Box2D(fpMin, fpMax)
            intBoxBig = geom.Box2I(fpBox, geom.Box2I.EXPAND)
            fpBoxBig = geom.Box2D(intBoxBig)
            intBoxSmall = geom.Box2I(fpBox, geom.Box2I.SHRINK)
            fpBoxSmall = geom.Box2D(intBoxSmall)
            self.assertTrue(fpBoxBig.contains(fpBox))
            self.assertTrue(fpBox.contains(fpBoxSmall))
            self.assertTrue(intBoxBig.contains(intBoxSmall))
            self.assertTrue(geom.Box2D(intBoxBig))
            self.assertEqual(geom.Box2I(fpBoxBig, geom.Box2I.EXPAND), intBoxBig)
            self.assertEqual(geom.Box2I(fpBoxSmall, geom.Box2I.SHRINK), intBoxSmall)
        self.assertTrue(geom.Box2I(geom.Box2D()).isEmpty())
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, geom.Box2I,
                          geom.Box2D(geom.Point2D(), geom.Point2D(float("inf"), float("inf"))))

    def testAccessors(self):
        xmin, xmax, ymin, ymax = [int(i) for i in np.random.randint(low=-5, high=5, size=4)]
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        pmin = geom.Point2I(xmin, ymin)
        pmax = geom.Point2I(xmax, ymax)
        box = geom.Box2I(pmin, pmax, True)
        self.assertEqual(pmin, box.getMin())
        self.assertEqual(pmax, box.getMax())
        self.assertEqual(box.getMinX(), xmin)
        self.assertEqual(box.getMinY(), ymin)
        self.assertEqual(box.getMaxX(), xmax)
        self.assertEqual(box.getMaxY(), ymax)
        self.assertEqual(box.getBegin(), pmin)
        self.assertEqual(box.getEnd(), (pmax + geom.Extent2I(1)))
        self.assertEqual(box.getBeginX(), xmin)
        self.assertEqual(box.getBeginY(), ymin)
        self.assertEqual(box.getEndX(), xmax + 1)
        self.assertEqual(box.getEndY(), ymax + 1)
        self.assertEqual(box.getDimensions(), (pmax - pmin + geom.Extent2I(1)))
        self.assertEqual(box.getWidth(), (xmax - xmin + 1))
        self.assertEqual(box.getHeight(), (ymax - ymin + 1))
        self.assertAlmostEqual(box.getArea(), box.getWidth() * box.getHeight(),
                               places=14)
        corners = box.getCorners()
        self.assertEqual(corners[0], box.getMin())
        self.assertEqual(corners[1].getX(), box.getMaxX())
        self.assertEqual(corners[1].getY(), box.getMinY())
        self.assertEqual(corners[2], box.getMax())
        self.assertEqual(corners[3].getX(), box.getMinX())
        self.assertEqual(corners[3].getY(), box.getMaxY())

    def testRelations(self):
        box = geom.Box2I(geom.Point2I(-2, -3), geom.Point2I(2, 1), True)
# TODO: decide if we want lists and boxes to compare equal
#        self.assertNotEqual(box, (3, 4, 5))  # should not throw
        self.assertTrue(box.contains(geom.Point2I(0, 0)))
        self.assertTrue(box.contains(geom.Point2I(-2, -3)))
        self.assertTrue(box.contains(geom.Point2I(2, -3)))
        self.assertTrue(box.contains(geom.Point2I(2, 1)))
        self.assertTrue(box.contains(geom.Point2I(-2, 1)))
        self.assertFalse(box.contains(geom.Point2I(-2, -4)))
        self.assertFalse(box.contains(geom.Point2I(-3, -3)))
        self.assertFalse(box.contains(geom.Point2I(2, -4)))
        self.assertFalse(box.contains(geom.Point2I(3, -3)))
        self.assertFalse(box.contains(geom.Point2I(3, 1)))
        self.assertFalse(box.contains(geom.Point2I(2, 2)))
        self.assertFalse(box.contains(geom.Point2I(-3, 1)))
        self.assertFalse(box.contains(geom.Point2I(-2, 2)))
        self.assertTrue(box.contains(geom.Box2I(geom.Point2I(-1, -2), geom.Point2I(1, 0))))
        self.assertTrue(box.contains(box))
        self.assertFalse(box.contains(geom.Box2I(geom.Point2I(-2, -3), geom.Point2I(2, 2))))
        self.assertFalse(box.contains(geom.Box2I(geom.Point2I(-2, -3), geom.Point2I(3, 1))))
        self.assertFalse(box.contains(geom.Box2I(geom.Point2I(-3, -3), geom.Point2I(2, 1))))
        self.assertFalse(box.contains(geom.Box2I(geom.Point2I(-3, -4), geom.Point2I(2, 1))))
        self.assertTrue(box.overlaps(geom.Box2I(geom.Point2I(-2, -3), geom.Point2I(2, 2))))
        self.assertTrue(box.overlaps(geom.Box2I(geom.Point2I(-2, -3), geom.Point2I(3, 1))))
        self.assertTrue(box.overlaps(geom.Box2I(geom.Point2I(-3, -3), geom.Point2I(2, 1))))
        self.assertTrue(box.overlaps(geom.Box2I(geom.Point2I(-3, -4), geom.Point2I(2, 1))))
        self.assertTrue(box.overlaps(geom.Box2I(geom.Point2I(-1, -2), geom.Point2I(1, 0))))
        self.assertTrue(box.overlaps(box))
        self.assertFalse(box.overlaps(geom.Box2I(geom.Point2I(-5, -3), geom.Point2I(-3, 1))))
        self.assertFalse(box.overlaps(geom.Box2I(geom.Point2I(-2, -6), geom.Point2I(2, -4))))
        self.assertFalse(box.overlaps(geom.Box2I(geom.Point2I(3, -3), geom.Point2I(4, 1))))
        self.assertFalse(box.overlaps(geom.Box2I(geom.Point2I(-2, 2), geom.Point2I(2, 2))))

    def testMutators(self):
        box = geom.Box2I(geom.Point2I(-2, -3), geom.Point2I(2, 1), True)
        box.grow(1)
        self.assertEqual(box, geom.Box2I(geom.Point2I(-3, -4), geom.Point2I(3, 2), True))
        box.grow(geom.Extent2I(2, 3))
        self.assertEqual(box, geom.Box2I(geom.Point2I(-5, -7), geom.Point2I(5, 5), True))
        box.shift(geom.Extent2I(3, 2))
        self.assertEqual(box, geom.Box2I(geom.Point2I(-2, -5), geom.Point2I(8, 7), True))
        box.include(geom.Point2I(-4, 2))
        self.assertEqual(box, geom.Box2I(geom.Point2I(-4, -5), geom.Point2I(8, 7), True))
        box.include(geom.Point2I(0, -6))
        self.assertEqual(box, geom.Box2I(geom.Point2I(-4, -6), geom.Point2I(8, 7), True))
        box.include(geom.Box2I(geom.Point2I(0, 0), geom.Point2I(10, 11), True))
        self.assertEqual(box, geom.Box2I(geom.Point2I(-4, -6), geom.Point2I(10, 11), True))
        box.clip(geom.Box2I(geom.Point2I(0, 0), geom.Point2I(11, 12), True))
        self.assertEqual(box, geom.Box2I(geom.Point2I(0, 0), geom.Point2I(10, 11), True))
        box.clip(geom.Box2I(geom.Point2I(-1, -2), geom.Point2I(5, 4), True))
        self.assertEqual(box, geom.Box2I(geom.Point2I(0, 0), geom.Point2I(5, 4), True))


class Box2DTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        np.random.seed(1)

    def testEmpty(self):
        box = geom.Box2D()
        self.assertTrue(box.isEmpty())
        self.assertEqual(box.getWidth(), 0.0)
        self.assertEqual(box.getHeight(), 0.0)
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                point = geom.Point2D(x, y)
                self.assertFalse(box.contains(point))
                box.include(point)
                self.assertTrue(box.contains(point))
                box = geom.Box2D()
        box.grow(3)
        self.assertTrue(box.isEmpty())

    def testConstruction(self):
        for n in range(10):
            xmin, xmax, ymin, ymax = np.random.uniform(low=-5, high=5, size=4)
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            pmin = geom.Point2D(xmin, ymin)
            pmax = geom.Point2D(xmax, ymax)
            # min/max constructor
            box = geom.Box2D(pmin, pmax)
            self.assertEqual(box.getMin(), pmin)
            self.assertEqual(box.getMax(), pmax)
            box = geom.Box2D(pmax, pmin)
            self.assertEqual(box.getMin(), pmin)
            self.assertEqual(box.getMax(), pmax)
            box = geom.Box2D(pmin, pmax, False)
            self.assertEqual(box.getMin(), pmin)
            self.assertEqual(box.getMax(), pmax)
            box = geom.Box2D(pmax, pmin, False)
            self.assertTrue(box.isEmpty())
            self.assertEqual(box, geom.Box2D(box))
            # min/dim constructor
            dim = pmax - pmin
            if any(dim.eq(0)):
                box = geom.Box2D(pmin, dim)
                self.assertTrue(box.isEmpty())
                box = geom.Box2D(pmin, dim, False)
                self.assertTrue(box.isEmpty())
            else:
                box = geom.Box2D(pmin, dim)
                self.assertEqual(box.getMin(), pmin)
                self.assertEqual(box.getDimensions(), dim)
                box = geom.Box2D(pmin, dim, False)
                self.assertEqual(box.getMin(), pmin)
                self.assertEqual(box.getDimensions(), dim)
                dim = -dim
                box = geom.Box2D(pmin, dim)
                self.assertEqual(box.getMin(), pmin + dim)
                self.assertFloatsAlmostEqual(box.getDimensions(),
                                             geom.Extent2D(abs(dim.getX()), abs(dim.getY())))

    def testSwap(self):
        x00, y00, x01, y01 = (0., 1., 2., 3.)
        x10, y10, x11, y11 = (4., 5., 6., 7.)

        box0 = geom.Box2D(geom.PointD(x00, y00), geom.PointD(x01, y01))
        box1 = geom.Box2D(geom.PointD(x10, y10), geom.PointD(x11, y11))
        box0.swap(box1)

        self.assertEqual(box0.getMinX(), x10)
        self.assertEqual(box0.getMinY(), y10)
        self.assertEqual(box0.getMaxX(), x11)
        self.assertEqual(box0.getMaxY(), y11)

        self.assertEqual(box1.getMinX(), x00)
        self.assertEqual(box1.getMinY(), y00)
        self.assertEqual(box1.getMaxX(), x01)
        self.assertEqual(box1.getMaxY(), y01)

    def testAccessors(self):
        xmin, xmax, ymin, ymax = np.random.uniform(low=-5, high=5, size=4)
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        pmin = geom.Point2D(xmin, ymin)
        pmax = geom.Point2D(xmax, ymax)
        box = geom.Box2D(pmin, pmax, True)
        self.assertEqual(pmin, box.getMin())
        self.assertEqual(pmax, box.getMax())
        self.assertEqual(box.getMinX(), xmin)
        self.assertEqual(box.getMinY(), ymin)
        self.assertEqual(box.getMaxX(), xmax)
        self.assertEqual(box.getMaxY(), ymax)
        self.assertEqual(box.getDimensions(), (pmax - pmin))
        self.assertEqual(box.getWidth(), (xmax - xmin))
        self.assertEqual(box.getHeight(), (ymax - ymin))
        self.assertEqual(box.getArea(), box.getWidth() * box.getHeight())
        self.assertEqual(box.getCenterX(), 0.5*(pmax.getX() + pmin.getX()))
        self.assertEqual(box.getCenterY(), 0.5*(pmax.getY() + pmin.getY()))
        self.assertEqual(box.getCenter().getX(), box.getCenterX())
        self.assertEqual(box.getCenter().getY(), box.getCenterY())
        corners = box.getCorners()
        self.assertEqual(corners[0], box.getMin())
        self.assertEqual(corners[1].getX(), box.getMaxX())
        self.assertEqual(corners[1].getY(), box.getMinY())
        self.assertEqual(corners[2], box.getMax())
        self.assertEqual(corners[3].getX(), box.getMinX())
        self.assertEqual(corners[3].getY(), box.getMaxY())

    def testRelations(self):
        box = geom.Box2D(geom.Point2D(-2, -3), geom.Point2D(2, 1), True)
        self.assertTrue(box.contains(geom.Point2D(0, 0)))
        self.assertTrue(box.contains(geom.Point2D(-2, -3)))
        self.assertFalse(box.contains(geom.Point2D(2, -3)))
        self.assertFalse(box.contains(geom.Point2D(2, 1)))
        self.assertFalse(box.contains(geom.Point2D(-2, 1)))
        self.assertTrue(box.contains(geom.Box2D(geom.Point2D(-1, -2), geom.Point2D(1, 0))))
        self.assertTrue(box.contains(box))
        self.assertFalse(box.contains(geom.Box2D(geom.Point2D(-2, -3), geom.Point2D(2, 2))))
        self.assertFalse(box.contains(geom.Box2D(geom.Point2D(-2, -3), geom.Point2D(3, 1))))
        self.assertFalse(box.contains(geom.Box2D(geom.Point2D(-3, -3), geom.Point2D(2, 1))))
        self.assertFalse(box.contains(geom.Box2D(geom.Point2D(-3, -4), geom.Point2D(2, 1))))
        self.assertTrue(box.overlaps(geom.Box2D(geom.Point2D(-2, -3), geom.Point2D(2, 2))))
        self.assertTrue(box.overlaps(geom.Box2D(geom.Point2D(-2, -3), geom.Point2D(3, 1))))
        self.assertTrue(box.overlaps(geom.Box2D(geom.Point2D(-3, -3), geom.Point2D(2, 1))))
        self.assertTrue(box.overlaps(geom.Box2D(geom.Point2D(-3, -4), geom.Point2D(2, 1))))
        self.assertTrue(box.overlaps(geom.Box2D(geom.Point2D(-1, -2), geom.Point2D(1, 0))))
        self.assertTrue(box.overlaps(box))
        self.assertFalse(box.overlaps(geom.Box2D(geom.Point2D(-5, -3), geom.Point2D(-3, 1))))
        self.assertFalse(box.overlaps(geom.Box2D(geom.Point2D(-2, -6), geom.Point2D(2, -4))))
        self.assertFalse(box.overlaps(geom.Box2D(geom.Point2D(3, -3), geom.Point2D(4, 1))))
        self.assertFalse(box.overlaps(geom.Box2D(geom.Point2D(-2, 2), geom.Point2D(2, 2))))
        self.assertFalse(box.overlaps(geom.Box2D(geom.Point2D(-2, -5), geom.Point2D(2, -3))))
        self.assertFalse(box.overlaps(geom.Box2D(geom.Point2D(-4, -3), geom.Point2D(-2, 1))))
        self.assertFalse(box.contains(geom.Box2D(geom.Point2D(-2, 1), geom.Point2D(2, 3))))
        self.assertFalse(box.contains(geom.Box2D(geom.Point2D(2, -3), geom.Point2D(4, 1))))

    def testMutators(self):
        box = geom.Box2D(geom.Point2D(-2, -3), geom.Point2D(2, 1), True)
        box.grow(1)
        self.assertEqual(box, geom.Box2D(geom.Point2D(-3, -4), geom.Point2D(3, 2), True))
        box.grow(geom.Extent2D(2, 3))
        self.assertEqual(box, geom.Box2D(geom.Point2D(-5, -7), geom.Point2D(5, 5), True))
        box.shift(geom.Extent2D(3, 2))
        self.assertEqual(box, geom.Box2D(geom.Point2D(-2, -5), geom.Point2D(8, 7), True))
        box.include(geom.Point2D(-4, 2))
        self.assertEqual(box, geom.Box2D(geom.Point2D(-4, -5), geom.Point2D(8, 7), True))
        self.assertTrue(box.contains(geom.Point2D(-4, 2)))
        box.include(geom.Point2D(0, -6))
        self.assertEqual(box, geom.Box2D(geom.Point2D(-4, -6), geom.Point2D(8, 7), True))
        box.include(geom.Box2D(geom.Point2D(0, 0), geom.Point2D(10, 11), True))
        self.assertEqual(box, geom.Box2D(geom.Point2D(-4, -6), geom.Point2D(10, 11), True))
        box.clip(geom.Box2D(geom.Point2D(0, 0), geom.Point2D(11, 12), True))
        self.assertEqual(box, geom.Box2D(geom.Point2D(0, 0), geom.Point2D(10, 11), True))
        box.clip(geom.Box2D(geom.Point2D(-1, -2), geom.Point2D(5, 4), True))
        self.assertEqual(box, geom.Box2D(geom.Point2D(0, 0), geom.Point2D(5, 4), True))

    def testFlipI(self):
        parentExtent = geom.Extent2I(15, 20)
        x00, y00, x11, y11 = (8, 11, 13, 16)
        lrx00, lry00, lrx11, lry11 = (1, 11, 6, 16)
        tbx00, tby00, tbx11, tby11 = (8, 3, 13, 8)

        box0 = geom.Box2I(geom.Point2I(x00, y00),
                          geom.Point2I(x11, y11))
        box1 = geom.Box2I(geom.Point2I(x00, y00),
                          geom.Point2I(x11, y11))
        box0.flipLR(parentExtent[0])
        box1.flipTB(parentExtent[1])

        # test flip RL
        self.assertEqual(box0.getMinX(), lrx00)
        self.assertEqual(box0.getMinY(), lry00)
        self.assertEqual(box0.getMaxX(), lrx11)
        self.assertEqual(box0.getMaxY(), lry11)

        # test flip TB
        self.assertEqual(box1.getMinX(), tbx00)
        self.assertEqual(box1.getMinY(), tby00)
        self.assertEqual(box1.getMaxX(), tbx11)
        self.assertEqual(box1.getMaxY(), tby11)

    def testFlipD(self):
        parentExtent = geom.Extent2D(15.1, 20.6)
        x00, y00, x11, y11 = (8.3, 11.4, 13.2, 16.9)
        lrx00, lry00, lrx11, lry11 = (1.9, 11.4, 6.8, 16.9)
        tbx00, tby00, tbx11, tby11 = (8.3, 3.7, 13.2, 9.2)

        box0 = geom.Box2D(geom.Point2D(x00, y00),
                          geom.Point2D(x11, y11))
        box1 = geom.Box2D(geom.Point2D(x00, y00),
                          geom.Point2D(x11, y11))
        box0.flipLR(parentExtent[0])
        box1.flipTB(parentExtent[1])

        # test flip RL
        self.assertAlmostEqual(box0.getMinX(), lrx00, places=6)
        self.assertAlmostEqual(box0.getMinY(), lry00, places=6)
        self.assertAlmostEqual(box0.getMaxX(), lrx11, places=6)
        self.assertAlmostEqual(box0.getMaxY(), lry11, places=6)

        # test flip TB
        self.assertAlmostEqual(box1.getMinX(), tbx00, places=6)
        self.assertAlmostEqual(box1.getMinY(), tby00, places=6)
        self.assertAlmostEqual(box1.getMaxX(), tbx11, places=6)
        self.assertAlmostEqual(box1.getMaxY(), tby11, places=6)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

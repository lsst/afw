#
# LSST Data Management System
# Copyright 2008-2017 LSST Corporation.
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

import unittest

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.geom
import lsst.afw.geom.ellipses
import lsst.afw.image


class EllipseTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        np.random.seed(500)
        self.cores = [
            lsst.afw.geom.ellipses.Axes(4, 3, 1),
            lsst.afw.geom.ellipses.Quadrupole(5, 3, -1),
            # A line segment at 45-degrees:
            lsst.afw.geom.ellipses.Quadrupole(ixx=99.99999763706363, iyy=99.99999763706357,
                                              ixy=99.99999763706),
            # A horizontal line segment:
            lsst.afw.geom.ellipses.Quadrupole(5.0, 0.0, 0.0),
            # A vertical line segment:
            lsst.afw.geom.ellipses.Quadrupole(0.0, 7.0, 0.0),
        ]
        self.classes = [lsst.afw.geom.ellipses.Axes,
                        lsst.afw.geom.ellipses.Quadrupole]
        for s in lsst.afw.geom.ellipses.Separable.values():
            self.cores.append(s(0.5, 0.3, 2.1))
            self.classes.append(s)

    def testRadii(self):
        for core, det, trace in zip(self.cores, [144, 14], [25, 8]):
            with self.subTest(core=core):
                detRadius = det**0.25
                traceRadius = (0.5 * trace)**0.5
                area = np.pi * det**0.5
                self.assertFloatsAlmostEqual(
                    core.getDeterminantRadius(), detRadius)
                self.assertFloatsAlmostEqual(core.getTraceRadius(), traceRadius)
                self.assertFloatsAlmostEqual(core.getArea(), area)
                for cls in self.classes:
                    conv = cls(core)
                    self.assertFloatsAlmostEqual(
                        conv.getDeterminantRadius(), detRadius)
                    self.assertFloatsAlmostEqual(
                        conv.getTraceRadius(), traceRadius)
                    self.assertFloatsAlmostEqual(conv.getArea(), area)
                    conv.scale(3.0)
                    self.assertFloatsAlmostEqual(
                        conv.getDeterminantRadius(), detRadius * 3)
                    self.assertFloatsAlmostEqual(
                        conv.getTraceRadius(), traceRadius * 3)
                    self.assertFloatsAlmostEqual(conv.getArea(), area * 9)

    def testAccessors(self):
        for core in self.cores:
            with self.subTest(core=core):
                vec = np.random.randn(3) * 1E-3 + core.getParameterVector()
                core.setParameterVector(vec)
                self.assertImagesEqual(core.getParameterVector(), vec)
                center = lsst.geom.Point2D(*np.random.randn(2))
                ellipse = lsst.afw.geom.ellipses.Ellipse(core, center)
                self.assertFloatsAlmostEqual(
                    core.getParameterVector(), ellipse.getParameterVector()[:3])
                self.assertEqual(tuple(center), tuple(ellipse.getCenter()))
                self.assertEqual(lsst.geom.Point2D, type(ellipse.getCenter()))
                newcore = lsst.afw.geom.ellipses.Axes(1, 2, 3)
                newcore.normalize()
                core.assign(newcore)
                ellipse.setCore(core)
                np.testing.assert_allclose(
                    core.getParameterVector(), ellipse.getCore().getParameterVector())
                self.assertFloatsAlmostEqual(
                    core.clone().getParameterVector(), core.getParameterVector())
                self.assertIsNot(core, core.clone())
                self.assertFloatsAlmostEqual(lsst.afw.geom.ellipses.Ellipse(ellipse).getParameterVector(),
                                             ellipse.getParameterVector())
                self.assertIsNot(ellipse, lsst.afw.geom.ellipses.Ellipse(ellipse))

    def testTransform(self):
        for core in self.cores:
            with self.subTest(core=core):
                transform = lsst.geom.LinearTransform(np.random.randn(2, 2))
                t1 = core.transform(transform)
                core.transformInPlace(transform)
                self.assertIsNot(t1, core)
                self.assertFloatsAlmostEqual(
                    t1.getParameterVector(), core.getParameterVector())

    def testPixelRegion(self):
        for core in self.cores:
            with self.subTest(core=core):
                e = lsst.afw.geom.ellipses.Ellipse(
                    core, lsst.geom.Point2D(*np.random.randn(2)))
                region = lsst.afw.geom.ellipses.PixelRegion(e)
                bbox = region.getBBox()
                bbox.grow(2)
                array = np.zeros((bbox.getHeight(), bbox.getWidth()), dtype=bool)
                for span in region:
                    for point in span:
                        adjusted = point - bbox.getMin()
                        array[adjusted.getY(), adjusted.getX()] = True
                    self.assertLessEqual(span.getMinX(), span.getMaxX())
                gt = e.getGridTransform()
                for i in range(bbox.getBeginY(), bbox.getEndY()):
                    for j in range(bbox.getBeginX(), bbox.getEndX()):
                        point = lsst.geom.Point2I(j, i)
                        adjusted = point - bbox.getMin()
                        transformed = gt(lsst.geom.Point2D(point))
                        r = (transformed.getX()**2 + transformed.getY()**2)**0.5
                        if array[adjusted.getY(), adjusted.getX()]:
                            self.assertLessEqual(
                                r, 1.0, "Point %s is in region but r=%f" % (point, r))
                        else:
                            self.assertGreater(
                                r, 1.0, "Point %s is outside region but r=%f" % (point, r))
                # Another ellipse at a different position, specifically to
                # reproduce the problem on DM-20246.
                e = lsst.afw.geom.Ellipse(core, lsst.geom.Point2D(100, 100))
                region = lsst.afw.geom.ellipses.PixelRegion(e)
                bbox = region.getBBox()
                for span in region:
                    self.assertLessEqual(span.getMinX(), span.getMaxX())
                    self.assertTrue(bbox.contains(span.getMin()))
                    self.assertTrue(bbox.contains(span.getMax()))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import zip
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
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
#pybind11#import unittest
#pybind11#import numpy as np
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.geom.ellipses
#pybind11#import lsst.afw.image
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class EllipseTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        np.random.seed(500)
#pybind11#        self.cores = [
#pybind11#            lsst.afw.geom.ellipses.Axes(4, 3, 1),
#pybind11#            lsst.afw.geom.ellipses.Quadrupole(5, 3, -1)
#pybind11#        ]
#pybind11#        self.classes = [lsst.afw.geom.ellipses.Axes, lsst.afw.geom.ellipses.Quadrupole]
#pybind11#        for s in lsst.afw.geom.ellipses.Separable.values():
#pybind11#            self.cores.append(s(0.5, 0.3, 2.1))
#pybind11#            self.classes.append(s)
#pybind11#
#pybind11#    def testRadii(self):
#pybind11#        for core, det, trace in zip(self.cores, [144, 14], [25, 8]):
#pybind11#            detRadius = det**0.25
#pybind11#            traceRadius = (0.5 * trace)**0.5
#pybind11#            area = np.pi * det**0.5
#pybind11#            self.assertClose(core.getDeterminantRadius(), detRadius)
#pybind11#            self.assertClose(core.getTraceRadius(), traceRadius)
#pybind11#            self.assertClose(core.getArea(), area)
#pybind11#            for cls in self.classes:
#pybind11#                conv = cls(core)
#pybind11#                self.assertClose(conv.getDeterminantRadius(), detRadius)
#pybind11#                self.assertClose(conv.getTraceRadius(), traceRadius)
#pybind11#                self.assertClose(conv.getArea(), area)
#pybind11#                conv.scale(3.0)
#pybind11#                self.assertClose(conv.getDeterminantRadius(), detRadius * 3)
#pybind11#                self.assertClose(conv.getTraceRadius(), traceRadius * 3)
#pybind11#                self.assertClose(conv.getArea(), area * 9)
#pybind11#
#pybind11#    def testAccessors(self):
#pybind11#        for core in self.cores:
#pybind11#            vec = np.random.randn(3) * 1E-3 + core.getParameterVector()
#pybind11#            core.setParameterVector(vec)
#pybind11#            self.assertImagesEqual(core.getParameterVector(), vec)
#pybind11#            center = lsst.afw.geom.Point2D(*np.random.randn(2))
#pybind11#            ellipse = lsst.afw.geom.ellipses.Ellipse(core, center)
#pybind11#            self.assertClose(core.getParameterVector(), ellipse.getParameterVector()[:3])
#pybind11#            self.assertEqual(tuple(center), tuple(ellipse.getCenter()))
#pybind11#            self.assertEqual(lsst.afw.geom.Point2D, type(ellipse.getCenter()))
#pybind11#            newcore = lsst.afw.geom.ellipses.Axes(1, 2, 3)
#pybind11#            newcore.normalize()
#pybind11#            core.assign(newcore)
#pybind11#            ellipse.setCore(core)
#pybind11#            np.testing.assert_allclose(core.getParameterVector(), ellipse.getCore().getParameterVector())
#pybind11#            self.assertClose(core.clone().getParameterVector(), core.getParameterVector())
#pybind11#            self.assertIsNot(core, core.clone())
#pybind11#            self.assertClose(lsst.afw.geom.ellipses.Ellipse(ellipse).getParameterVector(),
#pybind11#                ellipse.getParameterVector())
#pybind11#            self.assertIsNot(ellipse, lsst.afw.geom.ellipses.Ellipse(ellipse))
#pybind11#
#pybind11#    def testTransform(self):
#pybind11#        for core in self.cores:
#pybind11#            transform = lsst.afw.geom.LinearTransform(np.random.randn(2, 2))
#pybind11#            t1 = core.transform(transform)
#pybind11#            core.transformInPlace(transform)
#pybind11#            self.assertIsNot(t1, core)
#pybind11#            self.assertClose(t1.getParameterVector(), core.getParameterVector())
#pybind11#
#pybind11#    def testPixelRegion(self):
#pybind11#        for core in self.cores:
#pybind11#            e = lsst.afw.geom.ellipses.Ellipse(core, lsst.afw.geom.Point2D(*np.random.randn(2)))
#pybind11#            region = lsst.afw.geom.ellipses.PixelRegion(e)
#pybind11#            bbox = region.getBBox()
#pybind11#            bbox.grow(2)
#pybind11#            array = np.zeros((bbox.getHeight(), bbox.getWidth()), dtype=bool)
#pybind11#            for span in region:
#pybind11#                for point in span:
#pybind11#                    adjusted = point - bbox.getMin()
#pybind11#                    array[adjusted.getY(), adjusted.getX()] = True
#pybind11#            gt = e.getGridTransform()
#pybind11#            for i in range(bbox.getBeginY(), bbox.getEndY()):
#pybind11#                for j in range(bbox.getBeginX(), bbox.getEndX()):
#pybind11#                    point = lsst.afw.geom.Point2I(j, i)
#pybind11#                    adjusted = point - bbox.getMin()
#pybind11#                    transformed = gt(lsst.afw.geom.Point2D(point))
#pybind11#                    r = (transformed.getX()**2 + transformed.getY()**2)**0.5
#pybind11#                    if array[adjusted.getY(), adjusted.getX()]:
#pybind11#                        self.assertLessEqual(r, 1.0, "Point %s is in region but r=%f" % (point, r))
#pybind11#                    else:
#pybind11#                        self.assertGreater(r, 1.0, "Point %s is outside region but r=%f" % (point, r))
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

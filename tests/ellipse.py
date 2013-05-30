#!/usr/bin/env python

#
# LSST Data Management System
# Copyright 2008-2013 LSST Corporation.
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
Tests for geom.ellipses

Run with:
   ./ellipse.py
or
   python
   >>> import ellipse; ellipse.run()
"""

import unittest
import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.geom as geom
import lsst.afw.geom.ellipses as el
import lsst.afw.image

numpy.random.seed(500)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class EllipseTestCase(utilsTests.TestCase):

    def setUp(self):
        self.safe = el.Axes(4, 3, (numpy.pi/3.0) * geom.radians)
        self.circle = el.Axes(3.0)
        self.classes = ([el.Axes, el.Quadrupole] + list(el.Separable.values()))
        self.all = [self.safe, self.circle]

    def testRadii(self):
        for axes in self.all:
            quadrupole = el.Quadrupole(axes)
            matrix = quadrupole.getMatrix()
            det = matrix[0,0] * matrix[1,1] - matrix[0,1]*matrix[1,0]
            trace = matrix[0,0] + matrix[1,1]
            detRadius = det**0.25
            traceRadius = (0.5 * trace)**0.5
            area = numpy.pi * det**0.5
            for cls in self.classes:
                conv = cls(axes)
                self.assertClose(conv.getDeterminantRadius(), detRadius, rtol=1E-15)
                self.assertClose(conv.getTraceRadius(), traceRadius, rtol=1E-15)
                self.assertClose(conv.getArea(), area, rtol=1E-15)
                conv.scale(3.0)
                self.assertClose(conv.getDeterminantRadius(), detRadius * 3, rtol=1E-15)
                self.assertClose(conv.getTraceRadius(), traceRadius * 3, rtol=1E-15)
                self.assertClose(conv.getArea(), area * 9, rtol=1E-15)

    def computeJacobian(self, m, eps, func, initial):
        n = len(initial)
        x = numpy.zeros(n, dtype=float)
        x[:] = initial
        result = numpy.zeros((m,len(initial)), dtype=float)
        for i in range(n):
            x[i] += eps
            result[:,i] = func(x)
            x[i] -= 2.0 * eps
            result[:,i] -= func(x)
            x[i] += eps
            result[:,i] /= 2.0 * eps
        return result

    def testConversion(self):
        for cls1 in self.classes:
            for core1a, isCircle in [(cls1(self.safe), False), (cls1(self.circle), True)]:
                for cls2 in self.classes:
                    core2a = cls2(core1a)
                    core1b = cls1(core2a)
                    self.assertClose(core1a.getParameterVector(), core1b.getParameterVector(),
                                     rtol=1E-14)
                    core1c = core2a.as_(core1a.getName())
                    self.assertClose(core1a.getParameterVector(), core1c.getParameterVector(),
                                     rtol=1E-14)
                    core1d = core2a.as_(type(core1a))
                    self.assertClose(core1a.getParameterVector(), core1d.getParameterVector(),
                                     rtol=1E-14)
                    self.assertTrue(core2a.compare(core1a))
                    self.assertEqual(core2a.compare(core1a, True), cls1 is cls2)
                    self.assertTrue(core1a.compare(core1b))
                    self.assertTrue(core1a.compare(core1c))
                    self.assertTrue(core1a.compare(core1d))
                    if cls2 is el.Axes and isCircle:
                        continue
                    analytic = core2a.dAssign(core1a)
                    numeric = self.computeJacobian(3, 1E-8, lambda p: cls2(cls1(p)).getParameterVector(),
                                                   core1a.getParameterVector())
                    self.assertClose(analytic, numeric, rtol=1E-6, atol=1E-6)
                    if cls1 is el.Axes and isCircle:
                        continue
                    inverse = core1b.dAssign(core2a)
                    self.assertClose(numpy.dot(analytic, inverse), numpy.identity(3), rtol=1E-8, atol=1E-14)
                    self.assertClose(numpy.dot(inverse, analytic), numpy.identity(3), rtol=1E-8, atol=1E-14)

    def testAccessors(self):
        for cls in self.classes:
            for axes in self.all:
                core = cls(axes)
                vec = numpy.random.randn(3) * 1E-3 + core.getParameterVector()
                core.setParameterVector(vec)
                self.assert_((core.getParameterVector()==vec).all())
                center = geom.Point2D(*numpy.random.randn(2))
                ellipse = el.Ellipse(core, center)
                self.assertClose(core.getParameterVector(), ellipse.getParameterVector()[:3])
                self.assertEqual(tuple(center), tuple(ellipse.getCenter()))
                self.assertEqual(geom.Point2D, type(ellipse.getCenter()))
                newcore = el.Axes(1,2,3*geom.radians)
                newcore.normalize()
                core.assign(newcore)
                ellipse.setCore(core)
                self.assertClose(core.getParameterVector(), ellipse.getCore().getParameterVector(),
                                 rtol=1E-15)
                self.assert_((core.clone().getParameterVector()==core.getParameterVector()).all())
                self.assert_(core is not core.clone())
                self.assert_((geom.ellipses.Ellipse(ellipse).getParameterVector()
                              == ellipse.getParameterVector()).all())
                self.assert_(ellipse is not geom.ellipses.Ellipse(ellipse))

        q = el.Quadrupole(self.safe)
        matrix = q.getMatrix()
        self.assertEqual(matrix[0,0], q.getIxx())
        self.assertEqual(matrix[1,1], q.getIyy())
        self.assertEqual(matrix[0,1], q.getIxy())
        self.assertEqual(matrix[1,0], q.getIxy())
        self.assertEqual(q.getParameterVector()[q.IXX], q.getIxx())
        self.assertEqual(q.getParameterVector()[q.IYY], q.getIyy())
        self.assertEqual(q.getParameterVector()[q.IXY], q.getIxy())

        a = el.Axes(self.safe)
        self.assertEqual(a.getParameterVector()[a.A], a.getA())
        self.assertEqual(a.getParameterVector()[a.B], a.getB())
        self.assertEqual(a.getParameterVector()[a.THETA], a.getTheta().asRadians())

        for cls in el.Separable.values():
            s = cls(self.safe)
            self.assertEqual(s.getParameterVector()[s.E1], s.getE1())
            self.assertEqual(s.getParameterVector()[s.E2], s.getE2())
            self.assertEqual(s.getParameterVector()[s.RADIUS], s.getRadius())
            self.assertClose(s.getTraceRadius(), s.getRadius())
            self.assertClose(s.getEllipticity().getTheta().asRadians(), a.getTheta().asRadians())
            self.assertClose(s.getEllipticity().getAxisRatio(), a.getB() / a.getA())
            self.assertEqual(s.getEllipticity().getComplex(), s.getE1() + s.getE2()*1j)
            self.assertEqual(numpy.abs(s.getEllipticity().getComplex()), s.getEllipticity().getE())

    def testTransform(self):
        for cls in self.classes:
            for axes in self.all:
                core = cls(axes)
                transform = geom.LinearTransform(numpy.random.randn(2,2))
                t1 = core.transform(transform)
                core.transformInPlace(transform)
                self.assert_(t1 is not core)
                self.assertClose(t1.getParameterVector(), core.getParameterVector())

    def testPixelRegion(self):
        for cls in self.classes:
            for axes in self.all:
                core = cls(axes)
                e = geom.ellipses.Ellipse(core, geom.Point2D(*numpy.random.randn(2)))
                region = geom.ellipses.PixelRegion(e)
                bbox = region.getBBox()
                bbox.grow(2)
                array = numpy.zeros((bbox.getHeight(), bbox.getWidth()), dtype=bool)
                for span in region:
                    for point in span:
                        adjusted = point - bbox.getMin()
                        array[adjusted.getY(), adjusted.getX()] = True
                gt = e.getGridTransform()
                for i in range(bbox.getBeginY(), bbox.getEndY()):
                    for j in range(bbox.getBeginX(), bbox.getEndX()):
                        point = geom.Point2I(j, i)
                        adjusted = point - bbox.getMin()
                        transformed = gt(geom.Point2D(point))
                        r = (transformed.getX()**2 + transformed.getY()**2)**0.5
                        if array[adjusted.getY(), adjusted.getX()]:
                            self.assert_(r <= 1.0, "Point %s is in region but r=%f" % (point, r))
                        else:
                            self.assert_(r > 1.0, "Point %s is outside region but r=%f" % (point, r))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(EllipseTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

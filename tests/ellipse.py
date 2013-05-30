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

    def computeDerivative(self, m, eps, func, initial):
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
                        # TODO: fix this case or add warnings
                        continue
                    analytic = core2a.dAssign(core1a)
                    numeric = self.computeDerivative(3, 1E-8, lambda p: cls2(cls1(p)).getParameterVector(),
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

    def testModifiers(self):
        for cls in self.classes:
            for axes in self.all:
                core1 = cls(axes)
                core2 = core1.clone()
                core2.scale(2.0)
                core3 = core1.transform(geom.LinearTransform.makeScaling(2.0, 2.0))
                self.assertTrue(core2.compare(core3))
                self.assertClose(core2.getArea(), core1.getArea() * 4.0)
                core4 = core1.clone()
                core4.grow(3.0)
                self.assertTrue(core4.compare(el.Axes(axes.getA() + 3.0, axes.getB() + 3.0, axes.getTheta())))

    def testTransform(self):
        for cls in self.classes:
            for axes in self.all:
                core1 = cls(axes)

                # test transforming EllipseCore
                lt = geom.LinearTransform(numpy.random.randn(2,2))
                core2, dEllipseAnalytic, dTransformAnalytic = core1.transform(lt, doDerivatives=True)
                core3 = core1.clone()
                core3.transform(lt, inPlace=True)
                self.assertIsNot(core2, core3)
                self.assertClose(core2.getParameterVector(), core3.getParameterVector())

                # test derivative w.r.t. input EllipseCore
                self.assertEqual(type(dEllipseAnalytic), numpy.ndarray)
                def func1(p):
                    return cls(p).transform(lt).getParameterVector()
                dEllipseNumeric = self.computeDerivative(m=3, eps=1E-8, func=func1,
                                                         initial=core1.getParameterVector())
                self.assertClose(dEllipseAnalytic, dEllipseNumeric, rtol=1E-5, atol=1E-5)

                # test derivative w.r.t. LinearTransform
                self.assertEqual(type(dTransformAnalytic), numpy.ndarray)
                def func2(p):
                    lt1 = geom.LinearTransform()
                    lt1.setParameterVector(p)
                    return core1.transform(lt1).getParameterVector()
                dTransformNumeric = self.computeDerivative(m=3, eps=1E-8, func=func2,
                                                           initial=lt.getParameterVector())
                self.assertClose(dTransformAnalytic, dTransformNumeric, rtol=1E-5, atol=1E-5)

                # test transforming Ellipse
                ellipse1 = el.Ellipse(core1, geom.Point2D(numpy.random.randn(2)))
                at = geom.AffineTransform(numpy.random.randn(2,2), numpy.random.randn(2))
                ellipse2, dEllipseAnalytic, dTransformAnalytic = ellipse1.transform(at, doDerivatives=True)
                ellipse3 = el.Ellipse(ellipse1)
                ellipse3.transform(at, inPlace=True)
                self.assertIsNot(ellipse2, ellipse3)
                self.assertClose(ellipse2.getParameterVector(), ellipse3.getParameterVector())

                # test derivative w.r.t. input Ellipse
                self.assertEqual(type(dEllipseAnalytic), numpy.ndarray)
                def func3(p):
                    e = el.Ellipse(cls())
                    e.setParameterVector(p)
                    return e.transform(at).getParameterVector()
                dEllipseNumeric = self.computeDerivative(m=5, eps=1E-8, func=func3,
                                                         initial=ellipse1.getParameterVector())
                self.assertClose(dEllipseAnalytic, dEllipseNumeric, rtol=1E-5, atol=1E-5)

                # test derivative w.r.t. AffineTransform
                self.assertEqual(type(dTransformAnalytic), numpy.ndarray)
                def func4(p):
                    at1 = geom.AffineTransform()
                    at1.setParameterVector(p)
                    return ellipse1.transform(at1).getParameterVector()
                dTransformNumeric = self.computeDerivative(m=5, eps=1E-8, func=func4,
                                                           initial=at.getParameterVector())
                self.assertClose(dTransformAnalytic, dTransformNumeric, rtol=1E-5, atol=1E-5)

    def testGridTransform(self):
        unitCircleCore = el.Axes(1.0)
        unitCircleEllipse = el.Ellipse(unitCircleCore)
        for cls in self.classes:
            for axes in self.all:
                core1 = cls(axes)

                # test grid transform for EllipseCore
                lt1, analytic = core1.getGridTransform(doDerivatives=True)
                core2 = core1.transform(lt1)
                self.assertTrue(unitCircleCore.compare(core2))

                # test derivative w.r.t. input EllipseCore
                self.assertEqual(type(analytic), numpy.ndarray)
                def func1(p):
                    return cls(p).getGridTransform().getParameterVector()
                numeric = self.computeDerivative(m=4, eps=1E-8, func=func1,
                                                 initial=core1.getParameterVector())
                self.assertClose(analytic, numeric, rtol=1E-6, atol=1E-8)

                # test grid transform for Ellipse
                ellipse1 = el.Ellipse(core1, geom.Point2D(numpy.random.randn(2)))
                at1, analytic = ellipse1.getGridTransform(doDerivatives=True)
                ellipse2 = ellipse1.transform(at1)
                self.assertTrue(unitCircleEllipse.getCore().compare(ellipse2.getCore()))
                self.assertClose(ellipse2.getCenter().getX(), 0.0)
                self.assertClose(ellipse2.getCenter().getY(), 0.0)

                # test derivative w.r.t. input Ellipse
                self.assertEqual(type(analytic), numpy.ndarray)
                def func2(p):
                    e = el.Ellipse(cls())
                    e.setParameterVector(p)
                    return e.getGridTransform().getParameterVector()
                numeric = self.computeDerivative(m=6, eps=1E-8, func=func2,
                                                 initial=ellipse1.getParameterVector())
                self.assertClose(analytic, numeric, rtol=1E-6, atol=1E-8)

    def testConvolution(self):
        for axes1, isCircle1 in zip(self.all, (False, True)):
            matrix1 = el.Quadrupole(axes1).getMatrix()
            for axes2, isCircle2 in zip(self.all, (False, True)):
                matrix2 = el.Quadrupole(axes2).getMatrix()
                for cls1 in self.classes:
                    for cls2 in self.classes:
                        # test convolving EllipseCores
                        core1 = cls1(axes1)
                        core2 = cls2(axes2)
                        core3, analytic1 = core1.convolve(core2, doDerivatives=True)
                        core4 = core2.clone()
                        core4a = core4.convolve(core1, inPlace=True)
                        matrix3 = el.Quadrupole(core3).getMatrix()
                        matrix4 = el.Quadrupole(core4).getMatrix()
                        self.assertIs(core4, core4a)
                        self.assertTrue(core3.compare(core4))
                        self.assertClose(matrix1 + matrix2, matrix3, rtol=1E-14)
                        self.assertClose(matrix1 + matrix2, matrix4, rtol=1E-14)

                        # test convolving Ellipses
                        ellipse1 = el.Ellipse(core1, geom.Point2D(numpy.random.randn(2)))
                        ellipse2 = el.Ellipse(core2, geom.Point2D(numpy.random.randn(2)))
                        ellipse3, analytic2 = ellipse1.convolve(ellipse2, doDerivatives=True)
                        ellipse4 = el.Ellipse(ellipse2)
                        ellipse4a = ellipse4.convolve(ellipse1, inPlace=True)
                        matrix3 = el.Quadrupole(ellipse3.getCore()).getMatrix()
                        matrix4 = el.Quadrupole(ellipse4.getCore()).getMatrix()
                        self.assertIs(ellipse4, ellipse4a)
                        self.assertTrue(ellipse3.getCore().compare(ellipse4.getCore()))
                        self.assertClose(ellipse3.getCenter().getX(), ellipse4.getCenter().getX())
                        self.assertClose(ellipse3.getCenter().getY(), ellipse4.getCenter().getY())
                        self.assertClose(matrix1 + matrix2, matrix3, rtol=1E-14)
                        self.assertClose(matrix1 + matrix2, matrix4, rtol=1E-14)

                        if isCircle1 and isCircle2 and (cls1 is el.Axes or cls2 is el.Axes):
                            # TODO: fix this case or add warnings
                            continue

                        # test derivative w.r.t. input EllipseCore
                        self.assertEqual(type(analytic1), numpy.ndarray)
                        def func1(p):
                            return cls1(p).convolve(core2).getParameterVector()
                        numeric1 = self.computeDerivative(m=3, eps=1E-6, func=func1,
                                                          initial=core1.getParameterVector())
                        self.assertClose(analytic1, numeric1, rtol=1E-4, atol=1E-4)

                        # test derivative w.r.t. input Ellipse
                        self.assertEqual(type(analytic2), numpy.ndarray)
                        def func2(p):
                            e = el.Ellipse(cls1())
                            e.setParameterVector(p)
                            return e.convolve(ellipse2).getParameterVector()
                        numeric2 = self.computeDerivative(m=5, eps=1E-6, func=func2,
                                                          initial=ellipse1.getParameterVector())
                        self.assertClose(analytic2, numeric2, rtol=1E-4, atol=1E-4)

    def testParametric(self):
        """Spot-check ellipses::Parametric, using a few points calculated analytically with Mathematica
        """
        p = el.Parametric(el.Ellipse(el.Quadrupole(3,2,-0.65)))
        self.assertClose(numpy.array(p(1.45)), numpy.array((0.76537615289287353, 1.0573336496088439)))
        self.assertClose(numpy.array(p(-2.56)), numpy.array((-1.6804596457433354, 0.03378847788858419)))

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

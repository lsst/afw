#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
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
#pybind11#"""
#pybind11#Tests for Splines
#pybind11#
#pybind11#Run with:
#pybind11#   ./spline.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import spline; spline.run()
#pybind11#"""
#pybind11#import math
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.math as afwMath
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class SplineTestCase(unittest.TestCase):
#pybind11#    """A test case for Image"""
#pybind11#
#pybind11#    def smooth(self, x, differentiate=False):
#pybind11#        if differentiate:
#pybind11#            return math.cos(x)
#pybind11#        else:
#pybind11#            return math.sin(x)
#pybind11#
#pybind11#    def noDerivative(self, x):
#pybind11#        if False:
#pybind11#            return math.sin(x)
#pybind11#        else:
#pybind11#            if x < 1:
#pybind11#                return 1 - x
#pybind11#            else:
#pybind11#                return 0
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        x, x2, ySin, yND = [], [], [], []
#pybind11#        for i in range(0, 40):
#pybind11#            x.append(0.1*i)
#pybind11#            for j in range(4):
#pybind11#                x2.append(0.1*(i + 0.25*j))
#pybind11#
#pybind11#            ySin.append(self.smooth(x[i]))
#pybind11#            yND.append(self.noDerivative(x[i]))
#pybind11#
#pybind11#        self.x = x
#pybind11#        self.x2 = x2
#pybind11#        self.yND = yND
#pybind11#        self.ySin = ySin
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.x
#pybind11#        del self.x2
#pybind11#        del self.ySin
#pybind11#        del self.yND
#pybind11#
#pybind11#    def testNaturalSpline1(self):
#pybind11#        """Test fitting a natural spline to a smooth function"""
#pybind11#        gamma = 0
#pybind11#        sp = afwMath.TautSpline(self.x, self.ySin, gamma)
#pybind11#
#pybind11#        y2 = afwMath.vectorD()
#pybind11#        sp.interpolate(self.x2, y2)
#pybind11#
#pybind11#        for x, y in zip(self.x2, y2):
#pybind11#            self.assertAlmostEqual(y, self.smooth(x), 1)  # fails at 2 places!
#pybind11#
#pybind11#    def testNaturalSplineDerivative1(self):
#pybind11#        """Test fitting a natural spline to a smooth function and finding its derivative"""
#pybind11#        sp = afwMath.TautSpline(self.x, self.ySin)
#pybind11#
#pybind11#        y2 = afwMath.vectorD()
#pybind11#        sp.derivative(self.x2, y2)
#pybind11#
#pybind11#        for x, y in zip(self.x2, y2):
#pybind11#            self.assertAlmostEqual(y, self.smooth(x, True), delta=1.5e-3)
#pybind11#
#pybind11#    def testNaturalSpline2(self):
#pybind11#        """Test fitting a natural spline to a non-differentiable function (we basically fail)"""
#pybind11#        gamma = 0
#pybind11#        sp = afwMath.TautSpline(self.x, self.yND, gamma)
#pybind11#
#pybind11#        y2 = afwMath.vectorD()
#pybind11#        sp.interpolate(self.x2, y2)
#pybind11#
#pybind11#        for x, y in zip(self.x2, y2):
#pybind11#            self.assertAlmostEqual(y, self.noDerivative(x), 1)  # fails at 2 places!
#pybind11#
#pybind11#    def testTautSpline1(self):
#pybind11#        """Test fitting a taut spline to a smooth function"""
#pybind11#        gamma = 2.5
#pybind11#        sp = afwMath.TautSpline(self.x, self.ySin, gamma)
#pybind11#
#pybind11#        y2 = afwMath.vectorD()
#pybind11#        sp.interpolate(self.x2, y2)
#pybind11#
#pybind11#        for x, y in zip(self.x2, y2):
#pybind11#            self.assertAlmostEqual(y, self.smooth(x), 4)
#pybind11#
#pybind11#    def testTautSpline2(self):
#pybind11#        """Test fitting a taut spline to a non-differentiable function"""
#pybind11#        gamma = 2.5
#pybind11#        sp = afwMath.TautSpline(self.x, self.yND, gamma)
#pybind11#
#pybind11#        y2 = afwMath.vectorD()
#pybind11#        sp.interpolate(self.x2, y2)
#pybind11#
#pybind11#        for x, y in zip(self.x2, y2):
#pybind11#            self.assertAlmostEqual(y, self.noDerivative(x))
#pybind11#
#pybind11#    def testRootFinding(self):
#pybind11#        """Test finding roots of Spline = value"""
#pybind11#
#pybind11#        gamma = 2.5
#pybind11#        sp = afwMath.TautSpline(self.x, self.yND, gamma)
#pybind11#
#pybind11#        for value in (0.1, 0.5):
#pybind11#            self.assertEqual(sp.roots(value, self.x[0], self.x[-1])[0], 1 - value)
#pybind11#
#pybind11#        if False:
#pybind11#            y = afwMath.vectorD()
#pybind11#            sp.interpolate(self.x, y)
#pybind11#            for x, y in zip(self.x, y):
#pybind11#                print(x, y)
#pybind11#        #
#pybind11#        # Solve sin(x) = 0.5
#pybind11#        #
#pybind11#        sp = afwMath.TautSpline(self.x, self.ySin)
#pybind11#        roots = [math.degrees(x) for x in sp.roots(0.5, self.x[0], self.x[-1])]
#pybind11#        self.assertAlmostEqual(roots[0], 30, 5)
#pybind11#        self.assertAlmostEqual(roots[1], 150, 5)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
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

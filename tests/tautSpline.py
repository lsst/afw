#!/usr/bin/env python

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
Tests for Splines

Run with:
   ./spline.py
or
   python
   >>> import spline; spline.run()
"""

import math, os, sys
import numpy as np
import unittest

import lsst.utils.tests as utilsTests
import lsst.afw.math as afwMath

try:
    import matplotlib.pyplot as plt
    try:
        fig
    except NameError:
        fig = None
except ImportError:
    plt = None

try:
    display
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SplineTestCase(unittest.TestCase):
    """A test case for Image"""
    def smooth(self, x, differentiate=False):
        if differentiate:
            return math.cos(x)
        else:
            return math.sin(x)

    def noDerivative(self, x):
        if False:
            return math.sin(x)
        else:
            if x < 1:
                return 1 - x
            else:
                return 0

    def setUp(self):
        x, x2, ySin, yND = [], [], [], []
        for i in range(0,40):
            x.append(0.1*i)
            for j in range(4):
                x2.append(0.1*(i + 0.25*j))
                
            ySin.append(self.smooth(x[i]))
            yND.append(self.noDerivative(x[i]))

        self.x = x
        self.x2 = x2
        self.yND = yND
        self.ySin = ySin

    def tearDown(self):
        del self.x
        del self.x2
        del self.ySin
        del self.yND

    def testNaturalSpline1(self):
        """Test fitting a natural spline to a smooth function"""
        gamma = 0
        sp = afwMath.makeInterpolate(self.x, self.ySin, afwMath.InterpolateControlTautSpline(gamma))

        y2 = afwMath.vectorD()
        sp.interpolate(self.x2, y2)

        for x, y in zip(self.x2, y2):
            self.assertAlmostEqual(y, self.smooth(x), 1) # fails at 2 places!

    def testNaturalSplineDerivative1(self):
        """Test fitting a natural spline to a smooth function and finding its derivative"""
        gamma = 0
        sp = afwMath.makeInterpolate(self.x, self.ySin, afwMath.InterpolateControlTautSpline(gamma))

        y2 = afwMath.vectorD()
        sp.derivative(self.x2, y2)

        for x, y in zip(self.x2, y2):
            self.assertTrue(abs(y - self.smooth(x, True)) < 1.5e-3)

    def testNaturalSpline2(self):
        """Test fitting a natural spline to a non-differentiable function (we basically fail)"""
        gamma = 0
        sp = afwMath.makeInterpolate(self.x, self.yND, afwMath.InterpolateControlTautSpline(gamma))

        y2 = afwMath.vectorD()
        sp.interpolate(self.x2, y2)

        for x, y in zip(self.x2, y2):
            self.assertAlmostEqual(y, self.noDerivative(x), 1) # fails at 2 places!

    def testInterpolateTautSpline1(self):
        """Test fitting a taut spline to a smooth function"""
        gamma = 2.5
        sp = afwMath.makeInterpolate(self.x, self.ySin, afwMath.InterpolateControlTautSpline(gamma))

        y2 = afwMath.vectorD()
        sp.interpolate(self.x2, y2)

        for x, y in zip(self.x2, y2):
            self.assertAlmostEqual(y, self.smooth(x), 4)

    def testInterpolateTautSpline2(self):
        """Test fitting a taut spline to a non-differentiable function"""
        gamma = 2.5
        sp = afwMath.makeInterpolate(self.x, self.yND, afwMath.InterpolateControlTautSpline(gamma))

        y2 = afwMath.vectorD()
        sp.interpolate(self.x2, y2)

        for x, y in zip(self.x2, y2):
            self.assertAlmostEqual(y, self.noDerivative(x))

        if display and plt:
            global fig
            if fig is None:
                fig = plt.figure()
            else:
                fig.clf()
            axes = fig.add_axes((0.1, 0.1, 0.85, 0.80))
            axes.plot(self.x, self.yND, "b.", label="Data")
            axes.plot(self.x2, y2,      "r", label="Taut Spline")
            axes.set_ylim(-0.1, axes.get_ylim()[1] + 0.1)
            axes.set_xlim(-0.1, axes.get_xlim()[1] + 0.1)

            sp = afwMath.makeInterpolate(self.x, self.yND, afwMath.Interpolate.AKIMA_SPLINE)
            sp.interpolate(self.x2, y2)
            axes.plot(self.x2, y2,      "g", label="Akima Spline")

            sp = afwMath.makeInterpolate(self.x, self.yND, afwMath.Interpolate.NATURAL_SPLINE)
            sp.interpolate(self.x2, y2)
            axes.plot(self.x2, y2,      "b", label="Natural Spline")

            sp = afwMath.makeInterpolate(self.x, self.yND, afwMath.Interpolate.CUBIC_SPLINE)
            sp.interpolate(self.x2, y2)
            axes.plot(self.x2, y2,      "m", label="Cubic Spline")

            plt.legend(loc="upper right", ncol=1)
            fig.show()
            raw_input("Continue? ")

    def testInterpolateTaut_AkimaSpline(self):
        """Compare Taut and Akima splines"""
        gamma = 2.5
        taut  = afwMath.makeInterpolate(self.x, self.yND, afwMath.InterpolateControlTautSpline(gamma))
        akima = afwMath.makeInterpolate(self.x, self.yND, afwMath.Interpolate.AKIMA_SPLINE)

        yAkima, yTaut = afwMath.vectorD(), afwMath.vectorD()
        akima.interpolate(self.x2, yAkima)
        taut.interpolate(self.x2, yTaut)

        for ya, yt in zip(yAkima, yTaut):
            self.assertAlmostEqual(ya, yt)

    def testRootFinding(self):
        """Test finding roots of Spline = value"""

        gamma = 2.5
        sp = afwMath.makeInterpolate(self.x, self.yND, afwMath.InterpolateControlTautSpline(gamma))

        for value in (0.1, 0.5):
            self.assertEqual(sp.roots(value, self.x[0], self.x[-1])[0], 1 - value)

        if False:
            y = afwMath.vectorD()
            sp.interpolate(self.x, y)
            for x, y in zip(self.x, y):
                print x, y
        #
        # Solve sin(x) = 0.5
        #
        sp = afwMath.makeInterpolate(self.x, self.ySin, afwMath.InterpolateControlTautSpline(gamma))
        roots = [math.degrees(x) for x in sp.roots(0.5, self.x[0], self.x[-1])]
        self.assertAlmostEqual(roots[0]/30,  1, 6)
        self.assertAlmostEqual(roots[1]/150, 1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SplineTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

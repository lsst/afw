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
#pybind11#"""
#pybind11#Tests for Interpolate
#pybind11#
#pybind11#Run with:
#pybind11#   ./Interpolate.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import Interpolate; Interpolate.run()
#pybind11#"""
#pybind11#
#pybind11#
#pybind11#import unittest
#pybind11#import numpy as np
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class InterpolateTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    """A test case for Interpolate Linear"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.n = 10
#pybind11#        self.x = np.zeros(self.n, dtype=float)
#pybind11#        self.y1 = np.zeros(self.n, dtype=float)
#pybind11#        self.y2 = np.zeros(self.n, dtype=float)
#pybind11#        self.y0 = 1.0
#pybind11#        self.dydx = 1.0
#pybind11#        self.d2ydx2 = 0.5
#pybind11#
#pybind11#        for i in range(0, self.n, 1):
#pybind11#            self.x[i] = i
#pybind11#            self.y1[i] = self.dydx*self.x[i] + self.y0
#pybind11#            self.y2[i] = self.d2ydx2*self.x[i]*self.x[i] + self.dydx*self.x[i] + self.y0
#pybind11#
#pybind11#        self.xtest = 4.5
#pybind11#        self.y1test = self.dydx*self.xtest + self.y0
#pybind11#        self.y2test = self.d2ydx2*self.xtest*self.xtest + self.dydx*self.xtest + self.y0
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.x
#pybind11#        del self.y1
#pybind11#        del self.y2
#pybind11#
#pybind11#    def testLinearRamp(self):
#pybind11#
#pybind11#        # === test the Linear Interpolator ============================
#pybind11#        # default is akima spline
#pybind11#        yinterpL = afwMath.makeInterpolate(self.x, self.y1)
#pybind11#        youtL = yinterpL.interpolate(self.xtest)
#pybind11#
#pybind11#        self.assertEqual(youtL, self.y1test)
#pybind11#
#pybind11#    def testNaturalSplineRamp(self):
#pybind11#
#pybind11#        # === test the Spline interpolator =======================
#pybind11#        # specify interp type with the string interface
#pybind11#        yinterpS = afwMath.makeInterpolate(self.x, self.y1, afwMath.Interpolate.NATURAL_SPLINE)
#pybind11#        youtS = yinterpS.interpolate(self.xtest)
#pybind11#
#pybind11#        self.assertEqual(youtS, self.y1test)
#pybind11#
#pybind11#    def testAkimaSplineParabola(self):
#pybind11#        """test the Spline interpolator"""
#pybind11#        # specify interp type with the enum style interface
#pybind11#        yinterpS = afwMath.makeInterpolate(self.x, self.y2, afwMath.Interpolate.AKIMA_SPLINE)
#pybind11#        youtS = yinterpS.interpolate(self.xtest)
#pybind11#
#pybind11#        self.assertEqual(youtS, self.y2test)
#pybind11#
#pybind11#    def testConstant(self):
#pybind11#        """test the constant interpolator"""
#pybind11#        # [xy]vec:   point samples
#pybind11#        # [xy]vec_c: centered values
#pybind11#        xvec = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
#pybind11#        xvec_c = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
#pybind11#        yvec = np.array([1.0, 2.4, 5.0, 8.4, 13.0, 18.4, 25.0, 32.6, 41.0, 50.6])
#pybind11#        yvec_c = np.array([1.0, 1.7, 3.7, 6.7, 10.7, 15.7, 21.7, 28.8, 36.8, 45.8, 50.6])
#pybind11#
#pybind11#        interp = afwMath.makeInterpolate(xvec, yvec, afwMath.Interpolate.CONSTANT)
#pybind11#
#pybind11#        for x, y in zip(xvec_c, yvec_c):
#pybind11#            self.assertAlmostEqual(interp.interpolate(x + 0.1), y)
#pybind11#            self.assertAlmostEqual(interp.interpolate(x), y)
#pybind11#
#pybind11#        self.assertEqual(interp.interpolate(xvec[0] - 10), yvec[0])
#pybind11#        n = len(yvec)
#pybind11#        self.assertEqual(interp.interpolate(xvec[n - 1] + 10), yvec[n - 1])
#pybind11#
#pybind11#        for x, y in reversed(list(zip(xvec_c, yvec_c))):  # test caching as we go backwards
#pybind11#            self.assertAlmostEqual(interp.interpolate(x + 0.1), y)
#pybind11#            self.assertAlmostEqual(interp.interpolate(x), y)
#pybind11#
#pybind11#        i = 2
#pybind11#        for x in np.arange(xvec_c[i], xvec_c[i + 1], 10):
#pybind11#            self.assertEqual(interp.interpolate(x), yvec_c[i])
#pybind11#
#pybind11#    def testInvalidInputs(self):
#pybind11#        """Test that invalid inputs cause an abort"""
#pybind11#
#pybind11#        self.assertRaises(pexExcept.OutOfRangeError,
#pybind11#                          lambda: afwMath.makeInterpolate(np.array([], dtype=float), np.array([], dtype=float),
#pybind11#                                                          afwMath.Interpolate.CONSTANT)
#pybind11#                          )
#pybind11#
#pybind11#        afwMath.makeInterpolate(np.array([0], dtype=float), np.array([1], dtype=float),
#pybind11#                                afwMath.Interpolate.CONSTANT)
#pybind11#
#pybind11#        self.assertRaises(pexExcept.OutOfRangeError,
#pybind11#                          lambda: afwMath.makeInterpolate(np.array([0], dtype=float), np.array([1], dtype=float),
#pybind11#                                                          afwMath.Interpolate.LINEAR))
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
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

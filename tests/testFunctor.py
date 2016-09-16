#!/usr/bin/env python2
#
# LSST Data Management System
# Copyright 2015 LSST Corporation.
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
Tests for lsst.afw.geom.Functor classes.
"""
import unittest
import numpy as np
import lsst.utils.tests
import lsst.afw.geom as afwGeom
import lsst.pex.exceptions as pexExcept

def num_deriv(func, x, eps=1e-7):
    h = eps*max(abs(x), 1)
    xp = x + h
    dx = xp - x
    return (func(x + dx) - func(x))/dx

class FunctorTestCase(unittest.TestCase):
    def setUp(self):
        self.funcs = [afwGeom.LinearFunctor(5, 30)]
        self.xvals = np.logspace(-8, 1, 10)
        self.y0 = 1
    def tearDown(self):
        while self.funcs:
            func = self.funcs.pop()
            del func
    def testDerivatives(self):
        for func in self.funcs:
            for xx in self.xvals:
                self.assertAlmostEqual(func.derivative(xx), num_deriv(func, xx),
                                       places=6)
    def testInverse(self):
        for func in self.funcs:
            for xx in self.xvals:
                yy = func(xx)
                self.assertAlmostEqual(xx, func.inverse(yy), places=6)
    def testInverseTolOutOfRangeError(self):
        maxiter = 1000
        for func in self.funcs:
            self.assertRaises(pexExcept.OutOfRangeError, func.inverse,
                              self.y0, 10., maxiter)
            self.assertRaises(pexExcept.OutOfRangeError, func.inverse,
                              self.y0, -1, maxiter)
    def testInverseMaxiterOutOfRangeError(self):
        tol = 1e-5
        for func in self.funcs:
            # Check bad maxiter value.
            self.assertRaises(pexExcept.OutOfRangeError, func.inverse,
                              self.y0, tol, 0);
    def testInverseMaxiterRuntimeError(self):
        for func in self.funcs:
            # Check for exceeding maximum iterations.
            self.assertRaises(pexExcept.RuntimeError, func.inverse,
                              self.y0, 1e-10, 1)
    
def suite():
    """Return a suite containing all of the test cases in this module."""
    lsst.utils.tests.init()
    suites = []
    suites += unittest.makeSuite(FunctorTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == '__main__':
    run(True)

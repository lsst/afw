#pybind11##!/usr/bin/env python
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2015 LSST Corporation.
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
#pybind11#"""
#pybind11#Tests for lsst.afw.geom.Functor classes.
#pybind11#"""
#pybind11#from __future__ import division
#pybind11#import unittest
#pybind11#import numpy as np
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#
#pybind11#
#pybind11#def num_deriv(func, x, eps=1e-7):
#pybind11#    h = eps*max(abs(x), 1)
#pybind11#    xp = x + h
#pybind11#    dx = xp - x
#pybind11#    return (func(x + dx) - func(x))/dx
#pybind11#
#pybind11#
#pybind11#class FunctorTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.funcs = [afwGeom.LinearFunctor(5, 30)]
#pybind11#        self.xvals = np.logspace(-8, 1, 10)
#pybind11#        self.y0 = 1
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        while self.funcs:
#pybind11#            func = self.funcs.pop()
#pybind11#            del func
#pybind11#
#pybind11#    def testDerivatives(self):
#pybind11#        for func in self.funcs:
#pybind11#            for xx in self.xvals:
#pybind11#                self.assertAlmostEqual(func.derivative(xx), num_deriv(func, xx),
#pybind11#                                       places=6)
#pybind11#
#pybind11#    def testInverse(self):
#pybind11#        for func in self.funcs:
#pybind11#            for xx in self.xvals:
#pybind11#                yy = func(xx)
#pybind11#                self.assertAlmostEqual(xx, func.inverse(yy), places=6)
#pybind11#
#pybind11#    def testInverseTolOutOfRangeError(self):
#pybind11#        maxiter = 1000
#pybind11#        for func in self.funcs:
#pybind11#            self.assertRaises(pexExcept.OutOfRangeError, func.inverse,
#pybind11#                              self.y0, 10., maxiter)
#pybind11#            self.assertRaises(pexExcept.OutOfRangeError, func.inverse,
#pybind11#                              self.y0, -1, maxiter)
#pybind11#
#pybind11#    def testInverseMaxiterOutOfRangeError(self):
#pybind11#        tol = 1e-5
#pybind11#        for func in self.funcs:
#pybind11#            # Check bad maxiter value.
#pybind11#            self.assertRaises(pexExcept.OutOfRangeError, func.inverse,
#pybind11#                              self.y0, tol, 0)
#pybind11#
#pybind11#    def testInverseMaxiterRuntimeError(self):
#pybind11#        for func in self.funcs:
#pybind11#            # Check for exceeding maximum iterations.
#pybind11#            self.assertRaises(pexExcept.RuntimeError, func.inverse,
#pybind11#                              self.y0, 1e-10, 1)
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == '__main__':
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

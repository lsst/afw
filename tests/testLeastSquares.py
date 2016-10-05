#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
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
#pybind11#Tests for math.LeastSquares
#pybind11#
#pybind11#Run with:
#pybind11#   ./testLeastSquares.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import testLeastSquares; testLeastSquares.run()
#pybind11#"""
#pybind11#
#pybind11#import unittest
#pybind11#import numpy
#pybind11#import sys
#pybind11#
#pybind11#import lsst.utils.tests as utilsTests
#pybind11#import lsst.pex.exceptions
#pybind11#from lsst.afw.math import LeastSquares
#pybind11#from lsst.log import Log
#pybind11#
#pybind11#Log.getLogger("afw.math.LeastSquares").setLevel(Log.DEBUG)
#pybind11#
#pybind11#
#pybind11#class LeastSquaresTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def _assertClose(self, a, b, rtol=1E-5, atol=1E-8):
#pybind11#        self.assertFloatsAlmostEqual(a, b, rtol=rtol, atol=atol, msg="\n%s\n!=\n%s" % (a, b))
#pybind11#
#pybind11#    def _assertNotClose(self, a, b, rtol=1E-5, atol=1E-8):
#pybind11#        self.assertFloatsNotEqual(a, b, rtol=rtol, atol=atol, msg="\n%s\n==\n%s" % (a, b))
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        numpy.random.seed(500)
#pybind11#
#pybind11#    def check(self, solver, solution, rank, fisher, cov, sv):
#pybind11#        self.assertEqual(solver.getRank(), rank)
#pybind11#        self.assertEqual(solver.getDimension(), solution.shape[0])
#pybind11#        self._assertClose(solver.getSolution(), solution)
#pybind11#        self._assertClose(solver.getFisherMatrix(), fisher)
#pybind11#        self._assertClose(solver.getCovariance(), cov)
#pybind11#        if solver.getFactorization() != LeastSquares.NORMAL_CHOLESKY:
#pybind11#            self._assertClose(solver.getDiagnostic(LeastSquares.NORMAL_EIGENSYSTEM), sv**2)
#pybind11#            diagnostic = solver.getDiagnostic(solver.getFactorization())
#pybind11#            rcond = diagnostic[0] * solver.getThreshold()
#pybind11#            self.assertGreater(diagnostic[rank-1], rcond)
#pybind11#            if rank < solver.getDimension():
#pybind11#                self.assertLess(diagnostic[rank], rcond)
#pybind11#        else:
#pybind11#            self._assertClose(numpy.multiply.reduce(solver.getDiagnostic(LeastSquares.NORMAL_CHOLESKY)),
#pybind11#                             numpy.multiply.reduce(sv**2))
#pybind11#
#pybind11#    def testFullRank(self):
#pybind11#        dimension = 10
#pybind11#        nData = 500
#pybind11#        design = numpy.random.randn(dimension, nData).transpose()
#pybind11#        data = numpy.random.randn(nData)
#pybind11#        fisher = numpy.dot(design.transpose(), design)
#pybind11#        rhs = numpy.dot(design.transpose(), data)
#pybind11#        solution, residues, rank, sv = numpy.linalg.lstsq(design, data)
#pybind11#        cov = numpy.linalg.inv(fisher)
#pybind11#        s_svd = LeastSquares.fromDesignMatrix(design, data, LeastSquares.DIRECT_SVD)
#pybind11#        s_design_eigen = LeastSquares.fromDesignMatrix(design, data, LeastSquares.NORMAL_EIGENSYSTEM)
#pybind11#        s_design_cholesky = LeastSquares.fromDesignMatrix(design, data, LeastSquares.NORMAL_CHOLESKY)
#pybind11#        s_normal_eigen = LeastSquares.fromNormalEquations(fisher, rhs, LeastSquares.NORMAL_EIGENSYSTEM)
#pybind11#        s_normal_cholesky = LeastSquares.fromNormalEquations(fisher, rhs, LeastSquares.NORMAL_CHOLESKY)
#pybind11#        self.check(s_svd, solution, rank, fisher, cov, sv)
#pybind11#        self.check(s_design_eigen, solution, rank, fisher, cov, sv)
#pybind11#        self.check(s_design_cholesky, solution, rank, fisher, cov, sv)
#pybind11#        self.check(s_normal_eigen, solution, rank, fisher, cov, sv)
#pybind11#        self.check(s_normal_cholesky, solution, rank, fisher, cov, sv)
#pybind11#        # test updating solver in-place with the same kind of inputs
#pybind11#        design = numpy.random.randn(dimension, nData).transpose()
#pybind11#        data = numpy.random.randn(nData)
#pybind11#        fisher = numpy.dot(design.transpose(), design)
#pybind11#        rhs = numpy.dot(design.transpose(), data)
#pybind11#        solution, residues, rank, sv = numpy.linalg.lstsq(design, data)
#pybind11#        cov = numpy.linalg.inv(fisher)
#pybind11#        s_svd.setDesignMatrix(design, data)
#pybind11#        s_design_eigen.setDesignMatrix(design, data)
#pybind11#        s_design_cholesky.setDesignMatrix(design, data)
#pybind11#        s_normal_eigen.setNormalEquations(fisher, rhs)
#pybind11#        s_normal_cholesky.setNormalEquations(fisher, rhs)
#pybind11#        self.check(s_svd, solution, rank, fisher, cov, sv)
#pybind11#        self.check(s_design_eigen, solution, rank, fisher, cov, sv)
#pybind11#        self.check(s_design_cholesky, solution, rank, fisher, cov, sv)
#pybind11#        self.check(s_normal_eigen, solution, rank, fisher, cov, sv)
#pybind11#        self.check(s_normal_cholesky, solution, rank, fisher, cov, sv)
#pybind11#        # test updating solver in-place with the opposite kind of inputs
#pybind11#        design = numpy.random.randn(dimension, nData).transpose()
#pybind11#        data = numpy.random.randn(nData)
#pybind11#        fisher = numpy.dot(design.transpose(), design)
#pybind11#        rhs = numpy.dot(design.transpose(), data)
#pybind11#        solution, residues, rank, sv = numpy.linalg.lstsq(design, data)
#pybind11#        cov = numpy.linalg.inv(fisher)
#pybind11#        s_normal_eigen.setDesignMatrix(design, data)
#pybind11#        s_normal_cholesky.setDesignMatrix(design, data)
#pybind11#        s_design_eigen.setNormalEquations(fisher, rhs)
#pybind11#        s_design_cholesky.setNormalEquations(fisher, rhs)
#pybind11#        self.check(s_design_eigen, solution, rank, fisher, cov, sv)
#pybind11#        self.check(s_design_cholesky, solution, rank, fisher, cov, sv)
#pybind11#        self.check(s_normal_eigen, solution, rank, fisher, cov, sv)
#pybind11#        self.check(s_normal_cholesky, solution, rank, fisher, cov, sv)
#pybind11#
#pybind11#    def testSingular(self):
#pybind11#        dimension = 10
#pybind11#        nData = 100
#pybind11#        svIn = (numpy.random.randn(dimension) + 1.0)**2 + 1.0
#pybind11#        svIn = numpy.sort(svIn)[::-1]
#pybind11#        svIn[-1] = 0.0
#pybind11#        svIn[-2] = svIn[0] * 1E-4
#pybind11#        # Just use SVD to get a pair of orthogonal matrices; we'll use our own singular values
#pybind11#        # so we can control the stability of the matrix.
#pybind11#        u, s, vt = numpy.linalg.svd(numpy.random.randn(dimension, nData), full_matrices=False)
#pybind11#        design = numpy.dot(u * svIn, vt).transpose()
#pybind11#        data = numpy.random.randn(nData)
#pybind11#        fisher = numpy.dot(design.transpose(), design)
#pybind11#        rhs = numpy.dot(design.transpose(), data)
#pybind11#        threshold = 10 * sys.float_info.epsilon
#pybind11#        solution, residues, rank, sv = numpy.linalg.lstsq(design, data, rcond=threshold)
#pybind11#        self._assertClose(svIn, sv)
#pybind11#        cov = numpy.linalg.pinv(fisher, rcond=threshold)
#pybind11#        s_svd = LeastSquares.fromDesignMatrix(design, data, LeastSquares.DIRECT_SVD)
#pybind11#        s_design_eigen = LeastSquares.fromDesignMatrix(design, data, LeastSquares.NORMAL_EIGENSYSTEM)
#pybind11#        s_normal_eigen = LeastSquares.fromNormalEquations(fisher, rhs, LeastSquares.NORMAL_EIGENSYSTEM)
#pybind11#        self.check(s_svd, solution, rank, fisher, cov, sv)
#pybind11#        self.check(s_design_eigen, solution, rank, fisher, cov, sv)
#pybind11#        self.check(s_normal_eigen, solution, rank, fisher, cov, sv)
#pybind11#        s_svd.setThreshold(1E-3)
#pybind11#        s_design_eigen.setThreshold(1E-6)
#pybind11#        s_normal_eigen.setThreshold(1E-6)
#pybind11#        self.assertEqual(s_svd.getRank(), dimension - 2)
#pybind11#        self.assertEqual(s_design_eigen.getRank(), dimension - 2)
#pybind11#        self.assertEqual(s_normal_eigen.getRank(), dimension - 2)
#pybind11#        # Just check that solutions are different from before, but consistent with each other;
#pybind11#        # I can't figure out how get numpy.lstsq to deal with the thresholds appropriately to
#pybind11#        # test against that.
#pybind11#        self._assertNotClose(s_svd.getSolution(), solution)
#pybind11#        self._assertNotClose(s_design_eigen.getSolution(), solution)
#pybind11#        self._assertNotClose(s_normal_eigen.getSolution(), solution)
#pybind11#        self._assertClose(s_svd.getSolution(), s_design_eigen.getSolution())
#pybind11#        self._assertClose(s_svd.getSolution(), s_normal_eigen.getSolution())
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

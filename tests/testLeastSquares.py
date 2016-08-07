#!/usr/bin/env python2
from __future__ import absolute_import, division

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
Tests for math.LeastSquares

Run with:
   ./testLeastSquares.py
or
   python
   >>> import testLeastSquares; testLeastSquares.run()
"""

import unittest
import numpy
import sys

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.pex.logging
from lsst.afw.math import LeastSquares

numpy.random.seed(500)

lsst.pex.logging.getDefaultLog().setThresholdFor("afw.math.LeastSquares", -10)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class LeastSquaresTestCase(unittest.TestCase):

    def assertClose(self, a, b, rtol=1E-5, atol=1E-8):
        self.assert_(numpy.allclose(a, b, rtol=rtol, atol=atol), "\n%s\n!=\n%s" % (a, b))

    def assertNotClose(self, a, b, rtol=1E-5, atol=1E-8):
        self.assert_(not numpy.allclose(a, b, rtol=rtol, atol=atol), "\n%s\n==\n%s" % (a, b))

    def check(self, solver, solution, rank, fisher, cov, sv):
        self.assertEqual(solver.getRank(), rank)
        self.assertEqual(solver.getDimension(), solution.shape[0])
        self.assertClose(solver.getSolution(), solution)
        self.assertClose(solver.getFisherMatrix(), fisher)
        self.assertClose(solver.getCovariance(), cov)
        if solver.getFactorization() != LeastSquares.NORMAL_CHOLESKY:
            self.assertClose(solver.getDiagnostic(LeastSquares.NORMAL_EIGENSYSTEM), sv**2)
            diagnostic = solver.getDiagnostic(solver.getFactorization())
            rcond = diagnostic[0] * solver.getThreshold()
            self.assert_(diagnostic[rank-1] > rcond)
            if rank < solver.getDimension():
                self.assert_(diagnostic[rank] < rcond)
        else:
            self.assertClose(numpy.multiply.reduce(solver.getDiagnostic(LeastSquares.NORMAL_CHOLESKY)),
                             numpy.multiply.reduce(sv**2))

    def testFullRank(self):
        dimension = 10
        nData = 500
        design = numpy.random.randn(dimension, nData).transpose()
        data = numpy.random.randn(nData)
        fisher = numpy.dot(design.transpose(), design)
        rhs = numpy.dot(design.transpose(), data)
        solution, residues, rank, sv = numpy.linalg.lstsq(design, data)
        cov = numpy.linalg.inv(fisher)
        s_svd = LeastSquares.fromDesignMatrix(design, data, LeastSquares.DIRECT_SVD)
        s_design_eigen = LeastSquares.fromDesignMatrix(design, data, LeastSquares.NORMAL_EIGENSYSTEM)
        s_design_cholesky = LeastSquares.fromDesignMatrix(design, data, LeastSquares.NORMAL_CHOLESKY)
        s_normal_eigen = LeastSquares.fromNormalEquations(fisher, rhs, LeastSquares.NORMAL_EIGENSYSTEM)
        s_normal_cholesky = LeastSquares.fromNormalEquations(fisher, rhs, LeastSquares.NORMAL_CHOLESKY)
        self.check(s_svd, solution, rank, fisher, cov, sv)
        self.check(s_design_eigen, solution, rank, fisher, cov, sv)
        self.check(s_design_cholesky, solution, rank, fisher, cov, sv)
        self.check(s_normal_eigen, solution, rank, fisher, cov, sv)
        self.check(s_normal_cholesky, solution, rank, fisher, cov, sv)
        # test updating solver in-place with the same kind of inputs
        design = numpy.random.randn(dimension, nData).transpose()
        data = numpy.random.randn(nData)
        fisher = numpy.dot(design.transpose(), design)
        rhs = numpy.dot(design.transpose(), data)
        solution, residues, rank, sv = numpy.linalg.lstsq(design, data)
        cov = numpy.linalg.inv(fisher)
        s_svd.setDesignMatrix(design, data)
        s_design_eigen.setDesignMatrix(design, data)
        s_design_cholesky.setDesignMatrix(design, data)
        s_normal_eigen.setNormalEquations(fisher, rhs)
        s_normal_cholesky.setNormalEquations(fisher, rhs)
        self.check(s_svd, solution, rank, fisher, cov, sv)
        self.check(s_design_eigen, solution, rank, fisher, cov, sv)
        self.check(s_design_cholesky, solution, rank, fisher, cov, sv)
        self.check(s_normal_eigen, solution, rank, fisher, cov, sv)
        self.check(s_normal_cholesky, solution, rank, fisher, cov, sv)
        # test updating solver in-place with the opposite kind of inputs
        design = numpy.random.randn(dimension, nData).transpose()
        data = numpy.random.randn(nData)
        fisher = numpy.dot(design.transpose(), design)
        rhs = numpy.dot(design.transpose(), data)
        solution, residues, rank, sv = numpy.linalg.lstsq(design, data)
        cov = numpy.linalg.inv(fisher)
        s_normal_eigen.setDesignMatrix(design, data)
        s_normal_cholesky.setDesignMatrix(design, data)
        s_design_eigen.setNormalEquations(fisher, rhs)
        s_design_cholesky.setNormalEquations(fisher, rhs)
        self.check(s_design_eigen, solution, rank, fisher, cov, sv)
        self.check(s_design_cholesky, solution, rank, fisher, cov, sv)
        self.check(s_normal_eigen, solution, rank, fisher, cov, sv)
        self.check(s_normal_cholesky, solution, rank, fisher, cov, sv)

    def testSingular(self):
        dimension = 10
        nData = 100
        svIn = (numpy.random.randn(dimension) + 1.0)**2 + 1.0
        svIn = numpy.sort(svIn)[::-1]
        svIn[-1] = 0.0
        svIn[-2] = svIn[0] * 1E-4
        # Just use SVD to get a pair of orthogonal matrices; we'll use our own singular values
        # so we can control the stability of the matrix.
        u, s, vt = numpy.linalg.svd(numpy.random.randn(dimension, nData), full_matrices=False)
        design = numpy.dot(u * svIn, vt).transpose()
        data = numpy.random.randn(nData)
        fisher = numpy.dot(design.transpose(), design)
        rhs = numpy.dot(design.transpose(), data)
        threshold = 10 * sys.float_info.epsilon
        solution, residues, rank, sv = numpy.linalg.lstsq(design, data, rcond=threshold)
        self.assertClose(svIn, sv)
        cov = numpy.linalg.pinv(fisher, rcond=threshold)
        s_svd = LeastSquares.fromDesignMatrix(design, data, LeastSquares.DIRECT_SVD)
        s_design_eigen = LeastSquares.fromDesignMatrix(design, data, LeastSquares.NORMAL_EIGENSYSTEM)
        s_normal_eigen = LeastSquares.fromNormalEquations(fisher, rhs, LeastSquares.NORMAL_EIGENSYSTEM)
        self.check(s_svd, solution, rank, fisher, cov, sv)
        self.check(s_design_eigen, solution, rank, fisher, cov, sv)
        self.check(s_normal_eigen, solution, rank, fisher, cov, sv)
        s_svd.setThreshold(1E-3)
        s_design_eigen.setThreshold(1E-6)
        s_normal_eigen.setThreshold(1E-6)
        self.assertEqual(s_svd.getRank(), dimension - 2)
        self.assertEqual(s_design_eigen.getRank(), dimension - 2)
        self.assertEqual(s_normal_eigen.getRank(), dimension - 2)
        # Just check that solutions are different from before, but consistent with each other;
        # I can't figure out how get numpy.lstsq to deal with the thresholds appropriately to
        # test against that.
        self.assertNotClose(s_svd.getSolution(), solution)
        self.assertNotClose(s_design_eigen.getSolution(), solution)
        self.assertNotClose(s_normal_eigen.getSolution(), solution)
        self.assertClose(s_svd.getSolution(), s_design_eigen.getSolution())
        self.assertClose(s_svd.getSolution(), s_normal_eigen.getSolution())


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(LeastSquaresTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

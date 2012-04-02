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
        self.assert_(numpy.allclose(a, b, rtol=rtol, atol=atol), "%s != %s" % (a, b))

    def check(self, solver, solution, rank, fisher, cov):
        self.assertEqual(solver.getRank(), rank)
        self.assertEqual(solver.getDimension(), solution.shape[0])
        self.assertClose(solver.solve(), solution)
        self.assertClose(solver.computeFisherMatrix(), fisher)
        self.assertClose(solver.computeCovariance(), cov)

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
        self.check(s_svd, solution, rank, fisher, cov)
        self.check(s_design_eigen, solution, rank, fisher, cov)
        self.check(s_design_cholesky, solution, rank, fisher, cov)
        self.check(s_normal_eigen, solution, rank, fisher, cov)
        self.check(s_normal_cholesky, solution, rank, fisher, cov)
        self.assertClose(s_svd.getCondition(), sv)
        self.assertClose(s_design_eigen.getCondition(), sv**2)
        self.assertClose(s_normal_eigen.getCondition(), sv**2)
        self.assertClose(numpy.multiply.reduce(s_design_cholesky.getCondition()), 
                         numpy.multiply.reduce(sv**2))
        self.assertClose(numpy.multiply.reduce(s_normal_cholesky.getCondition()), 
                         numpy.multiply.reduce(sv**2))
        

    def testSingular(self):
        dimension = 10
        nData = 500
        indep = numpy.random.randn(dimension, dimension-1)
        factors = numpy.random.randn(dimension-1, nData)
        design = numpy.dot(indep, factors).transpose()
        data = numpy.random.randn(nData)
        fisher = numpy.dot(design.transpose(), design)
        rhs = numpy.dot(design.transpose(), data)
        threshold = sys.float_info.epsilon**0.5
        solution, residues, rank, sv = numpy.linalg.lstsq(design, data, rcond=threshold)
        cov = numpy.linalg.pinv(fisher, rcond=threshold)
        s_svd = LeastSquares.fromDesignMatrix(design, data, LeastSquares.DIRECT_SVD)
        s_design_eigen = LeastSquares.fromDesignMatrix(design, data, LeastSquares.NORMAL_EIGENSYSTEM)
        s_normal_eigen = LeastSquares.fromNormalEquations(fisher, rhs, LeastSquares.NORMAL_EIGENSYSTEM)
        self.check(s_svd, solution, rank, fisher, cov)
        self.check(s_design_eigen, solution, rank, fisher, cov)
        self.check(s_normal_eigen, solution, rank, fisher, cov)
        self.assertClose(s_svd.getCondition(), sv)
        self.assertClose(s_design_eigen.getCondition(), sv**2)
        self.assertClose(s_normal_eigen.getCondition(), sv**2)

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

#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import zip
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
#pybind11#
#pybind11#import unittest
#pybind11#
#pybind11#import numpy as np
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.math as afwMath
#pybind11#
#pybind11#
#pybind11#class MinimizeTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def testMinimize2(self):
#pybind11#
#pybind11#        variances = np.array([0.01, 0.01, 0.01, 0.01])
#pybind11#        xPositions = np.array([0.0, 1.0, 0.0, 1.0])
#pybind11#        yPositions = np.array([0.0, 0.0, 1.0, 1.0])
#pybind11#
#pybind11#        polyOrder = 1
#pybind11#        polyFunc = afwMath.PolynomialFunction2D(polyOrder)
#pybind11#
#pybind11#        modelParams = [0.1, 0.2, 0.3]
#pybind11#        polyFunc.setParameters(modelParams)
#pybind11#        measurements = []
#pybind11#        for x, y in zip(xPositions, yPositions):
#pybind11#            measurements.append(polyFunc(x, y))
#pybind11#        print("measurements=", measurements)
#pybind11#
#pybind11#        # Set up initial guesses
#pybind11#        nParameters = polyFunc.getNParameters()
#pybind11#        initialParameters = np.zeros(nParameters, float)
#pybind11#        stepsize = np.ones(nParameters, float)
#pybind11#        stepsize *= 0.1
#pybind11#
#pybind11#        # Minimize!
#pybind11#        fitResults = afwMath.minimize(
#pybind11#            polyFunc,
#pybind11#            initialParameters,
#pybind11#            stepsize,
#pybind11#            measurements,
#pybind11#            variances,
#pybind11#            xPositions,
#pybind11#            yPositions,
#pybind11#            0.1,
#pybind11#        )
#pybind11#
#pybind11#        print("modelParams=", modelParams)
#pybind11#        print("fitParams  =", fitResults.parameterList)
#pybind11#        self.assertTrue(fitResults.isValid, "fit failed")
#pybind11#        self.assertFloatsAlmostEqual(np.array(modelParams), np.array(fitResults.parameterList), 1e-11)
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

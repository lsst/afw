#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#


import unittest

import numpy

import lsst.utils.tests as utilsTests
import lsst.afw.math as afwMath

class MinimizeTestCase(unittest.TestCase):
    def testMinimize2(self):
    
        variances   = numpy.array([0.01, 0.01, 0.01, 0.01])
        xPositions   = numpy.array([0.0, 1.0, 0.0, 1.0])
        yPositions   = numpy.array([0.0, 0.0, 1.0, 1.0])
    
        polyOrder = 1
        polyFunc = afwMath.PolynomialFunction2D(polyOrder)
        
        modelParams = [0.1, 0.2, 0.3]
        polyFunc.setParameters(modelParams)
        measurements = []
        for x, y in zip(xPositions, yPositions):
            measurements.append(polyFunc(x, y))
        print "measurements=", measurements
    
        # Set up initial guesses
        nParameters = polyFunc.getNParameters()
        initialParameters = numpy.zeros(nParameters, float)    
        stepsize  = numpy.ones(nParameters, float)
        stepsize *= 0.1
            
        # Minimize!
        fitResults = afwMath.minimize(
            polyFunc,
            initialParameters,
            stepsize,
            measurements,
            variances,
            xPositions,
            yPositions,
            0.1,
        )
        
        print "modelParams=", modelParams
        print "fitParams  =", fitResults.parameterList
        self.assert_(fitResults.isValid, "fit failed")
        if not numpy.allclose(modelParams, fitResults.parameterList):
            self.fail("fit not accurate")


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(MinimizeTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

if __name__ == "__main__":
    utilsTests.run(suite())

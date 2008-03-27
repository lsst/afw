#!/usr/bin/env python
import pdb                          # we may want to say pdb.set_trace()
import unittest

import numpy

import lsst.daf.tests as tests
import lsst.afw.math as afwMath

__all__ = ["computePsfMatchingKernelForMaskedImage"]

class MinimizeTestCase(unittest.TestCase):
    def testMinimize2(self):
    
        variances   = numpy.array([0.01, 0.01, 0.01, 0.01])
        xPositions   = numpy.array([0.0, 1.0, 0.0, 1.0])
        yPositions   = numpy.array([0.0, 0.0, 1.0, 1.0])
        errorDef = 0.1
    
        polyOrder = 1
        polyFuncPtr = afwMath.Function2DPtr(afw.PolynomialFunction2D(polyOrder))
        
        modelParams = [0.1, 0.2, 0.3]
        polyFuncPtr.setParameters(modelParams)
        measurements = []
        for x, y in zip(xPositions, yPositions):
            measurements.append(polyFuncPtr(x,y))
        print "measurements=", measurements
    
        # Set up initial guesses
        nParameters = polyFuncPtr.getNParameters()
        initialParameters = numpy.zeros(nParameters, float)    
        stepsize  = numpy.ones(nParameters, float)
        stepsize *= 0.1
            
        # Minimize!
        fitResults = afw.minimize(
            polyFuncPtr,
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
    tests.init()

    suites = []
    suites += unittest.makeSuite(MinimizeTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)

    return unittest.TestSuite(suites)

if __name__ == "__main__":
    tests.run(suite())

#!/usr/bin/env python
import pdb                          # we may want to say pdb.set_trace()
import unittest

import numpy

import lsst.fw.Core.fwLib as fw
import lsst.mwi.tests as tests
import lsst.mwi.utils as mwiu

__all__ = ["computePsfMatchingKernelForMaskedImage"]

class MinimizeTestCase(unittest.TestCase):
    def testMinimize2(self):
    
        variances   = numpy.array([0.01, 0.01, 0.01, 0.01])
        xPositions   = numpy.array([0.0, 1.0, 0.0, 1.0])
        yPositions   = numpy.array([0.0, 0.0, 1.0, 1.0])
        errorDef = 0.1
    
        polyOrder = 1
        polyFunc = fw.PolynomialFunction2D(polyOrder)
        polyFuncPtr = fw.Function2PtrTypeD(polyFunc)
        polyFunc.this.disown() # pointer now owns this function
        
        modelParams = [0.1, 0.2, 0.3]
        polyFunc.setParameters(modelParams)
        measurements = []
        for x, y in zip(xPositions, yPositions):
            measurements.append(polyFunc(x,y))
        print "measurements=", measurements
    
        # Set up initial guesses
        nParameters = polyFunc.getNParameters()
        initialParameters = numpy.zeros(nParameters, float)    
        stepsize  = numpy.ones(nParameters, float)
        stepsize *= 0.1
            
        # Minimize!
        fitResults = fw.minimize(
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

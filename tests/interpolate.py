#!/usr/bin/env python
"""
Tests for Interpolate

Run with:
   ./Interpolate.py
or
   python
   >>> import Interpolate; Interpolate.run()
"""

import pdb  # we may want to say pdb.set_trace()
import unittest

import lsst.utils.tests as utilsTests
import lsst.afw.math as afwMath

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class InterpolateTestCase(unittest.TestCase):
    
    """A test case for Interpolate Lienar"""
    def setUp(self):
        self.n = 10
        self.x = afwMath.vectorD(self.n)
        self.y1 = afwMath.vectorD(self.n)
        self.y2 = afwMath.vectorD(self.n)
        self.y0 = 1.0
        self.dydx = 1.0
        self.d2ydx2 = 0.5

        for i in range(0, self.n, 1):
            self.x[i] = i
            self.y1[i] = self.dydx*self.x[i] + self.y0
            self.y2[i] = self.d2ydx2*self.x[i]*self.x[i] + self.dydx*self.x[i] + self.y0
            
        self.xtest = 4.5
        self.y1test = self.dydx*self.xtest + self.y0
        self.y2test = self.d2ydx2*self.xtest*self.xtest + self.dydx*self.xtest + self.y0

    def tearDown(self):
        del self.x
        del self.y1
        del self.y2

    def testLinearRamp(self):

        # === test the Linear Interpolator ============================
        # default is akima spline
        yinterpL = afwMath.Interpolate(self.x, self.y1)
        youtL = yinterpL.interpolate(self.xtest)

        self.assertEqual(youtL, self.y1test)


    def testNaturalSplineRamp(self):
        
        # === test the Spline interpolator =======================
        # specify interp type with the string interface
        yinterpS = afwMath.Interpolate(self.x, self.y1, "NATURAL_SPLINE")
        youtS = yinterpS.interpolate(self.xtest)
        
        self.assertEqual(youtS, self.y1test)

    def testAkimaSplineParabola(self):
        
        # === test the Spline interpolator =======================
        # specify interp type with the enum style interface
        yinterpS = afwMath.Interpolate(self.x, self.y2, afwMath.Interpolate.AKIMA_SPLINE)
        youtS = yinterpS.interpolate(self.xtest)
        
        self.assertEqual(youtS, self.y2test)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(InterpolateTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

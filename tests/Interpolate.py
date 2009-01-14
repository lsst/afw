#!/usr/bin/env python
"""
Tests for Interpolate

Run with:
   ./Interpolate.py
or
   python
   >>> import Interpolate; Interpolate.run()
"""

import math
import os
import pdb  # we may want to say pdb.set_trace()
import sys
import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class InterpolateTestCase(unittest.TestCase):
    
    """A test case for Interpolate Lienar"""
    def setUp(self):
	self.n = 10
	self.x = afwMath.vectorF(self.n)
	self.y1 = afwMath.vectorF(self.n)
	self.y2 = afwMath.vectorF(self.n)
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
        yinterpL = afwMath.LinearInterpolateFF(self.x, self.y1)
        youtL = yinterpL.interpolate(self.xtest)

	self.assertEqual(youtL, self.y1test)


    def testNaturalSplineRamp(self):
	
        # === test the Spline interpolator =======================
        yinterpS = afwMath.SplineInterpolateFF(self.x, self.y1)
        youtS = yinterpS.interpolate(self.xtest)
	
        self.assertEqual(youtS, self.y1test)

    def testNaturalSplineParabola(self):
	
        # === test the Spline interpolator =======================
	ictrl = afwMath.InterpControl()
	ictrl.setDydx0(2.0*self.d2ydx2*self.x[0] + self.dydx)
	ictrl.setDydxN(2.0*self.d2ydx2*self.x[self.n - 1] + self.dydx)
        yinterpS = afwMath.SplineInterpolateFF(self.x, self.y2, ictrl)
        youtS = yinterpS.interpolate(self.xtest)
	
        self.assertEqual(youtS, self.y2test)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(InterpolateTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

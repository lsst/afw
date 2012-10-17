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
Tests for Interpolate

Run with:
   ./Interpolate.py
or
   python
   >>> import Interpolate; Interpolate.run()
"""


import unittest

import lsst.utils.tests as utilsTests
import lsst.afw.math as afwMath
import lsst.pex.exceptions as pexExcept

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
        yinterpL = afwMath.makeInterpolate(self.x, self.y1)
        youtL = yinterpL.interpolate(self.xtest)

        self.assertEqual(youtL, self.y1test)


    def testNaturalSplineRamp(self):
        
        # === test the Spline interpolator =======================
        # specify interp type with the string interface
        yinterpS = afwMath.makeInterpolate(self.x, self.y1, afwMath.Interpolate.NATURAL_SPLINE)
        youtS = yinterpS.interpolate(self.xtest)
        
        self.assertEqual(youtS, self.y1test)

    def testAkimaSplineParabola(self):
        """test the Spline interpolator"""
        # specify interp type with the enum style interface
        yinterpS = afwMath.makeInterpolate(self.x, self.y2, afwMath.Interpolate.AKIMA_SPLINE)
        youtS = yinterpS.interpolate(self.xtest)
        

        self.assertEqual(youtS, self.y2test)

    def testInvalidInputs(self):
        """Test that invalid inputs cause an abort"""

        utilsTests.assertRaisesLsstCpp(self, pexExcept.MemoryException,
                                       lambda : afwMath.makeInterpolate([0], [1]))

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

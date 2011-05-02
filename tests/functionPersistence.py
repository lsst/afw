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

import os
import unittest

import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.logging as pexLog
import lsst.pex.policy as pexPolicy
import lsst.daf.base as dafBase
import lsst.daf.persistence as dafPersist
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.afw.image.testUtils as imTestUtils

Verbosity = 0 # increase to see trace
pexLog.Debug("lsst.afw", Verbosity)
# pexLog.Debug("afw.math.KernelFormatter", 30)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class FunctionPersistenceTestCase(unittest.TestCase):
    """A test case for persistence of Functions"""
    def test1DFunctions(self):
        for cName, funcArgs in (
            ("GaussianFunction1", (0.5,)),
            ("GaussianFunction1", (1.0,)),
            ("GaussianFunction1", (2.5,)),
            ("PolynomialFunction1", (0,)),
            ("PolynomialFunction1", (1,)),
            ("PolynomialFunction1", (4,)),
            ("Chebyshev1Function1", (0,)),
            ("Chebyshev1Function1", (1,)),
            ("Chebyshev1Function1", (4,)),
            ("LanczosFunction1", (1, 0.0)),
            ("LanczosFunction1", (1, 0.5)),
            ("LanczosFunction1", (1, -0.5)),
            ("LanczosFunction1", (2, 0.0)),
            ("LanczosFunction1", (2, 0.5)),
            ("LanczosFunction1", (2, -0.5)),
            ("LanczosFunction1", (3, 0.0)),
            ("LanczosFunction1", (3, 0.5)),
            ("LanczosFunction1", (3, -0.5)),
            ("NullFunction1", ()),
        ):
            for typeSuffix in ("F", "D"):
                pyName = cName + typeSuffix
                funcClass = getattr(afwMath, pyName)
                f = funcClass(*funcArgs)
                self.check1DFunction(f, cName)
                
    def test2DFunctions(self):
        for cName, funcArgs in (
            ("IntegerDeltaFunction2", (0.0, 0.0)),
            ("IntegerDeltaFunction2", (1.0, -1.0)),
            ("IntegerDeltaFunction2", (-1.0, 1.0)),
            ("GaussianFunction2", (0.5, 1.0)),
            ("GaussianFunction2", (1.0, 2.5)),
            ("GaussianFunction2", (2.5, 0.5)),
            ("PolynomialFunction2", (0,)),
            ("PolynomialFunction2", (1,)),
            ("PolynomialFunction2", (4,)),
            ("Chebyshev1Function2", (0,)),
            ("Chebyshev1Function2", (1,)),
            ("Chebyshev1Function2", (4,)),
            ("LanczosFunction2", (1, 0.0, 0.0)),
            ("LanczosFunction2", (1, 0.5, -0.5)),
            ("LanczosFunction2", (1, -0.5, 0.5)),
            ("LanczosFunction2", (2, 0.0, 0.0)),
            ("LanczosFunction2", (2, 0.5, -0.5)),
            ("LanczosFunction2", (2, -0.5, 0.5)),
            ("LanczosFunction2", (3, 0.0, 0.0)),
            ("LanczosFunction2", (3, 0.5, -0.5)),
            ("LanczosFunction2", (3, -0.5, 0.5)),
            ("NullFunction2", ()),
        ):
            for typeSuffix in ("F", "D"):
                pyName = cName + typeSuffix
                funcClass = getattr(afwMath, pyName)
                f = funcClass(*funcArgs)
                self.check2DFunction(f, cName)
                
    
    def check1DFunction(self, f, cName, deltaCoeff=0.1, nVal=10, xMin=-1.0, xMax=1.0):
        """Check a 1-D Function: set coeffs, persist, unpersist and compare to original
        """
        nCoeffs = f.getNParameters()
        coeffs = numpy.arange(deltaCoeff * nCoeffs, deltaCoeff / 2.0, -deltaCoeff)
        f.setParameters(coeffs)
        f2 = self.saveRestoreFunction(f, cName)
        self.assertSame1DFunction(f, f2, nVal=nVal, xMin=xMin, xMax=xMax)
        
    def check2DFunction(self, f, cName, deltaCoeff=0.1, nVal=10, xMin=-1.0, yMin=-1.0, xMax=1.0, yMax=1.0):
        """Check a 2-D Function: set coeffs, persist, unpersist and compare to original
        """
        nCoeffs = f.getNParameters()
        coeffs = numpy.arange(deltaCoeff * nCoeffs, deltaCoeff / 2.0, -deltaCoeff)
        f.setParameters(coeffs)
        f2 = self.saveRestoreFunction(f, cName)
        self.assertSame2DFunction(f, f2, nVal=nVal, xMin=xMin, yMin=yMin, xMax=xMax, yMax=yMax)
        
    def assertSame1DFunction(self, f1, f2, nVal=10, xMin=-1.0, xMax=1.0):
        """Assert that two 1D functions are the same
        """
        self.assertSameFunctions(f1, f2)
        
        xDelta = (xMax - xMin) / float(nVal - 1)
        xArr = numpy.arange(xMin, xMax + (xDelta / 2.0), xDelta)
        for x in xArr:
            if not numpy.allclose(f1(x), f2(x)):
                self.fail("%s(%s) = %s != %s(%s)" % \
                    (f1, x, f1(x), f2(x), f2, x))

    def assertSame2DFunction(self, f1, f2, nVal=10, xMin=-1.0, yMin=-1.0, xMax=1.0, yMax=1.0):
        """Assert that two 2D functions are the same
        """
        self.assertSameFunctions(f1, f2)
        
        xDelta = (xMax - xMin) / float(nVal - 1)
        xArr = numpy.arange(xMin, xMax + (xDelta / 2.0), xDelta)
        yArr = numpy.arange(yMin, yMax + (yDelta / 2.0), yDelta)
        for x in xArr:
            for y in yArr:
                if not numpy.allclose(f1(x, y), f2(x, y)):
                    self.fail("%s(%s, %s) = %s != %s(%s, %s)" % \
                        (f1, x, y, f1(x, y), f2(x, y), f2, x, y))

    def assertSameFunctions(self, f1, f2):
        """Assert that two functions are the same using tests that work for both 1D and 2D functions
        """
        self.assertEqual(type(f1), type(f2))
        self.assertEqual(f1.getNParameters(), f2.getNParameters())
        self.assertEqual(f1.getParameters(), f2.getParameters())
        self.assertEqual(f1.toString(), f2.toString())

    def saveRestoreFunction(self, f, cName):
        """Persist and unpersist f, returning the result
        
        @param f: Function to persist
        @param cName: C++ name of function class
        """
        funcPath = "tests/data/function.boost"
        pyName = type(f).__name__
        print "saveRestoreFunction(f=%s, cname=%s)" % (pyName, cName)

        pol = pexPolicy.Policy()
        additionalData = dafBase.PropertySet()
        loc = dafPersist.LogicalLocation(funcPath)
        persistence = dafPersist.Persistence.getPersistence(pol)

        storageList = dafPersist.StorageList()
        storage = persistence.getPersistStorage("XmlStorage", loc)
        storageList.append(storage)
        persistence.persist(f, storageList, additionalData)

        storageList2 = dafPersist.StorageList()
        storage2 = persistence.getRetrieveStorage("XmlStorage", loc)
        storageList2.append(storage2)
        x = persistence.unsafeRetrieve(cName, storageList2, additionalData)
        
        os.remove(funcPath)
        
        return type(f).swigConvert(x)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(FunctionPersistenceTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

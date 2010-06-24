#!/usr/bin/env python
"""
Tests for Footprints, and FootprintSets

Run with:
   Footprint_1.py
or
   python
   >>> import Footprint_1; Footprint_1.run()
"""


import unittest
import lsst.utils.tests as tests
import testEigenLib
import numpy

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class NumpyEigenTestCase(unittest.TestCase):
    """A test case for numpy/eigen bindings"""
    def setUp(self):
        self.a = {}
        n = 2
        for t in [numpy.int32, numpy.float32, numpy.float64]:
            self.a[t] = numpy.ones(n**2, dtype=t).reshape(n, n)

        self.identity = numpy.zeros(n**2).reshape(n, n)
        self.identity[range(n), range(n)] = 1

    def tearDown(self):
        pass

    def testByConstRef_Matrix2d_row(self):
        """Wcs expects a row-major Matrix2? const&"""
        for t, a in self.a.items():
            testEigenLib.Wcs(a)

    def testByConstRef_MatrixX_row(self):
        """printIt expects a row-major MatrixXf const&"""
        for t, a in self.a.items():
            testEigenLib.printIt(a)

    def testByPtr_MatrixX_row(self):
        """identity expects a row-major MatrixXf *"""
        for t, a in self.a.items():
            testEigenLib.identity(a)
            self.assertTrue(numpy.all(a == self.identity))

    def testGetIdentity(self):
        if False:
            print testEigenLib.getIdentity2() # a fixed dimension 2x2 matrix
            print testEigenLib.getIdentity(3) # a 3x3 dynamic matrix

        self.assertTrue(numpy.all(testEigenLib.getIdentity2() == testEigenLib.getIdentity(2)))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(NumpyEigenTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

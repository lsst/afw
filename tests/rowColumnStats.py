#!/usr/bin/env python
# -*- python -*-
"""
Tests for statisticsStack row/column statistics

Run with:
   ./rowColumnStats.py
or
   python
   >>> import rowColumnStats; rowColumnStats.run()
"""

##########################
# rowColumnStats.py
# Steve Bickerton
# An python test to check the row/column statistics from statisticsStack

import unittest
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexEx
import lsst.afw.display.ds9 as ds9

class RowColumnStatisticsTestCase(unittest.TestCase):

    def setUp(self):

        # fill an image with a gradient
        self.n      = 8
        self.column = [0.0 for i in range(self.n)]
        self.row    = [0.0 for i in range(self.n)]
        self.img    = afwImage.ImageF(self.n, self.n, 0)

        for y in range(self.n):
            for x in range(self.n):
                val = 1.0*x + 2.0*y
                self.img.set(x, y, val)
                self.column[y] += val
                self.row[x]    += val

        for i in range(self.n):
            self.row[i]    /= self.n
            self.column[i] /= self.n

    def tearDown(self):
        del self.img

    def testColumnStats(self):
        """Test the column statistics """
        imgProjectCol = afwMath.statisticsStack(self.img, afwMath.MEAN, 'x')
        for i in range(self.n):
            self.assertEqual(imgProjectCol.get(0, i)[0], self.column[i])

    def testRowStats(self):
        """Test the row statistics """
        imgProjectRow = afwMath.statisticsStack(self.img, afwMath.MEAN, 'y')
        for i in range(self.n):
            self.assertEqual(imgProjectRow.get(i, 0)[0], self.row[i])

    def testColumnPlus(self):
        """ Test sliceOperate on column addition """
        imgCol = afwImage.ImageF(1, self.n, 1)
        imPlus = afwMath.sliceOperate(self.img, imgCol, "column", '+')
        print self.img.get(self.n/2, self.n/2), imPlus.get(self.n/2, self.n/2)

#################################################################
# Test suite boiler plate
#################################################################
def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(RowColumnStatisticsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)


#!/usr/bin/env python
# -*- python -*-
"""
Tests for Stack

Run with:
   ./Stacker.py
or
   python
   >>> import Stacker; Stacker.run()
"""

##########################
# simpleStacker.py
# Steve Bickerton
# An example executible which calls the example 'stack' code 

import sys, os
import unittest
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.utils.tests as utilsTests
import eups

######################################
# main body of code
######################################
class StackTestCase(unittest.TestCase):

    def setUp(self):
        self.nImg = 10
        self.nX, self.nY = 64, 64
        
    def testMean(self):
        """ Test the meanStack() function """

        knownMean = 0.0
        imgList = afwImage.vectorImageF()
        for iImg in range(self.nImg):
            imgList.push_back(afwImage.ImageF(self.nX, self.nY, iImg))
            knownMean += iImg
            imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN)
        knownMean /= self.nImg
        
        self.assertEqual(imgStack.get(self.nX/2, self.nY/2), knownMean)

    def testStatistics(self):
        """ Test the statisticsStack() function """

        values = [1.0, 2.0, 2.0, 3.0, 8.0 ]

        imgList = afwImage.vectorImageF()
        for val in values:
            imgList.push_back(afwImage.ImageF(self.nX, self.nY, val))

        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN)
        mean = reduce(lambda x, y: x+y, values)/float(len(values))
        self.assertAlmostEqual(imgStack.get(self.nX/2, self.nY/2), mean)

        imgStack = afwMath.statisticsStack(imgList, afwMath.MEDIAN)
        median = sorted(values)[len(values)//2]
        self.assertEqual(imgStack.get(self.nX/2, self.nY/2), median)

#################################################################
# Test suite boiler plate
#################################################################
def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(StackTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit = False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

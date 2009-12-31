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
        self.values = [1.0, 2.0, 2.0, 3.0, 8.0 ]
        
    def testMean(self):
        """ Test the statisticsStack() function for a MEAN"""

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
        
        imgList = afwImage.vectorImageF()
        for val in self.values:
            imgList.push_back(afwImage.ImageF(self.nX, self.nY, val))
            
        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN)
        mean = reduce(lambda x, y: x+y, self.values)/float(len(self.values))
        self.assertAlmostEqual(imgStack.get(self.nX/2, self.nY/2), mean)

        imgStack = afwMath.statisticsStack(imgList, afwMath.MEDIAN)
        median = sorted(self.values)[len(self.values)//2]
        self.assertEqual(imgStack.get(self.nX/2, self.nY/2), median)

    def testWeightedStack(self):
        """ Test statisticsStack() function when weighting by a variance plane"""
        
        sctrl = afwMath.StatisticsControl()
        sctrl.setWeighted(True)
        mimgList = afwImage.vectorMaskedImageF()
        for val in self.values:
            mimg = afwImage.MaskedImageF(self.nX, self.nY)
            mimg.set(val, 0x0, val)
            mimgList.push_back(mimg)
        mimgStack = afwMath.statisticsStack(mimgList, afwMath.MEAN, sctrl)

        wvalues = map(lambda q: 1.0/q, self.values)
        wmean = float(len(self.values)) / reduce(lambda x, y: x + y, wvalues)
        self.assertAlmostEqual(mimgStack.getImage().get(self.nX/2, self.nY/2), wmean)


    def testConstantWeightedStack(self):
        """ Test statisticsStack() function when weighting by a vector of weights"""
        
        sctrl = afwMath.StatisticsControl()
        imgList = afwImage.vectorImageF()
        weights = afwMath.vectorF()
        for val in self.values:
            img = afwImage.ImageF(self.nX, self.nY, val)
            imgList.push_back(img)
            weights.push_back(val)
        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN, sctrl, weights)

        wsum = reduce(lambda x, y: x + y, self.values)
        wvalues = map(lambda x: x*x, self.values)
        wmean = reduce(lambda x, y: x + y, wvalues)/float(wsum)
        self.assertAlmostEqual(imgStack.get(self.nX/2, self.nY/2), wmean)


        
        
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

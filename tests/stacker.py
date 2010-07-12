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

import unittest
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexEx
import lsst.afw.display.ds9 as ds9

try:
    type(display)
except:
    display = False

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

        wvalues = [1.0/q for q in self.values]
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
        wvalues = [x*x for x in self.values]
        wmean = reduce(lambda x, y: x + y, wvalues)/float(wsum)
        self.assertAlmostEqual(imgStack.get(self.nX/2, self.nY/2), wmean)


    def testRequestMoreThanOneStat(self):
        """ Make sure we throw an exception if someone requests more than one type of statistics. """

        sctrl = afwMath.StatisticsControl()
        imgList = afwImage.vectorImageF()
        for val in self.values:
            img = afwImage.ImageF(self.nX, self.nY, val)
            imgList.push_back(img)

        def tst():
            imgStackBad = afwMath.statisticsStack(imgList, afwMath.MEAN | afwMath.MEANCLIP, sctrl)
            
        utilsTests.assertRaisesLsstCpp(self, pexEx.InvalidParameterException, tst)


    def testReturnInputs(self):
        """ Make sure that a single file put into the stacker is returned unscathed"""

        imgList = afwImage.vectorMaskedImageF()
        
        img = afwImage.MaskedImageF(10, 20)
        for y in range(img.getHeight()):
            simg = img.Factory(img, afwImage.BBox(afwImage.PointI(0, y), img.getWidth(), 1))
            simg.set(y)

        imgList.push_back(img)

        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN)

        if display:
            ds9.mtv(img, frame=1, title="input")
            ds9.mtv(imgStack, frame=2, title="stack")

        self.assertEqual(img.get(0, 0)[0], imgStack.get(0, 0)[0])

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

def run(shouldExit = False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

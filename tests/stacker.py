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
import numpy
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexEx
import lsst.afw.display.ds9 as ds9

numpy.random.seed(1)

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
            imgList.push_back(afwImage.ImageF(afwGeom.Extent2I(self.nX, self.nY), iImg))
            knownMean += iImg

        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN)
        knownMean /= self.nImg
        self.assertEqual(imgStack.get(self.nX/2, self.nY/2), knownMean)

        # Test in-place stacking
        afwMath.statisticsStack(imgStack, imgList, afwMath.MEAN)
        self.assertEqual(imgStack.get(self.nX/2, self.nY/2), knownMean)
        
    def testStatistics(self):
        """ Test the statisticsStack() function """
        
        imgList = afwImage.vectorImageF()
        for val in self.values:
            imgList.push_back(afwImage.ImageF(afwGeom.Extent2I(self.nX, self.nY), val))
            
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
            mimg = afwImage.MaskedImageF(afwGeom.Extent2I(self.nX, self.nY))
            mimg.set(val, 0x0, val)
            mimgList.push_back(mimg)
        mimgStack = afwMath.statisticsStack(mimgList, afwMath.MEAN, sctrl)

        wvalues = [1.0/q for q in self.values]
        wmean = float(len(self.values)) / reduce(lambda x, y: x + y, wvalues)
        self.assertAlmostEqual(mimgStack.getImage().get(self.nX/2, self.nY/2), wmean)

        # Test in-place stacking
        afwMath.statisticsStack(mimgStack, mimgList, afwMath.MEAN, sctrl)
        self.assertAlmostEqual(mimgStack.getImage().get(self.nX/2, self.nY/2), wmean)

    def testConstantWeightedStack(self):
        """ Test statisticsStack() function when weighting by a vector of weights"""
        
        sctrl = afwMath.StatisticsControl()
        imgList = afwImage.vectorImageF()
        weights = afwMath.vectorF()
        for val in self.values:
            img = afwImage.ImageF(afwGeom.Extent2I(self.nX, self.nY), val)
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
            img = afwImage.ImageF(afwGeom.Extent2I(self.nX, self.nY), val)
            imgList.push_back(img)

        def tst():
            imgStackBad = afwMath.statisticsStack(imgList, afwMath.MEAN | afwMath.MEANCLIP, sctrl)
            
        utilsTests.assertRaisesLsstCpp(self, pexEx.InvalidParameterError, tst)


    def testReturnInputs(self):
        """ Make sure that a single file put into the stacker is returned unscathed"""

        imgList = afwImage.vectorMaskedImageF()
        
        img = afwImage.MaskedImageF(afwGeom.Extent2I(10, 20))
        for y in range(img.getHeight()):
            simg = img.Factory(
                img,
                afwGeom.Box2I(afwGeom.Point2I(0, y), afwGeom.Extent2I(img.getWidth(), 1)),
                afwImage.LOCAL)
            simg.set(y)

        imgList.push_back(img)

        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN)

        if display:
            ds9.mtv(img, frame=1, title="input")
            ds9.mtv(imgStack, frame=2, title="stack")

        self.assertEqual(img.get(0, 0)[0], imgStack.get(0, 0)[0])

    def testStackBadPixels(self):
        """Check that we properly ignore masked pixels, and set noGoodPixelsMask where there are
        no good pixels"""
        mimgVec = afwImage.vectorMaskedImageF()

        DETECTED = afwImage.MaskU_getPlaneBitMask("DETECTED")
        EDGE =  afwImage.MaskU_getPlaneBitMask("EDGE")
        INTRP = afwImage.MaskU_getPlaneBitMask("INTRP")
        SAT = afwImage.MaskU_getPlaneBitMask("SAT")

        sctrl = afwMath.StatisticsControl()
        sctrl.setNanSafe(False)
        sctrl.setAndMask(INTRP | SAT)
        sctrl.setNoGoodPixelsMask(EDGE)

        edgeBBox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(20, 20)) # set these pixels to EDGE
        width, height = 512, 512
	dim=afwGeom.Extent2I(width, height)
        val, maskVal = 10, DETECTED
        for i in range(4):
            mimg = afwImage.MaskedImageF(dim)
            mimg.set(val, maskVal, 1)
            #
            # Set part of the image to NaN (with the INTRP bit set)
            #
            llc = afwGeom.Point2I(width//2*(i//2), height//2*(i%2))
            bbox = afwGeom.Box2I(llc, dim/2)

            smimg = mimg.Factory(mimg, bbox, afwImage.LOCAL)
            #smimg.set(numpy.nan, INTRP, numpy.nan)
            del smimg
            #
            # And the bottom corner to SAT
            #
            smask = mimg.getMask().Factory(mimg.getMask(), edgeBBox, afwImage.LOCAL)
            smask |= SAT
            del smask

            mimgVec.push_back(mimg)

            if display > 1:
                ds9.mtv(mimg, frame=i, title=str(i))

        mimgStack = afwMath.statisticsStack(mimgVec, afwMath.MEAN, sctrl)

        if display:
            i += 1
            ds9.mtv(mimgStack, frame=i, title="Stack")
            i += 1
            ds9.mtv(mimgStack.getVariance(), frame=i, title="var(Stack)")
        #
        # Check the output, ignoring EDGE pixels
        #
        sctrl = afwMath.StatisticsControl()
        sctrl.setAndMask(afwImage.MaskU_getPlaneBitMask("EDGE"))

        stats = afwMath.makeStatistics(mimgStack, afwMath.MIN | afwMath.MAX, sctrl)
        self.assertEqual(stats.getValue(afwMath.MIN), val)
        self.assertEqual(stats.getValue(afwMath.MAX), val)
        #
        # We have to clear EDGE in the known bad corner to check the mask
        #
        smask = mimgStack.getMask().Factory(mimgStack.getMask(), edgeBBox, afwImage.LOCAL)
        self.assertEqual(smask.get(edgeBBox.getMinX(), edgeBBox.getMinY()), EDGE)
        smask &= ~EDGE
        del smask

        self.assertEqual(afwMath.makeStatistics(mimgStack.getMask(), afwMath.SUM, sctrl).getValue(), maskVal)

    def testTicket1412(self):
        """Ticket 1412: ignored mask bits are propegated to output stack."""

        mimg1 = afwImage.MaskedImageF(afwGeom.Extent2I(1, 1))
        mimg1.set(0, 0, (1, 0x4, 1)) # set 0100
        mimg2 = afwImage.MaskedImageF(afwGeom.Extent2I(1, 1))
        mimg2.set(0, 0, (2, 0x3, 1)) # set 0010 and 0001
        
        imgList = afwImage.vectorMaskedImageF()
        imgList.push_back(mimg1)
        imgList.push_back(mimg2)
        
        sctrl = afwMath.StatisticsControl()
        sctrl.setAndMask(0x1) # andmask only 0001

        # try first with no sctrl (no andmask set), should see 0x0111 for all output mask pixels
        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN)
        self.assertEqual(imgStack.get(0, 0)[1], 0x7)

        # now try with sctrl (andmask = 0x0001), should see 0x0100 for all output mask pixels
        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN, sctrl)
        self.assertEqual(imgStack.get(0, 0)[1], 0x4)
        
    def test2145(self):
        """The how-to-repeat from #2145"""
        Size = 5
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setCalcErrorFromInputVariance(True)
        maskedImageList = afwImage.vectorMaskedImageF()
        weightList = []
        for i in range(3):
            mi = afwImage.MaskedImageF(Size, Size)
            imArr, maskArr, varArr = mi.getArrays()
            imArr[:] = numpy.random.normal(10, 0.1, (Size, Size))
            varArr[:] = numpy.random.normal(10, 0.1, (Size, Size))
            maskedImageList.append(mi)
            weightList.append(1.0)
            
        stack = afwMath.statisticsStack(maskedImageList, afwMath.MEAN, statsCtrl, weightList)
        if False:
            print "image=", stack.getImage().getArray()
            print "variance=", stack.getVariance().getArray()
        self.assertNotEqual(numpy.sum(stack.getVariance().getArray()), 0.0)

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

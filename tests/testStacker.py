#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import str
#pybind11#from builtins import range
#pybind11#from functools import reduce
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11## -*- python -*-
#pybind11#"""
#pybind11#Tests for Stack
#pybind11#
#pybind11#Run with:
#pybind11#   ./Stacker.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import Stacker; Stacker.run()
#pybind11#"""
#pybind11#
#pybind11###########################
#pybind11## simpleStacker.py
#pybind11## Steve Bickerton
#pybind11## An example executible which calls the example 'stack' code
#pybind11#
#pybind11#import unittest
#pybind11#import numpy
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions as pexEx
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except:
#pybind11#    display = False
#pybind11#
#pybind11#######################################
#pybind11## main body of code
#pybind11#######################################
#pybind11#
#pybind11#
#pybind11#class StackTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        numpy.random.seed(1)
#pybind11#        self.nImg = 10
#pybind11#        self.nX, self.nY = 64, 64
#pybind11#        self.values = [1.0, 2.0, 2.0, 3.0, 8.0]
#pybind11#
#pybind11#    def testMean(self):
#pybind11#        """ Test the statisticsStack() function for a MEAN"""
#pybind11#
#pybind11#        knownMean = 0.0
#pybind11#        imgList = afwImage.vectorImageF()
#pybind11#        for iImg in range(self.nImg):
#pybind11#            imgList.push_back(afwImage.ImageF(afwGeom.Extent2I(self.nX, self.nY), iImg))
#pybind11#            knownMean += iImg
#pybind11#
#pybind11#        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN)
#pybind11#        knownMean /= self.nImg
#pybind11#        self.assertEqual(imgStack.get(self.nX//2, self.nY//2), knownMean)
#pybind11#
#pybind11#        # Test in-place stacking
#pybind11#        afwMath.statisticsStack(imgStack, imgList, afwMath.MEAN)
#pybind11#        self.assertEqual(imgStack.get(self.nX//2, self.nY//2), knownMean)
#pybind11#
#pybind11#    def testStatistics(self):
#pybind11#        """ Test the statisticsStack() function """
#pybind11#
#pybind11#        imgList = afwImage.vectorImageF()
#pybind11#        for val in self.values:
#pybind11#            imgList.push_back(afwImage.ImageF(afwGeom.Extent2I(self.nX, self.nY), val))
#pybind11#
#pybind11#        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN)
#pybind11#        mean = reduce(lambda x, y: x+y, self.values)/float(len(self.values))
#pybind11#        self.assertAlmostEqual(imgStack.get(self.nX//2, self.nY//2), mean)
#pybind11#
#pybind11#        imgStack = afwMath.statisticsStack(imgList, afwMath.MEDIAN)
#pybind11#        median = sorted(self.values)[len(self.values)//2]
#pybind11#        self.assertEqual(imgStack.get(self.nX//2, self.nY//2), median)
#pybind11#
#pybind11#    def testWeightedStack(self):
#pybind11#        """ Test statisticsStack() function when weighting by a variance plane"""
#pybind11#
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#        sctrl.setWeighted(True)
#pybind11#        mimgList = afwImage.vectorMaskedImageF()
#pybind11#        for val in self.values:
#pybind11#            mimg = afwImage.MaskedImageF(afwGeom.Extent2I(self.nX, self.nY))
#pybind11#            mimg.set(val, 0x0, val)
#pybind11#            mimgList.push_back(mimg)
#pybind11#        mimgStack = afwMath.statisticsStack(mimgList, afwMath.MEAN, sctrl)
#pybind11#
#pybind11#        wvalues = [1.0/q for q in self.values]
#pybind11#        wmean = float(len(self.values)) / reduce(lambda x, y: x + y, wvalues)
#pybind11#        self.assertAlmostEqual(mimgStack.getImage().get(self.nX//2, self.nY//2), wmean)
#pybind11#
#pybind11#        # Test in-place stacking
#pybind11#        afwMath.statisticsStack(mimgStack, mimgList, afwMath.MEAN, sctrl)
#pybind11#        self.assertAlmostEqual(mimgStack.getImage().get(self.nX//2, self.nY//2), wmean)
#pybind11#
#pybind11#    def testConstantWeightedStack(self):
#pybind11#        """ Test statisticsStack() function when weighting by a vector of weights"""
#pybind11#
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#        imgList = afwImage.vectorImageF()
#pybind11#        weights = afwMath.vectorF()
#pybind11#        for val in self.values:
#pybind11#            img = afwImage.ImageF(afwGeom.Extent2I(self.nX, self.nY), val)
#pybind11#            imgList.push_back(img)
#pybind11#            weights.push_back(val)
#pybind11#        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN, sctrl, weights)
#pybind11#
#pybind11#        wsum = reduce(lambda x, y: x + y, self.values)
#pybind11#        wvalues = [x*x for x in self.values]
#pybind11#        wmean = reduce(lambda x, y: x + y, wvalues)/float(wsum)
#pybind11#        self.assertAlmostEqual(imgStack.get(self.nX//2, self.nY//2), wmean)
#pybind11#
#pybind11#    def testRequestMoreThanOneStat(self):
#pybind11#        """ Make sure we throw an exception if someone requests more than one type of statistics. """
#pybind11#
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#        imgList = afwImage.vectorImageF()
#pybind11#        for val in self.values:
#pybind11#            img = afwImage.ImageF(afwGeom.Extent2I(self.nX, self.nY), val)
#pybind11#            imgList.push_back(img)
#pybind11#
#pybind11#        def tst():
#pybind11#            afwMath.statisticsStack(imgList, afwMath.MEAN | afwMath.MEANCLIP, sctrl)
#pybind11#
#pybind11#        self.assertRaises(pexEx.InvalidParameterError, tst)
#pybind11#
#pybind11#    def testReturnInputs(self):
#pybind11#        """ Make sure that a single file put into the stacker is returned unscathed"""
#pybind11#
#pybind11#        imgList = afwImage.vectorMaskedImageF()
#pybind11#
#pybind11#        img = afwImage.MaskedImageF(afwGeom.Extent2I(10, 20))
#pybind11#        for y in range(img.getHeight()):
#pybind11#            simg = img.Factory(
#pybind11#                img,
#pybind11#                afwGeom.Box2I(afwGeom.Point2I(0, y), afwGeom.Extent2I(img.getWidth(), 1)),
#pybind11#                afwImage.LOCAL)
#pybind11#            simg.set(y)
#pybind11#
#pybind11#        imgList.push_back(img)
#pybind11#
#pybind11#        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN)
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(img, frame=1, title="input")
#pybind11#            ds9.mtv(imgStack, frame=2, title="stack")
#pybind11#
#pybind11#        self.assertEqual(img.get(0, 0)[0], imgStack.get(0, 0)[0])
#pybind11#
#pybind11#    def testStackBadPixels(self):
#pybind11#        """Check that we properly ignore masked pixels, and set noGoodPixelsMask where there are
#pybind11#        no good pixels"""
#pybind11#        mimgVec = afwImage.vectorMaskedImageF()
#pybind11#
#pybind11#        DETECTED = afwImage.MaskU_getPlaneBitMask("DETECTED")
#pybind11#        EDGE = afwImage.MaskU_getPlaneBitMask("EDGE")
#pybind11#        INTRP = afwImage.MaskU_getPlaneBitMask("INTRP")
#pybind11#        SAT = afwImage.MaskU_getPlaneBitMask("SAT")
#pybind11#
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#        sctrl.setNanSafe(False)
#pybind11#        sctrl.setAndMask(INTRP | SAT)
#pybind11#        sctrl.setNoGoodPixelsMask(EDGE)
#pybind11#
#pybind11#        edgeBBox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(20, 20))  # set these pixels to EDGE
#pybind11#        width, height = 512, 512
#pybind11#        dim = afwGeom.Extent2I(width, height)
#pybind11#        val, maskVal = 10, DETECTED
#pybind11#        for i in range(4):
#pybind11#            mimg = afwImage.MaskedImageF(dim)
#pybind11#            mimg.set(val, maskVal, 1)
#pybind11#            #
#pybind11#            # Set part of the image to NaN (with the INTRP bit set)
#pybind11#            #
#pybind11#            llc = afwGeom.Point2I(width//2*(i//2), height//2*(i % 2))
#pybind11#            bbox = afwGeom.Box2I(llc, dim//2)
#pybind11#
#pybind11#            smimg = mimg.Factory(mimg, bbox, afwImage.LOCAL)
#pybind11#            #smimg.set(numpy.nan, INTRP, numpy.nan)
#pybind11#            del smimg
#pybind11#            #
#pybind11#            # And the bottom corner to SAT
#pybind11#            #
#pybind11#            smask = mimg.getMask().Factory(mimg.getMask(), edgeBBox, afwImage.LOCAL)
#pybind11#            smask |= SAT
#pybind11#            del smask
#pybind11#
#pybind11#            mimgVec.push_back(mimg)
#pybind11#
#pybind11#            if display > 1:
#pybind11#                ds9.mtv(mimg, frame=i, title=str(i))
#pybind11#
#pybind11#        mimgStack = afwMath.statisticsStack(mimgVec, afwMath.MEAN, sctrl)
#pybind11#
#pybind11#        if display:
#pybind11#            i += 1
#pybind11#            ds9.mtv(mimgStack, frame=i, title="Stack")
#pybind11#            i += 1
#pybind11#            ds9.mtv(mimgStack.getVariance(), frame=i, title="var(Stack)")
#pybind11#        #
#pybind11#        # Check the output, ignoring EDGE pixels
#pybind11#        #
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#        sctrl.setAndMask(afwImage.MaskU_getPlaneBitMask("EDGE"))
#pybind11#
#pybind11#        stats = afwMath.makeStatistics(mimgStack, afwMath.MIN | afwMath.MAX, sctrl)
#pybind11#        self.assertEqual(stats.getValue(afwMath.MIN), val)
#pybind11#        self.assertEqual(stats.getValue(afwMath.MAX), val)
#pybind11#        #
#pybind11#        # We have to clear EDGE in the known bad corner to check the mask
#pybind11#        #
#pybind11#        smask = mimgStack.getMask().Factory(mimgStack.getMask(), edgeBBox, afwImage.LOCAL)
#pybind11#        self.assertEqual(smask.get(edgeBBox.getMinX(), edgeBBox.getMinY()), EDGE)
#pybind11#        smask &= ~EDGE
#pybind11#        del smask
#pybind11#
#pybind11#        self.assertEqual(afwMath.makeStatistics(mimgStack.getMask(), afwMath.SUM, sctrl).getValue(), maskVal)
#pybind11#
#pybind11#    def testTicket1412(self):
#pybind11#        """Ticket 1412: ignored mask bits are propegated to output stack."""
#pybind11#
#pybind11#        mimg1 = afwImage.MaskedImageF(afwGeom.Extent2I(1, 1))
#pybind11#        mimg1.set(0, 0, (1, 0x4, 1))  # set 0100
#pybind11#        mimg2 = afwImage.MaskedImageF(afwGeom.Extent2I(1, 1))
#pybind11#        mimg2.set(0, 0, (2, 0x3, 1))  # set 0010 and 0001
#pybind11#
#pybind11#        imgList = afwImage.vectorMaskedImageF()
#pybind11#        imgList.push_back(mimg1)
#pybind11#        imgList.push_back(mimg2)
#pybind11#
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#        sctrl.setAndMask(0x1)  # andmask only 0001
#pybind11#
#pybind11#        # try first with no sctrl (no andmask set), should see 0x0111 for all output mask pixels
#pybind11#        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN)
#pybind11#        self.assertEqual(imgStack.get(0, 0)[1], 0x7)
#pybind11#
#pybind11#        # now try with sctrl (andmask = 0x0001), should see 0x0100 for all output mask pixels
#pybind11#        imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN, sctrl)
#pybind11#        self.assertEqual(imgStack.get(0, 0)[1], 0x4)
#pybind11#
#pybind11#    def test2145(self):
#pybind11#        """The how-to-repeat from #2145"""
#pybind11#        Size = 5
#pybind11#        statsCtrl = afwMath.StatisticsControl()
#pybind11#        statsCtrl.setCalcErrorFromInputVariance(True)
#pybind11#        maskedImageList = afwImage.vectorMaskedImageF()
#pybind11#        weightList = []
#pybind11#        for i in range(3):
#pybind11#            mi = afwImage.MaskedImageF(Size, Size)
#pybind11#            imArr, maskArr, varArr = mi.getArrays()
#pybind11#            imArr[:] = numpy.random.normal(10, 0.1, (Size, Size))
#pybind11#            varArr[:] = numpy.random.normal(10, 0.1, (Size, Size))
#pybind11#            maskedImageList.append(mi)
#pybind11#            weightList.append(1.0)
#pybind11#
#pybind11#        stack = afwMath.statisticsStack(maskedImageList, afwMath.MEAN, statsCtrl, weightList)
#pybind11#        if False:
#pybind11#            print("image=", stack.getImage().getArray())
#pybind11#            print("variance=", stack.getVariance().getArray())
#pybind11#        self.assertNotEqual(numpy.sum(stack.getVariance().getArray()), 0.0)
#pybind11#
#pybind11#    def testRejectedMaskPropagation(self):
#pybind11#        """Test that we can propagate mask bits from rejected pixels, when the amount
#pybind11#        of rejection crosses a threshold."""
#pybind11#        rejectedBit = 1        # use this bit to determine whether to reject a pixel
#pybind11#        propagatedBit = 2  # propagate this bit if a pixel with it set is rejected
#pybind11#        statsCtrl = afwMath.StatisticsControl()
#pybind11#        statsCtrl.setMaskPropagationThreshold(propagatedBit, 0.3)
#pybind11#        statsCtrl.setAndMask(1 << rejectedBit)
#pybind11#        statsCtrl.setWeighted(True)
#pybind11#        maskedImageList = afwImage.vectorMaskedImageF()
#pybind11#
#pybind11#        # start with 4 images with no mask bits set
#pybind11#        partialSum = numpy.zeros((1, 4), dtype=numpy.float32)
#pybind11#        finalImage = numpy.array([12.0, 12.0, 12.0, 12.0], dtype=numpy.float32)
#pybind11#        for i in range(4):
#pybind11#            mi = afwImage.MaskedImageF(4, 1)
#pybind11#            imArr, maskArr, varArr = mi.getArrays()
#pybind11#            imArr[:, :] = numpy.ones((1, 4), dtype=numpy.float32)
#pybind11#            maskedImageList.append(mi)
#pybind11#            partialSum += imArr
#pybind11#        # add one more image with all permutations of the first two bits set in different pixels
#pybind11#        mi = afwImage.MaskedImageF(4, 1)
#pybind11#        imArr, maskArr, varArr = mi.getArrays()
#pybind11#        imArr[0, :] = finalImage
#pybind11#        maskArr[0, 1] |= (1 << rejectedBit)
#pybind11#        maskArr[0, 2] |= (1 << propagatedBit)
#pybind11#        maskArr[0, 3] |= (1 << rejectedBit)
#pybind11#        maskArr[0, 3] |= (1 << propagatedBit)
#pybind11#        maskedImageList.append(mi)
#pybind11#
#pybind11#        # these will always be rejected
#pybind11#        finalImage[1] = 0.0
#pybind11#        finalImage[3] = 0.0
#pybind11#
#pybind11#        # Uniform weights: we should only see pixel 2 set with propagatedBit, because it's not rejected;
#pybind11#        # pixel 3 is rejected, but its weight (0.2) below the propagation threshold (0.3)
#pybind11#        stack1 = afwMath.statisticsStack(maskedImageList, afwMath.MEAN, statsCtrl, [1.0, 1.0, 1.0, 1.0, 1.0])
#pybind11#        self.assertEqual(stack1.get(0, 0)[1], 0x0)
#pybind11#        self.assertEqual(stack1.get(1, 0)[1], 0x0)
#pybind11#        self.assertEqual(stack1.get(2, 0)[1], 1 << propagatedBit)
#pybind11#        self.assertEqual(stack1.get(3, 0)[1], 0x0)
#pybind11#        self.assertClose(stack1.getImage().getArray(),
#pybind11#                         (partialSum + finalImage) / numpy.array([5.0, 4.0, 5.0, 4.0]),
#pybind11#                         rtol=1E-7)
#pybind11#
#pybind11#        # Give the masked image more weight: we should see pixel 2 and pixel 3 set with propagatedBit,
#pybind11#        # pixel 2 because it's not rejected, and pixel 3 because the weight of the rejection (0.3333)
#pybind11#        # is above the threshold (0.3)
#pybind11#        # Note that rejectedBit is never propagated, because we didn't include it in statsCtrl (of course,
#pybind11#        # normally the bits we'd propagate and the bits we'd reject would be the same)
#pybind11#        stack2 = afwMath.statisticsStack(maskedImageList, afwMath.MEAN, statsCtrl, [1.0, 1.0, 1.0, 1.0, 2.0])
#pybind11#        self.assertEqual(stack2.get(0, 0)[1], 0x0)
#pybind11#        self.assertEqual(stack2.get(1, 0)[1], 0x0)
#pybind11#        self.assertEqual(stack2.get(2, 0)[1], 1 << propagatedBit)
#pybind11#        self.assertEqual(stack2.get(3, 0)[1], 1 << propagatedBit)
#pybind11#        self.assertClose(stack2.getImage().getArray(),
#pybind11#                         (partialSum + 2*finalImage) / numpy.array([6.0, 4.0, 6.0, 4.0]),
#pybind11#                         rtol=1E-7)
#pybind11#
#pybind11##################################################################
#pybind11## Test suite boiler plate
#pybind11##################################################################
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

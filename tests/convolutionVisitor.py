#!/usr/bin/env python
"""
Tests for ConvolutionVisitor

Run with:
   ./ConvolutionVisitor.py
or
   python
   >>> import ConvolutionVisitor; ConvolutionVisitor.run()
"""

import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math as afwMath

class ConvolutionVisitorTestCase(unittest.TestCase):
    """A test case for ConvolutionVisitor"""
    def setUp(self):
        self.width = 19
        self.height = 19
        self.fourierWidth = self.width/2 + 1
        self.center = (9, 9)
        self.image = afwImage.ImageD(self.width, self.height, 3)
        self.imageList = []
        self.imageList.append(afwImage.ImageD(self.width, self.height, 1))
        self.imageList.append(afwImage.ImageD(self.width, self.height, 2))

        self.paramList = [0.5, 1.5]

    def tearDown(self):
        del self.image
        del self.imageList

    def testBasic(self):
        imageVisitor = afwMath.ImageConvolutionVisitor(
                self.center, 
                self.paramList,
                self.image,
                self.imageList)
        fourierVisitor = afwMath.FourierConvolutionVisitor(imageVisitor)
        fourierVisitor.fft(self.width, self.height)

        cutout = fourierVisitor.getFourierImage()
        cutout.shift(self.center[0], self.center[1])
        cutout.differentiateX()
        cutout.differentiateY()


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ConvolutionVisitorTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

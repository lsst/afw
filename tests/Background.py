#!/usr/bin/env python
"""
Tests for Background

Run with:
   ./Background.py
or
   python
   >>> import Background; Background.run()
"""

import math
import os
import pdb  # we may want to say pdb.set_trace()
import sys
import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class BackgroundTestCase(unittest.TestCase):
    
    """A test case for Background"""
    def setUp(self):
        self.val = 10
        self.image = afwImage.ImageF(100, 200); self.image.set(self.val)

    def tearDown(self):
        del self.image


        #self.assertAlmostEqual(mean[1], sd/math.sqrt(image2.getWidth()*image2.getHeight()), 10)

    def testgetPixel(self):
	"""Test the getPixel() function"""


	xcen, ycen = 50, 100
	bgCtrl = afwMath.BackgroundControl(afwMath.NATURAL_SPLINE)
	bgCtrl.setNxSample(3)
	bgCtrl.setNySample(3)
	bgCtrl.sctrl.setNumIter(3)
	bgCtrl.sctrl.setNumSigmaClip(3)
	back = afwMath.BackgroundF(self.image, bgCtrl)
	mid = back.getPixel(xcen,ycen)
	
        self.assertEqual(back.getPixel(xcen,ycen), self.val)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(BackgroundTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

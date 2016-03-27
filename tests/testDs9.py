#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

"""
Tests for the legacy display code in ds9.py (iff lsst.display.ds9 is setup)

Run with:
   ds9.py
or
   python
   >>> import ds9
   >>> ds9.run()
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import unittest

import lsst.utils.tests as tests
import lsst.afw.image as afwImage
try:
    import  lsst.afw.display.ds9 as ds9
except Exception:
    ds9 = None

if ds9:
    try:
        ds9.mtv(afwImage.ImageF(1,1))
    except Exception as e:
        print("Unable to use ds9: %s" % e)
        ds9 = None

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class DisplayTestCase(unittest.TestCase):
    """A test case for Display"""
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
    def testMtv(self):
        """Test basic image display"""
        exp = afwImage.ImageF(10, 20)
        ds9.mtv(exp, title="parent")

    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
    def testMaskPlanes(self):
        """Test basic image display"""
        ds9.setMaskTransparency(50)
        ds9.setMaskPlaneColor("CROSSTALK", "orange")

    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
    def testTwoDisplays(self):
        """Test that we can do things with two frames"""

        exp = afwImage.ExposureF(300, 350)
        
        for frame in (0, 1):
            ds9.setMaskTransparency(50, frame=frame)

            if frame == 1:
                ds9.setMaskPlaneColor("CROSSTALK", "ignore", frame=frame)
            ds9.mtv(exp, title="parent", frame=frame)

            ds9.erase(frame=frame)
            ds9.dot('o', 205, 180, size=6, ctype=ds9.RED, frame=frame)

    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
    def testZoomPan(self):
        ds9.pan(205, 180)
        ds9.zoom(4)

        ds9.zoom(4, 205, 180, frame=1)

    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
    def testStackingOrder(self):
        """ Un-iconise and raise the display to the top of the stacking order if appropriate"""
        ds9.show()

    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
    def testDrawing(self):
        """Test drawing lines and glyphs"""
        ds9.erase()

        exp = afwImage.ExposureF(300, 350)
        ds9.mtv(exp, title="parent") # tells display0 about the image's xy0
        
        with ds9.Buffering():
            ds9.dot('o', 200, 220)
            vertices = [(200, 220), (210, 230), (224, 230), (214, 220), (200, 220)]
            ds9.line(vertices, ctype=ds9.CYAN)
            ds9.line(vertices[:-1], symbs="+x+x", size=3)

    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
    def testText(self):
        """Test drawing text"""
        ds9.erase()

        exp = afwImage.ExposureF(300, 350)
        ds9.mtv(exp, title="parent") # tells display0 about the image's xy0

        with ds9.Buffering():
            ds9.dot('hello', 200, 200)
            ds9.dot('hello', 200, 210, size=1.25)
            ds9.dot('hello', 200, 220, size=3, fontFamily="times")
            ds9.dot('hello', 200, 230, fontFamily="helvetica bold italic")

    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
    def testStretch(self):
        """Test playing with the lookup table"""
        ds9.show()

        ds9.scale("linear", "zscale")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(DisplayTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

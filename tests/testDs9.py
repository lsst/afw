#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2015 LSST Corporation.
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
#pybind11#"""
#pybind11#Tests for the legacy display code in ds9.py (iff lsst.display.ds9 is setup)
#pybind11#
#pybind11#Run with:
#pybind11#   ds9.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import ds9
#pybind11#   >>> ds9.run()
#pybind11#"""
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.image as afwImage
#pybind11#try:
#pybind11#    import lsst.afw.display.ds9 as ds9
#pybind11#except Exception:
#pybind11#    ds9 = None
#pybind11#
#pybind11#if ds9:
#pybind11#    try:
#pybind11#        ds9.mtv(afwImage.ImageF(1, 1))
#pybind11#    except Exception as e:
#pybind11#        print("Unable to use ds9: %s" % e)
#pybind11#        ds9 = None
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class DisplayTestCase(unittest.TestCase):
#pybind11#    """A test case for Display"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        pass
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        pass
#pybind11#
#pybind11#    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
#pybind11#    def testMtv(self):
#pybind11#        """Test basic image display"""
#pybind11#        exp = afwImage.ImageF(10, 20)
#pybind11#        ds9.mtv(exp, title="parent")
#pybind11#
#pybind11#    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
#pybind11#    def testMaskPlanes(self):
#pybind11#        """Test basic image display"""
#pybind11#        ds9.setMaskTransparency(50)
#pybind11#        ds9.setMaskPlaneColor("CROSSTALK", "orange")
#pybind11#
#pybind11#    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
#pybind11#    def testTwoDisplays(self):
#pybind11#        """Test that we can do things with two frames"""
#pybind11#
#pybind11#        exp = afwImage.ExposureF(300, 350)
#pybind11#
#pybind11#        for frame in (0, 1):
#pybind11#            ds9.setMaskTransparency(50, frame=frame)
#pybind11#
#pybind11#            if frame == 1:
#pybind11#                ds9.setMaskPlaneColor("CROSSTALK", "ignore", frame=frame)
#pybind11#            ds9.mtv(exp, title="parent", frame=frame)
#pybind11#
#pybind11#            ds9.erase(frame=frame)
#pybind11#            ds9.dot('o', 205, 180, size=6, ctype=ds9.RED, frame=frame)
#pybind11#
#pybind11#    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
#pybind11#    def testZoomPan(self):
#pybind11#        ds9.pan(205, 180)
#pybind11#        ds9.zoom(4)
#pybind11#
#pybind11#        ds9.zoom(4, 205, 180, frame=1)
#pybind11#
#pybind11#    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
#pybind11#    def testStackingOrder(self):
#pybind11#        """ Un-iconise and raise the display to the top of the stacking order if appropriate"""
#pybind11#        ds9.show()
#pybind11#
#pybind11#    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
#pybind11#    def testDrawing(self):
#pybind11#        """Test drawing lines and glyphs"""
#pybind11#        ds9.erase()
#pybind11#
#pybind11#        exp = afwImage.ExposureF(300, 350)
#pybind11#        ds9.mtv(exp, title="parent")  # tells display0 about the image's xy0
#pybind11#
#pybind11#        with ds9.Buffering():
#pybind11#            ds9.dot('o', 200, 220)
#pybind11#            vertices = [(200, 220), (210, 230), (224, 230), (214, 220), (200, 220)]
#pybind11#            ds9.line(vertices, ctype=ds9.CYAN)
#pybind11#            ds9.line(vertices[:-1], symbs="+x+x", size=3)
#pybind11#
#pybind11#    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
#pybind11#    def testText(self):
#pybind11#        """Test drawing text"""
#pybind11#        ds9.erase()
#pybind11#
#pybind11#        exp = afwImage.ExposureF(300, 350)
#pybind11#        ds9.mtv(exp, title="parent")  # tells display0 about the image's xy0
#pybind11#
#pybind11#        with ds9.Buffering():
#pybind11#            ds9.dot('hello', 200, 200)
#pybind11#            ds9.dot('hello', 200, 210, size=1.25)
#pybind11#            ds9.dot('hello', 200, 220, size=3, fontFamily="times")
#pybind11#            ds9.dot('hello', 200, 230, fontFamily="helvetica bold italic")
#pybind11#
#pybind11#    @unittest.skipUnless(ds9, "You must setup display.ds9 to run this test")
#pybind11#    def testStretch(self):
#pybind11#        """Test playing with the lookup table"""
#pybind11#        ds9.show()
#pybind11#
#pybind11#        ds9.scale("linear", "zscale")
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

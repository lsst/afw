#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import next
#pybind11#from builtins import range
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
#pybind11#Tests for displaying devices
#pybind11#
#pybind11#Run with:
#pybind11#   display.py [backend]
#pybind11#or
#pybind11#   python
#pybind11#   >>> import display
#pybind11#   >>> display.backend = "ds9"   # optional
#pybind11#   >>> display.run()
#pybind11#"""
#pybind11#import os
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.display as afwDisplay
#pybind11#
#pybind11#try:
#pybind11#    type(backend)
#pybind11#except NameError:
#pybind11#    backend = "virtualDevice"
#pybind11#    oldBackend = None
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class DisplayTestCase(unittest.TestCase):
#pybind11#    """A test case for Display"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        global oldBackend
#pybind11#        if backend != oldBackend:
#pybind11#            afwDisplay.setDefaultBackend(backend)
#pybind11#            afwDisplay.delAllDisplays()  # as some may use the old backend
#pybind11#
#pybind11#            oldBackend = backend
#pybind11#
#pybind11#        dirName = os.path.split(__file__)[0]
#pybind11#        self.fileName = os.path.join(dirName, "data", "HSC-0908120-056-small.fits")
#pybind11#        self.display0 = afwDisplay.getDisplay(frame=0, verbose=True)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        for d in self.display0._displays.values():
#pybind11#            d.verbose = False           # ensure that display9.close() call is quiet
#pybind11#
#pybind11#        del self.display0
#pybind11#        afwDisplay.delAllDisplays()
#pybind11#
#pybind11#    def testClose(self):
#pybind11#        """Test that we can close devices."""
#pybind11#        self.display0.close()
#pybind11#
#pybind11#    def testMtv(self):
#pybind11#        """Test basic image display"""
#pybind11#        exp = afwImage.ExposureF(self.fileName)
#pybind11#        self.display0.mtv(exp, title="parent")
#pybind11#
#pybind11#    def testMaskPlanes(self):
#pybind11#        """Test basic image display"""
#pybind11#        self.display0.setMaskTransparency(50)
#pybind11#        self.display0.setMaskPlaneColor("CROSSTALK", "orange")
#pybind11#
#pybind11#    def testWith(self):
#pybind11#        """Test using displays with with statement"""
#pybind11#        with afwDisplay.getDisplay(0) as disp:
#pybind11#            self.assertIsNotNone(disp)
#pybind11#
#pybind11#    def testTwoDisplays(self):
#pybind11#        """Test that we can do things with two frames"""
#pybind11#
#pybind11#        exp = afwImage.ExposureF(self.fileName)
#pybind11#
#pybind11#        for frame in (0, 1):
#pybind11#            with afwDisplay.Display(frame, verbose=False) as disp:
#pybind11#                disp.setMaskTransparency(50)
#pybind11#
#pybind11#                if frame == 1:
#pybind11#                    disp.setMaskPlaneColor("CROSSTALK", "ignore")
#pybind11#                disp.mtv(exp, title="parent")
#pybind11#
#pybind11#                disp.erase()
#pybind11#                disp.dot('o', 205, 180, size=6, ctype=afwDisplay.RED)
#pybind11#
#pybind11#    def testZoomPan(self):
#pybind11#        self.display0.pan(205, 180)
#pybind11#        self.display0.zoom(4)
#pybind11#
#pybind11#        afwDisplay.getDisplay(1).zoom(4, 205, 180)
#pybind11#
#pybind11#    def testStackingOrder(self):
#pybind11#        """ Un-iconise and raise the display to the top of the stacking order if appropriate"""
#pybind11#        self.display0.show()
#pybind11#
#pybind11#    def testDrawing(self):
#pybind11#        """Test drawing lines and glyphs"""
#pybind11#        self.display0.erase()
#pybind11#
#pybind11#        exp = afwImage.ExposureF(self.fileName)
#pybind11#        self.display0.mtv(exp, title="parent")  # tells display0 about the image's xy0
#pybind11#
#pybind11#        with self.display0.Buffering():
#pybind11#            self.display0.dot('o', 200, 220)
#pybind11#            vertices = [(200, 220), (210, 230), (224, 230), (214, 220), (200, 220)]
#pybind11#            self.display0.line(vertices, ctype=afwDisplay.CYAN)
#pybind11#            self.display0.line(vertices[:-1], symbs="+x+x", size=3)
#pybind11#
#pybind11#    def testStretch(self):
#pybind11#        """Test playing with the lookup table"""
#pybind11#        self.display0.show()
#pybind11#
#pybind11#        self.display0.scale("linear", "zscale")
#pybind11#
#pybind11#    def testMaskColorGeneration(self):
#pybind11#        """Demonstrate the utility routine to generate mask plane colours
#pybind11#        (used by e.g. the ds9 implementation of _mtv)"""
#pybind11#
#pybind11#        colorGenerator = self.display0.maskColorGenerator(omitBW=True)
#pybind11#        for i in range(10):
#pybind11#            print(i, next(colorGenerator), end=' ')
#pybind11#        print()
#pybind11#
#pybind11#    def testImageTypes(self):
#pybind11#        """Check that we can display a range of types of image"""
#pybind11#        with afwDisplay.getDisplay("dummy", "virtualDevice") as dummy:
#pybind11#            for imageType in [afwImage.DecoratedImageF,
#pybind11#                              afwImage.ExposureF,
#pybind11#                              afwImage.ImageU,
#pybind11#                              afwImage.ImageI,
#pybind11#                              afwImage.ImageF,
#pybind11#                              afwImage.MaskedImageF,
#pybind11#                              ]:
#pybind11#                im = imageType(self.fileName)
#pybind11#                dummy.mtv(im)
#pybind11#
#pybind11#            im = afwImage.MaskU(self.fileName, 3)
#pybind11#            dummy.mtv(im)
#pybind11#
#pybind11#    def testInteract(self):
#pybind11#        """Check that interact exits when a q, \c CR, or \c ESC is pressed, or if a callback function
#pybind11#        returns a ``True`` value.
#pybind11#        If this is run using the virtualDevice a "q" is automatically triggered.
#pybind11#        If running the tests using ds9 you will be expected to do this manually.
#pybind11#        """
#pybind11#        self.display0.interact()
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    import argparse
#pybind11#    import sys
#pybind11#
#pybind11#    parser = argparse.ArgumentParser(description="Run the image display test suite")
#pybind11#
#pybind11#    parser.add_argument('backend', type=str, nargs="?", default="virtualDevice",
#pybind11#                        help="The backend to use, e.g. ds9.  You may need to have the device setup")
#pybind11#    args = parser.parse_args()
#pybind11#
#pybind11#    # check that that backend is valid
#pybind11#    with afwDisplay.Display("test", backend=args.backend) as disp:
#pybind11#        pass
#pybind11#
#pybind11#    backend = args.backend              # backend is just a variable in this file
#pybind11#    lsst.utils.tests.init()
#pybind11#    del sys.argv[1:]
#pybind11#    unittest.main()

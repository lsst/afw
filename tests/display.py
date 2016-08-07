#!/usr/bin/env python2
from __future__ import absolute_import, division
from __future__ import print_function

#
# LSST Data Management System
# Copyright 2015 LSST Corporation.
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

"""
Tests for displaying devices

Run with:
   display.py [backend]
or
   python
   >>> import display
   >>> display.backend = "ds9"   # optional
   >>> display.run()
"""
import os
import unittest

import lsst.utils.tests as tests
import lsst.afw.image as afwImage
import lsst.afw.display as afwDisplay

try:
    type(backend)
except NameError:
    backend = "virtualDevice"
    oldBackend = None

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class DisplayTestCase(unittest.TestCase):
    """A test case for Display"""
    def setUp(self):
        global oldBackend
        if backend != oldBackend:
            afwDisplay.setDefaultBackend(backend)
            afwDisplay.delAllDisplays() # as some may use the old backend

            oldBackend = backend

        dirName = os.path.split(__file__)[0]
        self.fileName = os.path.join(dirName, "data", "HSC-0908120-056-small.fits")
        self.display0 = afwDisplay.getDisplay(frame=0, verbose=True)

    def tearDown(self):
        for d in self.display0._displays.values():
            d.verbose = False           # ensure that display9.close() call is quiet

        del self.display0
        afwDisplay.delAllDisplays()

    def testClose(self):
        """Test that we can close devices."""
        self.display0.close()

    def testMtv(self):
        """Test basic image display"""
        exp = afwImage.ExposureF(self.fileName)
        self.display0.mtv(exp, title="parent")

    def testMaskPlanes(self):
        """Test basic image display"""
        self.display0.setMaskTransparency(50)
        self.display0.setMaskPlaneColor("CROSSTALK", "orange")

    def testWith(self):
        """Test using displays with with statement"""
        with afwDisplay.getDisplay(0) as disp:
            self.assertTrue(disp is not None)

    def testTwoDisplays(self):
        """Test that we can do things with two frames"""

        exp = afwImage.ExposureF(self.fileName)

        for frame in (0, 1):
            with afwDisplay.Display(frame, verbose=False) as disp:
                disp.setMaskTransparency(50)

                if frame == 1:
                    disp.setMaskPlaneColor("CROSSTALK", "ignore")
                disp.mtv(exp, title="parent")

                disp.erase()
                disp.dot('o', 205, 180, size=6, ctype=afwDisplay.RED)

    def testZoomPan(self):
        self.display0.pan(205, 180)
        self.display0.zoom(4)

        afwDisplay.getDisplay(1).zoom(4, 205, 180)

    def testStackingOrder(self):
        """ Un-iconise and raise the display to the top of the stacking order if appropriate"""
        self.display0.show()

    def testDrawing(self):
        """Test drawing lines and glyphs"""
        self.display0.erase()

        exp = afwImage.ExposureF(self.fileName)
        self.display0.mtv(exp, title="parent") # tells display0 about the image's xy0

        with self.display0.Buffering():
            self.display0.dot('o', 200, 220)
            vertices = [(200, 220), (210, 230), (224, 230), (214, 220), (200, 220)]
            self.display0.line(vertices, ctype=afwDisplay.CYAN)
            self.display0.line(vertices[:-1], symbs="+x+x", size=3)

    def testStretch(self):
        """Test playing with the lookup table"""
        self.display0.show()

        self.display0.scale("linear", "zscale")

    def testMaskColorGeneration(self):
        """Demonstrate the utility routine to generate mask plane colours
        (used by e.g. the ds9 implementation of _mtv)"""

        colorGenerator = self.display0.maskColorGenerator(omitBW=True)
        for i in range(10):
            print(i, next(colorGenerator), end=' ')
        print()

    def testImageTypes(self):
        """Check that we can display a range of types of image"""
        with afwDisplay.getDisplay("dummy", "virtualDevice") as dummy:
            for imageType in [afwImage.DecoratedImageF,
                              afwImage.ExposureF,
                              afwImage.ImageU,
                              afwImage.ImageI,
                              afwImage.ImageF,
                              afwImage.MaskedImageF,
                              ]:
                im = imageType(self.fileName)
                dummy.mtv(im)

            im = afwImage.MaskU(self.fileName, 3)
            dummy.mtv(im)
    
    def testInteract(self):
        """Check that interact exits when a q, \c CR, or \c ESC is pressed, or if a callback function
        returns a ``True`` value.
        If this is run using the virtualDevice a "q" is automatically triggered.
        If running the tests using ds9 you will be expected to do this manually.
        """
        self.display0.interact()

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
    import argparse
    parser = argparse.ArgumentParser(description="Run the image display test suite")

    parser.add_argument('backend', type=str, nargs="?", default="virtualDevice",
                        help="The backend to use, e.g. ds9.  You may need to have the device setup")
    args = parser.parse_args()

    # check that that backend is valid
    with afwDisplay.Display("test", backend=args.backend) as disp:
        pass

    backend = args.backend              # backend is just a variable in this file
    run(True)

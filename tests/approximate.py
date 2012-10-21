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

"""
Tests for Approximate

Run with:
   ./Approximate.py
or
   python
   >>> import Approximate; Approximate.run()
"""
import unittest
import numpy as np
import lsst.utils.tests as utilsTests
import lsst.afw.display.ds9 as ds9
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.pex.exceptions as pexExcept

try:
    display
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ApproximateTestCase(unittest.TestCase):
    
    """A test case for Approximate Lienar"""
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def testLinearRamp(self):
        """Make a ramp and fit it"""
        #
        # Here's the image to fit
        #
        im = afwImage.MaskedImageF(20, 40)
        binsize = 2

        x = []
        for i in range(im.getWidth()):
            x.append((i + 0.5)*binsize)

        y = []
        for j in range(im.getHeight()):
            y.append((j + 0.5)*binsize)

        var = 1
        c = (1000, 1, 1)
        for i in range(im.getHeight()):
            for j in range(im.getWidth()):
                im.set(j, i, (c[0] + c[1]*x[j] + c[2]*y[i], 0x0, var))

        im.set(10, 20, (0, 0x1, np.nan))
        
        if display:
            ds9.mtv(im, title="Input", frame=0)
        #
        # Here's the bounding box of the 
        #
        bbox = afwGeom.BoxI(afwGeom.PointI(0, 0), afwGeom.PointI(binsize*im.getWidth() - 1,
                                                                 binsize*im.getHeight() - 1))

        approx = afwMath.makeApproximate(x, y, im, bbox,
                                         afwMath.ApproximateControl(afwMath.ApproximateControl.CHEBYSHEV, 1))
        aim = approx.getImage()
        w, h = aim.getDimensions()
        if display:
            ds9.mtv(aim, title="interpolated", frame=1)

        for x, y in [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1),]:
            self.assertEqual(aim.getImage().get(x, y), c[0] + c[1]*x + c[1]*y)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ApproximateTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

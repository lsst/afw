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
Tests for LocalKernel

Run with:
   ./LocalKernel.py
or
   python
   >>> import LocalKernel; LocalKernel.run()
"""

import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom

class LocalKernelTestCase(unittest.TestCase):
    """A test case for LocalKernel"""
    def setUp(self):
        self.width = 19
        self.height = 19
        self.fourierWidth = self.width/2 + 1
        self.center = afwGeom.makePointI(9, 9)
        self.image = afwImage.ImageD(self.width, self.height, 3)
        self.imageList = []
        self.imageList.append(afwImage.ImageD(self.width, self.height, 1))
        self.imageList.append(afwImage.ImageD(self.width, self.height, 2))

        self.paramList = [0.5, 1.5]

    def tearDown(self):
        del self.image
        del self.imageList

    def testBasic(self):
        imageKernel = afwMath.ImageLocalKernel(
                self.center,
                self.paramList,
                self.image,
                self.imageList)
        fourierKernel = afwMath.FftLocalKernel(imageKernel)
        fourierKernel.setDimensions(self.width, self.height)

        cutout = fourierKernel.getFourierImage()
        cutout.shift(self.center.getX(), self.center.getY())
        cutout.differentiateX()
        cutout.differentiateY()


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(LocalKernelTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

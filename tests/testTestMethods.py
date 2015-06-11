#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

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
import unittest

import lsst.utils.tests as utilsTests
import lsst.daf.base as dafBase
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage

class TestTestUtils(utilsTests.TestCase):
    """Test test methods added to lsst.utils.tests.TestCase
    """
    def setUp(self):
        self.bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(3001, 3001))
        ctrPix = afwGeom.Point2I(1500, 1500)
        metadata = dafBase.PropertySet()
        metadata.set("RADECSYS", "FK5")
        metadata.set("EQUINOX", 2000.0)
        metadata.set("CTYPE1", "RA---TAN")
        metadata.set("CTYPE2", "DEC--TAN")
        metadata.set("CUNIT1", "deg")
        metadata.set("CUNIT2", "deg")
        metadata.set("CRVAL1", 215.5)
        metadata.set("CRVAL2",  53.0)
        metadata.set("CRPIX1", ctrPix[0] + 1)
        metadata.set("CRPIX2", ctrPix[1] + 1)
        metadata.set("CD1_1",  5.1e-05)
        metadata.set("CD1_2",  0.0)
        metadata.set("CD2_2", -5.1e-05)
        metadata.set("CD2_1",  0.0)
        self.wcs0 = afwImage.cast_TanWcs(afwImage.makeWcs(metadata))
        metadata.set("CRVAL2",  53.000001) # tweak CRVAL2 for wcs1
        self.wcs1 = afwImage.cast_TanWcs(afwImage.makeWcs(metadata))

    def tearDown(self):
        del self.wcs0
        del self.wcs1

    def testAssertWcssNearlyEqualOverBBox(self):
        """Test assertWcssNearlyEqualOverBBox"""
        self.assertWcssNearlyEqualOverBBox(self.wcs0, self.wcs0, self.bbox,
            maxDiffSky=1e-7*afwGeom.arcseconds, maxDiffPix=1e-7)

        self.assertWcssNearlyEqualOverBBox(self.wcs0, self.wcs1, self.bbox,
            maxDiffSky=0.04*afwGeom.arcseconds, maxDiffPix=0.02)

        try:
            self.assertWcssNearlyEqualOverBBox(self.wcs0, self.wcs1, self.bbox,
                maxDiffSky=0.001*afwGeom.arcseconds, maxDiffPix=0.02)
        except Exception:
            pass
        else:
            self.fail("assertion not raised")

        try:
            self.assertWcssNearlyEqualOverBBox(self.wcs0, self.wcs1, self.bbox,
                maxDiffSky=0.04*afwGeom.arcseconds, maxDiffPix=0.001)
        except Exception:
            pass
        else:
            self.fail("assertion not raised")


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(TestTestUtils)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

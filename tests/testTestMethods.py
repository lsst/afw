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
import math
import unittest

import lsst.utils.tests as utilsTests
import lsst.daf.base as dafBase
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage

class TestTestUtils(utilsTests.TestCase):
    """Test test methods added to lsst.utils.tests.TestCase
    """
    def testAssertAnglesNearlyEqual(self):
        """Test assertAnglesNearlyEqual"""
        for angDeg in (0, 45, -75):
            ang0 = angDeg*afwGeom.degrees
            self.assertAnglesNearlyEqual(
                ang0,
                ang0 + 0.01*afwGeom.arcseconds,
                maxDiff = 0.010001*afwGeom.arcseconds,
            )
            self.assertRaises(AssertionError,
                self.assertAnglesNearlyEqual,
                    ang0,
                    ang0 + 0.01*afwGeom.arcseconds,
                    maxDiff = 0.009999*afwGeom.arcseconds,
            )

            self.assertAnglesNearlyEqual(
                ang0,
                ang0 - 0.01*afwGeom.arcseconds,
                maxDiff = 0.010001*afwGeom.arcseconds,
            )
            self.assertRaises(AssertionError,
                self.assertAnglesNearlyEqual,
                    ang0,
                    ang0 - 0.01*afwGeom.arcseconds,
                    maxDiff = 0.009999*afwGeom.arcseconds,
            )

            self.assertAnglesNearlyEqual(
                ang0 - 720*afwGeom.degrees,
                ang0 + 0.01*afwGeom.arcseconds,
                maxDiff = 0.010001*afwGeom.arcseconds,
            )
            self.assertRaises(AssertionError,
                self.assertAnglesNearlyEqual,
                ang0 - 720*afwGeom.degrees,
                ang0 + 0.01*afwGeom.arcseconds,
                ignoreWrap = False,
                maxDiff = 0.010001*afwGeom.arcseconds,
            )
            self.assertRaises(AssertionError,
                self.assertAnglesNearlyEqual,
                    ang0 - 720*afwGeom.degrees,
                    ang0 + 0.01*afwGeom.arcseconds,
                    maxDiff = 0.009999*afwGeom.arcseconds,
            )

            self.assertAnglesNearlyEqual(
                ang0,
                ang0 + 360*afwGeom.degrees + 0.01*afwGeom.arcseconds,
                maxDiff = 0.010001*afwGeom.arcseconds,
            )
            self.assertRaises(AssertionError,
                self.assertAnglesNearlyEqual,
                    ang0,
                    ang0 + 360*afwGeom.degrees + 0.01*afwGeom.arcseconds,
                    ignoreWrap = False,
                    maxDiff = 0.010001*afwGeom.arcseconds,
            )
            self.assertRaises(AssertionError,
                self.assertAnglesNearlyEqual,
                    ang0,
                    ang0 + 360*afwGeom.degrees + 0.01*afwGeom.arcseconds,
                    maxDiff = 0.009999*afwGeom.arcseconds,
            )

    def testAssertBoxesNearlyEqual(self):
        """Test assertBoxesNearlyEqual"""
        for min0 in ((0, 0), (-1000.5, 5000.1)):
            min0 = afwGeom.Point2D(*min0)
            for extent0 in ((2.01, 3.01), (5432, 2342)):
                extent0 = afwGeom.Extent2D(*extent0)
                box0 = afwGeom.Box2D(min0, extent0)
                self.assertBoxesNearlyEqual(box0, box0, maxDiff=1e-7)
                for deltaExtent in ((0.001, -0.001), (2, -3)):
                    deltaExtent = afwGeom.Extent2D(*deltaExtent)
                    box1 = afwGeom.Box2D(box0.getMin() + deltaExtent, box0.getMax())
                    radDiff = math.hypot(*deltaExtent)
                    self.assertBoxesNearlyEqual(box0, box1, maxDiff=radDiff*1.00001)
                    self.assertRaises(AssertionError, self.assertBoxesNearlyEqual,
                        box0, box1, maxDiff=radDiff*0.99999)

                    box2 = afwGeom.Box2D(box0.getMin() - deltaExtent, box0.getMax())
                    self.assertBoxesNearlyEqual(box0, box2, maxDiff=radDiff*1.00001)
                    self.assertRaises(AssertionError, self.assertBoxesNearlyEqual,
                        box0, box2, maxDiff=radDiff*0.999999)

                    box3 = afwGeom.Box2D(box0.getMin(), box0.getMax() + deltaExtent)
                    self.assertBoxesNearlyEqual(box0, box3, maxDiff=radDiff*1.00001)
                    self.assertRaises(AssertionError, self.assertBoxesNearlyEqual,
                        box0, box3, maxDiff=radDiff*0.999999)

    def testAssertCoordsNearlyEqual(self):
        """Test assertCoordsNearlyEqual"""
        for raDecDeg in ((45, 45), (-70, 89), (130, -89.5)):
            raDecDeg = [val*afwGeom.degrees for val in raDecDeg]
            coord0 = afwCoord.IcrsCoord(*raDecDeg)
            self.assertCoordsNearlyEqual(coord0, coord0, maxDiff=1e-7*afwGeom.arcseconds)

            for offAng in (0, 45, 90):
                offAng = offAng*afwGeom.degrees
                for offDist in (0.001, 0.1):
                    offDist = offDist*afwGeom.arcseconds
                    coord1 = coord0.toGalactic()
                    coord1.offset(offAng, offDist)
                    self.assertCoordsNearlyEqual(coord0, coord1, maxDiff=offDist*1.00001)
                    self.assertRaises(AssertionError,
                        self.assertCoordsNearlyEqual, coord0, coord1, maxDiff=offDist*0.99999)

            # test wraparound in RA
            coord2 = afwCoord.IcrsCoord(raDecDeg[0] + 360*afwGeom.degrees, raDecDeg[1])
            self.assertCoordsNearlyEqual(coord0, coord2, maxDiff=1e-7*afwGeom.arcseconds)

    def testAssertPairsNearlyEqual(self):
        """Test assertPairsNearlyEqual"""
        for pair0 in ((-5, 4), (-5, 0.001), (0, 0), (49, 0.1)):
            self.assertPairsNearlyEqual(pair0, pair0, maxDiff=1e-7)
            self.assertPairsNearlyEqual(afwGeom.Point2D(*pair0), afwGeom.Extent2D(*pair0), maxDiff=1e-7)
            for diff in ((0.001, 0), (-0.01, 0.03)):
                pair1 = [pair0[i] + diff[i] for i in range(2)]
                radialDiff = math.hypot(*diff)
                self.assertPairsNearlyEqual(pair0, pair1, maxDiff=radialDiff+1e-7)
                self.assertRaises(AssertionError,
                    self.assertPairsNearlyEqual,
                        pair0, pair1, maxDiff=radialDiff-1e-7)

    def testAssertWcssNearlyEqualOverBBox(self):
        """Test assertWcsNearlyEqualOverBBox"""
        bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(3001, 3001))
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
        wcs0 = afwImage.cast_TanWcs(afwImage.makeWcs(metadata))
        metadata.set("CRVAL2",  53.000001) # tweak CRVAL2 for wcs1
        wcs1 = afwImage.cast_TanWcs(afwImage.makeWcs(metadata))

        self.assertWcsNearlyEqualOverBBox(wcs0, wcs0, bbox,
            maxDiffSky=1e-7*afwGeom.arcseconds, maxDiffPix=1e-7)

        self.assertWcsNearlyEqualOverBBox(wcs0, wcs1, bbox,
            maxDiffSky=0.04*afwGeom.arcseconds, maxDiffPix=0.02)

        self.assertRaises(AssertionError, self.assertWcsNearlyEqualOverBBox,
            wcs0, wcs1, bbox, maxDiffSky=0.001*afwGeom.arcseconds, maxDiffPix=0.02)

        self.assertRaises(AssertionError, self.assertWcsNearlyEqualOverBBox,
            wcs0, wcs1, bbox, maxDiffSky=0.04*afwGeom.arcseconds, maxDiffPix=0.001)

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

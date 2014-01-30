#!/usr/bin/env python
# 
# LSST Data Management System
# Copyright 2014 LSST Corporation.
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
Tests for lsst.afw.cameraGeom.RawAmplifier
"""
import unittest

import lsst.utils.tests
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom

class RawAmplifierTestCase(unittest.TestCase):
    def testConstructor(self):
        """Test constructor
        """
        bbox = afwGeom.Box2I(afwGeom.Point2I(-25, 2), afwGeom.Extent2I(550, 629))
        dataBBox = afwGeom.Box2I(afwGeom.Point2I(-2, 29), afwGeom.Extent2I(123, 307))
        horizontalOverscanBBox = afwGeom.Box2I(afwGeom.Point2I(150, 29), afwGeom.Extent2I(25, 307))
        verticalOverscanBBox = afwGeom.Box2I(afwGeom.Point2I(-2, 201), afwGeom.Extent2I(123, 6))
        prescanBBox = afwGeom.Box2I(afwGeom.Point2I(-20, 2), afwGeom.Extent2I(5, 307))
        flipX = True
        flipY = False
        xyOffset = afwGeom.Extent2I(-97, 253)

        rawAmplifier = cameraGeom.RawAmplifier(
            bbox,
            dataBBox,
            horizontalOverscanBBox,
            verticalOverscanBBox,
            prescanBBox,
            flipX,
            flipY,
            xyOffset,
        )
        self.assertEquals(bbox, rawAmplifier.getBBox())
        self.assertEquals(dataBBox, rawAmplifier.getDataBBox())
        self.assertEquals(horizontalOverscanBBox, rawAmplifier.getHorizontalOverscanBBox())
        self.assertEquals(verticalOverscanBBox, rawAmplifier.getVerticalOverscanBBox())
        self.assertEquals(prescanBBox, rawAmplifier.getPrescanBBox())
        self.assertEquals(flipX, rawAmplifier.getFlipX())
        self.assertEquals(flipY, rawAmplifier.getFlipY())
        self.assertEquals(xyOffset, rawAmplifier.getXYOffset())


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(RawAmplifierTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

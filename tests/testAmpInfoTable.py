#!/usr/bin/env python2
from __future__ import absolute_import, division
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
Tests for lsst.afw.table.AmpInfoTable, etc.
"""
import unittest

import lsst.utils.tests
from lsst.pex.exceptions import LsstCppException
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable

class AmpInfoTableTestCase(unittest.TestCase):
    def setUp(self):
        self.schema = afwTable.AmpInfoTable.makeMinimalSchema()
        self.catalog = afwTable.AmpInfoCatalog(self.schema)

    def tearDown(self):
        self.catalog = None
        self.schema = None

    def testBasics(self):
        """Test basics
        """
        name = "Amp1"
        gain = 1.2345
        readNoise = -0.523
        linearityCoeffs = (1.1, 2.2, 3.3, 4.4)
        linearityType = "Polynomial"
        bbox = afwGeom.Box2I(afwGeom.Point2I(3, -2), afwGeom.Extent2I(231, 320))
        hasRawInfo = True
        rawFlipX = True
        rawFlipY = False
        rawBBox = afwGeom.Box2I(afwGeom.Point2I(-25, 2), afwGeom.Extent2I(550, 629))
        rawXYOffset = afwGeom.Extent2I(-97, 253)
        rawDataBBox = afwGeom.Box2I(afwGeom.Point2I(-2, 29), afwGeom.Extent2I(123, 307))
        rawHorizontalOverscanBBox = afwGeom.Box2I(afwGeom.Point2I(150, 29), afwGeom.Extent2I(25, 307))
        rawVerticalOverscanBBox = afwGeom.Box2I(afwGeom.Point2I(-2, 201), afwGeom.Extent2I(123, 6))
        rawPrescanBBox = afwGeom.Box2I(afwGeom.Point2I(-20, 2), afwGeom.Extent2I(5, 307))

        record = self.catalog.addNew()
        record.setBBox(bbox)
        record.setName(name)
        record.setGain(gain)
        record.setReadNoise(readNoise)
        record.setLinearityCoeffs(linearityCoeffs)
        record.setLinearityType(linearityType)
        record.setHasRawInfo(hasRawInfo)
        record.setRawFlipX(rawFlipX)
        record.setRawFlipY(rawFlipY)
        record.setRawBBox(rawBBox)
        record.setRawXYOffset(rawXYOffset)
        record.setRawDataBBox(rawDataBBox)
        record.setRawHorizontalOverscanBBox(rawHorizontalOverscanBBox)
        record.setRawVerticalOverscanBBox(rawVerticalOverscanBBox)
        record.setRawPrescanBBox(rawPrescanBBox)

        self.assertEquals(name, record.getName())
        self.assertEquals(gain, record.getGain())
        self.assertEquals(readNoise, record.getReadNoise())
        self.assertEquals(linearityCoeffs, record.getLinearityCoeffs())
        self.assertEquals(linearityType, record.getLinearityType())
        self.assertEquals(bbox, record.getBBox())
        self.assertEquals(rawBBox, record.getRawBBox())
        self.assertEquals(rawDataBBox, record.getRawDataBBox())
        self.assertEquals(rawHorizontalOverscanBBox, record.getRawHorizontalOverscanBBox())
        self.assertEquals(rawVerticalOverscanBBox, record.getRawVerticalOverscanBBox())
        self.assertEquals(rawPrescanBBox, record.getRawPrescanBBox())
        self.assertEquals(rawFlipX, record.getRawFlipX())
        self.assertEquals(rawFlipY, record.getRawFlipY())
        self.assertEquals(rawXYOffset, record.getRawXYOffset())

        self.assertEquals(len(self.catalog), 1)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(AmpInfoTableTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

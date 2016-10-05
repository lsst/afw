#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2014 LSST Corporation.
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
#pybind11#"""
#pybind11#Tests for lsst.afw.table.AmpInfoTable, etc.
#pybind11#"""
#pybind11#import itertools
#pybind11#import unittest
#pybind11#from builtins import zip
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.table as afwTable
#pybind11#
#pybind11#
#pybind11#class AmpInfoTableTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.schema = afwTable.AmpInfoTable.makeMinimalSchema()
#pybind11#        self.catalog = afwTable.AmpInfoCatalog(self.schema)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        self.catalog = None
#pybind11#        self.schema = None
#pybind11#
#pybind11#    def testEmptyBBox(self):
#pybind11#        record = self.catalog.addNew()
#pybind11#        emptyBox = afwGeom.Box2I()
#pybind11#        record.setBBox(emptyBox)
#pybind11#        record.setRawBBox(emptyBox)
#pybind11#        record.setRawDataBBox(emptyBox)
#pybind11#        record.setRawHorizontalOverscanBBox(emptyBox)
#pybind11#        record.setRawVerticalOverscanBBox(emptyBox)
#pybind11#        record.setRawPrescanBBox(emptyBox)
#pybind11#        self.assertEqual(emptyBox, record.getBBox())
#pybind11#        self.assertTrue(record.getBBox().isEmpty())
#pybind11#        self.assertEqual(emptyBox, record.getRawBBox())
#pybind11#        self.assertTrue(record.getRawBBox().isEmpty())
#pybind11#        self.assertEqual(emptyBox, record.getRawDataBBox())
#pybind11#        self.assertTrue(record.getRawDataBBox().isEmpty())
#pybind11#        self.assertEqual(emptyBox, record.getRawHorizontalOverscanBBox())
#pybind11#        self.assertTrue(record.getRawHorizontalOverscanBBox().isEmpty())
#pybind11#        self.assertEqual(emptyBox, record.getRawVerticalOverscanBBox())
#pybind11#        self.assertTrue(record.getRawVerticalOverscanBBox().isEmpty())
#pybind11#        self.assertEqual(emptyBox, record.getRawPrescanBBox())
#pybind11#        self.assertTrue(record.getRawPrescanBBox().isEmpty())
#pybind11#
#pybind11#    def testBasics(self):
#pybind11#        """Test basics
#pybind11#        """
#pybind11#        nameRawInfoList = (
#pybind11#            ("Amp1", True),
#pybind11#            ("Amp2", False),
#pybind11#        )
#pybind11#
#pybind11#        for name, hasRawInfo in nameRawInfoList:
#pybind11#            gain = 1.2345
#pybind11#            saturation = 65535
#pybind11#            readNoise = -0.523
#pybind11#            linearityCoeffs = (1.1, 2.2, 3.3, 4.4)
#pybind11#            linearityType = "Polynomial"
#pybind11#            bbox = afwGeom.Box2I(afwGeom.Point2I(3, -2), afwGeom.Extent2I(231, 320))
#pybind11#            hasRawInfo = hasRawInfo
#pybind11#            rawFlipX = True
#pybind11#            rawFlipY = False
#pybind11#            readoutCorner = afwTable.UL
#pybind11#            rawBBox = afwGeom.Box2I(afwGeom.Point2I(-25, 2), afwGeom.Extent2I(550, 629))
#pybind11#            rawXYOffset = afwGeom.Extent2I(-97, 253)
#pybind11#            rawDataBBox = afwGeom.Box2I(afwGeom.Point2I(-2, 29), afwGeom.Extent2I(123, 307))
#pybind11#            rawHorizontalOverscanBBox = afwGeom.Box2I(afwGeom.Point2I(150, 29), afwGeom.Extent2I(25, 307))
#pybind11#            rawVerticalOverscanBBox = afwGeom.Box2I(afwGeom.Point2I(-2, 201), afwGeom.Extent2I(123, 6))
#pybind11#            rawPrescanBBox = afwGeom.Box2I(afwGeom.Point2I(-20, 2), afwGeom.Extent2I(5, 307))
#pybind11#
#pybind11#            record = self.catalog.addNew()
#pybind11#            record.setBBox(bbox)
#pybind11#            record.setName(name)
#pybind11#            record.setGain(gain)
#pybind11#            record.setSaturation(saturation)
#pybind11#            record.setReadNoise(readNoise)
#pybind11#            record.setReadoutCorner(readoutCorner)
#pybind11#            record.setLinearityCoeffs(linearityCoeffs)
#pybind11#            record.setLinearityType(linearityType)
#pybind11#            record.setHasRawInfo(hasRawInfo)
#pybind11#            if hasRawInfo:
#pybind11#                record.setRawFlipX(rawFlipX)
#pybind11#                record.setRawFlipY(rawFlipY)
#pybind11#                record.setRawBBox(rawBBox)
#pybind11#                record.setRawXYOffset(rawXYOffset)
#pybind11#                record.setRawDataBBox(rawDataBBox)
#pybind11#                record.setRawHorizontalOverscanBBox(rawHorizontalOverscanBBox)
#pybind11#                record.setRawVerticalOverscanBBox(rawVerticalOverscanBBox)
#pybind11#                record.setRawPrescanBBox(rawPrescanBBox)
#pybind11#
#pybind11#            self.assertEquals(name, record.getName())
#pybind11#            self.assertEquals(gain, record.getGain())
#pybind11#            self.assertEquals(saturation, record.getSaturation())
#pybind11#            self.assertEquals(readNoise, record.getReadNoise())
#pybind11#            self.assertEquals(readoutCorner, record.getReadoutCorner())
#pybind11#            self.assertEquals(linearityCoeffs, record.getLinearityCoeffs())
#pybind11#            self.assertEquals(linearityType, record.getLinearityType())
#pybind11#            self.assertEquals(bbox, record.getBBox())
#pybind11#            self.assertEquals(hasRawInfo, record.getHasRawInfo())
#pybind11#            if hasRawInfo:
#pybind11#                self.assertEquals(rawBBox, record.getRawBBox())
#pybind11#                self.assertEquals(rawDataBBox, record.getRawDataBBox())
#pybind11#                self.assertEquals(rawHorizontalOverscanBBox, record.getRawHorizontalOverscanBBox())
#pybind11#                self.assertEquals(rawVerticalOverscanBBox, record.getRawVerticalOverscanBBox())
#pybind11#                self.assertEquals(rawPrescanBBox, record.getRawPrescanBBox())
#pybind11#                self.assertEquals(rawFlipX, record.getRawFlipX())
#pybind11#                self.assertEquals(rawFlipY, record.getRawFlipY())
#pybind11#                self.assertEquals(rawXYOffset, record.getRawXYOffset())
#pybind11#
#pybind11#        self.assertEquals(len(self.catalog), 2)
#pybind11#        for i, data in enumerate(zip(nameRawInfoList, self.catalog)):
#pybind11#            name, hasRawInfo = data[0]
#pybind11#            record = data[1]
#pybind11#            self.assertEquals(name, self.catalog[i].getName())
#pybind11#            self.assertEquals(name, record.getName())
#pybind11#            self.assertEquals(hasRawInfo, self.catalog[i].getHasRawInfo())
#pybind11#            self.assertEquals(hasRawInfo, record.getHasRawInfo())
#pybind11#
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as fileName:
#pybind11#            self.catalog.writeFits(fileName)
#pybind11#            catCopy = afwTable.AmpInfoCatalog.readFits(fileName)
#pybind11#            self.assertEquals(type(self.catalog), type(catCopy))
#pybind11#            for rec1, rec2 in zip(self.catalog, catCopy):
#pybind11#                self.assertEquals(rec1.getName(), rec2.getName())
#pybind11#                self.assertEquals(rec1.getHasRawInfo(), rec2.getHasRawInfo())
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
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

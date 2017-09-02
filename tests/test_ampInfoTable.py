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
from __future__ import absolute_import, division, print_function
import unittest

from builtins import zip

import lsst.utils.tests
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable


class AmpInfoTableTestCase(unittest.TestCase):

    def setUp(self):
        self.schema = afwTable.AmpInfoTable.makeMinimalSchema()
        self.catalog = afwTable.AmpInfoCatalog(self.schema)

    def tearDown(self):
        self.catalog = None
        self.schema = None

    def testEmptyBBox(self):
        record = self.catalog.addNew()
        emptyBox = afwGeom.Box2I()
        record.setBBox(emptyBox)
        record.setRawBBox(emptyBox)
        record.setRawDataBBox(emptyBox)
        record.setRawHorizontalOverscanBBox(emptyBox)
        record.setRawVerticalOverscanBBox(emptyBox)
        record.setRawPrescanBBox(emptyBox)
        self.assertEqual(emptyBox, record.getBBox())
        self.assertTrue(record.getBBox().isEmpty())
        self.assertEqual(emptyBox, record.getRawBBox())
        self.assertTrue(record.getRawBBox().isEmpty())
        self.assertEqual(emptyBox, record.getRawDataBBox())
        self.assertTrue(record.getRawDataBBox().isEmpty())
        self.assertEqual(emptyBox, record.getRawHorizontalOverscanBBox())
        self.assertTrue(record.getRawHorizontalOverscanBBox().isEmpty())
        self.assertEqual(emptyBox, record.getRawVerticalOverscanBBox())
        self.assertTrue(record.getRawVerticalOverscanBBox().isEmpty())
        self.assertEqual(emptyBox, record.getRawPrescanBBox())
        self.assertTrue(record.getRawPrescanBBox().isEmpty())

    def testBasics(self):
        """Test basics
        """
        nameRawInfoList = (
            ("Amp1", True),
            ("Amp2", False),
        )

        for name, hasRawInfo in nameRawInfoList:
            gain = 1.2345
            saturation = 65535
            readNoise = -0.523
            linearityCoeffs = (1.1, 2.2, 3.3, 4.4)
            linearityType = "Polynomial"
            bbox = afwGeom.Box2I(afwGeom.Point2I(3, -2),
                                 afwGeom.Extent2I(231, 320))
            hasRawInfo = hasRawInfo
            rawFlipX = True
            rawFlipY = False
            readoutCorner = afwTable.UL
            rawBBox = afwGeom.Box2I(afwGeom.Point2I(-25, 2),
                                    afwGeom.Extent2I(550, 629))
            rawXYOffset = afwGeom.Extent2I(-97, 253)
            rawDataBBox = afwGeom.Box2I(afwGeom.Point2I(-2, 29),
                                        afwGeom.Extent2I(123, 307))
            rawHorizontalOverscanBBox = afwGeom.Box2I(
                afwGeom.Point2I(150, 29), afwGeom.Extent2I(25, 307))
            rawVerticalOverscanBBox = afwGeom.Box2I(
                afwGeom.Point2I(-2, 201), afwGeom.Extent2I(123, 6))
            rawPrescanBBox = afwGeom.Box2I(
                afwGeom.Point2I(-20, 2), afwGeom.Extent2I(5, 307))

            record = self.catalog.addNew()
            record.setBBox(bbox)
            record.setName(name)
            record.setGain(gain)
            record.setSaturation(saturation)
            record.setReadNoise(readNoise)
            record.setReadoutCorner(readoutCorner)
            record.setLinearityCoeffs(linearityCoeffs)
            record.setLinearityType(linearityType)
            record.setHasRawInfo(hasRawInfo)
            if hasRawInfo:
                record.setRawFlipX(rawFlipX)
                record.setRawFlipY(rawFlipY)
                record.setRawBBox(rawBBox)
                record.setRawXYOffset(rawXYOffset)
                record.setRawDataBBox(rawDataBBox)
                record.setRawHorizontalOverscanBBox(rawHorizontalOverscanBBox)
                record.setRawVerticalOverscanBBox(rawVerticalOverscanBBox)
                record.setRawPrescanBBox(rawPrescanBBox)

            self.assertEqual(name, record.getName())
            self.assertEqual(gain, record.getGain())
            self.assertEqual(saturation, record.getSaturation())
            self.assertEqual(readNoise, record.getReadNoise())
            self.assertEqual(readoutCorner, record.getReadoutCorner())
            self.assertEqual(list(linearityCoeffs),
                             record.getLinearityCoeffs())
            self.assertEqual(linearityType, record.getLinearityType())
            self.assertEqual(bbox, record.getBBox())
            self.assertEqual(hasRawInfo, record.getHasRawInfo())
            if hasRawInfo:
                self.assertEqual(rawBBox, record.getRawBBox())
                self.assertEqual(rawDataBBox, record.getRawDataBBox())
                self.assertEqual(rawHorizontalOverscanBBox,
                                 record.getRawHorizontalOverscanBBox())
                self.assertEqual(rawVerticalOverscanBBox,
                                 record.getRawVerticalOverscanBBox())
                self.assertEqual(rawPrescanBBox, record.getRawPrescanBBox())
                self.assertEqual(rawFlipX, record.getRawFlipX())
                self.assertEqual(rawFlipY, record.getRawFlipY())
                self.assertEqual(rawXYOffset, record.getRawXYOffset())

        self.assertEqual(len(self.catalog), 2)
        for i, data in enumerate(zip(nameRawInfoList, self.catalog)):
            name, hasRawInfo = data[0]
            record = data[1]
            self.assertEqual(name, self.catalog[i].getName())
            self.assertEqual(name, record.getName())
            self.assertEqual(hasRawInfo, self.catalog[i].getHasRawInfo())
            self.assertEqual(hasRawInfo, record.getHasRawInfo())

        with lsst.utils.tests.getTempFilePath(".fits") as fileName:
            self.catalog.writeFits(fileName)
            catCopy = afwTable.AmpInfoCatalog.readFits(fileName)
            self.assertEqual(type(self.catalog), type(catCopy))
            for rec1, rec2 in zip(self.catalog, catCopy):
                self.assertEqual(rec1.getName(), rec2.getName())
                self.assertEqual(rec1.getHasRawInfo(), rec2.getHasRawInfo())

        self.assertEqual(self.catalog.table.schema, self.schema)
        self.assertEqual(self.catalog.schema, self.schema)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

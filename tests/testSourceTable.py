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
Tests for table.SourceTable

Run with:
   ./testSourceTable.py
or
   python
   >>> import testSourceTable; testSourceTable.run()
"""

import sys
import os
import unittest
import numpy

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.table
import lsst.afw.geom
import lsst.afw.coord
import lsst.afw.image

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def makeArray(size, dtype):
    return numpy.array(numpy.random.randn(*size), dtype=dtype)

def makeCov(size, dtype):
    m = numpy.array(numpy.random.randn(size, size), dtype=dtype)
    return numpy.dot(m, m.transpose())

def makeWcs():
    crval = lsst.afw.coord.Coord(lsst.afw.geom.Point2D(1.606631, 5.090329))
    crpix = lsst.afw.geom.Point2D(2036., 2000.)
    return lsst.afw.image.makeWcs(crval, crpix, 5.399452e-5, -1.30770e-5, 1.30770e-5, 5.399452e-5)

class SourceTableTestCase(unittest.TestCase):

    def fillRecord(self, record):
        record.set(self.fluxKey, numpy.random.randn())
        record.set(self.fluxErrKey, numpy.random.randn())
        record.set(self.centroidKey, lsst.afw.geom.Point2D(*numpy.random.randn(2)))
        record.set(self.centroidErrKey, makeCov(2, float))
        record.set(self.shapeKey, lsst.afw.geom.ellipses.Quadrupole(*numpy.random.randn(3)))
        record.set(self.shapeErrKey, makeCov(3, float))
        record.set(self.fluxFlagKey, numpy.random.randn() > 0)
        record.set(self.centroidFlagKey, numpy.random.randn() > 0)
        record.set(self.shapeFlagKey, numpy.random.randn() > 0)

    def setUp(self):
        self.schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        self.fluxKey = self.schema.addField("a", type="D")
        self.fluxErrKey = self.schema.addField("a.err", type="D")
        self.fluxFlagKey = self.schema.addField("a.flags", type="Flag")
        self.centroidKey = self.schema.addField("b", type="PointD")
        self.centroidErrKey = self.schema.addField("b.err", type="CovPointD")
        self.centroidFlagKey = self.schema.addField("b.flags", type="Flag")
        self.shapeKey = self.schema.addField("c", type="MomentsD")
        self.shapeErrKey = self.schema.addField("c.err", type="CovMomentsD")
        self.shapeFlagKey = self.schema.addField("c.flags", type="Flag")
        self.table = lsst.afw.table.SourceTable.make(self.schema)
        self.catalog = lsst.afw.table.SourceCatalog(self.table)
        self.record = self.catalog.addNew()
        self.fillRecord(self.record)
        self.record.setId(50)
        self.fillRecord(self.catalog.addNew())
        self.fillRecord(self.catalog.addNew())

    def tearDown(self):
        del self.schema
        del self.record
        del self.table
        del self.catalog

    def checkCanonical(self):
        self.assertEqual(self.table.getPsfFluxDefinition(), "a")
        self.assertEqual(self.record.get(self.fluxKey), self.record.getPsfFlux())
        self.assertEqual(self.record.get(self.fluxFlagKey), self.record.getPsfFluxFlag())
        self.assertEqual(self.table.getCentroidDefinition(), "b")
        self.assertEqual(self.record.get(self.centroidKey), self.record.getCentroid())
        self.assert_(numpy.all(self.record.get(self.centroidErrKey) == self.record.getCentroidErr()))
        self.assertEqual(self.table.getShapeDefinition(), "c")
        self.assertEqual(self.record.get(self.shapeKey), self.record.getShape())
        self.assert_(numpy.all(self.record.get(self.shapeErrKey) == self.record.getShapeErr()))

    def testCanonical1(self):
        self.table.definePsfFlux(self.fluxKey, self.fluxErrKey, self.fluxFlagKey)
        self.table.defineCentroid(self.centroidKey, self.centroidErrKey, self.centroidFlagKey)
        self.table.defineShape(self.shapeKey, self.shapeErrKey, self.shapeFlagKey)
        self.checkCanonical()

    def testCanonical2(self):
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        self.checkCanonical()

    def testTicket2165(self):
        """Check that we can define the slots without all the keys."""
        self.table.definePsfFlux(self.fluxKey)
        self.assertEqual(self.table.getPsfFluxDefinition(), "a")
        self.assertEqual(self.table.getPsfFluxKey(), self.fluxKey)
        self.assertFalse(self.table.getPsfFluxFlagKey().isValid())
        self.assertFalse(self.table.getPsfFluxErrKey().isValid())
        self.table.defineCentroid(self.centroidKey, flag=self.centroidFlagKey)
        self.assertEqual(self.table.getCentroidDefinition(), "b")
        self.assertEqual(self.table.getCentroidKey(), self.centroidKey)
        self.assertEqual(self.table.getCentroidFlagKey(), self.centroidFlagKey)
        self.assertFalse(self.table.getCentroidErrKey().isValid())
        schema2 = lsst.afw.table.SourceTable.makeMinimalSchema()
        fluxKey2 = schema2.addField("a", type="D")
        fluxFlagKey2 = schema2.addField("a.flags", type="Flag")
        table2 = lsst.afw.table.SourceTable.make(schema2)
        table2.definePsfFlux("a")
        self.assertEqual(table2.getPsfFluxDefinition(), "a")
        self.assertEqual(table2.getPsfFluxKey(), fluxKey2)
        self.assertEqual(table2.getPsfFluxFlagKey(), fluxFlagKey2)
        self.assertFalse(self.table.getPsfFluxErrKey().isValid())


    def testCoordUpdate(self):
        wcs = makeWcs()
        self.record.updateCoord(wcs, self.centroidKey)
        coord1 = self.record.getCoord()
        coord2 = wcs.pixelToSky(self.record.get(self.centroidKey))
        self.assertEqual(coord1, coord2)

    def testSorting(self):
        self.assertFalse(self.catalog.isSorted())
        self.catalog.sort()
        self.assert_(self.catalog.isSorted())
        r = self.catalog.find(2)
        self.assertEqual(r["id"], 2)
        r = self.catalog.find(500)
        self.assert_(r is None)

    def testConversion(self):
        catalog1 = self.catalog.cast(lsst.afw.table.SourceCatalog)
        catalog2 = self.catalog.cast(lsst.afw.table.SimpleCatalog)
        catalog3 = self.catalog.cast(lsst.afw.table.SourceCatalog, deep=True)
        catalog4 = self.catalog.cast(lsst.afw.table.SimpleCatalog, deep=True)
        self.assertEqual(self.catalog.table, catalog1.table)
        self.assertEqual(self.catalog.table, catalog2.table)
        self.assertNotEqual(self.catalog.table, catalog3.table)
        self.assertNotEqual(self.catalog.table, catalog3.table)
        for r, r1, r2, r3, r4 in zip(self.catalog, catalog1, catalog2, catalog3, catalog4):
            self.assertEqual(r, r1)
            self.assertEqual(r, r2)
            self.assertNotEqual(r, r3)
            self.assertNotEqual(r, r4)
            self.assertEqual(r.getId(), r3.getId())
            self.assertEqual(r.getId(), r4.getId())

    def testColumnView(self):
        cols1 = self.catalog.getColumnView()
        cols2 = self.catalog.columns
        self.assert_(cols1 is cols2)
        self.assert_(isinstance(cols1, lsst.afw.table.SourceColumnView))
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        self.assert_((cols2["a"] == cols2.getPsfFlux()).all())
        self.assert_((cols2["b.x"] == cols2.getX()).all())
        self.assert_((cols2["b.y"] == cols2.getY()).all())
        self.assert_((cols2["c.xx"] == cols2.getIxx()).all())
        self.assert_((cols2["c.yy"] == cols2.getIyy()).all())
        self.assert_((cols2["c.xy"] == cols2.getIxy()).all())

    def testForwarding(self):
        """Verify that Catalog forwards unknown methods to its table and/or columns."""
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        self.assert_((self.catalog.columns["a"] == self.catalog["a"]).all())
        self.assert_((self.catalog.columns[self.fluxKey] == self.catalog.get(self.fluxKey)).all())
        self.assert_((self.catalog.columns.get(self.fluxKey) == self.catalog.getPsfFlux()).all())
        self.assertEqual(self.fluxKey, self.catalog.getPsfFluxKey())
        self.assertRaises(AttributeError, lambda c: c.foo(), self.catalog)

    def testBitsColumn(self):
        allBits = self.catalog.getBits()
        someBits = self.catalog.getBits(["a.flags", "c.flags"])
        self.assertEqual(allBits.getMask("a.flags"), 0x1)
        self.assertEqual(allBits.getMask("b.flags"), 0x2)
        self.assertEqual(allBits.getMask("c.flags"), 0x4)
        self.assertEqual(someBits.getMask(self.fluxFlagKey), 0x1)
        self.assertEqual(someBits.getMask(self.shapeFlagKey), 0x2)
        self.assert_(((allBits.array & 0x1 != 0) == self.catalog.columns["a.flags"]).all())
        self.assert_(((allBits.array & 0x2 != 0) == self.catalog.columns["b.flags"]).all())
        self.assert_(((allBits.array & 0x4 != 0) == self.catalog.columns["c.flags"]).all())
        self.assert_(((someBits.array & 0x1 != 0) == self.catalog.columns["a.flags"]).all())
        self.assert_(((someBits.array & 0x2 != 0) == self.catalog.columns["c.flags"]).all())

    def testCast(self):
        baseCat = self.catalog.cast(lsst.afw.table.BaseCatalog)
        sourceCat = baseCat.cast(lsst.afw.table.SourceCatalog)

    def testIdFactory(self):
        expId = int(1257198)
        reserved = 32
        factory = lsst.afw.table.IdFactory.makeSource(expId, reserved)
        upper = expId
        id1 = factory()
        id2 = factory()
        self.assertEqual(id2 - id1, 1)
        factory.notify(0xFFFFFFFF)
        lsst.utils.tests.assertRaisesLsstCpp(self, lsst.pex.exceptions.LengthErrorException, factory)
        lsst.utils.tests.assertRaisesLsstCpp(self, lsst.pex.exceptions.InvalidParameterException,
                                             factory.notify, 0x1FFFFFFFF)
        lsst.utils.tests.assertRaisesLsstCpp(self, lsst.pex.exceptions.InvalidParameterException,
                                             lsst.afw.table.IdFactory.makeSource, 0x1FFFFFFFF, reserved)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(SourceTableTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

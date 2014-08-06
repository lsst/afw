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
import tempfile
import pickle

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.table
import lsst.afw.geom
import lsst.afw.coord
import lsst.afw.image
import lsst.afw.detection

numpy.random.seed(1)

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
        record.set(self.centroidErrKey, makeCov(2, numpy.float32))
        record.set(self.shapeKey, lsst.afw.geom.ellipses.Quadrupole(*numpy.random.randn(3)))
        record.set(self.shapeErrKey, makeCov(3, numpy.float32))
        record.set(self.fluxFlagKey, numpy.random.randn() > 0)
        record.set(self.centroidFlagKey, numpy.random.randn() > 0)
        record.set(self.shapeFlagKey, numpy.random.randn() > 0)

    def setUp(self):
        self.schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        self.fluxKey = self.schema.addField("a", type="D")
        self.fluxErrKey = self.schema.addField("a.err", type="D")
        self.fluxFlagKey = self.schema.addField("a.flags", type="Flag")
        self.centroidKey = self.schema.addField("b", type="PointD")
        self.centroidErrKey = self.schema.addField("b.err", type="CovPointF")
        self.centroidFlagKey = self.schema.addField("b.flags", type="Flag")
        self.shapeKey = self.schema.addField("c", type="MomentsD")
        self.shapeErrKey = self.schema.addField("c.err", type="CovMomentsF")
        self.shapeFlagKey = self.schema.addField("c.flags", type="Flag")
        self.table = lsst.afw.table.SourceTable.make(self.schema)
        self.table.setVersion(0)
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

    def testPersisted(self):
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        self.catalog.writeFits("test.fits")
        catalog = lsst.afw.table.SourceCatalog.readFits("test.fits")
        table = catalog.getTable()
        record = catalog[0]
        # I'm using the keys from the non-persisted table.  They should work at least in the current implementation
        self.assertEqual(table.getPsfFluxDefinition(), "a")
        self.assertEqual(record.get(self.fluxKey), record.getPsfFlux())
        self.assertEqual(record.get(self.fluxFlagKey), record.getPsfFluxFlag())
        self.assertEqual(table.getCentroidDefinition(), "b")
        self.assertEqual(record.get(self.centroidKey), record.getCentroid())
        self.assert_(numpy.all(record.get(self.centroidErrKey) == record.getCentroidErr()))
        self.assertEqual(table.getShapeDefinition(), "c")
        self.assertEqual(record.get(self.shapeKey), record.getShape())
        self.assert_(numpy.all(record.get(self.shapeErrKey) == record.getShapeErr()))
        os.unlink("test.fits")

    def testCanonical2(self):
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        self.checkCanonical()

    def testPickle(self):
        p = pickle.dumps(self.catalog)
        new = pickle.loads(p)

        self.assertEqual(self.catalog.schema.getNames(), new.schema.getNames())
        self.assertEqual(len(self.catalog), len(new))
        for r1, r2 in zip(self.catalog, new):
            for field in ("a", "a.err", "id"): # Columns that are easy to test
                k1 = self.catalog.schema.find(field).getKey()
                k2 = new.schema.find(field).getKey()
                self.assertTrue(r1[k1] == r2[k2])


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
        self.assert_((cols2["a.err"] == cols2.getPsfFluxErr()).all())
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

    def testFootprints(self):
        '''Test round-tripping Footprints (inc. HeavyFootprints) to FITS
        '''
        src1 = self.catalog.addNew()
        src2 = self.catalog.addNew()
        src3 = self.catalog.addNew()
        self.fillRecord(src1)
        self.fillRecord(src2)
        self.fillRecord(src3)
        src2.setParent(src1.getId())

        W,H = 100,100
        mim = lsst.afw.image.MaskedImageF(W,H)
        im = mim.getImage()
        msk = mim.getMask()
        var = mim.getVariance()
        for y in range(H):
            for x in range(W):
                im.set (x, y, y * 1e6 + x * 1e3)
                msk.set(x, y, (y << 8) | x)
                var.set(x, y, y * 1e2 + x)
        circ = lsst.afw.detection.Footprint(lsst.afw.geom.Point2I(50,50), 20)
        heavy = lsst.afw.detection.makeHeavyFootprint(circ, mim)
        src2.setFootprint(heavy)

        for i,src in enumerate(self.catalog):
            if src != src2:
                src.setFootprint(lsst.afw.detection.Footprint(lsst.afw.geom.Point2I(50,50), 1+i*2))

        # insert this HeavyFootprint into an otherwise blank image (for comparing the results)
        mim2 = lsst.afw.image.MaskedImageF(W,H)
        heavy.insert(mim2)

        f,fn = tempfile.mkstemp(prefix='testHeavyFootprint-', suffix='.fits')
        os.close(f)
        self.catalog.writeFits(fn)

        cat2 = lsst.afw.table.SourceCatalog.readFits(fn)
        r2 = cat2[-2]
        f2 = r2.getFootprint()
        self.assertTrue(f2.isHeavy())
        h2 = lsst.afw.detection.cast_HeavyFootprintF(f2)
        mim3 = lsst.afw.image.MaskedImageF(W, H)
        h2.insert(mim3)

        self.assertFalse(cat2[-1].getFootprint().isHeavy())
        self.assertFalse(cat2[-3].getFootprint().isHeavy())
        self.assertFalse(cat2[0].getFootprint().isHeavy())
        self.assertFalse(cat2[1].getFootprint().isHeavy())
        self.assertFalse(cat2[2].getFootprint().isHeavy())

        if False:
            # Write out before-n-after FITS images
            for MI in [mim, mim2, mim3]:
                f,fn2 = tempfile.mkstemp(prefix='testHeavyFootprint-', suffix='.fits')
                os.close(f)
                MI.writeFits(fn2)
                print 'wrote', fn2

        self.assertTrue(all((mim2.getImage().getArray() == mim3.getImage().getArray()).ravel()))
        self.assertTrue(all((mim2.getMask().getArray() == mim3.getMask().getArray()).ravel()))
        self.assertTrue(all((mim2.getVariance().getArray() == mim3.getVariance().getArray()).ravel()))

        im3 = mim3.getImage()
        ma3 = mim3.getMask()
        va3 = mim3.getVariance()
        for y in range(H):
            for x in range(W):
                if circ.contains(lsst.afw.geom.Point2I(x, y)):
                    self.assertEqual(im.get(x, y),  im3.get(x, y))
                    self.assertEqual(msk.get(x, y), ma3.get(x, y))
                    self.assertEqual(var.get(x, y), va3.get(x, y))
                else:
                    self.assertEqual(im3.get(x, y), 0.)
                    self.assertEqual(ma3.get(x, y), 0.)
                    self.assertEqual(va3.get(x, y), 0.)

        cat3 = lsst.afw.table.SourceCatalog.readFits(fn, 0, lsst.afw.table.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
        for src in cat3:
            self.assertFalse(src.getFootprint().isHeavy())
        cat4 = lsst.afw.table.SourceCatalog.readFits(fn, 0, lsst.afw.table.SOURCE_IO_NO_FOOTPRINTS)
        for src in cat4:
            self.assertIsNone(src.getFootprint())

        self.catalog.writeFits(fn, flags=lsst.afw.table.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
        cat5 = lsst.afw.table.SourceCatalog.readFits(fn)
        for src in cat5:
            self.assertFalse(src.getFootprint().isHeavy())

        self.catalog.writeFits(fn, flags=lsst.afw.table.SOURCE_IO_NO_FOOTPRINTS)
        cat6 = lsst.afw.table.SourceCatalog.readFits(fn)
        for src in cat6:
            self.assertIsNone(src.getFootprint())

    def testIdFactory(self):
        expId = int(1257198)
        reserved = 32
        factory = lsst.afw.table.IdFactory.makeSource(expId, reserved)
        upper = expId
        id1 = factory()
        id2 = factory()
        self.assertEqual(id2 - id1, 1)
        factory.notify(0xFFFFFFFF)
        self.assertRaises(lsst.pex.exceptions.LengthError, factory)
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, factory.notify, 0x1FFFFFFFF)
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError,
                                             lsst.afw.table.IdFactory.makeSource, 0x1FFFFFFFF, reserved)

    def testFamilies(self):
        self.catalog.sort()
        parents = self.catalog.getChildren(0)
        self.assertEqual(list(parents), list(self.catalog))
        parentKey = lsst.afw.table.SourceTable.getParentKey()
        for parent in parents:
            self.assertEqual(parent.get(parentKey), 0)
            for i in range(10):
                child = self.catalog.addNew()
                self.fillRecord(child)
                child.set(parentKey, parent.getId())
        for parent in parents:
            children, ids = self.catalog.getChildren(parent.getId(),
                                                     [record.getId() for record in self.catalog])
            self.assertEqual(len(children), 10)
            self.assertEqual(len(children), len(ids))
            for child, id in zip(children, ids):
                self.assertEqual(child.getParent(), parent.getId())
                self.assertEqual(child.getId(), id)

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

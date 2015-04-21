#!/usr/bin/env python2
from __future__ import absolute_import, division
#
# LSST Data Management System
# Copyright 2008-2014 LSST Corporation.
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

import os
import unittest
import numpy
import tempfile
import pickle
import math

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

class SourceTableTestCase(lsst.utils.tests.TestCase):

    def fillRecord(self, record):
        record.set(self.fluxKey, numpy.random.randn())
        record.set(self.fluxErrKey, numpy.random.randn())
        record.set(self.centroidKey.getX(), numpy.random.randn())
        record.set(self.centroidKey.getY(), numpy.random.randn())
        record.set(self.xErrKey, numpy.random.randn())
        record.set(self.yErrKey, numpy.random.randn())
        record.set(self.shapeKey.getIxx(), numpy.random.randn())
        record.set(self.shapeKey.getIyy(), numpy.random.randn())
        record.set(self.shapeKey.getIxy(), numpy.random.randn())
        record.set(self.xxErrKey, numpy.random.randn())
        record.set(self.yyErrKey, numpy.random.randn())
        record.set(self.xyErrKey, numpy.random.randn())
        record.set(self.fluxFlagKey, numpy.random.randn() > 0)
        record.set(self.centroidFlagKey, numpy.random.randn() > 0)
        record.set(self.shapeFlagKey, numpy.random.randn() > 0)

    def setUp(self):
        self.schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        self.fluxKey = self.schema.addField("a_flux", type = "D")
        self.fluxErrKey = self.schema.addField("a_fluxSigma", type = "D")
        self.fluxFlagKey = self.schema.addField("a_flag", type="Flag")

        # the meas field is added using a functor key, but the error is added
        # as scalars, as we lack a ResultKey functor as exists in meas_base
        self.centroidKey = lsst.afw.table.Point2DKey.addFields(self.schema,
            "b", "", "pixels")
        self.xErrKey = self.schema.addField("b_xSigma", type = "F")
        self.yErrKey = self.schema.addField("b_ySigma", type = "F")
        self.centroidFlagKey = self.schema.addField("b_flag", type="Flag")

        self.shapeKey = lsst.afw.table.QuadrupoleKey.addFields(self.schema,
            "c", "", lsst.afw.table.PIXEL)
        self.xxErrKey = self.schema.addField("c_xxSigma", type = "F")
        self.xyErrKey = self.schema.addField("c_xySigma", type = "F")
        self.yyErrKey = self.schema.addField("c_yySigma", type = "F")
        self.shapeFlagKey = self.schema.addField("c_flag", type="Flag")

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
        self.assertEqual(self.centroidKey.get(self.record), self.record.getCentroid())
        self.assertClose(math.fabs(self.record.get(self.xErrKey)),
            math.sqrt(self.record.getCentroidErr()[0,0]), rtol=1e-6)
        self.assertClose(math.fabs(self.record.get(self.yErrKey)),
            math.sqrt(self.record.getCentroidErr()[1,1]), rtol=1e-6)
        self.assertEqual(self.table.getShapeDefinition(), "c")
        self.assertEqual(self.shapeKey.get(self.record), self.record.getShape())
        self.assertClose(math.fabs(self.record.get(self.xxErrKey)),
            math.sqrt(self.record.getShapeErr()[0,0]), rtol=1e-6)
        self.assertClose(math.fabs(self.record.get(self.yyErrKey)),
            math.sqrt(self.record.getShapeErr()[1,1]), rtol=1e-6)
        self.assertClose(math.fabs(self.record.get(self.xyErrKey)),
            math.sqrt(self.record.getShapeErr()[2,2]), rtol=1e-6)

    def testPersisted(self):
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            self.catalog.writeFits(filename)
            catalog = lsst.afw.table.SourceCatalog.readFits(filename)
            table = catalog.getTable()
            record = catalog[0]
            # I'm using the keys from the non-persisted table.  They should work at least in the
            # current implementation
            self.assertEqual(table.getPsfFluxDefinition(), "a")
            self.assertEqual(record.get(self.fluxKey), record.getPsfFlux())
            self.assertEqual(record.get(self.fluxFlagKey), record.getPsfFluxFlag())
            self.assertEqual(table.getCentroidDefinition(), "b")
            centroid = self.centroidKey.get(self.record)
            self.assertEqual(centroid, record.getCentroid())
            self.assertClose(math.fabs(self.record.get(self.xErrKey)),
                math.sqrt(self.record.getCentroidErr()[0,0]), rtol=1e-6)
            self.assertClose(math.fabs(self.record.get(self.yErrKey)),
                math.sqrt(self.record.getCentroidErr()[1,1]), rtol=1e-6)
            shape = self.shapeKey.get(self.record)
            self.assertEqual(table.getShapeDefinition(), "c")
            self.assertEqual(shape, record.getShape())
            self.assertClose(math.fabs(self.record.get(self.xxErrKey)),
                math.sqrt(self.record.getShapeErr()[0,0]), rtol=1e-6)
            self.assertClose(math.fabs(self.record.get(self.yyErrKey)),
                math.sqrt(self.record.getShapeErr()[1,1]), rtol=1e-6)
            self.assertClose(math.fabs(self.record.get(self.xyErrKey)),
                math.sqrt(self.record.getShapeErr()[2,2]), rtol=1e-6)

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
            for field in ("a_flux", "a_fluxSigma", "id"): # Columns that are easy to test
                k1 = self.catalog.schema.find(field).getKey()
                k2 = new.schema.find(field).getKey()
                self.assertTrue(r1[k1] == r2[k2])

    def testCoordUpdate(self):
        self.table.defineCentroid("b")
        wcs = makeWcs()
        self.record.updateCoord(wcs)
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
        self.assert_((cols2["a_flux"] == cols2.getPsfFlux()).all())
        self.assert_((cols2["a_fluxSigma"] == cols2.getPsfFluxErr()).all())
        self.assert_((cols2["b_x"] == cols2.getX()).all())
        self.assert_((cols2["b_y"] == cols2.getY()).all())
        self.assert_((cols2["c_xx"] == cols2.getIxx()).all())
        self.assert_((cols2["c_yy"] == cols2.getIyy()).all())
        self.assert_((cols2["c_xy"] == cols2.getIxy()).all())

    def testForwarding(self):
        """Verify that Catalog forwards unknown methods to its table and/or columns."""
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        self.assert_((self.catalog.columns["a_flux"] == self.catalog["a_flux"]).all())
        self.assert_((self.catalog.columns[self.fluxKey] == self.catalog.get(self.fluxKey)).all())
        self.assert_((self.catalog.columns.get(self.fluxKey) == self.catalog.getPsfFlux()).all())
        self.assertEqual(self.fluxKey, self.catalog.getPsfFluxKey())
        self.assertRaises(AttributeError, lambda c: c.foo(), self.catalog)

    def testBitsColumn(self):
        allBits = self.catalog.getBits()
        someBits = self.catalog.getBits(["a_flag", "c_flag"])
        self.assertEqual(allBits.getMask("a_flag"), 0x1)
        self.assertEqual(allBits.getMask("b_flag"), 0x2)
        self.assertEqual(allBits.getMask("c_flag"), 0x4)
        self.assertEqual(someBits.getMask(self.fluxFlagKey), 0x1)
        self.assertEqual(someBits.getMask(self.shapeFlagKey), 0x2)
        self.assert_(((allBits.array & 0x1 != 0) == self.catalog.columns["a_flag"]).all())
        self.assert_(((allBits.array & 0x2 != 0) == self.catalog.columns["b_flag"]).all())
        self.assert_(((allBits.array & 0x4 != 0) == self.catalog.columns["c_flag"]).all())
        self.assert_(((someBits.array & 0x1 != 0) == self.catalog.columns["a_flag"]).all())
        self.assert_(((someBits.array & 0x2 != 0) == self.catalog.columns["c_flag"]).all())

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

        with lsst.utils.tests.getTempFilePath(".fits") as fn:
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

    def testFitsReadBackwardsCompatibility(self):
        cat = lsst.afw.table.SourceCatalog.readFits("tests/data/slotsVersion0.fits")
        self.assertTrue(cat.getPsfFluxSlot().isValid())
        self.assertTrue(cat.getApFluxSlot().isValid())
        self.assertTrue(cat.getInstFluxSlot().isValid())
        self.assertTrue(cat.getModelFluxSlot().isValid())
        self.assertTrue(cat.getCentroidSlot().isValid())
        self.assertTrue(cat.getShapeSlot().isValid())
        self.assertEqual(cat.getPsfFluxSlot().getMeasKey(), cat.schema.find("flux").key)
        self.assertEqual(cat.getApFluxSlot().getMeasKey(), cat.schema.find("flux").key)
        self.assertEqual(cat.getInstFluxSlot().getMeasKey(), cat.schema.find("flux").key)
        self.assertEqual(cat.getModelFluxSlot().getMeasKey(), cat.schema.find("flux").key)
        self.assertEqual(cat.getCentroidSlot().getMeasKey().getX(), cat.schema.find("centroid.x").key)
        self.assertEqual(cat.getCentroidSlot().getMeasKey().getY(), cat.schema.find("centroid.y").key)
        self.assertEqual(cat.getShapeSlot().getMeasKey().getIxx(), cat.schema.find("shape.xx").key)
        self.assertEqual(cat.getShapeSlot().getMeasKey().getIyy(), cat.schema.find("shape.yy").key)
        self.assertEqual(cat.getShapeSlot().getMeasKey().getIxy(), cat.schema.find("shape.xy").key)
        self.assertEqual(cat.getPsfFluxSlot().getErrKey(), cat.schema.find("flux.err").key)
        self.assertEqual(cat.getApFluxSlot().getErrKey(), cat.schema.find("flux.err").key)
        self.assertEqual(cat.getInstFluxSlot().getErrKey(), cat.schema.find("flux.err").key)
        self.assertEqual(cat.getModelFluxSlot().getErrKey(), cat.schema.find("flux.err").key)
        self.assertEqual(cat.getCentroidSlot().getErrKey(),
                         lsst.afw.table.makeCovarianceMatrixKey(cat.schema.find("centroid.err").key))
        self.assertEqual(cat.getShapeSlot().getErrKey(),
                         lsst.afw.table.makeCovarianceMatrixKey(cat.schema.find("shape.err").key))
        self.assertEqual(cat.getPsfFluxSlot().getFlagKey(), cat.schema.find("flux.flags").key)
        self.assertEqual(cat.getApFluxSlot().getFlagKey(), cat.schema.find("flux.flags").key)
        self.assertEqual(cat.getInstFluxSlot().getFlagKey(), cat.schema.find("flux.flags").key)
        self.assertEqual(cat.getModelFluxSlot().getFlagKey(), cat.schema.find("flux.flags").key)
        self.assertEqual(cat.getCentroidSlot().getFlagKey(), cat.schema.find("centroid.flags").key)
        self.assertEqual(cat.getShapeSlot().getFlagKey(), cat.schema.find("shape.flags").key)

    def testDM1083(self):
        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        st = lsst.afw.table.SourceTable.make(schema)
        cat = lsst.afw.table.SourceCatalog(st)
        tmp = lsst.afw.table.SourceCatalog(cat.getTable())
        record = tmp.addNew()
        cat.extend(tmp)
        self.assertEqual(cat[0].getId(), record.getId())
        # check that the same record is in both catalogs (not a copy)
        record.setId(15)
        self.assertEqual(cat[0].getId(), record.getId())

    def testSlotUndefine(self):
        """Test that we can correctly define and undefine a slot after a SourceTable has been created"""
        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        key = schema.addField("a_flux", type=float, doc="flux field")
        table = lsst.afw.table.SourceTable.make(schema)
        table.definePsfFlux("a")
        self.assertEqual(table.getPsfFluxKey(), key)
        table.schema.getAliasMap().erase("slot_PsfFlux")
        self.assertFalse(table.getPsfFluxKey().isValid())

    def testOldFootprintPersistence(self):
        """Test that we can still read SourceCatalogs with (Heavy)Footprints saved by an older
        version of the pipeline with a different format.
        """
        filename = os.path.join("tests", "data", "old-footprint-persistence.fits")
        catalog1 = lsst.afw.table.SourceCatalog.readFits(filename)
        self.assertEqual(len(catalog1), 2)
        self.assertRaises(KeyError, catalog1.schema.find, "footprint")
        fp1 = catalog1[0].getFootprint()
        fp2 = catalog1[1].getFootprint()
        self.assertEqual(fp1.getArea(), 495)
        self.assertEqual(fp2.getArea(), 767)
        self.assertFalse(fp1.isHeavy())
        self.assertTrue(fp2.isHeavy())
        self.assertEqual(len(fp1.getSpans()), 29)
        self.assertEqual(len(fp2.getSpans()), 44)
        self.assertEqual(len(fp1.getPeaks()), 1)
        self.assertEqual(len(fp2.getPeaks()), 1)
        self.assertEqual(fp1.getBBox(),
                         lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(129,2), lsst.afw.geom.Extent2I(25, 29)))
        self.assertEqual(fp2.getBBox(),
                         lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(1184,2), lsst.afw.geom.Extent2I(78, 38)))
        hfp = lsst.afw.detection.cast_HeavyFootprintF(fp2)
        self.assertEqual(len(hfp.getImageArray()), fp2.getArea())
        self.assertEqual(len(hfp.getMaskArray()), fp2.getArea())
        self.assertEqual(len(hfp.getVarianceArray()), fp2.getArea())
        catalog2 = lsst.afw.table.SourceCatalog.readFits(filename, 0,
                                                         lsst.afw.table.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
        self.assertEqual(list(fp1.getSpans()), list(catalog2[0].getFootprint().getSpans()))
        self.assertEqual(list(fp2.getSpans()), list(catalog2[1].getFootprint().getSpans()))
        self.assertFalse(catalog2[1].getFootprint().isHeavy())
        catalog3 = lsst.afw.table.SourceCatalog.readFits(filename, 0,
                                                         lsst.afw.table.SOURCE_IO_NO_FOOTPRINTS)
        self.assertEqual(catalog3[0].getFootprint(), None)
        self.assertEqual(catalog3[1].getFootprint(), None)

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

#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import zip
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008-2014 LSST Corporation.
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
#pybind11#
#pybind11#"""
#pybind11#Tests for table.SourceTable
#pybind11#
#pybind11#Run with:
#pybind11#   ./testSourceTable.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import testSourceTable; testSourceTable.run()
#pybind11#"""
#pybind11#import os
#pybind11#import unittest
#pybind11#import numpy
#pybind11#import tempfile
#pybind11#import pickle
#pybind11#import math
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.table
#pybind11#import lsst.afw.geom
#pybind11#import lsst.afw.coord
#pybind11#import lsst.afw.image
#pybind11#import lsst.afw.detection
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11#testPath = os.path.abspath(os.path.dirname(__file__))
#pybind11#
#pybind11#
#pybind11#def makeArray(size, dtype):
#pybind11#    return numpy.array(numpy.random.randn(*size), dtype=dtype)
#pybind11#
#pybind11#
#pybind11#def makeCov(size, dtype):
#pybind11#    m = numpy.array(numpy.random.randn(size, size), dtype=dtype)
#pybind11#    return numpy.dot(m, m.transpose())
#pybind11#
#pybind11#
#pybind11#def makeWcs():
#pybind11#    crval = lsst.afw.coord.Coord(lsst.afw.geom.Point2D(1.606631, 5.090329))
#pybind11#    crpix = lsst.afw.geom.Point2D(2036., 2000.)
#pybind11#    return lsst.afw.image.makeWcs(crval, crpix, 5.399452e-5, -1.30770e-5, 1.30770e-5, 5.399452e-5)
#pybind11#
#pybind11#
#pybind11#class SourceTableTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def fillRecord(self, record):
#pybind11#        record.set(self.fluxKey, numpy.random.randn())
#pybind11#        record.set(self.fluxErrKey, numpy.random.randn())
#pybind11#        record.set(self.centroidKey.getX(), numpy.random.randn())
#pybind11#        record.set(self.centroidKey.getY(), numpy.random.randn())
#pybind11#        record.set(self.xErrKey, numpy.random.randn())
#pybind11#        record.set(self.yErrKey, numpy.random.randn())
#pybind11#        record.set(self.shapeKey.getIxx(), numpy.random.randn())
#pybind11#        record.set(self.shapeKey.getIyy(), numpy.random.randn())
#pybind11#        record.set(self.shapeKey.getIxy(), numpy.random.randn())
#pybind11#        record.set(self.xxErrKey, numpy.random.randn())
#pybind11#        record.set(self.yyErrKey, numpy.random.randn())
#pybind11#        record.set(self.xyErrKey, numpy.random.randn())
#pybind11#        record.set(self.fluxFlagKey, numpy.random.randn() > 0)
#pybind11#        record.set(self.centroidFlagKey, numpy.random.randn() > 0)
#pybind11#        record.set(self.shapeFlagKey, numpy.random.randn() > 0)
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        numpy.random.seed(1)
#pybind11#        self.schema = lsst.afw.table.SourceTable.makeMinimalSchema()
#pybind11#        self.fluxKey = self.schema.addField("a_flux", type="D")
#pybind11#        self.fluxErrKey = self.schema.addField("a_fluxSigma", type="D")
#pybind11#        self.fluxFlagKey = self.schema.addField("a_flag", type="Flag")
#pybind11#
#pybind11#        # the meas field is added using a functor key, but the error is added
#pybind11#        # as scalars, as we lack a ResultKey functor as exists in meas_base
#pybind11#        self.centroidKey = lsst.afw.table.Point2DKey.addFields(self.schema,
#pybind11#                                                               "b", "", "pixel")
#pybind11#        self.xErrKey = self.schema.addField("b_xSigma", type="F")
#pybind11#        self.yErrKey = self.schema.addField("b_ySigma", type="F")
#pybind11#        self.centroidFlagKey = self.schema.addField("b_flag", type="Flag")
#pybind11#
#pybind11#        self.shapeKey = lsst.afw.table.QuadrupoleKey.addFields(self.schema,
#pybind11#                                                               "c", "", lsst.afw.table.CoordinateType_PIXEL)
#pybind11#        self.xxErrKey = self.schema.addField("c_xxSigma", type="F")
#pybind11#        self.xyErrKey = self.schema.addField("c_xySigma", type="F")
#pybind11#        self.yyErrKey = self.schema.addField("c_yySigma", type="F")
#pybind11#        self.shapeFlagKey = self.schema.addField("c_flag", type="Flag")
#pybind11#
#pybind11#        self.table = lsst.afw.table.SourceTable.make(self.schema)
#pybind11#        self.catalog = lsst.afw.table.SourceCatalog(self.table)
#pybind11#        self.record = self.catalog.addNew()
#pybind11#        self.fillRecord(self.record)
#pybind11#        self.record.setId(50)
#pybind11#        self.fillRecord(self.catalog.addNew())
#pybind11#        self.fillRecord(self.catalog.addNew())
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.schema
#pybind11#        del self.record
#pybind11#        del self.table
#pybind11#        del self.catalog
#pybind11#
#pybind11#    def checkCanonical(self):
#pybind11#        self.assertEqual(self.table.getPsfFluxDefinition(), "a")
#pybind11#        self.assertEqual(self.record.get(self.fluxKey), self.record.getPsfFlux())
#pybind11#        self.assertEqual(self.record.get(self.fluxFlagKey), self.record.getPsfFluxFlag())
#pybind11#        self.assertEqual(self.table.getCentroidDefinition(), "b")
#pybind11#        self.assertEqual(self.centroidKey.get(self.record), self.record.getCentroid())
#pybind11#        self.assertClose(math.fabs(self.record.get(self.xErrKey)),
#pybind11#                         math.sqrt(self.record.getCentroidErr()[0, 0]), rtol=1e-6)
#pybind11#        self.assertClose(math.fabs(self.record.get(self.yErrKey)),
#pybind11#                         math.sqrt(self.record.getCentroidErr()[1, 1]), rtol=1e-6)
#pybind11#        self.assertEqual(self.table.getShapeDefinition(), "c")
#pybind11#        self.assertEqual(self.shapeKey.get(self.record), self.record.getShape())
#pybind11#        self.assertClose(math.fabs(self.record.get(self.xxErrKey)),
#pybind11#                         math.sqrt(self.record.getShapeErr()[0, 0]), rtol=1e-6)
#pybind11#        self.assertClose(math.fabs(self.record.get(self.yyErrKey)),
#pybind11#                         math.sqrt(self.record.getShapeErr()[1, 1]), rtol=1e-6)
#pybind11#        self.assertClose(math.fabs(self.record.get(self.xyErrKey)),
#pybind11#                         math.sqrt(self.record.getShapeErr()[2, 2]), rtol=1e-6)
#pybind11#
#pybind11#    def testPersisted(self):
#pybind11#        self.table.definePsfFlux("a")
#pybind11#        self.table.defineCentroid("b")
#pybind11#        self.table.defineShape("c")
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as filename:
#pybind11#            self.catalog.writeFits(filename)
#pybind11#            catalog = lsst.afw.table.SourceCatalog.readFits(filename)
#pybind11#            table = catalog.getTable()
#pybind11#            record = catalog[0]
#pybind11#            # I'm using the keys from the non-persisted table.  They should work at least in the
#pybind11#            # current implementation
#pybind11#            self.assertEqual(table.getPsfFluxDefinition(), "a")
#pybind11#            self.assertEqual(record.get(self.fluxKey), record.getPsfFlux())
#pybind11#            self.assertEqual(record.get(self.fluxFlagKey), record.getPsfFluxFlag())
#pybind11#            self.assertEqual(table.getCentroidDefinition(), "b")
#pybind11#            centroid = self.centroidKey.get(self.record)
#pybind11#            self.assertEqual(centroid, record.getCentroid())
#pybind11#            self.assertClose(math.fabs(self.record.get(self.xErrKey)),
#pybind11#                             math.sqrt(self.record.getCentroidErr()[0, 0]), rtol=1e-6)
#pybind11#            self.assertClose(math.fabs(self.record.get(self.yErrKey)),
#pybind11#                             math.sqrt(self.record.getCentroidErr()[1, 1]), rtol=1e-6)
#pybind11#            shape = self.shapeKey.get(self.record)
#pybind11#            self.assertEqual(table.getShapeDefinition(), "c")
#pybind11#            self.assertEqual(shape, record.getShape())
#pybind11#            self.assertClose(math.fabs(self.record.get(self.xxErrKey)),
#pybind11#                             math.sqrt(self.record.getShapeErr()[0, 0]), rtol=1e-6)
#pybind11#            self.assertClose(math.fabs(self.record.get(self.yyErrKey)),
#pybind11#                             math.sqrt(self.record.getShapeErr()[1, 1]), rtol=1e-6)
#pybind11#            self.assertClose(math.fabs(self.record.get(self.xyErrKey)),
#pybind11#                             math.sqrt(self.record.getShapeErr()[2, 2]), rtol=1e-6)
#pybind11#
#pybind11#    def testCanonical2(self):
#pybind11#        self.table.definePsfFlux("a")
#pybind11#        self.table.defineCentroid("b")
#pybind11#        self.table.defineShape("c")
#pybind11#        self.checkCanonical()
#pybind11#
#pybind11#    def testPickle(self):
#pybind11#        p = pickle.dumps(self.catalog)
#pybind11#        new = pickle.loads(p)
#pybind11#
#pybind11#        self.assertEqual(self.catalog.schema.getNames(), new.schema.getNames())
#pybind11#        self.assertEqual(len(self.catalog), len(new))
#pybind11#        for r1, r2 in zip(self.catalog, new):
#pybind11#            for field in ("a_flux", "a_fluxSigma", "id"):  # Columns that are easy to test
#pybind11#                k1 = self.catalog.schema.find(field).getKey()
#pybind11#                k2 = new.schema.find(field).getKey()
#pybind11#                self.assertEqual(r1[k1], r2[k2])
#pybind11#
#pybind11#    def testCoordUpdate(self):
#pybind11#        self.table.defineCentroid("b")
#pybind11#        wcs = makeWcs()
#pybind11#        self.record.updateCoord(wcs)
#pybind11#        coord1 = self.record.getCoord()
#pybind11#        coord2 = wcs.pixelToSky(self.record.get(self.centroidKey))
#pybind11#        self.assertEqual(coord1, coord2)
#pybind11#
#pybind11#    def testSorting(self):
#pybind11#        self.assertFalse(self.catalog.isSorted())
#pybind11#        self.catalog.sort()
#pybind11#        self.assertTrue(self.catalog.isSorted())
#pybind11#        r = self.catalog.find(2)
#pybind11#        self.assertEqual(r["id"], 2)
#pybind11#        r = self.catalog.find(500)
#pybind11#        self.assertIsNone(r)
#pybind11#
#pybind11#    def testConversion(self):
#pybind11#        catalog1 = self.catalog.cast(lsst.afw.table.SourceCatalog)
#pybind11#        catalog2 = self.catalog.cast(lsst.afw.table.SimpleCatalog)
#pybind11#        catalog3 = self.catalog.cast(lsst.afw.table.SourceCatalog, deep=True)
#pybind11#        catalog4 = self.catalog.cast(lsst.afw.table.SimpleCatalog, deep=True)
#pybind11#        self.assertEqual(self.catalog.table, catalog1.table)
#pybind11#        self.assertEqual(self.catalog.table, catalog2.table)
#pybind11#        self.assertNotEqual(self.catalog.table, catalog3.table)
#pybind11#        self.assertNotEqual(self.catalog.table, catalog3.table)
#pybind11#        for r, r1, r2, r3, r4 in zip(self.catalog, catalog1, catalog2, catalog3, catalog4):
#pybind11#            self.assertEqual(r, r1)
#pybind11#            self.assertEqual(r, r2)
#pybind11#            self.assertNotEqual(r, r3)
#pybind11#            self.assertNotEqual(r, r4)
#pybind11#            self.assertEqual(r.getId(), r3.getId())
#pybind11#            self.assertEqual(r.getId(), r4.getId())
#pybind11#
#pybind11#    def testColumnView(self):
#pybind11#        cols1 = self.catalog.getColumnView()
#pybind11#        cols2 = self.catalog.columns
#pybind11#        self.assertIs(cols1, cols2)
#pybind11#        self.assertIsInstance(cols1, lsst.afw.table.SourceColumnView)
#pybind11#        self.table.definePsfFlux("a")
#pybind11#        self.table.defineCentroid("b")
#pybind11#        self.table.defineShape("c")
#pybind11#        self.assertFloatsEqual(cols2["a_flux"], cols2.getPsfFlux())
#pybind11#        self.assertFloatsEqual(cols2["a_fluxSigma"], cols2.getPsfFluxErr())
#pybind11#        self.assertFloatsEqual(cols2["b_x"], cols2.getX())
#pybind11#        self.assertFloatsEqual(cols2["b_y"], cols2.getY())
#pybind11#        self.assertFloatsEqual(cols2["c_xx"], cols2.getIxx())
#pybind11#        self.assertFloatsEqual(cols2["c_yy"], cols2.getIyy())
#pybind11#        self.assertFloatsEqual(cols2["c_xy"], cols2.getIxy())
#pybind11#
#pybind11#    def testForwarding(self):
#pybind11#        """Verify that Catalog forwards unknown methods to its table and/or columns."""
#pybind11#        self.table.definePsfFlux("a")
#pybind11#        self.table.defineCentroid("b")
#pybind11#        self.table.defineShape("c")
#pybind11#        self.assertFloatsEqual(self.catalog.columns["a_flux"], self.catalog["a_flux"])
#pybind11#        self.assertFloatsEqual(self.catalog.columns[self.fluxKey], self.catalog.get(self.fluxKey))
#pybind11#        self.assertFloatsEqual(self.catalog.columns.get(self.fluxKey), self.catalog.getPsfFlux())
#pybind11#        self.assertEqual(self.fluxKey, self.catalog.getPsfFluxKey())
#pybind11#        with self.assertRaises(AttributeError):
#pybind11#            self.catalog.foo()
#pybind11#
#pybind11#    def testBitsColumn(self):
#pybind11#        allBits = self.catalog.getBits()
#pybind11#        someBits = self.catalog.getBits(["a_flag", "c_flag"])
#pybind11#        self.assertEqual(allBits.getMask("a_flag"), 0x1)
#pybind11#        self.assertEqual(allBits.getMask("b_flag"), 0x2)
#pybind11#        self.assertEqual(allBits.getMask("c_flag"), 0x4)
#pybind11#        self.assertEqual(someBits.getMask(self.fluxFlagKey), 0x1)
#pybind11#        self.assertEqual(someBits.getMask(self.shapeFlagKey), 0x2)
#pybind11#        self.assertFloatsEqual((allBits.array & 0x1 != 0), self.catalog.columns["a_flag"])
#pybind11#        self.assertFloatsEqual((allBits.array & 0x2 != 0), self.catalog.columns["b_flag"])
#pybind11#        self.assertFloatsEqual((allBits.array & 0x4 != 0), self.catalog.columns["c_flag"])
#pybind11#        self.assertFloatsEqual((someBits.array & 0x1 != 0), self.catalog.columns["a_flag"])
#pybind11#        self.assertFloatsEqual((someBits.array & 0x2 != 0), self.catalog.columns["c_flag"])
#pybind11#
#pybind11#    def testCast(self):
#pybind11#        baseCat = self.catalog.cast(lsst.afw.table.BaseCatalog)
#pybind11#        baseCat.cast(lsst.afw.table.SourceCatalog)
#pybind11#
#pybind11#    def testFootprints(self):
#pybind11#        '''Test round-tripping Footprints (inc. HeavyFootprints) to FITS
#pybind11#        '''
#pybind11#        src1 = self.catalog.addNew()
#pybind11#        src2 = self.catalog.addNew()
#pybind11#        src3 = self.catalog.addNew()
#pybind11#        self.fillRecord(src1)
#pybind11#        self.fillRecord(src2)
#pybind11#        self.fillRecord(src3)
#pybind11#        src2.setParent(src1.getId())
#pybind11#
#pybind11#        W, H = 100, 100
#pybind11#        mim = lsst.afw.image.MaskedImageF(W, H)
#pybind11#        im = mim.getImage()
#pybind11#        msk = mim.getMask()
#pybind11#        var = mim.getVariance()
#pybind11#        for y in range(H):
#pybind11#            for x in range(W):
#pybind11#                im.set(x, y, y * 1e6 + x * 1e3)
#pybind11#                msk.set(x, y, (y << 8) | x)
#pybind11#                var.set(x, y, y * 1e2 + x)
#pybind11#        circ = lsst.afw.detection.Footprint(lsst.afw.geom.Point2I(50, 50), 20)
#pybind11#        heavy = lsst.afw.detection.makeHeavyFootprint(circ, mim)
#pybind11#        src2.setFootprint(heavy)
#pybind11#
#pybind11#        for i, src in enumerate(self.catalog):
#pybind11#            if src != src2:
#pybind11#                src.setFootprint(lsst.afw.detection.Footprint(lsst.afw.geom.Point2I(50, 50), 1+i*2))
#pybind11#
#pybind11#        # insert this HeavyFootprint into an otherwise blank image (for comparing the results)
#pybind11#        mim2 = lsst.afw.image.MaskedImageF(W, H)
#pybind11#        heavy.insert(mim2)
#pybind11#
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as fn:
#pybind11#            self.catalog.writeFits(fn)
#pybind11#
#pybind11#            cat2 = lsst.afw.table.SourceCatalog.readFits(fn)
#pybind11#            r2 = cat2[-2]
#pybind11#            f2 = r2.getFootprint()
#pybind11#            self.assertTrue(f2.isHeavy())
#pybind11#            h2 = lsst.afw.detection.cast_HeavyFootprintF(f2)
#pybind11#            mim3 = lsst.afw.image.MaskedImageF(W, H)
#pybind11#            h2.insert(mim3)
#pybind11#
#pybind11#            self.assertFalse(cat2[-1].getFootprint().isHeavy())
#pybind11#            self.assertFalse(cat2[-3].getFootprint().isHeavy())
#pybind11#            self.assertFalse(cat2[0].getFootprint().isHeavy())
#pybind11#            self.assertFalse(cat2[1].getFootprint().isHeavy())
#pybind11#            self.assertFalse(cat2[2].getFootprint().isHeavy())
#pybind11#
#pybind11#            if False:
#pybind11#                # Write out before-n-after FITS images
#pybind11#                for MI in [mim, mim2, mim3]:
#pybind11#                    f, fn2 = tempfile.mkstemp(prefix='testHeavyFootprint-', suffix='.fits')
#pybind11#                    os.close(f)
#pybind11#                    MI.writeFits(fn2)
#pybind11#                    print('wrote', fn2)
#pybind11#
#pybind11#            self.assertFloatsEqual(mim2.getImage().getArray(), mim3.getImage().getArray())
#pybind11#            self.assertFloatsEqual(mim2.getMask().getArray(), mim3.getMask().getArray())
#pybind11#            self.assertFloatsEqual(mim2.getVariance().getArray(), mim3.getVariance().getArray())
#pybind11#
#pybind11#            im3 = mim3.getImage()
#pybind11#            ma3 = mim3.getMask()
#pybind11#            va3 = mim3.getVariance()
#pybind11#            for y in range(H):
#pybind11#                for x in range(W):
#pybind11#                    if circ.contains(lsst.afw.geom.Point2I(x, y)):
#pybind11#                        self.assertEqual(im.get(x, y), im3.get(x, y))
#pybind11#                        self.assertEqual(msk.get(x, y), ma3.get(x, y))
#pybind11#                        self.assertEqual(var.get(x, y), va3.get(x, y))
#pybind11#                    else:
#pybind11#                        self.assertEqual(im3.get(x, y), 0.)
#pybind11#                        self.assertEqual(ma3.get(x, y), 0.)
#pybind11#                        self.assertEqual(va3.get(x, y), 0.)
#pybind11#
#pybind11#            cat3 = lsst.afw.table.SourceCatalog.readFits(fn, 0, lsst.afw.table.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
#pybind11#            for src in cat3:
#pybind11#                self.assertFalse(src.getFootprint().isHeavy())
#pybind11#            cat4 = lsst.afw.table.SourceCatalog.readFits(fn, 0, lsst.afw.table.SOURCE_IO_NO_FOOTPRINTS)
#pybind11#            for src in cat4:
#pybind11#                self.assertIsNone(src.getFootprint())
#pybind11#
#pybind11#            self.catalog.writeFits(fn, flags=lsst.afw.table.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
#pybind11#            cat5 = lsst.afw.table.SourceCatalog.readFits(fn)
#pybind11#            for src in cat5:
#pybind11#                self.assertFalse(src.getFootprint().isHeavy())
#pybind11#
#pybind11#            self.catalog.writeFits(fn, flags=lsst.afw.table.SOURCE_IO_NO_FOOTPRINTS)
#pybind11#            cat6 = lsst.afw.table.SourceCatalog.readFits(fn)
#pybind11#            for src in cat6:
#pybind11#                self.assertIsNone(src.getFootprint())
#pybind11#
#pybind11#    def testIdFactory(self):
#pybind11#        expId = int(1257198)
#pybind11#        reserved = 32
#pybind11#        factory = lsst.afw.table.IdFactory.makeSource(expId, reserved)
#pybind11#        id1 = factory()
#pybind11#        id2 = factory()
#pybind11#        self.assertEqual(id2 - id1, 1)
#pybind11#        factory.notify(0xFFFFFFFF)
#pybind11#        with self.assertRaises(lsst.pex.exceptions.LengthError):
#pybind11#            factory()
#pybind11#        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
#pybind11#            factory.notify(0x1FFFFFFFF)
#pybind11#        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
#pybind11#            lsst.afw.table.IdFactory.makeSource(0x1FFFFFFFF, reserved)
#pybind11#
#pybind11#    def testFamilies(self):
#pybind11#        self.catalog.sort()
#pybind11#        parents = self.catalog.getChildren(0)
#pybind11#        self.assertEqual(list(parents), list(self.catalog))
#pybind11#        parentKey = lsst.afw.table.SourceTable.getParentKey()
#pybind11#        for parent in parents:
#pybind11#            self.assertEqual(parent.get(parentKey), 0)
#pybind11#            for i in range(10):
#pybind11#                child = self.catalog.addNew()
#pybind11#                self.fillRecord(child)
#pybind11#                child.set(parentKey, parent.getId())
#pybind11#        for parent in parents:
#pybind11#            children, ids = self.catalog.getChildren(parent.getId(),
#pybind11#                                                     [record.getId() for record in self.catalog])
#pybind11#            self.assertEqual(len(children), 10)
#pybind11#            self.assertEqual(len(children), len(ids))
#pybind11#            for child, id in zip(children, ids):
#pybind11#                self.assertEqual(child.getParent(), parent.getId())
#pybind11#                self.assertEqual(child.getId(), id)
#pybind11#
#pybind11#        # Check detection of unsorted catalog
#pybind11#        self.catalog.sort(self.fluxKey)
#pybind11#        with self.assertRaises(AssertionError):
#pybind11#            self.catalog.getChildren(0)
#pybind11#        self.catalog.sort(parentKey)
#pybind11#        self.catalog.getChildren(0)  # Just care this succeeds
#pybind11#
#pybind11#    def testFitsReadBackwardsCompatibility(self):
#pybind11#        cat = lsst.afw.table.SourceCatalog.readFits(os.path.join(testPath, "data/empty-v0.fits"))
#pybind11#        self.assertTrue(cat.getPsfFluxSlot().isValid())
#pybind11#        self.assertTrue(cat.getApFluxSlot().isValid())
#pybind11#        self.assertTrue(cat.getInstFluxSlot().isValid())
#pybind11#        self.assertTrue(cat.getModelFluxSlot().isValid())
#pybind11#        self.assertTrue(cat.getCentroidSlot().isValid())
#pybind11#        self.assertTrue(cat.getShapeSlot().isValid())
#pybind11#        self.assertEqual(cat.getPsfFluxSlot().getMeasKey(), cat.schema.find("flux_psf").key)
#pybind11#        self.assertEqual(cat.getApFluxSlot().getMeasKey(), cat.schema.find("flux_sinc").key)
#pybind11#        self.assertEqual(cat.getInstFluxSlot().getMeasKey(), cat.schema.find("flux_naive").key)
#pybind11#        self.assertEqual(cat.getModelFluxSlot().getMeasKey(), cat.schema.find("cmodel_flux").key)
#pybind11#        self.assertEqual(cat.getCentroidSlot().getMeasKey().getX(), cat.schema.find("centroid_sdss_x").key)
#pybind11#        self.assertEqual(cat.getCentroidSlot().getMeasKey().getY(), cat.schema.find("centroid_sdss_y").key)
#pybind11#        self.assertEqual(cat.getShapeSlot().getMeasKey().getIxx(),
#pybind11#                         cat.schema.find("shape_hsm_moments_xx").key)
#pybind11#        self.assertEqual(cat.getShapeSlot().getMeasKey().getIyy(),
#pybind11#                         cat.schema.find("shape_hsm_moments_yy").key)
#pybind11#        self.assertEqual(cat.getShapeSlot().getMeasKey().getIxy(),
#pybind11#                         cat.schema.find("shape_hsm_moments_xy").key)
#pybind11#        self.assertEqual(cat.getPsfFluxSlot().getErrKey(), cat.schema.find("flux_psf_err").key)
#pybind11#        self.assertEqual(cat.getApFluxSlot().getErrKey(), cat.schema.find("flux_sinc_err").key)
#pybind11#        self.assertEqual(cat.getInstFluxSlot().getErrKey(), cat.schema.find("flux_naive_err").key)
#pybind11#        self.assertEqual(cat.getModelFluxSlot().getErrKey(), cat.schema.find("cmodel_flux_err").key)
#pybind11#        self.assertEqual(cat.getCentroidSlot().getErrKey(),
#pybind11#                         lsst.afw.table.CovarianceMatrix2fKey(cat.schema["centroid_sdss_err"], ["x", "y"]))
#pybind11#        self.assertEqual(cat.getShapeSlot().getErrKey(),
#pybind11#                         lsst.afw.table.CovarianceMatrix3fKey(cat.schema["shape_hsm_moments_err"], ["xx", "yy", "xy"]))
#pybind11#        self.assertEqual(cat.getPsfFluxSlot().getFlagKey(), cat.schema.find("flux_psf_flags").key)
#pybind11#        self.assertEqual(cat.getApFluxSlot().getFlagKey(), cat.schema.find("flux_sinc_flags").key)
#pybind11#        self.assertEqual(cat.getInstFluxSlot().getFlagKey(), cat.schema.find("flux_naive_flags").key)
#pybind11#        self.assertEqual(cat.getModelFluxSlot().getFlagKey(), cat.schema.find("cmodel_flux_flags").key)
#pybind11#        self.assertEqual(cat.getCentroidSlot().getFlagKey(), cat.schema.find("centroid_sdss_flags").key)
#pybind11#        self.assertEqual(cat.getShapeSlot().getFlagKey(), cat.schema.find("shape_hsm_moments_flags").key)
#pybind11#
#pybind11#    def testDM1083(self):
#pybind11#        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
#pybind11#        st = lsst.afw.table.SourceTable.make(schema)
#pybind11#        cat = lsst.afw.table.SourceCatalog(st)
#pybind11#        tmp = lsst.afw.table.SourceCatalog(cat.getTable())
#pybind11#        record = tmp.addNew()
#pybind11#        cat.extend(tmp)
#pybind11#        self.assertEqual(cat[0].getId(), record.getId())
#pybind11#        # check that the same record is in both catalogs (not a copy)
#pybind11#        record.setId(15)
#pybind11#        self.assertEqual(cat[0].getId(), record.getId())
#pybind11#
#pybind11#    def testSlotUndefine(self):
#pybind11#        """Test that we can correctly define and undefine a slot after a SourceTable has been created"""
#pybind11#        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
#pybind11#        key = schema.addField("a_flux", type=float, doc="flux field")
#pybind11#        table = lsst.afw.table.SourceTable.make(schema)
#pybind11#        table.definePsfFlux("a")
#pybind11#        self.assertEqual(table.getPsfFluxKey(), key)
#pybind11#        table.schema.getAliasMap().erase("slot_PsfFlux")
#pybind11#        self.assertFalse(table.getPsfFluxKey().isValid())
#pybind11#
#pybind11#    def testOldFootprintPersistence(self):
#pybind11#        """Test that we can still read SourceCatalogs with (Heavy)Footprints saved by an older
#pybind11#        version of the pipeline with a different format.
#pybind11#        """
#pybind11#        filename = os.path.join(testPath, "data", "old-footprint-persistence.fits")
#pybind11#        catalog1 = lsst.afw.table.SourceCatalog.readFits(filename)
#pybind11#        self.assertEqual(len(catalog1), 2)
#pybind11#        with self.assertRaises(KeyError):
#pybind11#            catalog1.schema.find("footprint")
#pybind11#        fp1 = catalog1[0].getFootprint()
#pybind11#        fp2 = catalog1[1].getFootprint()
#pybind11#        self.assertEqual(fp1.getArea(), 495)
#pybind11#        self.assertEqual(fp2.getArea(), 767)
#pybind11#        self.assertFalse(fp1.isHeavy())
#pybind11#        self.assertTrue(fp2.isHeavy())
#pybind11#        self.assertEqual(len(fp1.getSpans()), 29)
#pybind11#        self.assertEqual(len(fp2.getSpans()), 44)
#pybind11#        self.assertEqual(len(fp1.getPeaks()), 1)
#pybind11#        self.assertEqual(len(fp2.getPeaks()), 1)
#pybind11#        self.assertEqual(fp1.getBBox(),
#pybind11#                         lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(129, 2), lsst.afw.geom.Extent2I(25, 29)))
#pybind11#        self.assertEqual(fp2.getBBox(),
#pybind11#                         lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(1184, 2), lsst.afw.geom.Extent2I(78, 38)))
#pybind11#        hfp = lsst.afw.detection.cast_HeavyFootprintF(fp2)
#pybind11#        self.assertEqual(len(hfp.getImageArray()), fp2.getArea())
#pybind11#        self.assertEqual(len(hfp.getMaskArray()), fp2.getArea())
#pybind11#        self.assertEqual(len(hfp.getVarianceArray()), fp2.getArea())
#pybind11#        catalog2 = lsst.afw.table.SourceCatalog.readFits(filename, 0,
#pybind11#                                                         lsst.afw.table.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
#pybind11#        self.assertEqual(list(fp1.getSpans()), list(catalog2[0].getFootprint().getSpans()))
#pybind11#        self.assertEqual(list(fp2.getSpans()), list(catalog2[1].getFootprint().getSpans()))
#pybind11#        self.assertFalse(catalog2[1].getFootprint().isHeavy())
#pybind11#        catalog3 = lsst.afw.table.SourceCatalog.readFits(filename, 0,
#pybind11#                                                         lsst.afw.table.SOURCE_IO_NO_FOOTPRINTS)
#pybind11#        self.assertEqual(catalog3[0].getFootprint(), None)
#pybind11#        self.assertEqual(catalog3[1].getFootprint(), None)
#pybind11#
#pybind11#    def _testFluxSlot(self, slotName):
#pybind11#        """Demonstrate that we can create & use the named Flux slot."""
#pybind11#        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
#pybind11#        baseName = "afw_Test"
#pybind11#        fluxKey = schema.addField("%s_flux" % (baseName,), type=float, doc="flux")
#pybind11#        errKey = schema.addField("%s_fluxSigma" % (baseName,), type=float, doc="flux uncertainty")
#pybind11#        flagKey = schema.addField("%s_flag" % (baseName,), type="Flag", doc="flux flag")
#pybind11#        table = lsst.afw.table.SourceTable.make(schema)
#pybind11#
#pybind11#        # Initially, the slot is undefined.
#pybind11#        # For some reason this doesn't work with a context manager for assertRaises
#pybind11#        self.assertRaises(lsst.pex.exceptions.NotFoundError, getattr(table, "get%sDefinition" % (slotName,)))
#pybind11#
#pybind11#        # After definition, it maps to the keys defined above.
#pybind11#        getattr(table, "define%s" % (slotName,))(baseName)
#pybind11#        self.assertEqual(getattr(table, "get%sDefinition" % (slotName,))(), baseName)
#pybind11#        self.assertEqual(getattr(table, "get%sKey" % (slotName,))(), fluxKey)
#pybind11#        self.assertEqual(getattr(table, "get%sErrKey" % (slotName,))(), errKey)
#pybind11#        self.assertEqual(getattr(table, "get%sFlagKey" % (slotName,))(), flagKey)
#pybind11#
#pybind11#        # We should be able to retrieve arbitrary values set in records.
#pybind11#        record = table.makeRecord()
#pybind11#        flux, err, flag = 10.0, 1.0, False
#pybind11#        record.set(fluxKey, flux)
#pybind11#        record.set(errKey, err)
#pybind11#        record.set(flagKey, flag)
#pybind11#        self.assertEqual(getattr(record, "get%s" % (slotName,))(), flux)
#pybind11#        self.assertEqual(getattr(record, "get%sErr" % (slotName,))(), err)
#pybind11#        self.assertEqual(getattr(record, "get%sFlag" % (slotName,))(), flag)
#pybind11#
#pybind11#        # And we should be able to delete the slot, breaking the mapping.
#pybind11#        table.schema.getAliasMap().erase("slot_%s" % (slotName,))
#pybind11#        self.assertNotEqual(getattr(table, "get%sKey" % (slotName,))(), fluxKey)
#pybind11#        self.assertNotEqual(getattr(table, "get%sErrKey" % (slotName,))(), errKey)
#pybind11#        self.assertNotEqual(getattr(table, "get%sFlagKey" % (slotName,))(), flagKey)
#pybind11#
#pybind11#    def testFluxSlots(self):
#pybind11#        """Check that all the expected flux slots are present & correct."""
#pybind11#        for slotName in ["ApFlux", "CalibFlux", "InstFlux", "ModelFlux", "PsfFlux"]:
#pybind11#            self._testFluxSlot(slotName)
#pybind11#
#pybind11#        # But, of course, we should not accept a slot which hasn't be defined.
#pybind11#        with self.assertRaises(AttributeError):
#pybind11#            self._testFluxSlot("NotExtantFlux")
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

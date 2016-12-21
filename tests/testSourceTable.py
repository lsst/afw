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
from __future__ import absolute_import, division, print_function
import os
import unittest
import tempfile
import pickle
import math

from builtins import zip
from builtins import range
import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.table
import lsst.afw.geom
import lsst.afw.coord
import lsst.afw.image
import lsst.afw.detection

try:
    type(display)
except NameError:
    display = False

testPath = os.path.abspath(os.path.dirname(__file__))


def makeArray(size, dtype):
    return np.array(np.random.randn(*size), dtype=dtype)


def makeCov(size, dtype):
    m = np.array(np.random.randn(size, size), dtype=dtype)
    return np.dot(m, m.transpose())


def makeWcs():
    crval = lsst.afw.coord.Coord(lsst.afw.geom.Point2D(1.606631, 5.090329))
    crpix = lsst.afw.geom.Point2D(2036., 2000.)
    return lsst.afw.image.makeWcs(crval, crpix, 5.399452e-5, -1.30770e-5, 1.30770e-5, 5.399452e-5)


class SourceTableTestCase(lsst.utils.tests.TestCase):

    def fillRecord(self, record):
        record.set(self.fluxKey, np.random.randn())
        record.set(self.fluxErrKey, np.random.randn())
        record.set(self.centroidKey.getX(), np.random.randn())
        record.set(self.centroidKey.getY(), np.random.randn())
        record.set(self.xErrKey, np.random.randn())
        record.set(self.yErrKey, np.random.randn())
        record.set(self.shapeKey.getIxx(), np.random.randn())
        record.set(self.shapeKey.getIyy(), np.random.randn())
        record.set(self.shapeKey.getIxy(), np.random.randn())
        record.set(self.xxErrKey, np.random.randn())
        record.set(self.yyErrKey, np.random.randn())
        record.set(self.xyErrKey, np.random.randn())
        record.set(self.fluxFlagKey, np.random.randn() > 0)
        record.set(self.centroidFlagKey, np.random.randn() > 0)
        record.set(self.shapeFlagKey, np.random.randn() > 0)

    def setUp(self):
        np.random.seed(1)
        self.schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        self.fluxKey = self.schema.addField("a_flux", type="D")
        self.fluxErrKey = self.schema.addField("a_fluxSigma", type="D")
        self.fluxFlagKey = self.schema.addField("a_flag", type="Flag")

        # the meas field is added using a functor key, but the error is added
        # as scalars, as we lack a ResultKey functor as exists in meas_base
        self.centroidKey = lsst.afw.table.Point2DKey.addFields(self.schema,
                                                               "b", "", "pixel")
        self.xErrKey = self.schema.addField("b_xSigma", type="F")
        self.yErrKey = self.schema.addField("b_ySigma", type="F")
        self.centroidFlagKey = self.schema.addField("b_flag", type="Flag")

        self.shapeKey = lsst.afw.table.QuadrupoleKey.addFields(self.schema,
                                                               "c", "", lsst.afw.table.CoordinateType_PIXEL)
        self.xxErrKey = self.schema.addField("c_xxSigma", type="F")
        self.xyErrKey = self.schema.addField("c_xySigma", type="F")
        self.yyErrKey = self.schema.addField("c_yySigma", type="F")
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
                         math.sqrt(self.record.getCentroidErr()[0, 0]), rtol=1e-6)
        self.assertClose(math.fabs(self.record.get(self.yErrKey)),
                         math.sqrt(self.record.getCentroidErr()[1, 1]), rtol=1e-6)
        self.assertEqual(self.table.getShapeDefinition(), "c")
        self.assertEqual(self.shapeKey.get(self.record), self.record.getShape())
        self.assertClose(math.fabs(self.record.get(self.xxErrKey)),
                         math.sqrt(self.record.getShapeErr()[0, 0]), rtol=1e-6)
        self.assertClose(math.fabs(self.record.get(self.yyErrKey)),
                         math.sqrt(self.record.getShapeErr()[1, 1]), rtol=1e-6)
        self.assertClose(math.fabs(self.record.get(self.xyErrKey)),
                         math.sqrt(self.record.getShapeErr()[2, 2]), rtol=1e-6)

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
                             math.sqrt(self.record.getCentroidErr()[0, 0]), rtol=1e-6)
            self.assertClose(math.fabs(self.record.get(self.yErrKey)),
                             math.sqrt(self.record.getCentroidErr()[1, 1]), rtol=1e-6)
            shape = self.shapeKey.get(self.record)
            self.assertEqual(table.getShapeDefinition(), "c")
            self.assertEqual(shape, record.getShape())
            self.assertClose(math.fabs(self.record.get(self.xxErrKey)),
                             math.sqrt(self.record.getShapeErr()[0, 0]), rtol=1e-6)
            self.assertClose(math.fabs(self.record.get(self.yyErrKey)),
                             math.sqrt(self.record.getShapeErr()[1, 1]), rtol=1e-6)
            self.assertClose(math.fabs(self.record.get(self.xyErrKey)),
                             math.sqrt(self.record.getShapeErr()[2, 2]), rtol=1e-6)

    def testCanonical2(self):
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        self.checkCanonical()

    @unittest.skip("TODO support pickling with pybind11")
    def testPickle(self):
        p = pickle.dumps(self.catalog)
        new = pickle.loads(p)

        self.assertEqual(self.catalog.schema.getNames(), new.schema.getNames())
        self.assertEqual(len(self.catalog), len(new))
        for r1, r2 in zip(self.catalog, new):
            for field in ("a_flux", "a_fluxSigma", "id"):  # Columns that are easy to test
                k1 = self.catalog.schema.find(field).getKey()
                k2 = new.schema.find(field).getKey()
                self.assertEqual(r1[k1], r2[k2])

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
        self.assertTrue(self.catalog.isSorted())
        r = self.catalog.find(2)
        self.assertEqual(r["id"], 2)
        r = self.catalog.find(500)
        self.assertIsNone(r)

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
        self.assertEqual(cols1.schema, self.schema)
        self.assertEqual(cols1.table, self.table)
        cols2 = self.catalog.columns
        self.assertEqual(cols2.schema, self.schema)
        self.assertEqual(cols2.table, self.table)
        self.assertIsInstance(cols1, lsst.afw.table.SourceColumnView)
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        self.assertFloatsEqual(cols2["a_flux"], cols2.getPsfFlux())
        self.assertFloatsEqual(cols2["a_fluxSigma"], cols2.getPsfFluxErr())
        self.assertFloatsEqual(cols2["b_x"], cols2.getX())
        self.assertFloatsEqual(cols2["b_y"], cols2.getY())
        self.assertFloatsEqual(cols2["c_xx"], cols2.getIxx())
        self.assertFloatsEqual(cols2["c_yy"], cols2.getIyy())
        self.assertFloatsEqual(cols2["c_xy"], cols2.getIxy())

    def testForwarding(self):
        """Verify that Catalog forwards unknown methods to its table and/or columns."""
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        self.assertFloatsEqual(self.catalog.columns["a_flux"], self.catalog["a_flux"])
        self.assertFloatsEqual(self.catalog.columns[self.fluxKey], self.catalog.get(self.fluxKey))
        self.assertFloatsEqual(self.catalog.columns.get(self.fluxKey), self.catalog.getPsfFlux())
        self.assertEqual(self.fluxKey, self.catalog.getPsfFluxKey())
        with self.assertRaises(AttributeError):
            self.catalog.foo()

    def testBitsColumn(self):
        allBits = self.catalog.getBits()
        someBits = self.catalog.getBits(["a_flag", "c_flag"])
        self.assertEqual(allBits.getMask("a_flag"), 0x1)
        self.assertEqual(allBits.getMask("b_flag"), 0x2)
        self.assertEqual(allBits.getMask("c_flag"), 0x4)
        self.assertEqual(someBits.getMask(self.fluxFlagKey), 0x1)
        self.assertEqual(someBits.getMask(self.shapeFlagKey), 0x2)
        self.assertFloatsEqual((allBits.array & 0x1 != 0), self.catalog.columns["a_flag"])
        self.assertFloatsEqual((allBits.array & 0x2 != 0), self.catalog.columns["b_flag"])
        self.assertFloatsEqual((allBits.array & 0x4 != 0), self.catalog.columns["c_flag"])
        self.assertFloatsEqual((someBits.array & 0x1 != 0), self.catalog.columns["a_flag"])
        self.assertFloatsEqual((someBits.array & 0x2 != 0), self.catalog.columns["c_flag"])

    def testCast(self):
        baseCat = self.catalog.cast(lsst.afw.table.BaseCatalog)
        baseCat.cast(lsst.afw.table.SourceCatalog)

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

        W, H = 100, 100
        mim = lsst.afw.image.MaskedImageF(W, H)
        im = mim.getImage()
        msk = mim.getMask()
        var = mim.getVariance()
        for y in range(H):
            for x in range(W):
                im.set(x, y, y * 1e6 + x * 1e3)
                msk.set(x, y, (y << 8) | x)
                var.set(x, y, y * 1e2 + x)
        circ = lsst.afw.detection.Footprint(lsst.afw.geom.Point2I(50, 50), 20)
        heavy = lsst.afw.detection.makeHeavyFootprint(circ, mim)
        src2.setFootprint(heavy)

        for i, src in enumerate(self.catalog):
            if src != src2:
                src.setFootprint(lsst.afw.detection.Footprint(lsst.afw.geom.Point2I(50, 50), 1+i*2))

        # insert this HeavyFootprint into an otherwise blank image (for comparing the results)
        mim2 = lsst.afw.image.MaskedImageF(W, H)
        heavy.insert(mim2)

        with lsst.utils.tests.getTempFilePath(".fits") as fn:
            self.catalog.writeFits(fn)

            cat2 = lsst.afw.table.SourceCatalog.readFits(fn)
            r2 = cat2[-2]
            h2 = r2.getFootprint()
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
                    f, fn2 = tempfile.mkstemp(prefix='testHeavyFootprint-', suffix='.fits')
                    os.close(f)
                    MI.writeFits(fn2)
                    print('wrote', fn2)

            self.assertFloatsEqual(mim2.getImage().getArray(), mim3.getImage().getArray())
            self.assertFloatsEqual(mim2.getMask().getArray(), mim3.getMask().getArray())
            self.assertFloatsEqual(mim2.getVariance().getArray(), mim3.getVariance().getArray())

            im3 = mim3.getImage()
            ma3 = mim3.getMask()
            va3 = mim3.getVariance()
            for y in range(H):
                for x in range(W):
                    if circ.contains(lsst.afw.geom.Point2I(x, y)):
                        self.assertEqual(im.get(x, y), im3.get(x, y))
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
        id1 = factory()
        id2 = factory()
        self.assertEqual(id2 - id1, 1)
        factory.notify(0xFFFFFFFF)
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            factory()
        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
            factory.notify(0x1FFFFFFFF)
        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
            lsst.afw.table.IdFactory.makeSource(0x1FFFFFFFF, reserved)

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

        # Check detection of unsorted catalog
        self.catalog.sort(self.fluxKey)
        with self.assertRaises(AssertionError):
            self.catalog.getChildren(0)
        self.catalog.sort(parentKey)
        self.catalog.getChildren(0)  # Just care this succeeds

    def testFitsReadBackwardsCompatibility(self):
        cat = lsst.afw.table.SourceCatalog.readFits(os.path.join(testPath, "data/empty-v0.fits"))
        self.assertTrue(cat.getPsfFluxSlot().isValid())
        self.assertTrue(cat.getApFluxSlot().isValid())
        self.assertTrue(cat.getInstFluxSlot().isValid())
        self.assertTrue(cat.getModelFluxSlot().isValid())
        self.assertTrue(cat.getCentroidSlot().isValid())
        self.assertTrue(cat.getShapeSlot().isValid())
        self.assertEqual(cat.getPsfFluxSlot().getMeasKey(), cat.schema.find("flux_psf").key)
        self.assertEqual(cat.getApFluxSlot().getMeasKey(), cat.schema.find("flux_sinc").key)
        self.assertEqual(cat.getInstFluxSlot().getMeasKey(), cat.schema.find("flux_naive").key)
        self.assertEqual(cat.getModelFluxSlot().getMeasKey(), cat.schema.find("cmodel_flux").key)
        self.assertEqual(cat.getCentroidSlot().getMeasKey().getX(), cat.schema.find("centroid_sdss_x").key)
        self.assertEqual(cat.getCentroidSlot().getMeasKey().getY(), cat.schema.find("centroid_sdss_y").key)
        self.assertEqual(cat.getShapeSlot().getMeasKey().getIxx(),
                         cat.schema.find("shape_hsm_moments_xx").key)
        self.assertEqual(cat.getShapeSlot().getMeasKey().getIyy(),
                         cat.schema.find("shape_hsm_moments_yy").key)
        self.assertEqual(cat.getShapeSlot().getMeasKey().getIxy(),
                         cat.schema.find("shape_hsm_moments_xy").key)
        self.assertEqual(cat.getPsfFluxSlot().getErrKey(), cat.schema.find("flux_psf_err").key)
        self.assertEqual(cat.getApFluxSlot().getErrKey(), cat.schema.find("flux_sinc_err").key)
        self.assertEqual(cat.getInstFluxSlot().getErrKey(), cat.schema.find("flux_naive_err").key)
        self.assertEqual(cat.getModelFluxSlot().getErrKey(), cat.schema.find("cmodel_flux_err").key)
        self.assertEqual(cat.getCentroidSlot().getErrKey(),
                         lsst.afw.table.CovarianceMatrix2fKey(cat.schema["centroid_sdss_err"], ["x", "y"]))
        self.assertEqual(cat.getShapeSlot().getErrKey(),
                         lsst.afw.table.CovarianceMatrix3fKey(cat.schema["shape_hsm_moments_err"],
                         ["xx", "yy", "xy"]))
        self.assertEqual(cat.getPsfFluxSlot().getFlagKey(), cat.schema.find("flux_psf_flags").key)
        self.assertEqual(cat.getApFluxSlot().getFlagKey(), cat.schema.find("flux_sinc_flags").key)
        self.assertEqual(cat.getInstFluxSlot().getFlagKey(), cat.schema.find("flux_naive_flags").key)
        self.assertEqual(cat.getModelFluxSlot().getFlagKey(), cat.schema.find("cmodel_flux_flags").key)
        self.assertEqual(cat.getCentroidSlot().getFlagKey(), cat.schema.find("centroid_sdss_flags").key)
        self.assertEqual(cat.getShapeSlot().getFlagKey(), cat.schema.find("shape_hsm_moments_flags").key)

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
        key = schema.addField("a_flux", type=np.float64, doc="flux field")
        table = lsst.afw.table.SourceTable.make(schema)
        table.definePsfFlux("a")
        self.assertEqual(table.getPsfFluxKey(), key)
        table.schema.getAliasMap().erase("slot_PsfFlux")
        self.assertFalse(table.getPsfFluxKey().isValid())

    def testOldFootprintPersistence(self):
        """Test that we can still read SourceCatalogs with (Heavy)Footprints saved by an older
        version of the pipeline with a different format.
        """
        filename = os.path.join(testPath, "data", "old-footprint-persistence.fits")
        catalog1 = lsst.afw.table.SourceCatalog.readFits(filename)
        self.assertEqual(len(catalog1), 2)
        with self.assertRaises(KeyError):
            catalog1.schema.find("footprint")
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
                         lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(129, 2), lsst.afw.geom.Extent2I(25, 29)))
        self.assertEqual(fp2.getBBox(),
                         lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(1184, 2), lsst.afw.geom.Extent2I(78, 38)))
        hfp = lsst.afw.detection.HeavyFootprintF(fp2)
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

    def _testFluxSlot(self, slotName):
        """Demonstrate that we can create & use the named Flux slot."""
        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        baseName = "afw_Test"
        fluxKey = schema.addField("%s_flux" % (baseName,), type=np.float64, doc="flux")
        errKey = schema.addField("%s_fluxSigma" % (baseName,), type=np.float64, doc="flux uncertainty")
        flagKey = schema.addField("%s_flag" % (baseName,), type="Flag", doc="flux flag")
        table = lsst.afw.table.SourceTable.make(schema)

        # Initially, the slot is undefined.
        # For some reason this doesn't work with a context manager for assertRaises
        self.assertRaises(lsst.pex.exceptions.NotFoundError, getattr(table, "get%sDefinition" % (slotName,)))

        # After definition, it maps to the keys defined above.
        getattr(table, "define%s" % (slotName,))(baseName)
        self.assertEqual(getattr(table, "get%sDefinition" % (slotName,))(), baseName)
        self.assertEqual(getattr(table, "get%sKey" % (slotName,))(), fluxKey)
        self.assertEqual(getattr(table, "get%sErrKey" % (slotName,))(), errKey)
        self.assertEqual(getattr(table, "get%sFlagKey" % (slotName,))(), flagKey)

        # We should be able to retrieve arbitrary values set in records.
        record = table.makeRecord()
        flux, err, flag = 10.0, 1.0, False
        record.set(fluxKey, flux)
        record.set(errKey, err)
        record.set(flagKey, flag)
        self.assertEqual(getattr(record, "get%s" % (slotName,))(), flux)
        self.assertEqual(getattr(record, "get%sErr" % (slotName,))(), err)
        self.assertEqual(getattr(record, "get%sFlag" % (slotName,))(), flag)

        # And we should be able to delete the slot, breaking the mapping.
        table.schema.getAliasMap().erase("slot_%s" % (slotName,))
        self.assertNotEqual(getattr(table, "get%sKey" % (slotName,))(), fluxKey)
        self.assertNotEqual(getattr(table, "get%sErrKey" % (slotName,))(), errKey)
        self.assertNotEqual(getattr(table, "get%sFlagKey" % (slotName,))(), flagKey)

    def testFluxSlots(self):
        """Check that all the expected flux slots are present & correct."""
        for slotName in ["ApFlux", "CalibFlux", "InstFlux", "ModelFlux", "PsfFlux"]:
            self._testFluxSlot(slotName)

        # But, of course, we should not accept a slot which hasn't be defined.
        with self.assertRaises(AttributeError):
            self._testFluxSlot("NotExtantFlux")


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

#
# LSST Data Management System
# Copyright 2008-2017 LSST Corporation.
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
import tempfile
import pickle
import math

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.geom
import lsst.afw.table
import lsst.afw.geom
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
    crval = lsst.geom.SpherePoint(1.606631 * lsst.geom.degrees,
                                  5.090329 * lsst.geom.degrees)
    crpix = lsst.geom.Point2D(2036.0, 2000.0)
    cdMatrix = np.array([5.399452e-5, -1.30770e-5, 1.30770e-5, 5.399452e-5])
    cdMatrix.shape = (2, 2)
    return lsst.afw.geom.makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix)


class SourceTableTestCase(lsst.utils.tests.TestCase):

    def fillRecord(self, record):
        record.set(self.instFluxKey, np.random.randn())
        record.set(self.instFluxErrKey, np.random.randn())
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
        self.instFluxKey = self.schema.addField("a_instFlux", type="D")
        self.instFluxErrKey = self.schema.addField("a_instFluxErr", type="D")
        self.fluxFlagKey = self.schema.addField("a_flag", type="Flag")

        # the meas field is added using a functor key, but the error is added
        # as scalars, as we lack a ResultKey functor as exists in meas_base
        self.centroidKey = lsst.afw.table.Point2DKey.addFields(
            self.schema, "b", "", "pixel")
        self.xErrKey = self.schema.addField("b_xErr", type="F")
        self.yErrKey = self.schema.addField("b_yErr", type="F")
        self.centroidFlagKey = self.schema.addField("b_flag", type="Flag")

        self.shapeKey = lsst.afw.table.QuadrupoleKey.addFields(
            self.schema, "c", "", lsst.afw.table.CoordinateType.PIXEL)
        self.xxErrKey = self.schema.addField("c_xxErr", type="F")
        self.xyErrKey = self.schema.addField("c_xyErr", type="F")
        self.yyErrKey = self.schema.addField("c_yyErr", type="F")
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
        self.assertEqual(self.record.get(self.instFluxKey),
                         self.record.getPsfInstFlux())
        self.assertEqual(self.record.get(self.fluxFlagKey),
                         self.record.getPsfFluxFlag())
        self.assertEqual(self.table.getCentroidDefinition(), "b")
        self.assertEqual(self.centroidKey.get(self.record),
                         self.record.getCentroid())
        self.assertFloatsAlmostEqual(
            math.fabs(self.record.get(self.xErrKey)),
            math.sqrt(self.record.getCentroidErr()[0, 0]), rtol=1e-6)
        self.assertFloatsAlmostEqual(
            math.fabs(self.record.get(self.yErrKey)),
            math.sqrt(self.record.getCentroidErr()[1, 1]), rtol=1e-6)
        self.assertEqual(self.table.getShapeDefinition(), "c")
        self.assertEqual(self.shapeKey.get(self.record),
                         self.record.getShape())
        self.assertFloatsAlmostEqual(
            math.fabs(self.record.get(self.xxErrKey)),
            math.sqrt(self.record.getShapeErr()[0, 0]), rtol=1e-6)
        self.assertFloatsAlmostEqual(
            math.fabs(self.record.get(self.yyErrKey)),
            math.sqrt(self.record.getShapeErr()[1, 1]), rtol=1e-6)
        self.assertFloatsAlmostEqual(
            math.fabs(self.record.get(self.xyErrKey)),
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
            self.assertEqual(record.get(self.instFluxKey), record.getPsfInstFlux())
            self.assertEqual(record.get(self.fluxFlagKey),
                             record.getPsfFluxFlag())
            self.assertEqual(table.getCentroidDefinition(), "b")
            centroid = self.centroidKey.get(self.record)
            self.assertEqual(centroid, record.getCentroid())
            self.assertFloatsAlmostEqual(
                math.fabs(self.record.get(self.xErrKey)),
                math.sqrt(self.record.getCentroidErr()[0, 0]), rtol=1e-6)
            self.assertFloatsAlmostEqual(
                math.fabs(self.record.get(self.yErrKey)),
                math.sqrt(self.record.getCentroidErr()[1, 1]), rtol=1e-6)
            shape = self.shapeKey.get(self.record)
            self.assertEqual(table.getShapeDefinition(), "c")
            self.assertEqual(shape, record.getShape())
            self.assertFloatsAlmostEqual(
                math.fabs(self.record.get(self.xxErrKey)),
                math.sqrt(self.record.getShapeErr()[0, 0]), rtol=1e-6)
            self.assertFloatsAlmostEqual(
                math.fabs(self.record.get(self.yyErrKey)),
                math.sqrt(self.record.getShapeErr()[1, 1]), rtol=1e-6)
            self.assertFloatsAlmostEqual(
                math.fabs(self.record.get(self.xyErrKey)),
                math.sqrt(self.record.getShapeErr()[2, 2]), rtol=1e-6)

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
            # Columns that are easy to test
            for field in ("a_instFlux", "a_instFluxErr", "id"):
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
        for r, r1, r2, r3, r4 in \
                zip(self.catalog, catalog1, catalog2, catalog3, catalog4):
            self.assertEqual(r, r1)
            self.assertEqual(r, r2)
            self.assertNotEqual(r, r3)
            self.assertNotEqual(r, r4)
            self.assertEqual(r.getId(), r3.getId())
            self.assertEqual(r.getId(), r4.getId())

    def testColumnView(self):
        cols1 = self.catalog.getColumnView()
        cols2 = self.catalog.columns
        self.assertIs(cols1, cols2)
        self.assertIsInstance(cols1, lsst.afw.table.SourceColumnView)
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        self.assertFloatsEqual(cols2["a_instFlux"], cols2.getPsfInstFlux())
        self.assertFloatsEqual(cols2["a_instFluxErr"], cols2.getPsfInstFluxErr())
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
        self.assertFloatsEqual(self.catalog.columns["a_instFlux"],
                               self.catalog["a_instFlux"])
        self.assertFloatsEqual(self.catalog.columns[self.instFluxKey],
                               self.catalog.get(self.instFluxKey))
        self.assertFloatsEqual(self.catalog.columns.get(self.instFluxKey),
                               self.catalog.getPsfInstFlux())
        self.assertEqual(self.instFluxKey, self.catalog.getPsfFluxSlot().getMeasKey())
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
        np.testing.assert_array_equal(
            (allBits.array & 0x1 != 0), self.catalog.columns["a_flag"])
        np.testing.assert_array_equal(
            (allBits.array & 0x2 != 0), self.catalog.columns["b_flag"])
        np.testing.assert_array_equal(
            (allBits.array & 0x4 != 0), self.catalog.columns["c_flag"])
        np.testing.assert_array_equal(
            (someBits.array & 0x1 != 0), self.catalog.columns["a_flag"])
        np.testing.assert_array_equal(
            (someBits.array & 0x2 != 0), self.catalog.columns["c_flag"])

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
        x, y = np.meshgrid(np.arange(W, dtype=int), np.arange(H, dtype=int))
        im.array[:] = y*1E6 + x*1E3
        msk.array[:] = (y << 8) | x
        var.array[:] = y*1E2 + x
        spanSet = lsst.afw.geom.SpanSet.fromShape(20).shiftedBy(50, 50)
        circ = lsst.afw.detection.Footprint(spanSet)
        heavy = lsst.afw.detection.makeHeavyFootprint(circ, mim)
        src2.setFootprint(heavy)

        for i, src in enumerate(self.catalog):
            if src != src2:
                spanSet = lsst.afw.geom.SpanSet.fromShape(
                    1+i*2).shiftedBy(50, 50)
                src.setFootprint(lsst.afw.detection.Footprint(spanSet))

        # insert this HeavyFootprint into an otherwise blank image (for comparing the results)
        mim2 = lsst.afw.image.MaskedImageF(W, H)
        heavy.insert(mim2)

        with lsst.utils.tests.getTempFilePath(".fits") as fn:
            self.catalog.writeFits(fn)

            cat2 = lsst.afw.table.SourceCatalog.readFits(fn)
            r2 = cat2[-2]
            h2 = r2.getFootprint()
            self.assertTrue(h2.isHeavy())
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
                    f, fn2 = tempfile.mkstemp(prefix='testHeavyFootprint-',
                                              suffix='.fits')
                    os.close(f)
                    MI.writeFits(fn2)
                    print('wrote', fn2)

            self.assertFloatsEqual(mim2.getImage().getArray(),
                                   mim3.getImage().getArray())
            self.assertFloatsEqual(mim2.getMask().getArray(),
                                   mim3.getMask().getArray())
            self.assertFloatsEqual(mim2.getVariance().getArray(),
                                   mim3.getVariance().getArray())

            im3 = mim3.getImage()
            ma3 = mim3.getMask()
            va3 = mim3.getVariance()
            for y in range(H):
                for x in range(W):
                    if circ.contains(lsst.geom.Point2I(x, y)):
                        self.assertEqual(im[x, y, lsst.afw.image.PARENT], im3[x, y, lsst.afw.image.PARENT])
                        self.assertEqual(msk[x, y, lsst.afw.image.PARENT], ma3[x, y, lsst.afw.image.PARENT])
                        self.assertEqual(var[x, y, lsst.afw.image.PARENT], va3[x, y, lsst.afw.image.PARENT])
                    else:
                        self.assertEqual(im3[x, y, lsst.afw.image.PARENT], 0.)
                        self.assertEqual(ma3[x, y, lsst.afw.image.PARENT], 0.)
                        self.assertEqual(va3[x, y, lsst.afw.image.PARENT], 0.)

            cat3 = lsst.afw.table.SourceCatalog.readFits(
                fn, flags=lsst.afw.table.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
            for src in cat3:
                self.assertFalse(src.getFootprint().isHeavy())
            cat4 = lsst.afw.table.SourceCatalog.readFits(
                fn, flags=lsst.afw.table.SOURCE_IO_NO_FOOTPRINTS)
            for src in cat4:
                self.assertIsNone(src.getFootprint())

            self.catalog.writeFits(
                fn, flags=lsst.afw.table.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
            cat5 = lsst.afw.table.SourceCatalog.readFits(fn)
            for src in cat5:
                self.assertFalse(src.getFootprint().isHeavy())

            self.catalog.writeFits(
                fn, flags=lsst.afw.table.SOURCE_IO_NO_FOOTPRINTS)
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
            children, ids = self.catalog.getChildren(
                parent.getId(), [record.getId() for record in self.catalog])
            self.assertEqual(len(children), 10)
            self.assertEqual(len(children), len(ids))
            for child, id in zip(children, ids):
                self.assertEqual(child.getParent(), parent.getId())
                self.assertEqual(child.getId(), id)

        # Check detection of unsorted catalog
        self.catalog.sort(self.instFluxKey)
        with self.assertRaises(AssertionError):
            self.catalog.getChildren(0)
        self.catalog.sort(parentKey)
        self.catalog.getChildren(0)  # Just care this succeeds

    def testFitsReadVersion0Compatibility(self):
        cat = lsst.afw.table.SourceCatalog.readFits(
            os.path.join(testPath, "data/empty-v0.fits"))
        self.assertTrue(cat.getPsfFluxSlot().isValid())
        self.assertTrue(cat.getApFluxSlot().isValid())
        self.assertTrue(cat.getGaussianFluxSlot().isValid())
        self.assertTrue(cat.getModelFluxSlot().isValid())
        self.assertTrue(cat.getCentroidSlot().isValid())
        self.assertTrue(cat.getShapeSlot().isValid())
        self.assertEqual(cat.getPsfFluxSlot().getMeasKey(),
                         cat.schema.find("flux_psf").key)
        self.assertEqual(cat.getApFluxSlot().getMeasKey(),
                         cat.schema.find("flux_sinc").key)
        self.assertEqual(cat.getGaussianFluxSlot().getMeasKey(),
                         cat.schema.find("flux_naive").key)
        self.assertEqual(cat.getModelFluxSlot().getMeasKey(),
                         cat.schema.find("cmodel_flux").key)
        self.assertEqual(cat.getCentroidSlot().getMeasKey().getX(),
                         cat.schema.find("centroid_sdss_x").key)
        self.assertEqual(cat.getCentroidSlot().getMeasKey().getY(),
                         cat.schema.find("centroid_sdss_y").key)
        self.assertEqual(cat.getShapeSlot().getMeasKey().getIxx(),
                         cat.schema.find("shape_hsm_moments_xx").key)
        self.assertEqual(cat.getShapeSlot().getMeasKey().getIyy(),
                         cat.schema.find("shape_hsm_moments_yy").key)
        self.assertEqual(cat.getShapeSlot().getMeasKey().getIxy(),
                         cat.schema.find("shape_hsm_moments_xy").key)
        self.assertEqual(cat.getPsfFluxSlot().getErrKey(),
                         cat.schema.find("flux_psf_err").key)
        self.assertEqual(cat.getApFluxSlot().getErrKey(),
                         cat.schema.find("flux_sinc_err").key)
        self.assertEqual(cat.getGaussianFluxSlot().getErrKey(),
                         cat.schema.find("flux_naive_err").key)
        self.assertEqual(cat.getModelFluxSlot().getErrKey(),
                         cat.schema.find("cmodel_flux_err").key)
        self.assertEqual(
            cat.getCentroidSlot().getErrKey(),
            lsst.afw.table.CovarianceMatrix2fKey(
                cat.schema["centroid_sdss_err"],
                ["x", "y"]))
        self.assertEqual(
            cat.getShapeSlot().getErrKey(),
            lsst.afw.table.CovarianceMatrix3fKey(
                cat.schema["shape_hsm_moments_err"],
                ["xx", "yy", "xy"]))
        self.assertEqual(cat.getPsfFluxSlot().getFlagKey(),
                         cat.schema.find("flux_psf_flags").key)
        self.assertEqual(cat.getApFluxSlot().getFlagKey(),
                         cat.schema.find("flux_sinc_flags").key)
        self.assertEqual(cat.getGaussianFluxSlot().getFlagKey(),
                         cat.schema.find("flux_naive_flags").key)
        self.assertEqual(cat.getModelFluxSlot().getFlagKey(),
                         cat.schema.find("cmodel_flux_flags").key)
        self.assertEqual(cat.getCentroidSlot().getFlagKey(),
                         cat.schema.find("centroid_sdss_flags").key)
        self.assertEqual(cat.getShapeSlot().getFlagKey(),
                         cat.schema.find("shape_hsm_moments_flags").key)

    def testFitsReadVersion1Compatibility(self):
        """Test reading of catalogs with version 1 schema

        Version 2 catalogs (the current version) provide aliases to
        fields whose names end in Sigma: xErr -> xSigma for any x
        """
        cat = lsst.afw.table.SourceCatalog.readFits(
            os.path.join(testPath, "data", "sourceTable-v1.fits"))
        self.assertEqual(cat.schema["a_fluxSigma"].asKey(), cat.schema["a_fluxErr"].asKey())
        self.assertEqual(
            cat.getCentroidSlot().getErrKey(),
            lsst.afw.table.CovarianceMatrix2fKey(
                cat.schema["slot_Centroid"],
                ["x", "y"]))
        self.assertEqual(
            cat.getShapeSlot().getErrKey(),
            lsst.afw.table.CovarianceMatrix3fKey(
                cat.schema["slot_Shape"],
                ["xx", "yy", "xy"]))

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
        key = schema.addField("a_instFlux", type=np.float64, doc="flux field")
        table = lsst.afw.table.SourceTable.make(schema)
        table.definePsfFlux("a")
        self.assertEqual(table.getPsfFluxSlot().getMeasKey(), key)
        table.schema.getAliasMap().erase("slot_PsfFlux")
        self.assertFalse(table.getPsfFluxSlot().isValid())

    def testOldFootprintPersistence(self):
        """Test that we can still read SourceCatalogs with (Heavy)Footprints saved by an older
        version of the pipeline with a different format.
        """
        filename = os.path.join(testPath, "data",
                                "old-footprint-persistence.fits")
        catalog1 = lsst.afw.table.SourceCatalog.readFits(filename)
        self.assertEqual(len(catalog1), 2)
        with self.assertRaises(LookupError):
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
                         lsst.geom.Box2I(lsst.geom.Point2I(129, 2),
                                         lsst.geom.Extent2I(25, 29)))
        self.assertEqual(fp2.getBBox(),
                         lsst.geom.Box2I(lsst.geom.Point2I(1184, 2),
                                         lsst.geom.Extent2I(78, 38)))
        hfp = lsst.afw.detection.HeavyFootprintF(fp2)
        self.assertEqual(len(hfp.getImageArray()), fp2.getArea())
        self.assertEqual(len(hfp.getMaskArray()), fp2.getArea())
        self.assertEqual(len(hfp.getVarianceArray()), fp2.getArea())
        catalog2 = lsst.afw.table.SourceCatalog.readFits(
            filename, flags=lsst.afw.table.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
        self.assertEqual(list(fp1.getSpans()),
                         list(catalog2[0].getFootprint().getSpans()))
        self.assertEqual(list(fp2.getSpans()),
                         list(catalog2[1].getFootprint().getSpans()))
        self.assertFalse(catalog2[1].getFootprint().isHeavy())
        catalog3 = lsst.afw.table.SourceCatalog.readFits(
            filename, flags=lsst.afw.table.SOURCE_IO_NO_FOOTPRINTS)
        self.assertEqual(catalog3[0].getFootprint(), None)
        self.assertEqual(catalog3[1].getFootprint(), None)

    def _testFluxSlot(self, slotName):
        """Demonstrate that we can create & use the named Flux slot."""
        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        baseName = "afw_Test"
        instFluxKey = schema.addField("%s_instFlux" % (baseName,),
                                      type=np.float64, doc="flux")
        errKey = schema.addField("%s_instFluxErr" % (baseName,),
                                 type=np.float64, doc="flux uncertainty")
        flagKey = schema.addField("%s_flag" % (baseName,),
                                  type="Flag", doc="flux flag")
        table = lsst.afw.table.SourceTable.make(schema)

        # Initially, the slot is undefined.
        self.assertFalse(getattr(table, "get%sSlot" % (slotName,))().isValid())

        # After definition, it maps to the keys defined above.
        getattr(table, "define%s" % (slotName,))(baseName)
        self.assertTrue(getattr(table, "get%sSlot" % (slotName,))().isValid())
        self.assertEqual(getattr(table, "get%sSlot" % (slotName,))().getMeasKey(), instFluxKey)
        self.assertEqual(getattr(table, "get%sSlot" % (slotName,))().getErrKey(), errKey)
        self.assertEqual(getattr(table, "get%sSlot" % (slotName,))().getFlagKey(), flagKey)

        # We should be able to retrieve arbitrary values set in records.
        record = table.makeRecord()
        instFlux, err, flag = 10.0, 1.0, False
        record.set(instFluxKey, instFlux)
        record.set(errKey, err)
        record.set(flagKey, flag)
        instFluxName = slotName.replace("Flux", "InstFlux")
        self.assertEqual(getattr(record, "get%s" % (instFluxName,))(), instFlux)
        self.assertEqual(getattr(record, "get%sErr" % (instFluxName,))(), err)
        self.assertEqual(getattr(record, "get%sFlag" % (slotName,))(), flag)

        # And we should be able to delete the slot, breaking the mapping.
        table.schema.getAliasMap().erase("slot_%s" % (slotName,))
        self.assertFalse(getattr(table, "get%sSlot" % (slotName,))().isValid())
        self.assertNotEqual(getattr(table, "get%sSlot" % (slotName,))().getMeasKey(), instFluxKey)
        self.assertNotEqual(getattr(table, "get%sSlot" % (slotName,))().getErrKey(), errKey)
        self.assertNotEqual(getattr(table, "get%sSlot" % (slotName,))().getFlagKey(), flagKey)

    def testFluxSlots(self):
        """Check that all the expected flux slots are present & correct."""
        for slotName in ["ApFlux", "CalibFlux", "GaussianFlux", "ModelFlux",
                         "PsfFlux"]:
            self._testFluxSlot(slotName)

        # But, of course, we should not accept a slot which hasn't be defined.
        with self.assertRaises(AttributeError):
            self._testFluxSlot("NotExtantFlux")

    def testStr(self):
        """Check that the str() produced on a catalog contains expected things."""
        string = str(self.catalog)
        for field in ('id', 'coord_ra', 'coord_dec'):
            self.assertIn(field, string)

    def testRepr(self):
        """Check that the repr() produced on a catalog contains expected things."""
        string = repr(self.catalog)
        self.assertIn(str(type(self.catalog)), string)
        for field in ('id', 'coord_ra', 'coord_dec'):
            self.assertIn(field, string)

    def testStrNonContiguous(self):
        """Check that str() doesn't fail on non-contiguous tables."""
        del self.catalog[1]
        string = str(self.catalog)
        self.assertIn('Non-contiguous afw.Catalog of 2 rows.', string)
        for field in ('id', 'coord_ra', 'coord_dec'):
            self.assertIn(field, string)

    def testRecordStr(self):
        """Test that str(record) contains expected things."""
        string = str(self.catalog[0])
        for field in ('id: 50', 'coord_ra: nan', 'coord_dec: nan'):
            self.assertIn(field, string)

    def testRecordRepr(self):
        """Test that repr(record) contains expected things."""
        string = repr(self.catalog[0])
        self.assertIn(str(type(self.catalog[0])), string)
        for field in ('id: 50', 'coord_ra: nan', 'coord_dec: nan'):
            self.assertIn(field, string)

    def testGetNonContiguous(self):
        """Check that we can index on non-contiguous tables"""
        # Make a non-contiguous catalog
        nonContiguous = type(self.catalog)(self.catalog.table)
        for rr in reversed(self.catalog):
            nonContiguous.append(rr)
        num = len(self.catalog)
        # Check assumptions
        self.assertFalse(nonContiguous.isContiguous())  # We managed to produce a non-contiguous catalog
        self.assertEqual(len(set(self.catalog["id"])), num)  # ID values are unique
        # Indexing with boolean array
        select = np.zeros(num, dtype=bool)
        select[1] = True
        self.assertEqual(nonContiguous[np.flip(select, 0)]["id"], self.catalog[select]["id"])
        # Extracting a number column
        column = "a_instFlux"
        array = nonContiguous[column]
        self.assertFloatsEqual(np.flip(array, 0), self.catalog[column])
        with self.assertRaises(ValueError):
            array[1] = 1.2345  # Should be immutable
        # Extracting a flag column
        column = "a_flag"
        array = nonContiguous[column]
        np.testing.assert_equal(np.flip(array, 0), self.catalog[column])
        with self.assertRaises(ValueError):
            array[1] = True  # Should be immutable


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

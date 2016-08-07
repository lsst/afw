#!/usr/bin/env python2
from __future__ import absolute_import, division
from __future__ import print_function

#
# LSST Data Management System
# Copyright 2008-2014 AURA/LSST
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
Tests for table.SimpleTable

Run with:
   ./testSimpleTable.py
or
   python
   >>> import testSimpleTable; testSimpleTable.run()
"""
import os.path
import unittest
import numpy

try:
    import pyfits
except ImportError:
    pyfits = None
    print("WARNING: pyfits not available; some tests will not be run")

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.daf.base
import lsst.afw.table
import lsst.afw.geom
import lsst.afw.coord
import lsst.afw.fits

numpy.random.seed(1)

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def makeArray(size, dtype):
    return numpy.array(numpy.random.randn(size), dtype=dtype)

def makeCov(size, dtype):
    m = numpy.array(numpy.random.randn(size, size), dtype=dtype)
    r = numpy.dot(m, m.transpose())  # not quite symmetric for single-precision on some platforms
    for i in range(r.shape[0]):
        for j in range(i):
            r[i,j] = r[j,i]
    return r

class SimpleTableTestCase(lsst.utils.tests.TestCase):

    def checkScalarAccessors(self, record, key, name, value1, value2):
        fastSetter = getattr(record, "set" + key.getTypeString())
        fastGetter = getattr(record, "get" + key.getTypeString())
        record[key] = value1
        self.assertEqual(record[key], value1)
        self.assertEqual(record.get(key), value1)
        self.assertEqual(record[name], value1)
        self.assertEqual(record.get(name), value1)
        self.assertEqual(fastGetter(key), value1)
        record.set(key, value2)
        self.assertEqual(record[key], value2)
        self.assertEqual(record.get(key), value2)
        self.assertEqual(record[name], value2)
        self.assertEqual(record.get(name), value2)
        self.assertEqual(fastGetter(key), value2)
        record[name] = value1
        self.assertEqual(record[key], value1)
        self.assertEqual(record.get(key), value1)
        self.assertEqual(record[name], value1)
        self.assertEqual(record.get(name), value1)
        self.assertEqual(fastGetter(key), value1)
        record.set(name, value2)
        self.assertEqual(record[key], value2)
        self.assertEqual(record.get(key), value2)
        self.assertEqual(record[name], value2)
        self.assertEqual(record.get(name), value2)
        self.assertEqual(fastGetter(key), value2)
        fastSetter(key, value1)
        self.assertEqual(record[key], value1)
        self.assertEqual(record.get(key), value1)
        self.assertEqual(record[name], value1)
        self.assertEqual(record.get(name), value1)
        self.assertEqual(fastGetter(key), value1)
        self.assert_(key.subfields is None)

    def checkArrayAccessors(self, record, key, name, value):
        fastSetter = getattr(record, "set" + key.getTypeString())
        fastGetter = getattr(record, "get" + key.getTypeString())
        record.set(key, value)
        self.assert_(numpy.all(record.get(key) == value))
        record.set(name, value)
        self.assert_(numpy.all(record.get(name) == value))
        fastSetter(key, value)
        self.assert_(numpy.all(fastGetter(key) == value))

    def testRecordAccess(self):
        schema = lsst.afw.table.Schema()
        k0 = schema.addField("f0", type="U")
        k1 = schema.addField("f1", type="I")
        k2 = schema.addField("f2", type="L")
        k3 = schema.addField("f3", type="F")
        k4 = schema.addField("f4", type="D")
        k10b = schema.addField("f10b", type="ArrayU", size=2)
        k10a = schema.addField("f10a", type="ArrayI", size=3)
        k10 = schema.addField("f10", type="ArrayF", size=4)
        k11 = schema.addField("f11", type="ArrayD", size=5)
        k18 = schema.addField("f18", type="Angle")
        schema.addField("f20", type="String", size=4)
        table = lsst.afw.table.BaseTable.make(schema)
        record = table.makeRecord()
        self.assertEqual(record[k1], 0)
        self.assertEqual(record[k2], 0)
        self.assert_(numpy.isnan(record[k3]))
        self.assert_(numpy.isnan(record[k4]))
        self.checkScalarAccessors(record, k0, "f0", 5, 6)
        self.checkScalarAccessors(record, k1, "f1", 2, 3)
        self.checkScalarAccessors(record, k2, "f2", 2, 3)
        self.checkScalarAccessors(record, k3, "f3", 2.5, 3.5)
        self.checkScalarAccessors(record, k4, "f4", 2.5, 3.5)
        self.checkArrayAccessors(record, k10b, "f10b", makeArray(k10b.getSize(), dtype=numpy.uint16))
        self.checkArrayAccessors(record, k10a, "f10a", makeArray(k10a.getSize(), dtype=numpy.int32))
        self.checkArrayAccessors(record, k10, "f10", makeArray(k10.getSize(), dtype=numpy.float32))
        self.checkArrayAccessors(record, k11, "f11", makeArray(k11.getSize(), dtype=numpy.float64))
        for k in (k10, k11): self.assertEqual(k.subfields, tuple(range(k.getSize())))
        sub1 = k11.slice(1, 3)
        sub2 = k11[0:2]
        self.assertClose(record.get(sub1), record.get(k11)[1:3], rtol=0, atol=0)
        self.assertClose(record.get(sub2), record.get(k11)[0:2], rtol=0, atol=0)
        self.assertEqual(sub1[0], sub2[1])
        self.assert_(k18.subfields is None)
        k0a = lsst.afw.table.Key["D"]()
        k0b = lsst.afw.table.Key["Flag"]()
        self.assertRaises(lsst.pex.exceptions.LogicError, record.get, k0a)
        self.assertRaises(lsst.pex.exceptions.LogicError, record.get, k0b)

    def _testBaseFits(self, target):
        schema = lsst.afw.table.Schema()
        k = schema.addField("f", type="D")
        cat1 = lsst.afw.table.BaseCatalog(schema)
        for i in range(50):
            record = cat1.addNew()
            record.set(k, numpy.random.randn())
        cat1.writeFits(target)
        cat2 = lsst.afw.table.BaseCatalog.readFits(target)
        self.assertEqual(len(cat1), len(cat2))
        for r1, r2 in zip(cat1, cat2):
            self.assertEqual(r1.get(k), r2.get(k))

    def testBaseFits(self):
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            self._testBaseFits(tmpFile)
        self.assertRaises(Exception, lsst.afw.table.BaseCatalog.readFits, "nonexistentfile.fits")

    def testMemoryFits(self):
        mem = lsst.afw.fits.MemFileManager()
        self._testBaseFits(mem)

    def testColumnView(self):
        schema = lsst.afw.table.Schema()
        k0 = schema.addField("f0", type="U")
        k1 = schema.addField("f1", type="I")
        kb1 = schema.addField("fb1", type="Flag")
        k2 = schema.addField("f2", type="F")
        kb2 = schema.addField("fb2", type="Flag")
        k3 = schema.addField("f3", type="D")
        kb3 = schema.addField("fb3", type="Flag")
        k4 = schema.addField("f4", type="ArrayF", size=2)
        k5 = schema.addField("f5", type="ArrayD", size=3)
        k6 = schema.addField("f6", type="Angle")
        k7 = schema.addField("f7", type="ArrayU", size=4)
        catalog = lsst.afw.table.BaseCatalog(schema)
        catalog.addNew()
        catalog[0].set(k0, 1)
        catalog[0].set(k1, 2)
        catalog[0].set(k2, 0.5)
        catalog[0].set(k3, 0.25)
        catalog[0].set(kb1, False)
        catalog[0].set(kb2, True)
        catalog[0].set(kb3, False)
        catalog[0].set(k4, numpy.array([-0.5, -0.25], dtype=numpy.float32))
        catalog[0].set(k5, numpy.array([-1.5, -1.25, 3.375], dtype=numpy.float64))
        catalog[0].set(k6, lsst.afw.geom.Angle(0.25))
        catalog[0].set(k7, numpy.array([2,3,4,1], dtype=numpy.uint16))
        col1a = catalog[k1]
        self.assertEqual(col1a.shape, (1,))
        catalog.addNew()
        catalog[1].set(k0, 4)
        catalog[1].set(k1, 3)
        catalog[1].set(k2, 2.5)
        catalog[1].set(k3, 0.75)
        catalog[1].set(kb1, True)
        catalog[1].set(kb2, False)
        catalog[1].set(kb3, True)
        catalog[1].set(k4, numpy.array([-3.25, -0.75], dtype=numpy.float32))
        catalog[1].set(k5, numpy.array([-1.25, -2.75, 0.625], dtype=numpy.float64))
        catalog[1].set(k6, lsst.afw.geom.Angle(0.15))
        catalog[1].set(k7, numpy.array([5,6,8,7], dtype=numpy.uint16))
        col1b = catalog[k1]
        self.assertEqual(col1b.shape, (2,))
        columns = catalog.getColumnView()
        for key in [k0, k1, k2, k3, kb1, kb2, kb3]:
            array = columns[key]
            for i in [0, 1]:
                self.assertEqual(array[i], catalog[i].get(key))
        for key in [k4, k5, k7]:
            array = columns[key]
            for i in [0, 1]:
                self.assert_(numpy.all(array[i] == catalog[i].get(key)))
        for key in [k6]:
            array = columns[key]
            for i in [0, 1]:
                self.assertEqual(lsst.afw.geom.Angle(array[i]), catalog[i].get(key))
        for key in [k1, k2, k3]:
            vals = columns[key].copy()
            vals *= 2
            array = columns[key]
            array *= 2
            for i in [0, 1]:
                self.assertEqual(catalog[i].get(key), vals[i])
                self.assertEqual(array[i], vals[i])
        catalog[k1] = 4
        f3v = numpy.random.randn(2)
        catalog["f3"] = f3v
        for i in [0, 1]:
            self.assertEqual(catalog[i].get(k1), 4)
            self.assertEqual(catalog[i].get(k3), f3v[i])

    def testUnsignedFitsPersistence(self):
        """Test FITS round-trip of unsigned short ints, since FITS handles unsigned columns differently
        from signed columns
        """
        schema = lsst.afw.table.Schema()
        k1 = schema.addField("f1", type=numpy.uint16, doc="scalar uint16")
        k2 = schema.addField("f2", type="ArrayU", doc="array uint16", size=4)
        cat1 = lsst.afw.table.BaseCatalog(schema)
        record1 = cat1.addNew()
        record1.set(k1, 4)
        record1.set(k2, numpy.array([5, 6, 7, 8], dtype=numpy.uint16))
        filename = "testSimpleTable-testUnsignedFitsPersistence.fits"
        cat1.writeFits(filename)
        cat2 = lsst.afw.table.BaseCatalog.readFits(filename)
        record2 = cat2[0]
        self.assertEqual(cat1.schema, cat2.schema)
        self.assertEqual(record1.get(k1), record2.get(k1))
        self.assertTrue(numpy.all(record1.get(k2) == record2.get(k2)))
        os.remove(filename)

    def testIteration(self):
        schema = lsst.afw.table.Schema()
        k = schema.addField("a", type=int)
        catalog = lsst.afw.table.BaseCatalog(schema)
        for n in range(5):
            record = catalog.addNew()
            record[k] = n
        for n, r in enumerate(catalog):
            self.assertEqual(n, r[k])

    def testTicket2262(self):
        """Test that we can construct an array field in Python"""
        f1 = lsst.afw.table.Field["ArrayF"]("name", "doc", "barn", 5)
        f2 = lsst.afw.table.Field["ArrayD"]("name", "doc", 5)
        self.assertEqual(f1.getSize(), 5)
        self.assertEqual(f2.getSize(), 5)

    def testExtract(self):
        schema = lsst.afw.table.Schema()
        schema.addField("a_b_c1", type=numpy.float64)
        schema.addField("a_b_c2", type="Flag")
        schema.addField("a_d1", type=numpy.int32)
        schema.addField("a_d2", type=numpy.float32)
        pointKey = lsst.afw.table.Point2IKey.addFields(schema, "q_e1", "doc for point field", "pixel")
        schema.addField("q_e2_xxSigma", type=numpy.float32)
        schema.addField("q_e2_yySigma", type=numpy.float32)
        schema.addField("q_e2_xySigma", type=numpy.float32)
        schema.addField("q_e2_xx_yy_Cov", type=numpy.float32)
        schema.addField("q_e2_xx_xy_Cov", type=numpy.float32)
        schema.addField("q_e2_yy_xy_Cov", type=numpy.float32)
        covKey = lsst.afw.table.CovarianceMatrix3fKey(schema["q_e2"], ["xx", "yy", "xy"])
        self.assertEqual(schema.extract("a_b_*", ordered=True).keys(), ["a_b_c1", "a_b_c2"])
        self.assertEqual(schema.extract("*1", ordered=True).keys(), ["a_b_c1", "a_d1"])
        self.assertEqual(schema.extract("a_b_*", "*2", ordered=True).keys(),
                         ["a_b_c1", "a_b_c2", "a_d2"])
        self.assertEqual(schema.extract(regex=r"a_(.+)1", sub=r"\1f", ordered=True).keys(), ["b_cf", "df"])
        catalog = lsst.afw.table.BaseCatalog(schema)
        for i in range(5):
            record = catalog.addNew()
            record.set("a_b_c1", numpy.random.randn())
            record.set("a_b_c2", True)
            record.set("a_d1", numpy.random.randint(100))
            record.set("a_d2", numpy.random.randn())
            record.set(pointKey, lsst.afw.geom.Point2I(numpy.random.randint(10), numpy.random.randint(10)))
            record.set(covKey, numpy.random.randn(3,3).astype(numpy.float32))
        d = record.extract("*")
        self.assertEqual(set(d.keys()), set(schema.getNames()))
        self.assertEqual(d["a_b_c1"], record.get("a_b_c1"))
        self.assertEqual(d["a_b_c2"], record.get("a_b_c2"))
        self.assertEqual(d["a_d1"], record.get("a_d1"))
        self.assertEqual(d["a_d2"], record.get("a_d2"))
        self.assertEqual(d["q_e1_x"], record.get(pointKey.getX()))
        self.assertEqual(d["q_e1_y"], record.get(pointKey.getY()))
        allIdx = slice(None)
        sliceIdx = slice(0, 4, 2)
        boolIdx = numpy.array([True, False, False, True, True])
        for kwds, idx in [
            ({}, allIdx),
            ({"copy": True}, allIdx),
            ({"where": boolIdx}, boolIdx),
            ({"where": sliceIdx}, sliceIdx),
            ({"where": boolIdx, "copy": True}, boolIdx),
            ({"where": sliceIdx, "copy": True}, sliceIdx),
            ]:
            d = catalog.extract("*", **kwds)
            self.assert_(numpy.all(d["a_b_c1"] == catalog.get("a_b_c1")[idx]))
            self.assert_(numpy.all(d["a_b_c2"] == catalog.get("a_b_c2")[idx]))
            self.assert_(numpy.all(d["a_d1"] == catalog.get("a_d1")[idx]))
            self.assert_(numpy.all(d["a_d2"] == catalog.get("a_d2")[idx]))
            self.assert_(numpy.all(d["q_e1_x"] == catalog.get("q_e1_x")[idx]))
            self.assert_(numpy.all(d["q_e1_y"] == catalog.get("q_e1_y")[idx]))
            if "copy" in kwds or idx is boolIdx:
                for col in d.values():
                    self.assert_(col.flags.c_contiguous)
        # Test that aliases are included in extract()
        schema.getAliasMap().set("b_f", "a_b")
        d = schema.extract("b_f*")
        self.assertEqual(sorted(d.keys()), ["b_f_c1", "b_f_c2"])

    def testExtend(self):
        schema1 = lsst.afw.table.SourceTable.makeMinimalSchema()
        k1 = schema1.addField("f1", type=int)
        k2 = schema1.addField("f2", type=float)
        cat1 = lsst.afw.table.BaseCatalog(schema1)
        for i in range(1000):
            record = cat1.addNew()
            record.setI(k1, i)
            record.setD(k2, numpy.random.randn())
        self.assertFalse(cat1.isContiguous())
        cat2 = lsst.afw.table.BaseCatalog(schema1)
        cat2.extend(cat1, deep=True)
        self.assertEqual(len(cat1), len(cat2))
        self.assert_(cat2.isContiguous())
        cat3 = lsst.afw.table.BaseCatalog(cat1.table)
        cat3.extend(cat1, deep=False)
        self.assertFalse(cat3.isContiguous())
        cat4 = lsst.afw.table.BaseCatalog(cat1.table)
        cat4.extend(list(cat1), deep=False)
        self.assertFalse(cat4.isContiguous())
        cat4 = lsst.afw.table.BaseCatalog(schema1)
        cat4.extend(list(cat1), deep=True)
        self.assertFalse(cat4.isContiguous())
        mapper = lsst.afw.table.SchemaMapper(schema1)
        mapper.addMinimalSchema(lsst.afw.table.SourceTable.makeMinimalSchema())
        mapper.addMapping(k2)
        schema2 = mapper.getOutputSchema()
        self.assert_(mapper.getOutputSchema().contains(lsst.afw.table.SourceTable.makeMinimalSchema()))
        cat5 = lsst.afw.table.BaseCatalog(schema2)
        cat5.extend(cat1, mapper=mapper)
        self.assert_(cat5.isContiguous())
        cat6 = lsst.afw.table.SourceCatalog(schema2)
        cat6.extend(list(cat1), mapper=mapper)
        self.assertFalse(cat6.isContiguous())
        cat7 = lsst.afw.table.SourceCatalog(schema2)
        cat7.reserve(len(cat1) * 3)
        cat7.extend(list(cat1), mapper=mapper)
        cat7.extend(cat1, mapper)
        cat7.extend(list(cat1), mapper)
        self.assert_(cat7.isContiguous())
        cat8 = lsst.afw.table.BaseCatalog(schema2)
        cat8.extend(list(cat7), True)
        cat8.extend(list(cat7), deep=True)

    def testTicket2308(self):
        inputSchema = lsst.afw.table.SourceTable.makeMinimalSchema()
        mapper1 = lsst.afw.table.SchemaMapper(inputSchema)
        mapper1.addMinimalSchema(lsst.afw.table.SourceTable.makeMinimalSchema(), True)
        mapper2 = lsst.afw.table.SchemaMapper(inputSchema)
        mapper2.addMinimalSchema(lsst.afw.table.SourceTable.makeMinimalSchema(), False)
        inputTable = lsst.afw.table.SourceTable.make(inputSchema)
        inputRecord = inputTable.makeRecord()
        inputRecord.set("id", 42)
        outputTable1 = lsst.afw.table.SourceTable.make(mapper1.getOutputSchema())
        outputTable2 = lsst.afw.table.SourceTable.make(mapper2.getOutputSchema())
        outputRecord1 = outputTable1.makeRecord()
        outputRecord2 = outputTable2.makeRecord()
        self.assertEqual(outputRecord1.getId(), outputRecord2.getId())
        self.assertNotEqual(outputRecord1.getId(), inputRecord.getId())
        outputRecord1.assign(inputRecord, mapper1)
        self.assertEqual(outputRecord1.getId(), inputRecord.getId())
        outputRecord2.assign(inputRecord, mapper2)
        self.assertNotEqual(outputRecord2.getId(), inputRecord.getId())

    def testTicket2393(self):
        schema = lsst.afw.table.Schema()
        k = schema.addField(lsst.afw.table.Field[int]("i", "doc for i"))
        item = schema.find("i")
        self.assertEqual(k, item.key)

    def testTicket2850(self):
        schema = lsst.afw.table.Schema()
        table = lsst.afw.table.BaseTable.make(schema)
        self.assertEqual(table.getBufferSize(), 0)

    def testTicket2894(self):
        """Test boolean-array indexing of catalogs"""
        schema = lsst.afw.table.Schema()
        key = schema.addField(lsst.afw.table.Field[int]("i", "doc for i"))
        cat1 = lsst.afw.table.BaseCatalog(schema)
        cat1.addNew().set(key, 1)
        cat1.addNew().set(key, 2)
        cat1.addNew().set(key, 3)
        cat2 = cat1[numpy.array([True, False, False], dtype=bool)]
        self.assertTrue((cat2[key] == numpy.array([1], dtype=int)).all())
        self.assertEqual(cat2[0], cat1[0])  # records compare using pointer equality
        cat3 = cat1[numpy.array([True, True, False], dtype=bool)]
        self.assertTrue((cat3[key] == numpy.array([1,2], dtype=int)).all())
        cat4 = cat1[numpy.array([True, False, True], dtype=bool)]
        self.assertTrue((cat4.copy(deep=True)[key] == numpy.array([1,3], dtype=int)).all())

    def testTicket2938(self):
        """Test heterogenous catalogs that have records from multiple tables"""
        schema = lsst.afw.table.Schema()
        schema.addField("i", type=int, doc="doc for i")
        cat = lsst.afw.table.BaseCatalog(schema)
        cat.addNew()
        t1 = lsst.afw.table.BaseTable.make(schema)
        cat.append(t1.makeRecord())
        self.assertEqual(cat[-1].getTable(), t1)
        self.assertRaises(lsst.pex.exceptions.RuntimeError,
                                             cat.getColumnView)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            cat.writeFits(filename)  # shouldn't throw
            schema.addField("d", type=float, doc="doc for d")
            t2 = lsst.afw.table.BaseTable.make(schema)
            cat.append(t2.makeRecord())
            self.assertRaises(lsst.pex.exceptions.LogicError, cat.writeFits, filename)

    def testTicket3056(self):
        """Test sorting and sort-based searches of Catalogs"""
        schema = lsst.afw.table.SimpleTable.makeMinimalSchema()
        ki = schema.addField("i", type=int, doc="doc for i")
        kl = schema.addField("l", type=numpy.int64, doc="doc for l")
        kf = schema.addField("f", type=float, doc="doc for f")
        cat = lsst.afw.table.SimpleCatalog(schema)
        for j in range(50, 0, -1):
            record = cat.addNew()
            record.set(ki, j//10)
            record.set(kl, j)
            record.set(kf, numpy.random.randn())
        self.assertFalse(cat.isSorted(ki))
        self.assertFalse(cat.isSorted(kl))
        # sort by unique int64 field, try unique lookups
        cat.sort(kl)
        self.assertTrue(cat.isSorted(kl))
        r10 = cat.find(10, kl)
        self.assertEqual(r10.get(kl), 10)
        # sort by probably-unique float field, try unique and range lookups
        cat.sort(kf)
        self.assertTrue(cat.isSorted(kf))
        r10 = cat.find(10, kf)
        self.assertTrue(r10 is None or r10.get(kf) == 10.0) # latter case virtually impossible
        i0 = cat.lower_bound(-0.5, kf)
        i1 = cat.upper_bound(0.5, kf)
        for i in range(i0, i1):
            self.assertGreaterEqual(cat[i].get(kf), -0.5)
            self.assertLess(cat[i].get(kf), 0.5)
        for r in cat[cat.between(-0.5, 0.5, kf)]:
            self.assertGreaterEqual(r.get(kf), -0.5)
            self.assertLess(r.get(kf), 0.5)
        # sort by nonunique int32 field, try range lookups
        cat.sort(ki)
        self.assertTrue(cat.isSorted(ki))
        s = cat.equal_range(3, ki)
        self.assertTrue(cat[s].isSorted(kf))  # test for stable sort
        for r in cat[s]:
            self.assertEqual(r.get(ki), 3)
        self.assertEqual(s.start, cat.lower_bound(3, ki))
        self.assertEqual(s.stop, cat.upper_bound(3, ki))

    def testRename(self):
        """Test field-renaming functionality in Field, SchemaMapper"""
        field1i = lsst.afw.table.Field[int]("i1", "doc for i", "m")
        field2i = field1i.copyRenamed("i2")
        self.assertEqual(field1i.getName(), "i1")
        self.assertEqual(field2i.getName(), "i2")
        self.assertEqual(field1i.getDoc(), field2i.getDoc())
        self.assertEqual(field1i.getUnits(), field2i.getUnits())
        field1a = lsst.afw.table.Field["ArrayF"]("a1", "doc for a", "s", 3)
        field2a = field1a.copyRenamed("a2")
        self.assertEqual(field1a.getName(), "a1")
        self.assertEqual(field2a.getName(), "a2")
        self.assertEqual(field1a.getDoc(), field2a.getDoc())
        self.assertEqual(field1a.getUnits(), field2a.getUnits())
        self.assertEqual(field1a.getSize(), field2a.getSize())
        schema1 = lsst.afw.table.Schema()
        k1i = schema1.addField(field1i)
        k1a = schema1.addField(field1a)
        mapper = lsst.afw.table.SchemaMapper(schema1)
        k2i = mapper.addMapping(k1i, "i2")
        k2a = mapper.addMapping(k1a, "a2")
        schema2 = mapper.getOutputSchema()
        self.assertEqual(schema1.find(k1i).field.getName(), "i1")
        self.assertEqual(schema2.find(k2i).field.getName(), "i2")
        self.assertEqual(schema1.find(k1a).field.getName(), "a1")
        self.assertEqual(schema2.find(k2a).field.getName(), "a2")
        self.assertEqual(schema1.find(k1i).field.getDoc(), schema2.find(k2i).field.getDoc())
        self.assertEqual(schema1.find(k1a).field.getDoc(), schema2.find(k2a).field.getDoc())
        self.assertEqual(schema1.find(k1i).field.getUnits(), schema2.find(k2i).field.getUnits())
        self.assertEqual(schema1.find(k1a).field.getUnits(), schema2.find(k2a).field.getUnits())
        self.assertEqual(schema1.find(k1a).field.getSize(), schema2.find(k2a).field.getSize())
        k3i = mapper.addMapping(k1i, "i3")
        k3a = mapper.addMapping(k1a, "a3")
        schema3 = mapper.getOutputSchema()
        self.assertEqual(schema1.find(k1i).field.getName(), "i1")
        self.assertEqual(schema3.find(k3i).field.getName(), "i3")
        self.assertEqual(schema1.find(k1a).field.getName(), "a1")
        self.assertEqual(schema3.find(k3a).field.getName(), "a3")
        self.assertEqual(schema1.find(k1i).field.getDoc(), schema3.find(k3i).field.getDoc())
        self.assertEqual(schema1.find(k1a).field.getDoc(), schema3.find(k3a).field.getDoc())
        self.assertEqual(schema1.find(k1i).field.getUnits(), schema3.find(k3i).field.getUnits())
        self.assertEqual(schema1.find(k1a).field.getUnits(), schema3.find(k3a).field.getUnits())
        self.assertEqual(schema1.find(k1a).field.getSize(), schema3.find(k3a).field.getSize())

    def testTicket3066(self):
        """Test the doReplace option on Schema.addField
        """
        schema = lsst.afw.table.Schema()
        k1a = schema.addField("f1", doc="f1a", type="I")
        k2a = schema.addField("f2", doc="f2a", type="Flag")
        k3a = schema.addField("f3", doc="f3a", type="ArrayF", size=4)
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError,
                                             schema.addField, "f1", doc="f1b", type="I")
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError,
                                             schema.addField, "f2", doc="f2b", type="Flag")
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError,
                                             schema.addField, "f1", doc="f1b", type="F")
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError,
                                             schema.addField, "f2", doc="f2b", type="F")
        self.assertRaises(lsst.pex.exceptions.TypeError,
                                             schema.addField, "f1", doc="f1b", type="F", doReplace=True)
        self.assertRaises(lsst.pex.exceptions.TypeError,
                                             schema.addField, "f2", doc="f2b", type="F", doReplace=True)
        self.assertRaises(lsst.pex.exceptions.TypeError,
                                             schema.addField, "f3", doc="f3b", type="ArrayF",
                                             size=3, doReplace=True)
        k1b = schema.addField("f1", doc="f1b", type="I", doReplace=True)
        self.assertEqual(k1a, k1b)
        self.assertEqual(schema.find(k1a).field.getDoc(), "f1b")
        k2b = schema.addField("f2", doc="f2b", type="Flag", doReplace=True)
        self.assertEqual(k2a, k2b)
        self.assertEqual(schema.find(k2a).field.getDoc(), "f2b")
        k3b = schema.addField("f3", doc="f3b", type="ArrayF", size=4, doReplace=True)
        self.assertEqual(k3a, k3b)
        self.assertEqual(schema.find(k3a).field.getDoc(), "f3b")

    def testDM352(self):
        filename = os.path.join(os.path.split(__file__)[0], "data", "great3.fits")
        cat = lsst.afw.table.BaseCatalog.readFits(filename)
        self.assertEqual(len(cat), 1)

    def testDM1710(self):
        # Extending without specifying a mapper or a deep argument should not
        # raise.
        schema = lsst.afw.table.Schema()
        cat1 = lsst.afw.table.BaseCatalog(schema)
        cat2 = lsst.afw.table.BaseCatalog(schema)
        cat1.extend(cat2)

    def testVariableLengthArrays(self):
        schema = lsst.afw.table.Schema()
        ka = schema.addField("a", doc="integer", type="ArrayI", size=0)
        kb = schema.addField("b", doc="single-precision", type="ArrayF")
        kc = schema.addField("c", doc="double-precision", type="ArrayD")
        cat1 = lsst.afw.table.BaseCatalog(schema)
        record1 = cat1.addNew()
        self.assertEqual(list(record1.get(ka)), [])
        self.assertEqual(list(record1.get(kb)), [])
        self.assertEqual(list(record1.get(kc)), [])
        a1 = numpy.random.randint(low=3, high=6, size=4).astype(numpy.int32)
        b1 = numpy.random.randn(5).astype(numpy.float32)
        c1 = numpy.random.randn(6).astype(numpy.float64)
        # Test get/set
        record1.set(ka, a1)
        record1.set(kb, b1)
        record1.set(kc, c1)
        self.assertTrue(numpy.all(record1.get(ka) == a1))
        self.assertTrue(numpy.all(record1.get(kb) == b1))
        self.assertTrue(numpy.all(record1.get(kc) == c1))
        # Test __getitem__ and view semantics
        record1[kb][2] = 3.5
        self.assertEqual(b1[2], 3.5)
        # Check that we throw when we try to index a variable-length array Key
        self.assertRaisesLsstCpp(lsst.pex.exceptions.LogicError, lambda x: ka[x], 0)
        self.assertRaisesLsstCpp(lsst.pex.exceptions.LogicError, lambda x, y: ka[x:y], 0, 1)
        # Test copying records, both with and without SchemaMapper
        record2 = cat1.addNew()
        record2.assign(record1)
        self.assertTrue(numpy.all(record1.get(ka) == a1))
        self.assertTrue(numpy.all(record1.get(kb) == b1))
        self.assertTrue(numpy.all(record1.get(kc) == c1))
        record1[kb][2] = 4.5
        self.assertEqual(record2[kb][2], 3.5) # copy in assign() should be deep
        mapper = lsst.afw.table.SchemaMapper(schema)
        kb2 = mapper.addMapping(kb)
        cat2 = lsst.afw.table.BaseCatalog(mapper.getOutputSchema())
        record3 = cat2.addNew()
        record3.assign(record1, mapper)
        self.assertTrue(numpy.all(record3.get(kb2) == b1))
        # Test that we throw if we try to get a column view of a variable-length arry
        self.assertRaisesLsstCpp(lsst.pex.exceptions.LogicError, cat1.get, ka)
        # Test that we can round-trip variable-length arrays through FITS
        filename = "testSimpleTable_testVariableLengthArrays.fits"
        cat1.writeFits(filename)
        cat3 = lsst.afw.table.BaseCatalog.readFits(filename)
        self.assertEqual(schema.compare(cat3.schema, lsst.afw.table.Schema.IDENTICAL),
                         lsst.afw.table.Schema.IDENTICAL)
        record4 = cat3[0]
        self.assertTrue(numpy.all(record4.get(ka) == a1))
        self.assertTrue(numpy.all(record4.get(kb) == b1))
        self.assertTrue(numpy.all(record4.get(kc) == c1))
        os.remove(filename)

    def testCompoundFieldFitsConversion(self):
        """Test that we convert compound fields saved with an older version of the pipeline
        into the set of multiple fields used by their replacement FunctorKeys.
        """
        geomValues = {
            "point_i_x": 4, "point_i_y": 5,
            "point_d_x": 3.5, "point_d_y": 2.0,
            "moments_xx": 5.0, "moments_yy": 6.5, "moments_xy": 2.25,
            "coord_ra": 1.0*lsst.afw.geom.radians, "coord_dec": 0.5*lsst.afw.geom.radians,
        }
        covValues = {
            "cov_z": numpy.array([[4.00, 1.25, 1.50, 0.75],
                                  [1.25, 2.25, 0.50, 0.25],
                                  [1.50, 0.50, 6.25, 1.75],
                                  [0.75, 0.25, 1.75, 9.00]], dtype=numpy.float32),
            "cov_p": numpy.array([[5.50, -2.0],
                                  [-2.0, 3.25]], dtype=numpy.float32),
            "cov_m": numpy.array([[3.75, -0.5, 1.25],
                                  [-0.5, 4.50, 0.75],
                                  [1.25, 0.75, 6.25]], dtype=numpy.float32),
        }
        filename = os.path.join(os.path.split(__file__)[0], "data", "CompoundFieldConversion.fits")
        cat2 = lsst.afw.table.BaseCatalog.readFits(filename)
        record2 = cat2[0]
        for k, v in geomValues.iteritems():
            self.assertEqual(record2.get(k), v, msg=k)
        covZKey = lsst.afw.table.CovarianceMatrixXfKey(cat2.schema["cov_z"], ["0", "1", "2", "3"])
        covPKey = lsst.afw.table.CovarianceMatrix2fKey(cat2.schema["cov_p"], ["x", "y"])
        covMKey = lsst.afw.table.CovarianceMatrix3fKey(cat2.schema["cov_m"], ["xx", "yy", "xy"])
        self.assertClose(record2.get(covZKey), covValues["cov_z"], rtol=1E-6)
        self.assertClose(record2.get(covPKey), covValues["cov_p"], rtol=1E-6)
        self.assertClose(record2.get(covMKey), covValues["cov_m"], rtol=1E-6)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(SimpleTableTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

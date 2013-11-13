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
Tests for table.SimpleTable

Run with:
   ./testSimpleTable.py
or
   python
   >>> import testSimpleTable; testSimpleTable.run()
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

class SimpleTableTestCase(unittest.TestCase):

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

    def checkGeomAccessors(self, record, key, name, value):
        fastSetter = getattr(record, "set" + key.getTypeString())
        fastGetter = getattr(record, "get" + key.getTypeString())
        record.set(key, value)
        self.assertEqual(record.get(key), value)
        record.set(name, value)
        self.assertEqual(record.get(name), value)
        fastSetter(key, value)
        self.assertEqual(fastGetter(key), value)

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
        k1 = schema.addField("f1", type="I")
        k2 = schema.addField("f2", type="L")
        k3 = schema.addField("f3", type="F")
        k4 = schema.addField("f4", type="D")
        k5 = schema.addField("f5", type="PointI")
        k6 = schema.addField("f6", type="PointF")
        k7 = schema.addField("f7", type="PointD")
        k8 = schema.addField("f8", type="MomentsF")
        k9 = schema.addField("f9", type="MomentsD")
        k10 = schema.addField("f10", type="ArrayF", size=4)
        k11 = schema.addField("f11", type="ArrayD", size=5)
        k12 = schema.addField("f12", type="CovF", size=3)
        k14 = schema.addField("f14", type="CovPointF")
        k16 = schema.addField("f16", type="CovMomentsF")
        k18 = schema.addField("f18", type="Angle")
        k19 = schema.addField("f19", type="Coord")
        k20 = schema.addField("f20", type="String", size=4)
        table = lsst.afw.table.BaseTable.make(schema)
        record = table.makeRecord()
        self.assertEqual(record[k1], 0)
        self.assertEqual(record[k2], 0)
        self.assert_(numpy.isnan(record[k3]))
        self.assert_(numpy.isnan(record[k4]))
        self.assertEqual(record.get(k5), lsst.afw.geom.Point2I())
        self.assert_(numpy.isnan(record[k6.getX()]))
        self.assert_(numpy.isnan(record[k6.getY()]))
        self.assert_(numpy.isnan(record[k7.getX()]))
        self.assert_(numpy.isnan(record[k7.getY()]))
        self.checkScalarAccessors(record, k1, "f1", 2, 3)
        self.checkScalarAccessors(record, k2, "f2", 2, 3)
        self.checkScalarAccessors(record, k3, "f3", 2.5, 3.5)
        self.checkScalarAccessors(record, k4, "f4", 2.5, 3.5)
        self.checkGeomAccessors(record, k5, "f5", lsst.afw.geom.Point2I(5, 3))
        self.checkGeomAccessors(record, k6, "f6", lsst.afw.geom.Point2D(5.5, 3.5))
        self.checkGeomAccessors(record, k7, "f7", lsst.afw.geom.Point2D(5.5, 3.5))
        for k in (k5, k6, k7): self.assertEqual(k.subfields, ("x", "y"))
        self.checkGeomAccessors(record, k8, "f8", lsst.afw.geom.ellipses.Quadrupole(5.5, 3.5, -1.0))
        self.checkGeomAccessors(record, k9, "f9", lsst.afw.geom.ellipses.Quadrupole(5.5, 3.5, -1.0))
        for k in (k8, k9): self.assertEqual(k.subfields, ("xx", "yy", "xy"))
        self.checkArrayAccessors(record, k10, "f10", makeArray(k10.getSize(), dtype=numpy.float32))
        self.checkArrayAccessors(record, k11, "f11", makeArray(k11.getSize(), dtype=numpy.float64))
        for k in (k10, k11): self.assertEqual(k.subfields, tuple(range(k.getSize())))
        self.checkArrayAccessors(record, k12, "f12", makeCov(k12.getSize(), dtype=numpy.float32))
        self.checkArrayAccessors(record, k14, "f14", makeCov(k14.getSize(), dtype=numpy.float32))
        self.checkArrayAccessors(record, k16, "f16", makeCov(k16.getSize(), dtype=numpy.float32))
        sub = k11.slice(1, 3)
        self.assert_((record.get(sub) == record.get(k11)[1:3]).all())
        for k in (k12, k14, k16):
            n = 0
            for idx, subkey in zip(k.subfields, k.subkeys):
                self.assertEqual(k[idx], subkey)
                n += 1
            self.assertEqual(n, k.getElementCount())
        self.checkGeomAccessors(record, k18, "f18", lsst.afw.geom.Angle(1.2))
        self.assert_(k18.subfields is None)
        self.checkGeomAccessors(
            record, k19, "f19", 
            lsst.afw.coord.IcrsCoord(lsst.afw.geom.Angle(1.3), lsst.afw.geom.Angle(0.5))
            )
        self.assertEqual(k19.subfields, ("ra", "dec"))
        self.checkScalarAccessors(record, k20, "f20", "foo", "bar")
        k0a = lsst.afw.table.Key["D"]()
        k0b = lsst.afw.table.Key["Flag"]()
        lsst.utils.tests.assertRaisesLsstCpp(self, lsst.pex.exceptions.LogicErrorException, record.get, k0a)
        lsst.utils.tests.assertRaisesLsstCpp(self, lsst.pex.exceptions.LogicErrorException, record.get, k0b)

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
        self._testBaseFits("testBaseTable.fits")
        os.remove("testBaseTable.fits")
        self.assertRaises(Exception, lsst.afw.table.BaseCatalog.readFits, "nonexistentfile.fits")

    def testMemoryFits(self):
        mem = lsst.afw.table.MemFileManager()
        self._testBaseFits(mem)

    def testColumnView(self):
        schema = lsst.afw.table.Schema()
        k1 = schema.addField("f1", type="I")
        kb1 = schema.addField("fb1", type="Flag")
        k2 = schema.addField("f2", type="F")
        kb2 = schema.addField("fb2", type="Flag")
        k3 = schema.addField("f3", type="D")
        kb3 = schema.addField("fb3", type="Flag")
        k4 = schema.addField("f4", type="ArrayF", size=2)
        k5 = schema.addField("f5", type="ArrayD", size=3)
        k6 = schema.addField("f6", type="Angle")
        catalog = lsst.afw.table.BaseCatalog(schema)
        catalog.addNew()
        catalog[0].set(k1, 2)
        catalog[0].set(k2, 0.5)
        catalog[0].set(k3, 0.25)
        catalog[0].set(kb1, False)
        catalog[0].set(kb2, True)
        catalog[0].set(kb3, False)
        catalog[0].set(k4, numpy.array([-0.5, -0.25], dtype=numpy.float32))
        catalog[0].set(k5, numpy.array([-1.5, -1.25, 3.375], dtype=numpy.float64))
        catalog[0].set(k6, lsst.afw.geom.Angle(0.25))
        col1a = catalog[k1]
        self.assertEqual(col1a.shape, (1,))
        catalog.addNew()
        catalog[1].set(k1, 3)
        catalog[1].set(k2, 2.5)
        catalog[1].set(k3, 0.75)
        catalog[1].set(kb1, True)
        catalog[1].set(kb2, False)
        catalog[1].set(kb3, True)
        catalog[1].set(k4, numpy.array([-3.25, -0.75], dtype=numpy.float32))
        catalog[1].set(k5, numpy.array([-1.25, -2.75, 0.625], dtype=numpy.float64))
        catalog[1].set(k6, lsst.afw.geom.Angle(0.15))
        col1b = catalog[k1]
        self.assertEqual(col1b.shape, (2,))
        columns = catalog.getColumnView()
        for key in [k1, k2, k3, kb1, kb2, kb3]:
            array = columns[key]
            for i in [0, 1]:
                self.assertEqual(array[i], catalog[i].get(key))
        for key in [k4, k5]:
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
        f1 = lsst.afw.table.Field["ArrayF"]("name", "doc", "units", 5)
        f2 = lsst.afw.table.Field["ArrayD"]("name", "doc", 5)
        self.assertEqual(f1.getSize(), 5)
        self.assertEqual(f2.getSize(), 5)
        
    def testExtract(self):
        schema = lsst.afw.table.Schema()
        schema.addField("a.b.c1", type=numpy.float64)
        schema.addField("a.b.c2", type="Flag")
        schema.addField("a.d1", type=numpy.int32)
        schema.addField("a.d2", type="ArrayF", size=2)
        schema.addField("q.e1", type="PointI")
        covKey = schema.addField("q.e2", type="CovF", size=3)
        self.assertEqual(schema.extract("a.b.*", ordered=True).keys(), ["a.b.c1", "a.b.c2"])
        self.assertEqual(schema.extract("*1", ordered=True).keys(), ["a.b.c1", "a.d1", "q.e1"])
        self.assertEqual(schema.extract("a.b.*", "*2", ordered=True).keys(),
                         ["a.b.c1", "a.b.c2", "a.d2", "q.e2"])
        self.assertEqual(schema.extract(regex=r"a\.(.+)1", sub=r"\1f", ordered=True).keys(), ["b.cf", "df"])
        catalog = lsst.afw.table.BaseCatalog(schema)
        for i in range(5):
            record = catalog.addNew()
            record.set("a.b.c1", numpy.random.randn())
            record.set("a.b.c2", True)
            record.set("a.d1", numpy.random.randint(100))
            record.set("a.d2", numpy.random.randn(2).astype(numpy.float32))
            record.set("q.e1", lsst.afw.geom.Point2I(numpy.random.randint(10), numpy.random.randint(10)))
            record.set("q.e2", numpy.random.randn(3,3).astype(numpy.float32))
        d = record.extract("*")
        self.assertEqual(set(d.keys()), set(schema.getNames()))
        self.assertEqual(d["a.b.c1"], record.get("a.b.c1"))
        self.assertEqual(d["a.b.c2"], record.get("a.b.c2"))
        self.assertEqual(d["a.d1"], record.get("a.d1"))
        self.assert_(numpy.all(d["a.d2"] == record.get("a.d2")))
        self.assertEqual(d["q.e1"], record.get("q.e1"))
        self.assert_(numpy.all(d["q.e2"] == record.get("q.e2")))
        d = record.extract("q.e1", split=True)
        self.assertEqual(d["q.e1.x"], record.get("q.e1.x"))
        self.assertEqual(d["q.e1.y"], record.get("q.e1.y"))
        self.assert_("q.e1" not in d)
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
            self.assert_(numpy.all(d["a.b.c1"] == catalog.get("a.b.c1")[idx]))
            self.assert_(numpy.all(d["a.b.c2"] == catalog.get("a.b.c2")[idx]))
            self.assert_(numpy.all(d["a.d1"] == catalog.get("a.d1")[idx]))
            self.assert_(numpy.all(d["a.d2"] == catalog.get("a.d2")[idx]))
            self.assert_(numpy.all(d["q.e1.x"] == catalog.get("q.e1.x")[idx]))
            self.assert_(numpy.all(d["q.e1.y"] == catalog.get("q.e1.y")[idx]))
            cov = d["q.e2"]
            for i in range(covKey.getSize()):
                for j in range(covKey.getSize()):
                    self.assert_(numpy.all(cov[:,i,j] == catalog.get(covKey[i,j])[idx]))
            if "copy" in kwds or idx is boolIdx:
                for col in d.values():
                    self.assert_(col.flags.c_contiguous)

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
        k2a = mapper.addMapping(k2)
        schema2 = mapper.getOutputSchema()
        self.assert_(mapper.getOutputSchema().contains(lsst.afw.table.SourceTable.makeMinimalSchema()))
        cat5 = lsst.afw.table.BaseCatalog(schema2)
        cat5.extend(cat1, mapper=mapper)
        self.assert_(cat5.isContiguous())
        cat6 = lsst.afw.table.SourceCatalog(schema2)
        cat6.extend(list(cat1), mapper=mapper)
        self.assertFalse(cat6.isContiguous())
        cat7 = lsst.afw.table.SourceCatalog(schema2)
        cat7.reserve(len(cat1) * 2)
        cat7.extend(list(cat1), mapper=mapper)
        cat7.extend(cat1, mapper=mapper)
        self.assert_(cat7.isContiguous())

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
        lsst.utils.tests.assertRaisesLsstCpp(self, lsst.pex.exceptions.RuntimeErrorException,
                                             cat.getColumnView)
        filename = "testTicket2938.fits"
        cat.writeFits(filename)  # shouldn't throw
        schema.addField("d", type=float, doc="doc for d")
        t2 = lsst.afw.table.BaseTable.make(schema)
        cat.append(t2.makeRecord())
        lsst.utils.tests.assertRaisesLsstCpp(self, lsst.pex.exceptions.LogicErrorException,
                                             cat.writeFits, filename)
        os.remove(filename)

    def testRename(self):
        """Test field-renaming functionality in Field, SchemaMapper"""
        field1i = lsst.afw.table.Field[int]("i1", "doc for i", "units for i")
        field2i = field1i.copyRenamed("i2")
        self.assertEqual(field1i.getName(), "i1")
        self.assertEqual(field2i.getName(), "i2")
        self.assertEqual(field1i.getDoc(), field2i.getDoc())
        self.assertEqual(field1i.getUnits(), field2i.getUnits())
        field1a = lsst.afw.table.Field["ArrayF"]("a1", "doc for a", "units for a", 3)
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

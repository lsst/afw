#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import zip
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008-2014 AURA/LSST
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
#pybind11#Tests for table.SimpleTable
#pybind11#
#pybind11#Run with:
#pybind11#   ./testSimpleTable.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import testSimpleTable; testSimpleTable.run()
#pybind11#"""
#pybind11#import os.path
#pybind11#import unittest
#pybind11#import numpy
#pybind11#
#pybind11#try:
#pybind11#    import pyfits
#pybind11#except ImportError:
#pybind11#    pyfits = None
#pybind11#    print("WARNING: pyfits not available; some tests will not be run")
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.daf.base
#pybind11#import lsst.afw.table
#pybind11#import lsst.afw.geom
#pybind11#import lsst.afw.coord
#pybind11#import lsst.afw.fits
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11#def makeArray(size, dtype):
#pybind11#    return numpy.array(numpy.random.randn(size), dtype=dtype)
#pybind11#
#pybind11#
#pybind11#def makeCov(size, dtype):
#pybind11#    m = numpy.array(numpy.random.randn(size, size), dtype=dtype)
#pybind11#    r = numpy.dot(m, m.transpose())  # not quite symmetric for single-precision on some platforms
#pybind11#    for i in range(r.shape[0]):
#pybind11#        for j in range(i):
#pybind11#            r[i, j] = r[j, i]
#pybind11#    return r
#pybind11#
#pybind11#
#pybind11#class SimpleTableTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        numpy.random.seed(1)
#pybind11#
#pybind11#    def checkScalarAccessors(self, record, key, name, value1, value2):
#pybind11#        fastSetter = getattr(record, "set" + key.getTypeString())
#pybind11#        fastGetter = getattr(record, "get" + key.getTypeString())
#pybind11#        record[key] = value1
#pybind11#        self.assertEqual(record[key], value1)
#pybind11#        self.assertEqual(record.get(key), value1)
#pybind11#        self.assertEqual(record[name], value1)
#pybind11#        self.assertEqual(record.get(name), value1)
#pybind11#        self.assertEqual(fastGetter(key), value1)
#pybind11#        record.set(key, value2)
#pybind11#        self.assertEqual(record[key], value2)
#pybind11#        self.assertEqual(record.get(key), value2)
#pybind11#        self.assertEqual(record[name], value2)
#pybind11#        self.assertEqual(record.get(name), value2)
#pybind11#        self.assertEqual(fastGetter(key), value2)
#pybind11#        record[name] = value1
#pybind11#        self.assertEqual(record[key], value1)
#pybind11#        self.assertEqual(record.get(key), value1)
#pybind11#        self.assertEqual(record[name], value1)
#pybind11#        self.assertEqual(record.get(name), value1)
#pybind11#        self.assertEqual(fastGetter(key), value1)
#pybind11#        record.set(name, value2)
#pybind11#        self.assertEqual(record[key], value2)
#pybind11#        self.assertEqual(record.get(key), value2)
#pybind11#        self.assertEqual(record[name], value2)
#pybind11#        self.assertEqual(record.get(name), value2)
#pybind11#        self.assertEqual(fastGetter(key), value2)
#pybind11#        fastSetter(key, value1)
#pybind11#        self.assertEqual(record[key], value1)
#pybind11#        self.assertEqual(record.get(key), value1)
#pybind11#        self.assertEqual(record[name], value1)
#pybind11#        self.assertEqual(record.get(name), value1)
#pybind11#        self.assertEqual(fastGetter(key), value1)
#pybind11#        self.assertIsNone(key.subfields)
#pybind11#
#pybind11#    def checkArrayAccessors(self, record, key, name, value):
#pybind11#        fastSetter = getattr(record, "set" + key.getTypeString())
#pybind11#        fastGetter = getattr(record, "get" + key.getTypeString())
#pybind11#        record.set(key, value)
#pybind11#        self.assertFloatsEqual(record.get(key), value)
#pybind11#        record.set(name, value)
#pybind11#        self.assertFloatsEqual(record.get(name), value)
#pybind11#        fastSetter(key, value)
#pybind11#        self.assertFloatsEqual(fastGetter(key), value)
#pybind11#
#pybind11#    def testRecordAccess(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        k0 = schema.addField("f0", type="U")
#pybind11#        k1 = schema.addField("f1", type="I")
#pybind11#        k2 = schema.addField("f2", type="L")
#pybind11#        k3 = schema.addField("f3", type="F")
#pybind11#        k4 = schema.addField("f4", type="D")
#pybind11#        k10b = schema.addField("f10b", type="ArrayU", size=2)
#pybind11#        k10a = schema.addField("f10a", type="ArrayI", size=3)
#pybind11#        k10 = schema.addField("f10", type="ArrayF", size=4)
#pybind11#        k11 = schema.addField("f11", type="ArrayD", size=5)
#pybind11#        k18 = schema.addField("f18", type="Angle")
#pybind11#        schema.addField("f20", type="String", size=4)
#pybind11#        table = lsst.afw.table.BaseTable.make(schema)
#pybind11#        record = table.makeRecord()
#pybind11#        self.assertEqual(record[k1], 0)
#pybind11#        self.assertEqual(record[k2], 0)
#pybind11#        self.assertTrue(numpy.isnan(record[k3]))
#pybind11#        self.assertTrue(numpy.isnan(record[k4]))
#pybind11#        self.checkScalarAccessors(record, k0, "f0", 5, 6)
#pybind11#        self.checkScalarAccessors(record, k1, "f1", 2, 3)
#pybind11#        self.checkScalarAccessors(record, k2, "f2", 2, 3)
#pybind11#        self.checkScalarAccessors(record, k3, "f3", 2.5, 3.5)
#pybind11#        self.checkScalarAccessors(record, k4, "f4", 2.5, 3.5)
#pybind11#        self.checkArrayAccessors(record, k10b, "f10b", makeArray(k10b.getSize(), dtype=numpy.uint16))
#pybind11#        self.checkArrayAccessors(record, k10a, "f10a", makeArray(k10a.getSize(), dtype=numpy.int32))
#pybind11#        self.checkArrayAccessors(record, k10, "f10", makeArray(k10.getSize(), dtype=numpy.float32))
#pybind11#        self.checkArrayAccessors(record, k11, "f11", makeArray(k11.getSize(), dtype=numpy.float64))
#pybind11#        for k in (k10, k11):
#pybind11#            self.assertEqual(k.subfields, tuple(range(k.getSize())))
#pybind11#        sub1 = k11.slice(1, 3)
#pybind11#        sub2 = k11[0:2]
#pybind11#        self.assertFloatsAlmostEqual(record.get(sub1), record.get(k11)[1:3], rtol=0, atol=0)
#pybind11#        self.assertFloatsAlmostEqual(record.get(sub2), record.get(k11)[0:2], rtol=0, atol=0)
#pybind11#        self.assertEqual(sub1[0], sub2[1])
#pybind11#        self.assertIsNone(k18.subfields)
#pybind11#        k0a = lsst.afw.table.Key["D"]()
#pybind11#        k0b = lsst.afw.table.Key["Flag"]()
#pybind11#        with self.assertRaises(lsst.pex.exceptions.LogicError):
#pybind11#            record.get(k0a)
#pybind11#        with self.assertRaises(lsst.pex.exceptions.LogicError):
#pybind11#            record.get(k0b)
#pybind11#
#pybind11#    def _testBaseFits(self, target):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        k = schema.addField("f", type="D")
#pybind11#        cat1 = lsst.afw.table.BaseCatalog(schema)
#pybind11#        for i in range(50):
#pybind11#            record = cat1.addNew()
#pybind11#            record.set(k, numpy.random.randn())
#pybind11#        cat1.writeFits(target)
#pybind11#        cat2 = lsst.afw.table.BaseCatalog.readFits(target)
#pybind11#        self.assertEqual(len(cat1), len(cat2))
#pybind11#        for r1, r2 in zip(cat1, cat2):
#pybind11#            self.assertEqual(r1.get(k), r2.get(k))
#pybind11#
#pybind11#    def testBaseFits(self):
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            self._testBaseFits(tmpFile)
#pybind11#        with self.assertRaises(Exception):
#pybind11#            lsst.afw.table.BaseCatalog.readFits("nonexistentfile.fits")
#pybind11#
#pybind11#    def testMemoryFits(self):
#pybind11#        mem = lsst.afw.fits.MemFileManager()
#pybind11#        self._testBaseFits(mem)
#pybind11#
#pybind11#    def testColumnView(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        k0 = schema.addField("f0", type="U")
#pybind11#        k1 = schema.addField("f1", type="I")
#pybind11#        kb1 = schema.addField("fb1", type="Flag")
#pybind11#        k2 = schema.addField("f2", type="F")
#pybind11#        kb2 = schema.addField("fb2", type="Flag")
#pybind11#        k3 = schema.addField("f3", type="D")
#pybind11#        kb3 = schema.addField("fb3", type="Flag")
#pybind11#        k4 = schema.addField("f4", type="ArrayF", size=2)
#pybind11#        k5 = schema.addField("f5", type="ArrayD", size=3)
#pybind11#        k6 = schema.addField("f6", type="Angle")
#pybind11#        k7 = schema.addField("f7", type="ArrayU", size=4)
#pybind11#        catalog = lsst.afw.table.BaseCatalog(schema)
#pybind11#        catalog.addNew()
#pybind11#        catalog[0].set(k0, 1)
#pybind11#        catalog[0].set(k1, 2)
#pybind11#        catalog[0].set(k2, 0.5)
#pybind11#        catalog[0].set(k3, 0.25)
#pybind11#        catalog[0].set(kb1, False)
#pybind11#        catalog[0].set(kb2, True)
#pybind11#        catalog[0].set(kb3, False)
#pybind11#        catalog[0].set(k4, numpy.array([-0.5, -0.25], dtype=numpy.float32))
#pybind11#        catalog[0].set(k5, numpy.array([-1.5, -1.25, 3.375], dtype=numpy.float64))
#pybind11#        catalog[0].set(k6, lsst.afw.geom.Angle(0.25))
#pybind11#        catalog[0].set(k7, numpy.array([2, 3, 4, 1], dtype=numpy.uint16))
#pybind11#        col1a = catalog[k1]
#pybind11#        self.assertEqual(col1a.shape, (1,))
#pybind11#        catalog.addNew()
#pybind11#        catalog[1].set(k0, 4)
#pybind11#        catalog[1].set(k1, 3)
#pybind11#        catalog[1].set(k2, 2.5)
#pybind11#        catalog[1].set(k3, 0.75)
#pybind11#        catalog[1].set(kb1, True)
#pybind11#        catalog[1].set(kb2, False)
#pybind11#        catalog[1].set(kb3, True)
#pybind11#        catalog[1].set(k4, numpy.array([-3.25, -0.75], dtype=numpy.float32))
#pybind11#        catalog[1].set(k5, numpy.array([-1.25, -2.75, 0.625], dtype=numpy.float64))
#pybind11#        catalog[1].set(k6, lsst.afw.geom.Angle(0.15))
#pybind11#        catalog[1].set(k7, numpy.array([5, 6, 8, 7], dtype=numpy.uint16))
#pybind11#        col1b = catalog[k1]
#pybind11#        self.assertEqual(col1b.shape, (2,))
#pybind11#        columns = catalog.getColumnView()
#pybind11#        for key in [k0, k1, k2, k3, kb1, kb2, kb3]:
#pybind11#            array = columns[key]
#pybind11#            for i in [0, 1]:
#pybind11#                self.assertEqual(array[i], catalog[i].get(key))
#pybind11#        for key in [k4, k5, k7]:
#pybind11#            array = columns[key]
#pybind11#            for i in [0, 1]:
#pybind11#                self.assertFloatsEqual(array[i], catalog[i].get(key))
#pybind11#        for key in [k6]:
#pybind11#            array = columns[key]
#pybind11#            for i in [0, 1]:
#pybind11#                self.assertEqual(lsst.afw.geom.Angle(array[i]), catalog[i].get(key))
#pybind11#        for key in [k1, k2, k3]:
#pybind11#            vals = columns[key].copy()
#pybind11#            vals *= 2
#pybind11#            array = columns[key]
#pybind11#            array *= 2
#pybind11#            for i in [0, 1]:
#pybind11#                self.assertEqual(catalog[i].get(key), vals[i])
#pybind11#                self.assertEqual(array[i], vals[i])
#pybind11#        catalog[k1] = 4
#pybind11#        f3v = numpy.random.randn(2)
#pybind11#        catalog["f3"] = f3v
#pybind11#        for i in [0, 1]:
#pybind11#            self.assertEqual(catalog[i].get(k1), 4)
#pybind11#            self.assertEqual(catalog[i].get(k3), f3v[i])
#pybind11#
#pybind11#    def testUnsignedFitsPersistence(self):
#pybind11#        """Test FITS round-trip of unsigned short ints, since FITS handles unsigned columns differently
#pybind11#        from signed columns
#pybind11#        """
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        k1 = schema.addField("f1", type=numpy.uint16, doc="scalar uint16")
#pybind11#        k2 = schema.addField("f2", type="ArrayU", doc="array uint16", size=4)
#pybind11#        cat1 = lsst.afw.table.BaseCatalog(schema)
#pybind11#        record1 = cat1.addNew()
#pybind11#        record1.set(k1, 4)
#pybind11#        record1.set(k2, numpy.array([5, 6, 7, 8], dtype=numpy.uint16))
#pybind11#        filename = "testSimpleTable-testUnsignedFitsPersistence.fits"
#pybind11#        cat1.writeFits(filename)
#pybind11#        cat2 = lsst.afw.table.BaseCatalog.readFits(filename)
#pybind11#        record2 = cat2[0]
#pybind11#        self.assertEqual(cat1.schema, cat2.schema)
#pybind11#        self.assertEqual(record1.get(k1), record2.get(k1))
#pybind11#        self.assertFloatsEqual(record1.get(k2), record2.get(k2))
#pybind11#        os.remove(filename)
#pybind11#
#pybind11#    def testIteration(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        k = schema.addField("a", type=int)
#pybind11#        catalog = lsst.afw.table.BaseCatalog(schema)
#pybind11#        for n in range(5):
#pybind11#            record = catalog.addNew()
#pybind11#            record[k] = n
#pybind11#        for n, r in enumerate(catalog):
#pybind11#            self.assertEqual(n, r[k])
#pybind11#
#pybind11#    def testTicket2262(self):
#pybind11#        """Test that we can construct an array field in Python"""
#pybind11#        f1 = lsst.afw.table.Field["ArrayF"]("name", "doc", "barn", 5)
#pybind11#        f2 = lsst.afw.table.Field["ArrayD"]("name", "doc", 5)
#pybind11#        self.assertEqual(f1.getSize(), 5)
#pybind11#        self.assertEqual(f2.getSize(), 5)
#pybind11#
#pybind11#    def testExtract(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        schema.addField("a_b_c1", type=numpy.float64)
#pybind11#        schema.addField("a_b_c2", type="Flag")
#pybind11#        schema.addField("a_d1", type=numpy.int32)
#pybind11#        schema.addField("a_d2", type=numpy.float32)
#pybind11#        pointKey = lsst.afw.table.Point2IKey.addFields(schema, "q_e1", "doc for point field", "pixel")
#pybind11#        schema.addField("q_e2_xxSigma", type=numpy.float32)
#pybind11#        schema.addField("q_e2_yySigma", type=numpy.float32)
#pybind11#        schema.addField("q_e2_xySigma", type=numpy.float32)
#pybind11#        schema.addField("q_e2_xx_yy_Cov", type=numpy.float32)
#pybind11#        schema.addField("q_e2_xx_xy_Cov", type=numpy.float32)
#pybind11#        schema.addField("q_e2_yy_xy_Cov", type=numpy.float32)
#pybind11#        covKey = lsst.afw.table.CovarianceMatrix3fKey(schema["q_e2"], ["xx", "yy", "xy"])
#pybind11#        self.assertEqual(list(schema.extract("a_b_*", ordered=True).keys()), ["a_b_c1", "a_b_c2"])
#pybind11#        self.assertEqual(list(schema.extract("*1", ordered=True).keys()), ["a_b_c1", "a_d1"])
#pybind11#        self.assertEqual(list(schema.extract("a_b_*", "*2", ordered=True).keys()),
#pybind11#                         ["a_b_c1", "a_b_c2", "a_d2"])
#pybind11#        self.assertEqual(list(schema.extract(regex=r"a_(.+)1", sub=r"\1f",
#pybind11#                                             ordered=True).keys()), ["b_cf", "df"])
#pybind11#        catalog = lsst.afw.table.BaseCatalog(schema)
#pybind11#        for i in range(5):
#pybind11#            record = catalog.addNew()
#pybind11#            record.set("a_b_c1", numpy.random.randn())
#pybind11#            record.set("a_b_c2", True)
#pybind11#            record.set("a_d1", numpy.random.randint(100))
#pybind11#            record.set("a_d2", numpy.random.randn())
#pybind11#            record.set(pointKey, lsst.afw.geom.Point2I(numpy.random.randint(10), numpy.random.randint(10)))
#pybind11#            record.set(covKey, numpy.random.randn(3, 3).astype(numpy.float32))
#pybind11#        d = record.extract("*")
#pybind11#        self.assertEqual(set(d.keys()), set(schema.getNames()))
#pybind11#        self.assertEqual(d["a_b_c1"], record.get("a_b_c1"))
#pybind11#        self.assertEqual(d["a_b_c2"], record.get("a_b_c2"))
#pybind11#        self.assertEqual(d["a_d1"], record.get("a_d1"))
#pybind11#        self.assertEqual(d["a_d2"], record.get("a_d2"))
#pybind11#        self.assertEqual(d["q_e1_x"], record.get(pointKey.getX()))
#pybind11#        self.assertEqual(d["q_e1_y"], record.get(pointKey.getY()))
#pybind11#        allIdx = slice(None)
#pybind11#        sliceIdx = slice(0, 4, 2)
#pybind11#        boolIdx = numpy.array([True, False, False, True, True])
#pybind11#        for kwds, idx in [
#pybind11#            ({}, allIdx),
#pybind11#            ({"copy": True}, allIdx),
#pybind11#            ({"where": boolIdx}, boolIdx),
#pybind11#            ({"where": sliceIdx}, sliceIdx),
#pybind11#            ({"where": boolIdx, "copy": True}, boolIdx),
#pybind11#            ({"where": sliceIdx, "copy": True}, sliceIdx),
#pybind11#        ]:
#pybind11#            d = catalog.extract("*", **kwds)
#pybind11#            self.assertFloatsEqual(d["a_b_c1"], catalog.get("a_b_c1")[idx])
#pybind11#            self.assertFloatsEqual(d["a_b_c2"], catalog.get("a_b_c2")[idx])
#pybind11#            self.assertFloatsEqual(d["a_d1"], catalog.get("a_d1")[idx])
#pybind11#            self.assertFloatsEqual(d["a_d2"], catalog.get("a_d2")[idx])
#pybind11#            self.assertFloatsEqual(d["q_e1_x"], catalog.get("q_e1_x")[idx])
#pybind11#            self.assertFloatsEqual(d["q_e1_y"], catalog.get("q_e1_y")[idx])
#pybind11#            if "copy" in kwds or idx is boolIdx:
#pybind11#                for col in d.values():
#pybind11#                    self.assertTrue(col.flags.c_contiguous)
#pybind11#        # Test that aliases are included in extract()
#pybind11#        schema.getAliasMap().set("b_f", "a_b")
#pybind11#        d = schema.extract("b_f*")
#pybind11#        self.assertEqual(sorted(d.keys()), ["b_f_c1", "b_f_c2"])
#pybind11#
#pybind11#    def testExtend(self):
#pybind11#        schema1 = lsst.afw.table.SourceTable.makeMinimalSchema()
#pybind11#        k1 = schema1.addField("f1", type=int)
#pybind11#        k2 = schema1.addField("f2", type=float)
#pybind11#        cat1 = lsst.afw.table.BaseCatalog(schema1)
#pybind11#        for i in range(1000):
#pybind11#            record = cat1.addNew()
#pybind11#            record.setI(k1, i)
#pybind11#            record.setD(k2, numpy.random.randn())
#pybind11#        self.assertFalse(cat1.isContiguous())
#pybind11#        cat2 = lsst.afw.table.BaseCatalog(schema1)
#pybind11#        cat2.extend(cat1, deep=True)
#pybind11#        self.assertEqual(len(cat1), len(cat2))
#pybind11#        self.assertTrue(cat2.isContiguous())
#pybind11#        cat3 = lsst.afw.table.BaseCatalog(cat1.table)
#pybind11#        cat3.extend(cat1, deep=False)
#pybind11#        self.assertFalse(cat3.isContiguous())
#pybind11#        cat4 = lsst.afw.table.BaseCatalog(cat1.table)
#pybind11#        cat4.extend(list(cat1), deep=False)
#pybind11#        self.assertFalse(cat4.isContiguous())
#pybind11#        cat4 = lsst.afw.table.BaseCatalog(schema1)
#pybind11#        cat4.extend(list(cat1), deep=True)
#pybind11#        self.assertFalse(cat4.isContiguous())
#pybind11#        mapper = lsst.afw.table.SchemaMapper(schema1)
#pybind11#        mapper.addMinimalSchema(lsst.afw.table.SourceTable.makeMinimalSchema())
#pybind11#        mapper.addMapping(k2)
#pybind11#        schema2 = mapper.getOutputSchema()
#pybind11#        self.assertTrue(mapper.getOutputSchema().contains(lsst.afw.table.SourceTable.makeMinimalSchema()))
#pybind11#        cat5 = lsst.afw.table.BaseCatalog(schema2)
#pybind11#        cat5.extend(cat1, mapper=mapper)
#pybind11#        self.assertTrue(cat5.isContiguous())
#pybind11#        cat6 = lsst.afw.table.SourceCatalog(schema2)
#pybind11#        cat6.extend(list(cat1), mapper=mapper)
#pybind11#        self.assertFalse(cat6.isContiguous())
#pybind11#        cat7 = lsst.afw.table.SourceCatalog(schema2)
#pybind11#        cat7.reserve(len(cat1) * 3)
#pybind11#        cat7.extend(list(cat1), mapper=mapper)
#pybind11#        cat7.extend(cat1, mapper)
#pybind11#        cat7.extend(list(cat1), mapper)
#pybind11#        self.assertTrue(cat7.isContiguous())
#pybind11#        cat8 = lsst.afw.table.BaseCatalog(schema2)
#pybind11#        cat8.extend(list(cat7), True)
#pybind11#        cat8.extend(list(cat7), deep=True)
#pybind11#
#pybind11#    def testTicket2308(self):
#pybind11#        inputSchema = lsst.afw.table.SourceTable.makeMinimalSchema()
#pybind11#        mapper1 = lsst.afw.table.SchemaMapper(inputSchema)
#pybind11#        mapper1.addMinimalSchema(lsst.afw.table.SourceTable.makeMinimalSchema(), True)
#pybind11#        mapper2 = lsst.afw.table.SchemaMapper(inputSchema)
#pybind11#        mapper2.addMinimalSchema(lsst.afw.table.SourceTable.makeMinimalSchema(), False)
#pybind11#        inputTable = lsst.afw.table.SourceTable.make(inputSchema)
#pybind11#        inputRecord = inputTable.makeRecord()
#pybind11#        inputRecord.set("id", 42)
#pybind11#        outputTable1 = lsst.afw.table.SourceTable.make(mapper1.getOutputSchema())
#pybind11#        outputTable2 = lsst.afw.table.SourceTable.make(mapper2.getOutputSchema())
#pybind11#        outputRecord1 = outputTable1.makeRecord()
#pybind11#        outputRecord2 = outputTable2.makeRecord()
#pybind11#        self.assertEqual(outputRecord1.getId(), outputRecord2.getId())
#pybind11#        self.assertNotEqual(outputRecord1.getId(), inputRecord.getId())
#pybind11#        outputRecord1.assign(inputRecord, mapper1)
#pybind11#        self.assertEqual(outputRecord1.getId(), inputRecord.getId())
#pybind11#        outputRecord2.assign(inputRecord, mapper2)
#pybind11#        self.assertNotEqual(outputRecord2.getId(), inputRecord.getId())
#pybind11#
#pybind11#    def testTicket2393(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        k = schema.addField(lsst.afw.table.Field[int]("i", "doc for i"))
#pybind11#        item = schema.find("i")
#pybind11#        self.assertEqual(k, item.key)
#pybind11#
#pybind11#    def testTicket2850(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        table = lsst.afw.table.BaseTable.make(schema)
#pybind11#        self.assertEqual(table.getBufferSize(), 0)
#pybind11#
#pybind11#    def testTicket2894(self):
#pybind11#        """Test boolean-array indexing of catalogs"""
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        key = schema.addField(lsst.afw.table.Field[int]("i", "doc for i"))
#pybind11#        cat1 = lsst.afw.table.BaseCatalog(schema)
#pybind11#        cat1.addNew().set(key, 1)
#pybind11#        cat1.addNew().set(key, 2)
#pybind11#        cat1.addNew().set(key, 3)
#pybind11#        cat2 = cat1[numpy.array([True, False, False], dtype=bool)]
#pybind11#        self.assertFloatsEqual(cat2[key], numpy.array([1], dtype=int))
#pybind11#        self.assertEqual(cat2[0], cat1[0])  # records compare using pointer equality
#pybind11#        cat3 = cat1[numpy.array([True, True, False], dtype=bool)]
#pybind11#        self.assertFloatsEqual(cat3[key], numpy.array([1, 2], dtype=int))
#pybind11#        cat4 = cat1[numpy.array([True, False, True], dtype=bool)]
#pybind11#        self.assertFloatsEqual(cat4.copy(deep=True)[key], numpy.array([1, 3], dtype=int))
#pybind11#
#pybind11#    def testTicket2938(self):
#pybind11#        """Test heterogenous catalogs that have records from multiple tables"""
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        schema.addField("i", type=int, doc="doc for i")
#pybind11#        cat = lsst.afw.table.BaseCatalog(schema)
#pybind11#        cat.addNew()
#pybind11#        t1 = lsst.afw.table.BaseTable.make(schema)
#pybind11#        cat.append(t1.makeRecord())
#pybind11#        self.assertEqual(cat[-1].getTable(), t1)
#pybind11#        with self.assertRaises(lsst.pex.exceptions.RuntimeError):
#pybind11#            cat.getColumnView()
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as filename:
#pybind11#            cat.writeFits(filename)  # shouldn't throw
#pybind11#            schema.addField("d", type=float, doc="doc for d")
#pybind11#            t2 = lsst.afw.table.BaseTable.make(schema)
#pybind11#            cat.append(t2.makeRecord())
#pybind11#            with self.assertRaises(lsst.pex.exceptions.LogicError):
#pybind11#                cat.writeFits(filename)
#pybind11#
#pybind11#    def testTicket3056(self):
#pybind11#        """Test sorting and sort-based searches of Catalogs"""
#pybind11#        schema = lsst.afw.table.SimpleTable.makeMinimalSchema()
#pybind11#        ki = schema.addField("i", type=int, doc="doc for i")
#pybind11#        kl = schema.addField("l", type=numpy.int64, doc="doc for l")
#pybind11#        kf = schema.addField("f", type=float, doc="doc for f")
#pybind11#        cat = lsst.afw.table.SimpleCatalog(schema)
#pybind11#        for j in range(50, 0, -1):
#pybind11#            record = cat.addNew()
#pybind11#            record.set(ki, j//10)
#pybind11#            record.set(kl, j)
#pybind11#            record.set(kf, numpy.random.randn())
#pybind11#        self.assertFalse(cat.isSorted(ki))
#pybind11#        self.assertFalse(cat.isSorted(kl))
#pybind11#        # sort by unique int64 field, try unique lookups
#pybind11#        cat.sort(kl)
#pybind11#        self.assertTrue(cat.isSorted(kl))
#pybind11#        r10 = cat.find(10, kl)
#pybind11#        self.assertEqual(r10.get(kl), 10)
#pybind11#        # sort by probably-unique float field, try unique and range lookups
#pybind11#        cat.sort(kf)
#pybind11#        self.assertTrue(cat.isSorted(kf))
#pybind11#        r10 = cat.find(10, kf)
#pybind11#        self.assertTrue(r10 is None or r10.get(kf) == 10.0)  # latter case virtually impossible
#pybind11#        i0 = cat.lower_bound(-0.5, kf)
#pybind11#        i1 = cat.upper_bound(0.5, kf)
#pybind11#        for i in range(i0, i1):
#pybind11#            self.assertGreaterEqual(cat[i].get(kf), -0.5)
#pybind11#            self.assertLess(cat[i].get(kf), 0.5)
#pybind11#        for r in cat[cat.between(-0.5, 0.5, kf)]:
#pybind11#            self.assertGreaterEqual(r.get(kf), -0.5)
#pybind11#            self.assertLess(r.get(kf), 0.5)
#pybind11#        # sort by nonunique int32 field, try range lookups
#pybind11#        cat.sort(ki)
#pybind11#        self.assertTrue(cat.isSorted(ki))
#pybind11#        s = cat.equal_range(3, ki)
#pybind11#        self.assertTrue(cat[s].isSorted(kf))  # test for stable sort
#pybind11#        for r in cat[s]:
#pybind11#            self.assertEqual(r.get(ki), 3)
#pybind11#        self.assertEqual(s.start, cat.lower_bound(3, ki))
#pybind11#        self.assertEqual(s.stop, cat.upper_bound(3, ki))
#pybind11#
#pybind11#    def testRename(self):
#pybind11#        """Test field-renaming functionality in Field, SchemaMapper"""
#pybind11#        field1i = lsst.afw.table.Field[int]("i1", "doc for i", "m")
#pybind11#        field2i = field1i.copyRenamed("i2")
#pybind11#        self.assertEqual(field1i.getName(), "i1")
#pybind11#        self.assertEqual(field2i.getName(), "i2")
#pybind11#        self.assertEqual(field1i.getDoc(), field2i.getDoc())
#pybind11#        self.assertEqual(field1i.getUnits(), field2i.getUnits())
#pybind11#        field1a = lsst.afw.table.Field["ArrayF"]("a1", "doc for a", "s", 3)
#pybind11#        field2a = field1a.copyRenamed("a2")
#pybind11#        self.assertEqual(field1a.getName(), "a1")
#pybind11#        self.assertEqual(field2a.getName(), "a2")
#pybind11#        self.assertEqual(field1a.getDoc(), field2a.getDoc())
#pybind11#        self.assertEqual(field1a.getUnits(), field2a.getUnits())
#pybind11#        self.assertEqual(field1a.getSize(), field2a.getSize())
#pybind11#        schema1 = lsst.afw.table.Schema()
#pybind11#        k1i = schema1.addField(field1i)
#pybind11#        k1a = schema1.addField(field1a)
#pybind11#        mapper = lsst.afw.table.SchemaMapper(schema1)
#pybind11#        k2i = mapper.addMapping(k1i, "i2")
#pybind11#        k2a = mapper.addMapping(k1a, "a2")
#pybind11#        schema2 = mapper.getOutputSchema()
#pybind11#        self.assertEqual(schema1.find(k1i).field.getName(), "i1")
#pybind11#        self.assertEqual(schema2.find(k2i).field.getName(), "i2")
#pybind11#        self.assertEqual(schema1.find(k1a).field.getName(), "a1")
#pybind11#        self.assertEqual(schema2.find(k2a).field.getName(), "a2")
#pybind11#        self.assertEqual(schema1.find(k1i).field.getDoc(), schema2.find(k2i).field.getDoc())
#pybind11#        self.assertEqual(schema1.find(k1a).field.getDoc(), schema2.find(k2a).field.getDoc())
#pybind11#        self.assertEqual(schema1.find(k1i).field.getUnits(), schema2.find(k2i).field.getUnits())
#pybind11#        self.assertEqual(schema1.find(k1a).field.getUnits(), schema2.find(k2a).field.getUnits())
#pybind11#        self.assertEqual(schema1.find(k1a).field.getSize(), schema2.find(k2a).field.getSize())
#pybind11#        k3i = mapper.addMapping(k1i, "i3")
#pybind11#        k3a = mapper.addMapping(k1a, "a3")
#pybind11#        schema3 = mapper.getOutputSchema()
#pybind11#        self.assertEqual(schema1.find(k1i).field.getName(), "i1")
#pybind11#        self.assertEqual(schema3.find(k3i).field.getName(), "i3")
#pybind11#        self.assertEqual(schema1.find(k1a).field.getName(), "a1")
#pybind11#        self.assertEqual(schema3.find(k3a).field.getName(), "a3")
#pybind11#        self.assertEqual(schema1.find(k1i).field.getDoc(), schema3.find(k3i).field.getDoc())
#pybind11#        self.assertEqual(schema1.find(k1a).field.getDoc(), schema3.find(k3a).field.getDoc())
#pybind11#        self.assertEqual(schema1.find(k1i).field.getUnits(), schema3.find(k3i).field.getUnits())
#pybind11#        self.assertEqual(schema1.find(k1a).field.getUnits(), schema3.find(k3a).field.getUnits())
#pybind11#        self.assertEqual(schema1.find(k1a).field.getSize(), schema3.find(k3a).field.getSize())
#pybind11#
#pybind11#    def testTicket3066(self):
#pybind11#        """Test the doReplace option on Schema.addField
#pybind11#        """
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        k1a = schema.addField("f1", doc="f1a", type="I")
#pybind11#        k2a = schema.addField("f2", doc="f2a", type="Flag")
#pybind11#        k3a = schema.addField("f3", doc="f3a", type="ArrayF", size=4)
#pybind11#        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
#pybind11#            schema.addField("f1", doc="f1b", type="I")
#pybind11#        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
#pybind11#            schema.addField("f2", doc="f2b", type="Flag")
#pybind11#        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
#pybind11#            schema.addField("f1", doc="f1b", type="F")
#pybind11#        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
#pybind11#            schema.addField("f2", doc="f2b", type="F")
#pybind11#        with self.assertRaises(lsst.pex.exceptions.TypeError):
#pybind11#            schema.addField("f1", doc="f1b", type="F", doReplace=True)
#pybind11#        with self.assertRaises(lsst.pex.exceptions.TypeError):
#pybind11#            schema.addField("f2", doc="f2b", type="F", doReplace=True)
#pybind11#        with self.assertRaises(lsst.pex.exceptions.TypeError):
#pybind11#            schema.addField("f3", doc="f3b", type="ArrayF",
#pybind11#                          size=3, doReplace=True)
#pybind11#        k1b = schema.addField("f1", doc="f1b", type="I", doReplace=True)
#pybind11#        self.assertEqual(k1a, k1b)
#pybind11#        self.assertEqual(schema.find(k1a).field.getDoc(), "f1b")
#pybind11#        k2b = schema.addField("f2", doc="f2b", type="Flag", doReplace=True)
#pybind11#        self.assertEqual(k2a, k2b)
#pybind11#        self.assertEqual(schema.find(k2a).field.getDoc(), "f2b")
#pybind11#        k3b = schema.addField("f3", doc="f3b", type="ArrayF", size=4, doReplace=True)
#pybind11#        self.assertEqual(k3a, k3b)
#pybind11#        self.assertEqual(schema.find(k3a).field.getDoc(), "f3b")
#pybind11#
#pybind11#    def testDM352(self):
#pybind11#        filename = os.path.join(os.path.split(__file__)[0], "data", "great3.fits")
#pybind11#        cat = lsst.afw.table.BaseCatalog.readFits(filename)
#pybind11#        self.assertEqual(len(cat), 1)
#pybind11#
#pybind11#    def testDM1710(self):
#pybind11#        # Extending without specifying a mapper or a deep argument should not
#pybind11#        # raise.
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        cat1 = lsst.afw.table.BaseCatalog(schema)
#pybind11#        cat2 = lsst.afw.table.BaseCatalog(schema)
#pybind11#        cat1.extend(cat2)
#pybind11#
#pybind11#    def testVariableLengthArrays(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        ka = schema.addField("a", doc="integer", type="ArrayI", size=0)
#pybind11#        kb = schema.addField("b", doc="single-precision", type="ArrayF")
#pybind11#        kc = schema.addField("c", doc="double-precision", type="ArrayD")
#pybind11#        cat1 = lsst.afw.table.BaseCatalog(schema)
#pybind11#        record1 = cat1.addNew()
#pybind11#        self.assertEqual(list(record1.get(ka)), [])
#pybind11#        self.assertEqual(list(record1.get(kb)), [])
#pybind11#        self.assertEqual(list(record1.get(kc)), [])
#pybind11#        a1 = numpy.random.randint(low=3, high=6, size=4).astype(numpy.int32)
#pybind11#        b1 = numpy.random.randn(5).astype(numpy.float32)
#pybind11#        c1 = numpy.random.randn(6).astype(numpy.float64)
#pybind11#        # Test get/set
#pybind11#        record1.set(ka, a1)
#pybind11#        record1.set(kb, b1)
#pybind11#        record1.set(kc, c1)
#pybind11#        self.assertFloatsEqual(record1.get(ka), a1)
#pybind11#        self.assertFloatsEqual(record1.get(kb), b1)
#pybind11#        self.assertFloatsEqual(record1.get(kc), c1)
#pybind11#        # Test __getitem__ and view semantics
#pybind11#        record1[kb][2] = 3.5
#pybind11#        self.assertEqual(b1[2], 3.5)
#pybind11#        # Check that we throw when we try to index a variable-length array Key
#pybind11#        self.assertRaisesLsstCpp(lsst.pex.exceptions.LogicError, lambda x: ka[x], 0)
#pybind11#        self.assertRaisesLsstCpp(lsst.pex.exceptions.LogicError, lambda x, y: ka[x:y], 0, 1)
#pybind11#        # Test copying records, both with and without SchemaMapper
#pybind11#        record2 = cat1.addNew()
#pybind11#        record2.assign(record1)
#pybind11#        self.assertFloatsEqual(record1.get(ka), a1)
#pybind11#        self.assertFloatsEqual(record1.get(kb), b1)
#pybind11#        self.assertFloatsEqual(record1.get(kc), c1)
#pybind11#        record1[kb][2] = 4.5
#pybind11#        self.assertEqual(record2[kb][2], 3.5)  # copy in assign() should be deep
#pybind11#        mapper = lsst.afw.table.SchemaMapper(schema)
#pybind11#        kb2 = mapper.addMapping(kb)
#pybind11#        cat2 = lsst.afw.table.BaseCatalog(mapper.getOutputSchema())
#pybind11#        record3 = cat2.addNew()
#pybind11#        record3.assign(record1, mapper)
#pybind11#        self.assertFloatsEqual(record3.get(kb2), b1)
#pybind11#        # Test that we throw if we try to get a column view of a variable-length arry
#pybind11#        self.assertRaisesLsstCpp(lsst.pex.exceptions.LogicError, cat1.get, ka)
#pybind11#        # Test that we can round-trip variable-length arrays through FITS
#pybind11#        filename = "testSimpleTable_testVariableLengthArrays.fits"
#pybind11#        cat1.writeFits(filename)
#pybind11#        cat3 = lsst.afw.table.BaseCatalog.readFits(filename)
#pybind11#        self.assertEqual(schema.compare(cat3.schema, lsst.afw.table.Schema.IDENTICAL),
#pybind11#                         lsst.afw.table.Schema.IDENTICAL)
#pybind11#        record4 = cat3[0]
#pybind11#        self.assertFloatsEqual(record4.get(ka), a1)
#pybind11#        self.assertFloatsEqual(record4.get(kb), b1)
#pybind11#        self.assertFloatsEqual(record4.get(kc), c1)
#pybind11#        os.remove(filename)
#pybind11#
#pybind11#    def testCompoundFieldFitsConversion(self):
#pybind11#        """Test that we convert compound fields saved with an older version of the pipeline
#pybind11#        into the set of multiple fields used by their replacement FunctorKeys.
#pybind11#        """
#pybind11#        geomValues = {
#pybind11#            "point_i_x": 4, "point_i_y": 5,
#pybind11#            "point_d_x": 3.5, "point_d_y": 2.0,
#pybind11#            "moments_xx": 5.0, "moments_yy": 6.5, "moments_xy": 2.25,
#pybind11#            "coord_ra": 1.0*lsst.afw.geom.radians, "coord_dec": 0.5*lsst.afw.geom.radians,
#pybind11#        }
#pybind11#        covValues = {
#pybind11#            "cov_z": numpy.array([[4.00, 1.25, 1.50, 0.75],
#pybind11#                                  [1.25, 2.25, 0.50, 0.25],
#pybind11#                                  [1.50, 0.50, 6.25, 1.75],
#pybind11#                                  [0.75, 0.25, 1.75, 9.00]], dtype=numpy.float32),
#pybind11#            "cov_p": numpy.array([[5.50, -2.0],
#pybind11#                                  [-2.0, 3.25]], dtype=numpy.float32),
#pybind11#            "cov_m": numpy.array([[3.75, -0.5, 1.25],
#pybind11#                                  [-0.5, 4.50, 0.75],
#pybind11#                                  [1.25, 0.75, 6.25]], dtype=numpy.float32),
#pybind11#        }
#pybind11#        filename = os.path.join(os.path.split(__file__)[0], "data", "CompoundFieldConversion.fits")
#pybind11#        cat2 = lsst.afw.table.BaseCatalog.readFits(filename)
#pybind11#        record2 = cat2[0]
#pybind11#        for k, v in geomValues.items():
#pybind11#            self.assertEqual(record2.get(k), v, msg=k)
#pybind11#        covZKey = lsst.afw.table.CovarianceMatrixXfKey(cat2.schema["cov_z"], ["0", "1", "2", "3"])
#pybind11#        covPKey = lsst.afw.table.CovarianceMatrix2fKey(cat2.schema["cov_p"], ["x", "y"])
#pybind11#        covMKey = lsst.afw.table.CovarianceMatrix3fKey(cat2.schema["cov_m"], ["xx", "yy", "xy"])
#pybind11#        self.assertFloatsAlmostEqual(record2.get(covZKey), covValues["cov_z"], rtol=1E-6)
#pybind11#        self.assertFloatsAlmostEqual(record2.get(covPKey), covValues["cov_p"], rtol=1E-6)
#pybind11#        self.assertFloatsAlmostEqual(record2.get(covMKey), covValues["cov_m"], rtol=1E-6)
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

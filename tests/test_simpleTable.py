# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os.path
import unittest

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.geom
import lsst.daf.base
import lsst.afw.table
import lsst.afw.fits

# Testing files live under this, in `data/`.
testPath = os.path.abspath(os.path.dirname(__file__))


def makeArray(size, dtype):
    return np.array(np.random.randn(size), dtype=dtype)


def makeCov(size, dtype):
    m = np.array(np.random.randn(size, size), dtype=dtype)
    # not quite symmetric for single-precision on some platforms
    r = np.dot(m, m.transpose())
    for i in range(r.shape[0]):
        for j in range(i):
            r[i, j] = r[j, i]
    return r


class SimpleTableTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        np.random.seed(1)

    def checkScalarAccessors(self, record, key, name, value1, value2):
        fastSetter = getattr(record, "set" + key.getTypeString())
        fastGetter = getattr(record, "get")
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
        self.assertIsNone(key.subfields)

    def checkArrayAccessors(self, record, key, name, value):
        fastSetter = getattr(record, "set" + key.getTypeString())
        fastGetter = getattr(record, "get")
        record.set(key, value)
        self.assertFloatsEqual(record.get(key), value)
        record.set(name, value)
        self.assertFloatsEqual(record.get(name), value)
        fastSetter(key, value)
        self.assertFloatsEqual(fastGetter(key), value)

    def testRecordAccess(self):
        schema = lsst.afw.table.Schema()
        kB = schema.addField("fB", type="B")
        kU = schema.addField("fU", type="U")
        kI = schema.addField("fI", type="I")
        kL = schema.addField("fL", type="L")
        kF = schema.addField("fF", type="F")
        kD = schema.addField("fD", type="D")
        kAngle = schema.addField("fAngle", type="Angle")
        kString = schema.addField("fString", type="String", size=4)
        kArrayB = schema.addField("fArrayB", type="ArrayB", size=6)
        kArrayU = schema.addField("fArrayU", type="ArrayU", size=2)
        kArrayI = schema.addField("fArrayI", type="ArrayI", size=3)
        kArrayF = schema.addField("fArrayF", type="ArrayF", size=4)
        kArrayD = schema.addField("fArrayD", type="ArrayD", size=5)
        table = lsst.afw.table.BaseTable.make(schema)
        record = table.makeRecord()
        self.assertEqual(record[kB], 0)
        self.assertEqual(record[kU], 0)
        self.assertEqual(record[kI], 0)
        self.assertEqual(record[kL], 0)
        self.assertTrue(np.isnan(record[kF]))
        self.assertTrue(np.isnan(record[kD]))
        self.checkScalarAccessors(record, kB, "fB", 4, 5)
        self.checkScalarAccessors(record, kU, "fU", 5, 6)
        self.checkScalarAccessors(record, kI, "fI", 2, 3)
        self.checkScalarAccessors(record, kL, "fL", 2, 3)
        self.checkScalarAccessors(record, kF, "fF", 2.5, 3.5)
        self.checkScalarAccessors(record, kD, "fD", 2.5, 3.5)
        self.checkScalarAccessors(record, kAngle, "fAngle",
                                  5.1*lsst.geom.degrees, -4.1*lsst.geom.degrees)
        self.checkScalarAccessors(record, kString, "fString", "ab", "abcd")
        self.checkArrayAccessors(record, kArrayB, "fArrayB",
                                 makeArray(kArrayB.getSize(), dtype=np.uint8))
        self.checkArrayAccessors(record, kArrayU, "fArrayU",
                                 makeArray(kArrayU.getSize(), dtype=np.uint16))
        self.checkArrayAccessors(record, kArrayI, "fArrayI",
                                 makeArray(kArrayI.getSize(), dtype=np.int32))
        self.checkArrayAccessors(record, kArrayF, "fArrayF",
                                 makeArray(kArrayF.getSize(), dtype=np.float32))
        self.checkArrayAccessors(record, kArrayD, "fArrayD",
                                 makeArray(kArrayD.getSize(), dtype=np.float64))
        for k in (kArrayF, kArrayD):
            self.assertEqual(k.subfields, tuple(range(k.getSize())))
        sub1 = kArrayD.slice(1, 3)
        sub2 = kArrayD[0:2]
        self.assertFloatsAlmostEqual(record.get(sub1),
                                     record.get(kArrayD)[1:3], rtol=0, atol=0)
        self.assertFloatsAlmostEqual(record.get(sub2),
                                     record.get(kArrayD)[0:2], rtol=0, atol=0)
        self.assertEqual(sub1[0], sub2[1])
        self.assertIsNone(kAngle.subfields)
        k0a = lsst.afw.table.Key["D"]()
        k0b = lsst.afw.table.Key["Flag"]()
        with self.assertRaises(lsst.pex.exceptions.LogicError):
            record.get(k0a)
        with self.assertRaises(lsst.pex.exceptions.LogicError):
            record.get(k0b)

    def _testBaseFits(self, target):
        schema = lsst.afw.table.Schema()
        k = schema.addField("f", type="D")
        cat1 = lsst.afw.table.BaseCatalog(schema)
        for i in range(50):
            record = cat1.addNew()
            record.set(k, np.random.randn())
        cat1.writeFits(target)
        cat2 = lsst.afw.table.BaseCatalog.readFits(target)
        self.assertEqual(len(cat1), len(cat2))
        for r1, r2 in zip(cat1, cat2):
            self.assertEqual(r1.get(k), r2.get(k))

    def testBaseFits(self):
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            self._testBaseFits(tmpFile)
        with self.assertRaises(Exception):
            lsst.afw.table.BaseCatalog.readFits("nonexistentfile.fits")

    def testMemoryFits(self):
        mem = lsst.afw.fits.MemFileManager()
        self._testBaseFits(mem)

    def testColumnView(self):
        schema = lsst.afw.table.Schema()
        kB = schema.addField("fB", type="B")
        kU = schema.addField("fU", type="U")
        kI = schema.addField("fI", type="I")
        kF = schema.addField("fF", type="F")
        kD = schema.addField("fD", type="D")
        kArrayF = schema.addField("fArrayF", type="ArrayF", size=2)
        kArrayD = schema.addField("fArrayD", type="ArrayD", size=3)
        kAngle = schema.addField("fAngle", type="Angle")
        kArrayU = schema.addField("fArrayU", type="ArrayU", size=4)
        catalog = lsst.afw.table.BaseCatalog(schema)
        catalog.addNew()
        catalog[0].set(kB, 5)
        catalog[0].set(kU, 1)
        catalog[0].set(kI, 2)
        catalog[0].set(kF, 0.5)
        catalog[0].set(kD, 0.25)
        catalog[0].set(kArrayF, np.array([-0.5, -0.25], dtype=np.float32))
        catalog[0].set(kArrayD, np.array([-1.5, -1.25, 3.375], dtype=np.float64))
        catalog[0].set(kAngle, lsst.geom.Angle(0.25))
        catalog[0].set(kArrayU, np.array([2, 3, 4, 1], dtype=np.uint16))
        col1a = catalog[kI]
        self.assertEqual(col1a.shape, (1,))
        catalog.addNew()
        catalog[1].set(kB, 6)
        catalog[1].set(kU, 4)
        catalog[1].set(kI, 3)
        catalog[1].set(kF, 2.5)
        catalog[1].set(kD, 0.75)
        catalog[1].set(kArrayF, np.array([-3.25, -0.75], dtype=np.float32))
        catalog[1].set(kArrayD, np.array([-1.25, -2.75, 0.625], dtype=np.float64))
        catalog[1].set(kAngle, lsst.geom.Angle(0.15))
        catalog[1].set(kArrayU, np.array([5, 6, 8, 7], dtype=np.uint16))
        col1b = catalog[kI]
        self.assertEqual(col1b.shape, (2,))
        columns = catalog.getColumnView()
        for key in [kB, kU, kI, kF, kD]:
            array = columns[key]
            for i in [0, 1]:
                self.assertEqual(array[i], catalog[i].get(key))
        for key in [kArrayF, kArrayD, kArrayU]:
            array = columns[key]
            for i in [0, 1]:
                self.assertFloatsEqual(array[i], catalog[i].get(key))
        for key in [kAngle]:
            array = columns[key]
            for i in [0, 1]:
                self.assertEqual(lsst.geom.Angle(array[i]),
                                 catalog[i].get(key))
        for key in [kB, kU, kI, kF, kD]:
            vals = columns[key].copy()
            vals *= 2
            array = columns[key]
            array *= 2
            for i in [0, 1]:
                self.assertEqual(catalog[i].get(key), vals[i])
                self.assertEqual(array[i], vals[i])
        catalog[kI] = 4
        f3v = np.random.randn(2)
        catalog["fD"] = f3v
        for i in [0, 1]:
            self.assertEqual(catalog[i].get(kI), 4)
            self.assertEqual(catalog[i].get(kD), f3v[i])

        # Accessing an invalid key should raise.
        for keyType in ["Angle", "ArrayB", "ArrayD", "ArrayF", "ArrayI",
                        "ArrayU", "B", "D", "F", "I", "L", "U"]:
            # Default-constructed key is invalid
            invalidKey = getattr(lsst.afw.table, f"Key{keyType}")()
            self.assertFalse(invalidKey.isValid())
            with self.assertRaises(lsst.pex.exceptions.LogicError):
                catalog.get(invalidKey)

    def testUnsignedFitsPersistence(self):
        """Test FITS round-trip of unsigned short ints, since FITS handles unsigned columns differently
        from signed columns.
        """
        schema = lsst.afw.table.Schema()
        k1 = schema.addField("f1", type=np.uint16, doc="scalar uint16")
        k2 = schema.addField("f2", type="ArrayU", doc="array uint16", size=4)
        cat1 = lsst.afw.table.BaseCatalog(schema)
        record1 = cat1.addNew()
        record1.set(k1, 4)
        record1.set(k2, np.array([5, 6, 7, 8], dtype=np.uint16))
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            cat1.writeFits(filename)
            cat2 = lsst.afw.table.BaseCatalog.readFits(filename)
        record2 = cat2[0]
        self.assertEqual(cat1.schema, cat2.schema)
        self.assertEqual(record1.get(k1), record2.get(k1))
        self.assertFloatsEqual(record1.get(k2), record2.get(k2))

    def testIteration(self):
        schema = lsst.afw.table.Schema()
        k = schema.addField("a", type=np.int32)
        catalog = lsst.afw.table.BaseCatalog(schema)
        for n in range(5):
            record = catalog.addNew()
            record[k] = n
        for n, r in enumerate(catalog):
            self.assertEqual(n, r[k])

    def testTicket2262(self):
        """Test that we can construct an array field in Python.
        """
        f1 = lsst.afw.table.Field["ArrayF"]("name", "doc", "barn", size=5)
        f2 = lsst.afw.table.Field["ArrayD"]("name", "doc", size=5)
        self.assertEqual(f1.getSize(), 5)
        self.assertEqual(f2.getSize(), 5)

    def testExtract(self):
        schema = lsst.afw.table.Schema()
        schema.addField("a_b_c1", type=np.float64)
        schema.addField("a_b_c2", type="Flag")
        schema.addField("a_d1", type=np.int32)
        schema.addField("a_d2", type=np.float32)
        pointKey = lsst.afw.table.Point2IKey.addFields(
            schema, "q_e1", "doc for point field", "pixel")
        schema.addField("q_e2_xxErr", type=np.float32)
        schema.addField("q_e2_yyErr", type=np.float32)
        schema.addField("q_e2_xyErr", type=np.float32)
        schema.addField("q_e2_xx_yy_Cov", type=np.float32)
        schema.addField("q_e2_xx_xy_Cov", type=np.float32)
        schema.addField("q_e2_yy_xy_Cov", type=np.float32)
        covKey = lsst.afw.table.CovarianceMatrix3fKey(
            schema["q_e2"], ["xx", "yy", "xy"])
        self.assertEqual(
            list(schema.extract("a_b_*", ordered=True).keys()), ["a_b_c1", "a_b_c2"])
        self.assertEqual(
            list(schema.extract("*1", ordered=True).keys()), ["a_b_c1", "a_d1"])
        self.assertEqual(list(schema.extract("a_b_*", "*2", ordered=True).keys()),
                         ["a_b_c1", "a_b_c2", "a_d2"])
        self.assertEqual(list(schema.extract(regex=r"a_(.+)1", sub=r"\1f",
                                             ordered=True).keys()), ["b_cf", "df"])
        catalog = lsst.afw.table.BaseCatalog(schema)
        for i in range(5):
            record = catalog.addNew()
            record.set("a_b_c1", np.random.randn())
            record.set("a_b_c2", True)
            record.set("a_d1", np.random.randint(100))
            record.set("a_d2", np.random.randn())
            record.set(pointKey,
                       lsst.geom.Point2I(np.random.randint(10), np.random.randint(10)))
            record.set(covKey, np.random.randn(3, 3).astype(np.float32))
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
        boolIdx = np.array([True, False, False, True, True])
        for kwds, idx in [
            ({}, allIdx),
            ({"copy": True}, allIdx),
            ({"where": boolIdx}, boolIdx),
            ({"where": sliceIdx}, sliceIdx),
            ({"where": boolIdx, "copy": True}, boolIdx),
            ({"where": sliceIdx, "copy": True}, sliceIdx),
        ]:
            d = catalog.extract("*", **kwds)
            np.testing.assert_array_equal(
                d["a_b_c1"], catalog.get("a_b_c1")[idx])
            np.testing.assert_array_equal(
                d["a_d1"], catalog.get("a_d1")[idx])
            np.testing.assert_array_equal(
                d["a_d2"], catalog.get("a_d2")[idx])
            np.testing.assert_array_equal(
                d["q_e1_x"], catalog.get("q_e1_x")[idx])
            np.testing.assert_array_equal(
                d["q_e1_y"], catalog.get("q_e1_y")[idx])
            if "copy" in kwds or idx is boolIdx:
                for col in d.values():
                    self.assertTrue(col.flags.c_contiguous)
        # Test that aliases are included in extract()
        schema.getAliasMap().set("b_f", "a_b")
        d = schema.extract("b_f*")
        self.assertEqual(sorted(d.keys()), ["b_f_c1", "b_f_c2"])

    def testExtend(self):
        schema1 = lsst.afw.table.SourceTable.makeMinimalSchema()
        k1 = schema1.addField("f1", type=np.int32)
        k2 = schema1.addField("f2", type=np.float64)
        cat1 = lsst.afw.table.BaseCatalog(schema1)
        for i in range(1000):
            record = cat1.addNew()
            record.setI(k1, i)
            record.setD(k2, np.random.randn())
        self.assertFalse(cat1.isContiguous())
        cat2 = lsst.afw.table.BaseCatalog(schema1)
        cat2.extend(cat1, deep=True)
        self.assertEqual(len(cat1), len(cat2))
        self.assertTrue(cat2.isContiguous())
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
        self.assertTrue(mapper.getOutputSchema().contains(
            lsst.afw.table.SourceTable.makeMinimalSchema()))
        cat5 = lsst.afw.table.BaseCatalog(schema2)
        cat5.extend(cat1, mapper=mapper)
        self.assertTrue(cat5.isContiguous())
        cat6 = lsst.afw.table.SourceCatalog(schema2)
        cat6.extend(list(cat1), mapper=mapper)
        self.assertFalse(cat6.isContiguous())
        cat7 = lsst.afw.table.SourceCatalog(schema2)
        cat7.reserve(len(cat1)*3)
        cat7.extend(list(cat1), mapper=mapper)
        cat7.extend(cat1, mapper)
        cat7.extend(list(cat1), mapper)
        self.assertTrue(cat7.isContiguous())
        cat8 = lsst.afw.table.BaseCatalog(schema2)
        cat8.extend(list(cat7), True)
        cat8.extend(list(cat7), deep=True)

    def testTicket2308(self):
        inputSchema = lsst.afw.table.SourceTable.makeMinimalSchema()
        mapper1 = lsst.afw.table.SchemaMapper(inputSchema)
        mapper1.addMinimalSchema(
            lsst.afw.table.SourceTable.makeMinimalSchema(), True)
        mapper2 = lsst.afw.table.SchemaMapper(inputSchema)
        mapper2.addMinimalSchema(
            lsst.afw.table.SourceTable.makeMinimalSchema(), False)
        inputTable = lsst.afw.table.SourceTable.make(inputSchema)
        inputRecord = inputTable.makeRecord()
        inputRecord.set("id", 42)
        outputTable1 = lsst.afw.table.SourceTable.make(
            mapper1.getOutputSchema())
        outputTable2 = lsst.afw.table.SourceTable.make(
            mapper2.getOutputSchema())
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
        k = schema.addField(lsst.afw.table.Field[np.int32]("i", "doc for i"))
        item = schema.find("i")
        self.assertEqual(k, item.key)

    def testTicket2850(self):
        schema = lsst.afw.table.Schema()
        table = lsst.afw.table.BaseTable.make(schema)
        self.assertEqual(table.getBufferSize(), 0)

    def testTicket2894(self):
        """Test boolean-array indexing of catalogs.
        """
        schema = lsst.afw.table.Schema()
        key = schema.addField(lsst.afw.table.Field[np.int32]("i", "doc for i"))
        cat1 = lsst.afw.table.BaseCatalog(schema)
        cat1.addNew().set(key, 1)
        cat1.addNew().set(key, 2)
        cat1.addNew().set(key, 3)
        cat2 = cat1[np.array([True, False, False], dtype=bool)]
        self.assertFloatsEqual(cat2[key], np.array([1], dtype=np.int32))
        # records compare using pointer equality
        self.assertEqual(cat2[0], cat1[0])
        cat3 = cat1[np.array([True, True, False], dtype=bool)]
        self.assertFloatsEqual(cat3[key], np.array([1, 2], dtype=np.int32))
        cat4 = cat1[np.array([True, False, True], dtype=bool)]
        self.assertFloatsEqual(cat4.copy(deep=True)[
                               key], np.array([1, 3], dtype=np.int32))

    def testTicket2938(self):
        """Test heterogenous catalogs that have records from multiple tables.
        """
        schema = lsst.afw.table.Schema()
        schema.addField("i", type=np.int32, doc="doc for i")
        cat = lsst.afw.table.BaseCatalog(schema)
        cat.addNew()
        t1 = lsst.afw.table.BaseTable.make(schema)
        cat.append(t1.makeRecord())
        self.assertEqual(cat[-1].getTable(), t1)
        with self.assertRaises(lsst.pex.exceptions.RuntimeError):
            cat.getColumnView()
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            cat.writeFits(filename)  # shouldn't throw
            schema.addField("d", type=np.float64, doc="doc for d")
            t2 = lsst.afw.table.BaseTable.make(schema)
            cat.append(t2.makeRecord())
            with self.assertRaises(lsst.pex.exceptions.LogicError):
                cat.writeFits(filename)

    def testTicket3056(self):
        """Test sorting and sort-based searches of Catalogs.
        """
        schema = lsst.afw.table.SimpleTable.makeMinimalSchema()
        ki = schema.addField("i", type=np.int32, doc="doc for i")
        kl = schema.addField("l", type=np.int64, doc="doc for l")
        kf = schema.addField("f", type=np.float64, doc="doc for f")
        cat = lsst.afw.table.SimpleCatalog(schema)
        for j in range(50, 0, -1):
            record = cat.addNew()
            record.set(ki, j//10)
            record.set(kl, j)
            record.set(kf, np.random.randn())
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
        # latter case virtually impossible
        self.assertTrue(r10 is None or r10.get(kf) == 10.0)
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
        """Test field-renaming functionality in Field, SchemaMapper.
        """
        field1i = lsst.afw.table.Field[np.int32]("i1", "doc for i", "m")
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
        self.assertEqual(schema1.find(k1i).field.getDoc(),
                         schema2.find(k2i).field.getDoc())
        self.assertEqual(schema1.find(k1a).field.getDoc(),
                         schema2.find(k2a).field.getDoc())
        self.assertEqual(schema1.find(k1i).field.getUnits(),
                         schema2.find(k2i).field.getUnits())
        self.assertEqual(schema1.find(k1a).field.getUnits(),
                         schema2.find(k2a).field.getUnits())
        self.assertEqual(schema1.find(k1a).field.getSize(),
                         schema2.find(k2a).field.getSize())
        k3i = mapper.addMapping(k1i, "i3")
        k3a = mapper.addMapping(k1a, "a3")
        schema3 = mapper.getOutputSchema()
        self.assertEqual(schema1.find(k1i).field.getName(), "i1")
        self.assertEqual(schema3.find(k3i).field.getName(), "i3")
        self.assertEqual(schema1.find(k1a).field.getName(), "a1")
        self.assertEqual(schema3.find(k3a).field.getName(), "a3")
        self.assertEqual(schema1.find(k1i).field.getDoc(),
                         schema3.find(k3i).field.getDoc())
        self.assertEqual(schema1.find(k1a).field.getDoc(),
                         schema3.find(k3a).field.getDoc())
        self.assertEqual(schema1.find(k1i).field.getUnits(),
                         schema3.find(k3i).field.getUnits())
        self.assertEqual(schema1.find(k1a).field.getUnits(),
                         schema3.find(k3a).field.getUnits())
        self.assertEqual(schema1.find(k1a).field.getSize(),
                         schema3.find(k3a).field.getSize())

    def testTicket3066(self):
        """Test the doReplace option on Schema.addField.
        """
        schema = lsst.afw.table.Schema()
        k1a = schema.addField("f1", doc="f1a", type="I")
        k2a = schema.addField("f2", doc="f2a", type="Flag")
        k3a = schema.addField("f3", doc="f3a", type="ArrayF", size=4)
        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
            schema.addField("f1", doc="f1b", type="I")
        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
            schema.addField("f2", doc="f2b", type="Flag")
        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
            schema.addField("f1", doc="f1b", type="F")
        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
            schema.addField("f2", doc="f2b", type="F")
        with self.assertRaises(lsst.pex.exceptions.TypeError):
            schema.addField("f1", doc="f1b", type="F", doReplace=True)
        with self.assertRaises(lsst.pex.exceptions.TypeError):
            schema.addField("f2", doc="f2b", type="F", doReplace=True)
        with self.assertRaises(lsst.pex.exceptions.TypeError):
            schema.addField("f3", doc="f3b", type="ArrayF",
                            size=3, doReplace=True)
        k1b = schema.addField("f1", doc="f1b", type="I", doReplace=True)
        self.assertEqual(k1a, k1b)
        self.assertEqual(schema.find(k1a).field.getDoc(), "f1b")
        k2b = schema.addField("f2", doc="f2b", type="Flag", doReplace=True)
        self.assertEqual(k2a, k2b)
        self.assertEqual(schema.find(k2a).field.getDoc(), "f2b")
        k3b = schema.addField(
            "f3", doc="f3b", type="ArrayF", size=4, doReplace=True)
        self.assertEqual(k3a, k3b)
        self.assertEqual(schema.find(k3a).field.getDoc(), "f3b")

    def testDM352(self):
        filename = os.path.join(os.path.split(__file__)[0],
                                "data", "great3.fits")
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
        kArrayB = schema.addField("fArrayB", doc="uint8", type="ArrayB", size=0)
        kArrayU = schema.addField("fArrayU", doc="uint16", type="ArrayU", size=0)
        kArrayI = schema.addField("fArrayI", doc="int32", type="ArrayI", size=0)
        kArrayF = schema.addField("fArrayF", doc="single-precision", type="ArrayF")
        kArrayD = schema.addField("fArrayD", doc="double-precision", type="ArrayD")
        kString = schema.addField("fString", doc="string", type="String", size=0)
        cat1 = lsst.afw.table.BaseCatalog(schema)
        record1 = cat1.addNew()
        self.assertEqual(list(record1.get(kArrayB)), [])
        self.assertEqual(list(record1.get(kArrayU)), [])
        self.assertEqual(list(record1.get(kArrayI)), [])
        self.assertEqual(list(record1.get(kArrayF)), [])
        self.assertEqual(list(record1.get(kArrayD)), [])
        self.assertEqual(record1.get(kString), "")
        dataB = np.random.randint(low=3, high=6, size=4).astype(np.uint8)
        dataU = np.random.randint(low=3, high=6, size=4).astype(np.uint16)
        dataI = np.random.randint(low=3, high=6, size=4).astype(np.int32)
        dataF = np.random.randn(5).astype(np.float32)
        dataD = np.random.randn(6).astype(np.float64)
        dataString = "the\nquick\tbrown\rfox jumps over the lazy dog"
        # Test get/set
        record1.set(kArrayB, dataB)
        record1.set(kArrayU, dataU)
        record1.set(kArrayI, dataI)
        record1.set(kArrayF, dataF)
        record1.set(kArrayD, dataD)
        record1.set(kString, dataString)
        self.assertFloatsEqual(record1.get(kArrayB), dataB)
        self.assertFloatsEqual(record1.get(kArrayU), dataU)
        self.assertFloatsEqual(record1.get(kArrayI), dataI)
        self.assertFloatsEqual(record1.get(kArrayF), dataF)
        self.assertFloatsEqual(record1.get(kArrayD), dataD)
        self.assertEqual(record1.get(kString), dataString)
        # Test __getitem__ and view semantics
        record1[kArrayF][2] = 3.5
        self.assertEqual(dataF[2], 3.5)
        # Check that we throw when we try to index a variable-length array Key
        with self.assertRaises(lsst.pex.exceptions.LogicError):
            kArrayI[0]
        with self.assertRaises(lsst.pex.exceptions.LogicError):
            kArrayI[0:1]
        # Test copying records, both with and without SchemaMapper
        record2 = cat1.addNew()
        record2.assign(record1)
        self.assertFloatsEqual(record2.get(kArrayB), dataB)
        self.assertFloatsEqual(record2.get(kArrayU), dataU)
        self.assertFloatsEqual(record2.get(kArrayI), dataI)
        self.assertFloatsEqual(record2.get(kArrayF), dataF)
        self.assertFloatsEqual(record2.get(kArrayD), dataD)
        self.assertEqual(record2.get(kString), dataString)
        record1[kArrayF][2] = 4.5
        # copy in assign() should be deep
        self.assertEqual(record2[kArrayF][2], 3.5)
        mapper = lsst.afw.table.SchemaMapper(schema)
        kb2 = mapper.addMapping(kArrayF)
        cat2 = lsst.afw.table.BaseCatalog(mapper.getOutputSchema())
        record3 = cat2.addNew()
        record3.assign(record1, mapper)
        self.assertFloatsEqual(record3.get(kb2), dataF)
        # Test that we throw if we try to get a column view of a
        # variable-length arry
        with self.assertRaises(lsst.pex.exceptions.LogicError):
            cat1.get(kArrayI)
        # Test that we can round-trip variable-length arrays through FITS
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            cat1.writeFits(filename)
            cat3 = lsst.afw.table.BaseCatalog.readFits(filename)
        self.assertEqual(schema.compare(cat3.schema, lsst.afw.table.Schema.IDENTICAL),
                         lsst.afw.table.Schema.IDENTICAL)
        record4 = cat3[0]
        np.testing.assert_array_equal(record4.get(kArrayB), dataB)
        np.testing.assert_array_equal(record4.get(kArrayU), dataU)
        np.testing.assert_array_equal(record4.get(kArrayI), dataI)
        np.testing.assert_array_equal(record4.get(kArrayF), dataF)
        np.testing.assert_array_equal(record4.get(kArrayD), dataD)
        self.assertEqual(record4.get(kString), dataString)

    def testCompoundFieldFitsConversion(self):
        """Test that we convert compound fields saved with an older version of the pipeline
        into the set of multiple fields used by their replacement FunctorKeys.
        """
        geomValues = {
            "point_i_x": 4, "point_i_y": 5,
            "point_d_x": 3.5, "point_d_y": 2.0,
            "moments_xx": 5.0, "moments_yy": 6.5, "moments_xy": 2.25,
            "coord_ra": 1.0*lsst.geom.radians, "coord_dec": 0.5*lsst.geom.radians,
        }
        covValues = {
            "cov_z": np.array([[4.00, 1.25, 1.50, 0.75],
                               [1.25, 2.25, 0.50, 0.25],
                               [1.50, 0.50, 6.25, 1.75],
                               [0.75, 0.25, 1.75, 9.00]], dtype=np.float32),
            "cov_p": np.array([[5.50, -2.0],
                               [-2.0, 3.25]], dtype=np.float32),
            "cov_m": np.array([[3.75, -0.5, 1.25],
                               [-0.5, 4.50, 0.75],
                               [1.25, 0.75, 6.25]], dtype=np.float32),
        }
        filename = os.path.join(os.path.split(__file__)[0],
                                "data", "CompoundFieldConversion.fits")
        cat2 = lsst.afw.table.BaseCatalog.readFits(filename)
        record2 = cat2[0]
        for k, v in geomValues.items():
            self.assertEqual(record2.get(k), v, msg=k)
        covZKey = lsst.afw.table.CovarianceMatrixXfKey(
            cat2.schema["cov_z"], ["0", "1", "2", "3"])
        covPKey = lsst.afw.table.CovarianceMatrix2fKey(
            cat2.schema["cov_p"], ["x", "y"])
        covMKey = lsst.afw.table.CovarianceMatrix3fKey(
            cat2.schema["cov_m"], ["xx", "yy", "xy"])
        self.assertFloatsAlmostEqual(record2.get(covZKey),
                                     covValues["cov_z"], rtol=1E-6)
        self.assertFloatsAlmostEqual(record2.get(covPKey),
                                     covValues["cov_p"], rtol=1E-6)
        self.assertFloatsAlmostEqual(record2.get(covMKey),
                                     covValues["cov_m"], rtol=1E-6)

    def testFitsReadVersion1Compatibility(self):
        """Test that v1 SimpleCatalogs read from FITS get correct aliases.
        """
        filename = os.path.join(testPath, "data", "ps1-refcat-v1.fits")
        catalog = lsst.afw.table.SimpleCatalog.readFits(filename)
        self.assertIn('g_flux', catalog.schema)
        self.assertNotIn('g_instFlux', catalog.schema)

        self.assertIn('g_fluxErr', catalog.schema)
        self.assertNotIn('g_instFluxErr', catalog.schema)

    def testDelete(self):
        schema = lsst.afw.table.Schema()
        key = schema.addField("a", type=np.float64, doc="doc for 'a'")
        catalog = lsst.afw.table.BaseCatalog(schema)
        for i in range(10):
            catalog.addNew().set(key, i)
        del catalog[4]
        self.assertEqual(len(catalog), 9)
        self.assertEqual([r.get(key) for r in catalog],
                         [0, 1, 2, 3, 5, 6, 7, 8, 9])
        del catalog[4:7]
        self.assertEqual(len(catalog), 6)
        self.assertEqual([r.get(key) for r in catalog],
                         [0, 1, 2, 3, 8, 9])
        with self.assertRaises(IndexError):
            del catalog[1:3:-1]
        with self.assertRaises(IndexError):
            del catalog[:4:2]
        with self.assertRaises(IndexError):
            del catalog[50]

    def testSetFlagColumn(self):
        schema = lsst.afw.table.Schema()
        key = schema.addField("a", type="Flag", doc="doc for 'a'")
        catalog = lsst.afw.table.BaseCatalog(schema)
        catalog.resize(5)
        # Set scalar with key.
        catalog[key] = True
        self.assertEqual(list(catalog[key]), [True] * 5)
        # Set scalar with name.
        catalog["a"] = False
        self.assertEqual(list(catalog[key]), [False] * 5)
        # Set array with key.
        v1 = np.array([True, False, True, False, True], dtype=bool)
        catalog[key] = v1
        self.assertEqual(list(catalog[key]), list(v1))
        # Set array with name.
        v2 = np.array([False, True, False, True, False], dtype=bool)
        catalog["a"] = v2
        self.assertEqual(list(catalog[key]), list(v2))

    def testAngleColumnArrayAccess(self):
        """Test column-array access to Angle columns on both contiguous and
        non-contiguous arrays.
        """
        schema = lsst.afw.table.Schema()
        key = schema.addField("a", type="Angle", doc="doc for a")
        catalog = lsst.afw.table.BaseCatalog(schema)
        catalog.resize(2)
        self.assertTrue(catalog.isContiguous())
        catalog[key] = np.array([3.0, 4.0])
        self.assertFloatsEqual(catalog[key], np.array([3.0, 4.0]))
        record = catalog.addNew()
        self.assertFalse(catalog.isContiguous())
        record[key] = 5.0 * lsst.geom.radians
        self.assertFloatsEqual(catalog[key], np.array([3.0, 4.0, 5.0]))
        self.assertFalse(catalog[key].flags.writeable)

        # Test that non-contiugous catalog can support hasattr of missing
        # attributes.
        self.assertFalse(hasattr(catalog, "__qualname__"))
        self.assertTrue(hasattr(type(catalog), "__qualname__"))

    def testArrayColumnArrayAccess(self):
        """Test column-array access to Array columns on both contiguous and
        non-contiguous arrays.
        """
        schema = lsst.afw.table.Schema()
        key = schema.addField("a", type="ArrayD", doc="doc for a", size=3)
        catalog = lsst.afw.table.BaseCatalog(schema)
        catalog.resize(2)
        self.assertTrue(catalog.isContiguous())
        catalog[key] = np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        self.assertFloatsEqual(catalog[key], np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]))
        record = catalog.addNew()
        self.assertFalse(catalog.isContiguous())
        record[key] = np.array([9.0, 10.0, 11.0])
        self.assertFloatsEqual(catalog[key], np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]))
        self.assertFalse(catalog[key].flags.writeable)

    def testMetadataProperty(self):
        """Test that the metadata property of BaseTable works as expected.
        """
        schema = lsst.afw.table.Schema()
        table = lsst.afw.table.BaseTable.make(schema)
        # BaseTable should not have a metadata property on construction.
        self.assertIsNone(table.metadata)
        metadata = lsst.daf.base.PropertyList()
        metadata["one"] = 1
        table.metadata = metadata
        self.assertEqual(table.metadata["one"], 1)

        # SimpleTable should have an empty metadata property on construction.
        schema = lsst.afw.table.SimpleTable.makeMinimalSchema()
        table = lsst.afw.table.SimpleTable.make(schema)
        self.assertEqual(table.metadata, lsst.daf.base.PropertyList())


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import zip
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
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
#pybind11#Tests for table.Schema
#pybind11#
#pybind11#Run with:
#pybind11#   ./testSchema.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import testSchema; testSchema.run()
#pybind11#"""
#pybind11#
#pybind11#import unittest
#pybind11#import numpy
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.table
#pybind11#import lsst.afw.geom
#pybind11#import lsst.afw.coord
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class SchemaTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def testSchema(self):
#pybind11#        def testKey(name, key):
#pybind11#            col = schema.find(name)
#pybind11#            self.assertEqual(col.key, key)
#pybind11#            self.assertEqual(col.field.getName(), name)
#pybind11#
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        ab_k = lsst.afw.table.CoordKey.addFields(schema, "a_b", "parent coord")
#pybind11#        abp_k = lsst.afw.table.Point2DKey.addFields(schema, "a_b_p", "point", "pixel")
#pybind11#        abi_k = schema.addField("a_b_i", type=int, doc="int")
#pybind11#        acf_k = schema.addField("a_c_f", type=numpy.float32, doc="float")
#pybind11#        egd_k = schema.addField("e_g_d", type=lsst.afw.geom.Angle, doc="angle")
#pybind11#
#pybind11#        # Basic test for all native key types.
#pybind11#        for name, key in (("a_b_i", abi_k), ("a_c_f", acf_k), ("e_g_d", egd_k)):
#pybind11#            testKey(name, key)
#pybind11#
#pybind11#        # Extra tests for special types
#pybind11#        self.assertEqual(ab_k.getRa(), schema["a_b_ra"].asKey())
#pybind11#        abpx_si = schema.find("a_b_p_x")
#pybind11#        self.assertEqual(abp_k.getX(), abpx_si.key)
#pybind11#        self.assertEqual(abpx_si.field.getName(), "a_b_p_x")
#pybind11#        self.assertEqual(abpx_si.field.getDoc(), "point")
#pybind11#        self.assertEqual(abp_k.getX(), schema["a_b_p_x"].asKey())
#pybind11#        self.assertEqual(schema.getNames(), ('a_b_dec', 'a_b_i', 'a_b_p_x', 'a_b_p_y', 'a_b_ra', 'a_c_f',
#pybind11#                                             'e_g_d'))
#pybind11#        self.assertEqual(schema.getNames(True), ("a", "e"))
#pybind11#        self.assertEqual(schema["a"].getNames(), ('b_dec', 'b_i', 'b_p_x', 'b_p_y', 'b_ra', 'c_f'))
#pybind11#        self.assertEqual(schema["a"].getNames(True), ("b", "c"))
#pybind11#        schema2 = lsst.afw.table.Schema(schema)
#pybind11#        self.assertEqual(schema, schema2)
#pybind11#        schema2.addField("q", type=float, doc="another double")
#pybind11#        self.assertNotEqual(schema, schema2)
#pybind11#        schema3 = lsst.afw.table.Schema()
#pybind11#        schema3.addField("ra", type="Angle", doc="coord_ra")
#pybind11#        schema3.addField("dec", type="Angle", doc="coord_dec")
#pybind11#        schema3.addField("x", type="D", doc="position_x")
#pybind11#        schema3.addField("y", type="D", doc="position_y")
#pybind11#        schema3.addField("i", type="I", doc="int")
#pybind11#        schema3.addField("f", type="F", doc="float")
#pybind11#        schema3.addField("d", type="Angle", doc="angle")
#pybind11#        self.assertEqual(schema3, schema)
#pybind11#        schema4 = lsst.afw.table.Schema()
#pybind11#        keys = []
#pybind11#        keys.append(schema4.addField("a", type="Angle", doc="a"))
#pybind11#        keys.append(schema4.addField("b", type="Flag", doc="b"))
#pybind11#        keys.append(schema4.addField("c", type=int, doc="c"))
#pybind11#        keys.append(schema4.addField("d", type="Flag", doc="d"))
#pybind11#        self.assertEqual(keys[1].getBit(), 0)
#pybind11#        self.assertEqual(keys[3].getBit(), 1)
#pybind11#        for n1, n2 in zip(schema4.getOrderedNames(), "abcd"):
#pybind11#            self.assertEqual(n1, n2)
#pybind11#        keys2 = [x.key for x in schema4]
#pybind11#        self.assertEqual(keys, keys2)
#pybind11#
#pybind11#    def testUnits(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        # first insert some valid units
#pybind11#        schema.addField("a", type="I", units="pixel")
#pybind11#        schema.addField("b", type="I", units="m2")
#pybind11#        schema.addField("c", type="I", units="electron / adu")
#pybind11#        schema.addField("d", type="I", units="kg m s^(-2)")
#pybind11#        schema.addField("e", type="I", units="GHz / Mpc")
#pybind11#        schema.addField("f", type="Angle", units="deg")
#pybind11#        schema.addField("g", type="Angle", units="rad")
#pybind11#        schema.checkUnits()
#pybind11#        # now try inserting invalid units
#pybind11#        with self.assertRaises(ValueError):
#pybind11#            schema.addField("a", type="I", units="camel")
#pybind11#        with self.assertRaises(ValueError):
#pybind11#            schema.addField("b", type="I", units="pixels^2^2")
#pybind11#        # add invalid units in silent mode, should work fine
#pybind11#        schema.addField("h", type="I", units="lala", parse_strict='silent')
#pybind11#        # Now this check should raise because there is an invalid unit
#pybind11#        with self.assertRaises(ValueError):
#pybind11#            schema.checkUnits()
#pybind11#
#pybind11#    def testInspection(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        keys = []
#pybind11#        keys.append(schema.addField("d", type=int))
#pybind11#        keys.append(schema.addField("c", type=float))
#pybind11#        keys.append(schema.addField("b", type="ArrayF", size=3))
#pybind11#        keys.append(schema.addField("a", type="F"))
#pybind11#        for key, item in zip(keys, schema):
#pybind11#            self.assertEqual(item.key, key)
#pybind11#            self.assertIn(key, schema)
#pybind11#        for name in ("a", "b", "c", "d"):
#pybind11#            self.assertIn(name, schema)
#pybind11#        self.assertNotIn("e", schema)
#pybind11#        otherSchema = lsst.afw.table.Schema()
#pybind11#        otherKey = otherSchema.addField("d", type=float)
#pybind11#        self.assertNotIn(otherKey, schema)
#pybind11#        self.assertNotEqual(keys[0], keys[1])
#pybind11#
#pybind11#    def testKeyAccessors(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        arrayKey = schema.addField("a", type="ArrayF", doc="doc for array field", size=5)
#pybind11#        arrayElementKey = arrayKey[1]
#pybind11#        self.assertEqual(lsst.afw.table.Key["F"], type(arrayElementKey))
#pybind11#
#pybind11#    def testComparison(self):
#pybind11#        schema1 = lsst.afw.table.Schema()
#pybind11#        schema1.addField("a", type=float, doc="doc for a", units="m")
#pybind11#        schema1.addField("b", type=int, doc="doc for b", units="s")
#pybind11#        schema2 = lsst.afw.table.Schema()
#pybind11#        schema2.addField("a", type=int, doc="doc for a", units="m")
#pybind11#        schema2.addField("b", type=float, doc="doc for b", units="s")
#pybind11#        cmp1 = schema1.compare(schema2, lsst.afw.table.Schema.IDENTICAL)
#pybind11#        self.assertTrue(cmp1 & lsst.afw.table.Schema.EQUAL_NAMES)
#pybind11#        self.assertTrue(cmp1 & lsst.afw.table.Schema.EQUAL_DOCS)
#pybind11#        self.assertTrue(cmp1 & lsst.afw.table.Schema.EQUAL_UNITS)
#pybind11#        self.assertFalse(cmp1 & lsst.afw.table.Schema.EQUAL_KEYS)
#pybind11#        schema3 = lsst.afw.table.Schema(schema1)
#pybind11#        schema3.addField("c", type=str, doc="doc for c", size=4)
#pybind11#        self.assertFalse(schema1.compare(schema3))
#pybind11#        self.assertFalse(schema1.contains(schema3))
#pybind11#        self.assertTrue(schema3.contains(schema1))
#pybind11#        schema1.addField("d", type=str, doc="no docs!", size=4)
#pybind11#        cmp2 = schema1.compare(schema3, lsst.afw.table.Schema.IDENTICAL)
#pybind11#        self.assertFalse(cmp2 & lsst.afw.table.Schema.EQUAL_NAMES)
#pybind11#        self.assertFalse(cmp2 & lsst.afw.table.Schema.EQUAL_DOCS)
#pybind11#        self.assertTrue(cmp2 & lsst.afw.table.Schema.EQUAL_KEYS)
#pybind11#        self.assertTrue(cmp2 & lsst.afw.table.Schema.EQUAL_UNITS)
#pybind11#        self.assertFalse(schema1.compare(schema3, lsst.afw.table.Schema.EQUAL_NAMES))
#pybind11#
#pybind11#
#pybind11#class SchemaMapperTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def testJoin(self):
#pybind11#        inputs = [lsst.afw.table.Schema(), lsst.afw.table.Schema(), lsst.afw.table.Schema()]
#pybind11#        inputs = lsst.afw.table.SchemaVector(inputs)
#pybind11#        prefixes = ["u", "v", "w"]
#pybind11#        ka = inputs[0].addField("a", type=numpy.float64, doc="doc for a")
#pybind11#        kb = inputs[0].addField("b", type=numpy.int32, doc="doc for b")
#pybind11#        kc = inputs[1].addField("c", type=numpy.float32, doc="doc for c")
#pybind11#        kd = inputs[2].addField("d", type=numpy.int64, doc="doc for d")
#pybind11#        flags1 = lsst.afw.table.Schema.IDENTICAL
#pybind11#        flags2 = flags1 & ~lsst.afw.table.Schema.EQUAL_NAMES
#pybind11#        mappers1 = lsst.afw.table.SchemaMapper.join(inputs)
#pybind11#        mappers2 = lsst.afw.table.SchemaMapper.join(inputs, prefixes)
#pybind11#        records = [lsst.afw.table.BaseTable.make(schema).makeRecord() for schema in inputs]
#pybind11#        records[0].set(ka, 3.14159)
#pybind11#        records[0].set(kb, 21623)
#pybind11#        records[1].set(kc, 1.5616)
#pybind11#        records[2].set(kd, 1261236)
#pybind11#        for mappers, flags in zip((mappers1, mappers2), (flags1, flags2)):
#pybind11#            output = lsst.afw.table.BaseTable.make(mappers[0].getOutputSchema()).makeRecord()
#pybind11#            for mapper, record in zip(mappers, records):
#pybind11#                output.assign(record, mapper)
#pybind11#                self.assertEqual(mapper.getOutputSchema().compare(output.getSchema(), flags), flags)
#pybind11#                self.assertEqual(mapper.getInputSchema().compare(record.getSchema(), flags), flags)
#pybind11#            names = output.getSchema().getOrderedNames()
#pybind11#            self.assertEqual(output.get(names[0]), records[0].get(ka))
#pybind11#            self.assertEqual(output.get(names[1]), records[0].get(kb))
#pybind11#            self.assertEqual(output.get(names[2]), records[1].get(kc))
#pybind11#            self.assertEqual(output.get(names[3]), records[2].get(kd))
#pybind11#
#pybind11#    def testMinimalSchema(self):
#pybind11#        front = lsst.afw.table.Schema()
#pybind11#        ka = front.addField("a", type=numpy.float64, doc="doc for a")
#pybind11#        kb = front.addField("b", type=numpy.int32, doc="doc for b")
#pybind11#        full = lsst.afw.table.Schema(front)
#pybind11#        kc = full.addField("c", type=numpy.float32, doc="doc for c")
#pybind11#        kd = full.addField("d", type=numpy.int64, doc="doc for d")
#pybind11#        mapper1 = lsst.afw.table.SchemaMapper(full)
#pybind11#        mapper2 = lsst.afw.table.SchemaMapper(full)
#pybind11#        mapper3 = lsst.afw.table.SchemaMapper.removeMinimalSchema(full, front)
#pybind11#        mapper1.addMinimalSchema(front)
#pybind11#        mapper2.addMinimalSchema(front, False)
#pybind11#        self.assertIn(ka, mapper1.getOutputSchema())
#pybind11#        self.assertIn(kb, mapper1.getOutputSchema())
#pybind11#        self.assertNotIn(kc, mapper1.getOutputSchema())
#pybind11#        self.assertNotIn(kd, mapper1.getOutputSchema())
#pybind11#        self.assertIn(ka, mapper2.getOutputSchema())
#pybind11#        self.assertIn(kb, mapper2.getOutputSchema())
#pybind11#        self.assertNotIn(kc, mapper2.getOutputSchema())
#pybind11#        self.assertNotIn(kd, mapper2.getOutputSchema())
#pybind11#        self.assertNotIn(ka, mapper3.getOutputSchema())
#pybind11#        self.assertNotIn(kb, mapper3.getOutputSchema())
#pybind11#        self.assertNotIn(kc, mapper3.getOutputSchema())
#pybind11#        self.assertNotIn(kd, mapper3.getOutputSchema())
#pybind11#        inputRecord = lsst.afw.table.BaseTable.make(full).makeRecord()
#pybind11#        inputRecord.set(ka, numpy.pi)
#pybind11#        inputRecord.set(kb, 2)
#pybind11#        inputRecord.set(kc, numpy.exp(1))
#pybind11#        inputRecord.set(kd, 4)
#pybind11#        outputRecord1 = lsst.afw.table.BaseTable.make(mapper1.getOutputSchema()).makeRecord()
#pybind11#        outputRecord1.assign(inputRecord, mapper1)
#pybind11#        self.assertEqual(inputRecord.get(ka), outputRecord1.get(ka))
#pybind11#        self.assertEqual(inputRecord.get(kb), outputRecord1.get(kb))
#pybind11#
#pybind11#    def testOutputSchema(self):
#pybind11#        mapper = lsst.afw.table.SchemaMapper(lsst.afw.table.Schema())
#pybind11#        out1 = mapper.getOutputSchema()
#pybind11#        out2 = mapper.editOutputSchema()
#pybind11#        k1 = out1.addField("a1", type=int)
#pybind11#        self.assertNotIn(k1, mapper.getOutputSchema())
#pybind11#        self.assertIn(k1, out1)
#pybind11#        self.assertNotIn(k1, out2)
#pybind11#        k2 = mapper.addOutputField(lsst.afw.table.Field[float]("a2", "doc for a2"))
#pybind11#        self.assertNotIn(k2, out1)
#pybind11#        self.assertIn(k2, mapper.getOutputSchema())
#pybind11#        self.assertIn(k2, out2)
#pybind11#        k3 = out2.addField("a3", type=numpy.float32, doc="doc for a3")
#pybind11#        self.assertNotIn(k3, out1)
#pybind11#        self.assertIn(k3, mapper.getOutputSchema())
#pybind11#        self.assertIn(k3, out2)
#pybind11#
#pybind11#    def testDoReplace(self):
#pybind11#        inSchema = lsst.afw.table.Schema()
#pybind11#        ka = inSchema.addField("a", type=int)
#pybind11#        outSchema = lsst.afw.table.Schema(inSchema)
#pybind11#        kb = outSchema.addField("b", type=int)
#pybind11#        kc = outSchema.addField("c", type=int)
#pybind11#        mapper1 = lsst.afw.table.SchemaMapper(inSchema, outSchema)
#pybind11#        mapper1.addMapping(ka, True)
#pybind11#        self.assertEqual(mapper1.getMapping(ka), ka)
#pybind11#        mapper2 = lsst.afw.table.SchemaMapper(inSchema, outSchema)
#pybind11#        mapper2.addMapping(ka, lsst.afw.table.Field[int]("b", "doc for b"), True)
#pybind11#        self.assertEqual(mapper2.getMapping(ka), kb)
#pybind11#        mapper3 = lsst.afw.table.SchemaMapper(inSchema, outSchema)
#pybind11#        mapper3.addMapping(ka, "c", True)
#pybind11#        self.assertEqual(mapper3.getMapping(ka), kc)
#pybind11#
#pybind11#    def testJoin2(self):
#pybind11#        s1 = lsst.afw.table.Schema()
#pybind11#        self.assertEqual(s1.join("a", "b"), "a_b")
#pybind11#        self.assertEqual(s1.join("a", "b", "c"), "a_b_c")
#pybind11#        self.assertEqual(s1.join("a", "b", "c", "d"), "a_b_c_d")
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

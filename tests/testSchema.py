#!/usr/bin/env python2
from __future__ import absolute_import, division
from builtins import zip

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
Tests for table.Schema

Run with:
   ./testSchema.py
or
   python
   >>> import testSchema; testSchema.run()
"""

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


class SchemaTestCase(unittest.TestCase):

    def testSchema(self):
        def testKey(name, key):
            col = schema.find(name)
            self.assertEqual(col.key, key)
            self.assertEqual(col.field.getName(), name)

        schema = lsst.afw.table.Schema();
        ab_k = lsst.afw.table.CoordKey.addFields(schema, "a_b", "parent coord")
        abp_k = lsst.afw.table.Point2DKey.addFields(schema, "a_b_p", "point", "pixel")
        abi_k = schema.addField("a_b_i", type=int, doc="int")
        acf_k = schema.addField("a_c_f", type=numpy.float32, doc="float")
        egd_k = schema.addField("e_g_d", type=lsst.afw.geom.Angle, doc="angle")

        #Basic test for all native key types.
        for name, key in (("a_b_i", abi_k), ("a_c_f", acf_k), ("e_g_d", egd_k)):
            testKey(name, key)

        #Extra tests for special types
        self.assertEqual(ab_k.getRa(), schema["a_b_ra"].asKey());
        abpx_si = schema.find("a_b_p_x")
        self.assertEqual(abp_k.getX(), abpx_si.key);
        self.assertEqual(abpx_si.field.getName(), "a_b_p_x")
        self.assertEqual(abpx_si.field.getDoc(), "point")
        self.assertEqual(abp_k.getX(), schema["a_b_p_x"].asKey());
        self.assertEqual(schema.getNames(), ('a_b_dec', 'a_b_i', 'a_b_p_x', 'a_b_p_y', 'a_b_ra', 'a_c_f',
                                             'e_g_d'))
        self.assertEqual(schema.getNames(True), ("a", "e"))
        self.assertEqual(schema["a"].getNames(), ('b_dec', 'b_i', 'b_p_x', 'b_p_y', 'b_ra', 'c_f'))
        self.assertEqual(schema["a"].getNames(True), ("b", "c"))
        schema2 = lsst.afw.table.Schema(schema)
        self.assertEqual(schema, schema2)
        schema2.addField("q", type=float, doc="another double")
        self.assertNotEqual(schema, schema2)
        schema3 = lsst.afw.table.Schema()
        schema3.addField("ra", type="Angle", doc="coord_ra")
        schema3.addField("dec", type="Angle", doc="coord_dec")
        schema3.addField("x", type="D", doc="position_x")
        schema3.addField("y", type="D", doc="position_y")
        schema3.addField("i", type="I", doc="int")
        schema3.addField("f", type="F", doc="float")
        schema3.addField("d", type="Angle", doc="angle")
        self.assertEqual(schema3, schema)
        schema4 = lsst.afw.table.Schema()
        keys = []
        keys.append(schema4.addField("a", type="Angle", doc="a"))
        keys.append(schema4.addField("b", type="Flag", doc="b"))
        keys.append(schema4.addField("c", type=int, doc="c"))
        keys.append(schema4.addField("d", type="Flag", doc="d"))
        self.assertEqual(keys[1].getBit(), 0)
        self.assertEqual(keys[3].getBit(), 1)
        for n1, n2 in zip(schema4.getOrderedNames(), "abcd"):
            self.assertEqual(n1, n2)
        keys2 = [x.key for x in schema4.asList()]
        self.assertEqual(keys, keys2)

    def testUnits(self):
        schema = lsst.afw.table.Schema();
	# first insert some valid units
        schema.addField("a", type="I", units="pixel")
        schema.addField("b", type="I", units="m2")
        schema.addField("c", type="I", units="electron / adu")
        schema.addField("d", type="I", units="kg m s^(-2)")
        schema.addField("e", type="I", units="GHz / Mpc")
        schema.addField("f", type="Angle", units="deg")
        schema.addField("g", type="Angle", units="rad")
        schema.checkUnits()
	# now try inserting invalid units
        self.assertRaises(ValueError, schema.addField, "a", type="I", units="camel")
        self.assertRaises(ValueError, schema.addField, "b", type="I", units="pixels^2^2")
	# add invalid units in silent mode, should work fine
        schema.addField("h", type="I", units="lala", parse_strict='silent')
	# Now this check should raise because there is an invalid unit
	self.assertRaises(ValueError, schema.checkUnits)

    def testInspection(self):
        schema = lsst.afw.table.Schema()
        keys = []
        keys.append(schema.addField("d", type=int))
        keys.append(schema.addField("c", type=float))
        keys.append(schema.addField("b", type="ArrayF", size=3))
        keys.append(schema.addField("a", type="F"))
        for key, item in zip(keys, schema):
            self.assertEqual(item.key, key)
            self.assert_(key in schema)
        for name in ("a", "b", "c", "d"):
            self.assert_(name in schema)
        self.assertFalse("e" in schema)
        otherSchema = lsst.afw.table.Schema()
        otherKey = otherSchema.addField("d", type=float)
        self.assertFalse(otherKey in schema)
        self.assertNotEqual(keys[0], keys[1])

    def testKeyAccessors(self):
        schema = lsst.afw.table.Schema()
        arrayKey = schema.addField("a", type="ArrayF", doc="doc for array field", size=5)
        arrayElementKey = arrayKey[1]
        self.assertEqual(lsst.afw.table.Key["F"], type(arrayElementKey))

    def testComparison(self):
        schema1 = lsst.afw.table.Schema()
        schema1.addField("a", type=float, doc="doc for a", units="m")
        schema1.addField("b", type=int, doc="doc for b", units="s")
        schema2 = lsst.afw.table.Schema()
        schema2.addField("a", type=int, doc="doc for a", units="m")
        schema2.addField("b", type=float, doc="doc for b", units="s")
        cmp1 = schema1.compare(schema2, lsst.afw.table.Schema.IDENTICAL)
        self.assertTrue(cmp1 & lsst.afw.table.Schema.EQUAL_NAMES)
        self.assertTrue(cmp1 & lsst.afw.table.Schema.EQUAL_DOCS)
        self.assertTrue(cmp1 & lsst.afw.table.Schema.EQUAL_UNITS)
        self.assertFalse(cmp1 & lsst.afw.table.Schema.EQUAL_KEYS)
        schema3 = lsst.afw.table.Schema(schema1)
        schema3.addField("c", type=str, doc="doc for c", size=4)
        self.assertFalse(schema1.compare(schema3))
        self.assertFalse(schema1.contains(schema3))
        self.assertTrue(schema3.contains(schema1))
        schema1.addField("d", type=str, doc="no docs!", size=4)
        cmp2 = schema1.compare(schema3, lsst.afw.table.Schema.IDENTICAL)
        self.assertFalse(cmp2 & lsst.afw.table.Schema.EQUAL_NAMES)
        self.assertFalse(cmp2 & lsst.afw.table.Schema.EQUAL_DOCS)
        self.assertTrue(cmp2 & lsst.afw.table.Schema.EQUAL_KEYS)
        self.assertTrue(cmp2 & lsst.afw.table.Schema.EQUAL_UNITS)
        self.assertFalse(schema1.compare(schema3, lsst.afw.table.Schema.EQUAL_NAMES))


class SchemaMapperTestCase(unittest.TestCase):

    def testJoin(self):
        inputs = [lsst.afw.table.Schema(), lsst.afw.table.Schema(), lsst.afw.table.Schema()]
        inputs = lsst.afw.table.SchemaVector(inputs)
        prefixes = ["u", "v", "w"]
        ka = inputs[0].addField("a", type=numpy.float64, doc="doc for a")
        kb = inputs[0].addField("b", type=numpy.int32, doc="doc for b")
        kc = inputs[1].addField("c", type=numpy.float32, doc="doc for c")
        kd = inputs[2].addField("d", type=numpy.int64, doc="doc for d")
        flags1 = lsst.afw.table.Schema.IDENTICAL
        flags2 = flags1 & ~lsst.afw.table.Schema.EQUAL_NAMES
        mappers1 = lsst.afw.table.SchemaMapper.join(inputs)
        mappers2 = lsst.afw.table.SchemaMapper.join(inputs, prefixes)
        records = [lsst.afw.table.BaseTable.make(schema).makeRecord() for schema in inputs]
        records[0].set(ka, 3.14159)
        records[0].set(kb, 21623)
        records[1].set(kc, 1.5616)
        records[2].set(kd, 1261236)
        for mappers, flags in zip((mappers1, mappers2), (flags1, flags2)):
            output = lsst.afw.table.BaseTable.make(mappers[0].getOutputSchema()).makeRecord()
            for mapper, record in zip(mappers, records):
                output.assign(record, mapper)
                self.assertEqual(mapper.getOutputSchema().compare(output.getSchema(), flags), flags)
                self.assertEqual(mapper.getInputSchema().compare(record.getSchema(), flags), flags)
            names = output.getSchema().getOrderedNames()
            self.assertEqual(output.get(names[0]), records[0].get(ka))
            self.assertEqual(output.get(names[1]), records[0].get(kb))
            self.assertEqual(output.get(names[2]), records[1].get(kc))
            self.assertEqual(output.get(names[3]), records[2].get(kd))

    def testMinimalSchema(self):
        front = lsst.afw.table.Schema()
        ka = front.addField("a", type=numpy.float64, doc="doc for a")
        kb = front.addField("b", type=numpy.int32, doc="doc for b")
        full = lsst.afw.table.Schema(front)
        kc = full.addField("c", type=numpy.float32, doc="doc for c")
        kd = full.addField("d", type=numpy.int64, doc="doc for d")
        mapper1 = lsst.afw.table.SchemaMapper(full)
        mapper2 = lsst.afw.table.SchemaMapper(full)
        mapper3 = lsst.afw.table.SchemaMapper.removeMinimalSchema(full, front)
        mapper1.addMinimalSchema(front)
        mapper2.addMinimalSchema(front, False)
        self.assert_(ka in mapper1.getOutputSchema())
        self.assert_(kb in mapper1.getOutputSchema())
        self.assert_(kc not in mapper1.getOutputSchema())
        self.assert_(kd not in mapper1.getOutputSchema())
        self.assert_(ka in mapper2.getOutputSchema())
        self.assert_(kb in mapper2.getOutputSchema())
        self.assert_(kc not in mapper2.getOutputSchema())
        self.assert_(kd not in mapper2.getOutputSchema())
        self.assert_(ka not in mapper3.getOutputSchema())
        self.assert_(kb not in mapper3.getOutputSchema())
        self.assert_(kc not in mapper3.getOutputSchema())
        self.assert_(kd not in mapper3.getOutputSchema())
        inputRecord = lsst.afw.table.BaseTable.make(full).makeRecord()
        inputRecord.set(ka, numpy.pi)
        inputRecord.set(kb, 2)
        inputRecord.set(kc, numpy.exp(1))
        inputRecord.set(kd, 4)
        outputRecord1 = lsst.afw.table.BaseTable.make(mapper1.getOutputSchema()).makeRecord()
        outputRecord1.assign(inputRecord, mapper1)
        self.assertEqual(inputRecord.get(ka), outputRecord1.get(ka))
        self.assertEqual(inputRecord.get(kb), outputRecord1.get(kb))

    def testOutputSchema(self):
        mapper = lsst.afw.table.SchemaMapper(lsst.afw.table.Schema())
        out1 = mapper.getOutputSchema()
        out2 = mapper.editOutputSchema()
        k1 = out1.addField("a1", type=int)
        self.assert_(k1 not in mapper.getOutputSchema())
        self.assert_(k1 in out1)
        self.assert_(k1 not in out2)
        k2 = mapper.addOutputField(lsst.afw.table.Field[float]("a2", "doc for a2"))
        self.assert_(k2 not in out1)
        self.assert_(k2 in mapper.getOutputSchema())
        self.assert_(k2 in out2)
        k3 = out2.addField("a3", type=numpy.float32, doc="doc for a3")
        self.assert_(k3 not in out1)
        self.assert_(k3 in mapper.getOutputSchema())
        self.assert_(k3 in out2)

    def testDoReplace(self):
        inSchema = lsst.afw.table.Schema()
        ka = inSchema.addField("a", type=int)
        outSchema = lsst.afw.table.Schema(inSchema)
        kb = outSchema.addField("b", type=int)
        kc = outSchema.addField("c", type=int)
        mapper1 = lsst.afw.table.SchemaMapper(inSchema, outSchema)
        mapper1.addMapping(ka, True)
        self.assertEqual(mapper1.getMapping(ka), ka)
        mapper2 = lsst.afw.table.SchemaMapper(inSchema, outSchema)
        mapper2.addMapping(ka, lsst.afw.table.Field[int]("b", "doc for b"), True)
        self.assertEqual(mapper2.getMapping(ka), kb)
        mapper3 = lsst.afw.table.SchemaMapper(inSchema, outSchema)
        mapper3.addMapping(ka, "c", True)
        self.assertEqual(mapper3.getMapping(ka), kc)

    def testJoin2(self):
        s1 = lsst.afw.table.Schema()
        self.assertEqual(s1.join("a", "b"), "a_b")
        self.assertEqual(s1.join("a", "b", "c"), "a_b_c")
        self.assertEqual(s1.join("a", "b", "c", "d"), "a_b_c_d")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(SchemaTestCase)
    suites += unittest.makeSuite(SchemaMapperTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

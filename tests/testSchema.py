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
Tests for table.Schema

Run with:
   ./testSchema.py
or
   python
   >>> import testSchema; testSchema.run()
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

class SchemaTestCase(unittest.TestCase):

    def testSchema(self):
        schema = lsst.afw.table.Schema();
        ab_k = schema.addField("a.b", type="Coord", doc="parent coord")
        abi_k = schema.addField("a.b.i", type=int, doc="int")
        acf_k = schema.addField("a.c.f", type=numpy.float32, doc="float")
        egd_k = schema.addField("e.g.d", type=lsst.afw.geom.Angle, doc="angle")
        abp_k = schema.addField("a.b.p", type="PointF", doc="point")
        ab_si = schema.find("a.b")
        self.assertEqual(ab_si.key, ab_k)
        self.assertEqual(ab_si.field.getName(), "a.b")
        self.assertEqual(ab_k.getRa(), schema["a.b.ra"].asKey());
        abp_si = schema.find("a.b.p")
        self.assertEqual(abp_si.key, abp_k)
        self.assertEqual(abp_si.field.getName(), "a.b.p")
        abpx_si = schema.find("a.b.p.x")
        self.assertEqual(abp_k.getX(), abpx_si.key);
        self.assertEqual(abpx_si.field.getName(), "a.b.p.x")
        self.assertEqual(abpx_si.field.getDoc(), "point")
        self.assertEqual(abp_k, schema["a.b.p"].asKey())
        self.assertEqual(abp_k.getX(), schema["a.b.p.x"].asKey());
        self.assertEqual(schema.getNames(), ("a.b", "a.b.i", "a.b.p", "a.c.f", "e.g.d"))
        self.assertEqual(schema.getNames(True), ("a", "e"))
        self.assertEqual(schema["a"].getNames(), ("b", "b.i", "b.p", "c.f"))
        self.assertEqual(schema["a"].getNames(True), ("b", "c"))
        schema2 = lsst.afw.table.Schema(schema)
        self.assertEqual(schema, schema2)
        schema2.addField("q", type=float, doc="another double")
        self.assertNotEqual(schema, schema2)
        schema3 = lsst.afw.table.Schema()
        schema3.addField("j", type=lsst.afw.coord.Coord, doc="coord")
        schema3.addField("i", type="I", doc="int")
        schema3.addField("f", type="F", doc="float")
        schema3.addField("d", type="Angle", doc="angle")
        schema3.addField("p", type="PointF", doc="point")
        self.assertEqual(schema3, schema)
        schema4 = lsst.afw.table.Schema()
        keys = []
        keys.append(schema4.addField("a", type="Coord", doc="a"))
        keys.append(schema4.addField("b", type="Flag", doc="b"))
        keys.append(schema4.addField("c", type=int, doc="c"))
        keys.append(schema4.addField("d", type="Flag", doc="d"))
        self.assertEqual(keys[1].getBit(), 0)
        self.assertEqual(keys[3].getBit(), 1)
        for n1, n2 in zip(schema4.getOrderedNames(), "abcd"):
            self.assertEqual(n1, n2)
        keys2 = map(lambda x: x.key, schema4.asList())
        self.assertEqual(keys, keys2)

    def testInspection(self):
        schema = lsst.afw.table.Schema()
        keys = []
        keys.append(schema.addField("d", type=int))
        keys.append(schema.addField("c", type=float))
        keys.append(schema.addField("b", type="ArrayF", size=3))
        keys.append(schema.addField("a", type="CovPointF"))
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
        covKey = schema.addField("c", type="CovF", doc="doc for cov field", size=5)
        covElementKey = covKey[1,2]
        self.assertEqual(lsst.afw.table.Key["F"], type(covElementKey))
        pointKey = schema.addField("p", type="PointF", doc="doc for point field")
        pointElementKey = pointKey.getX()
        self.assertEqual(lsst.afw.table.Key["F"], type(pointElementKey))
        shapeKey = schema.addField("s", type="MomentsF", doc="doc for shape field")
        shapeElementKey = shapeKey.getIxx()
        self.assertEqual(lsst.afw.table.Key["F"], type(shapeElementKey))

    def testComparison(self):
        schema1 = lsst.afw.table.Schema()
        schema1.addField("a", type=float, doc="doc for a", units="units for a")
        schema1.addField("b", type=int, doc="doc for b", units="units for b")
        schema2 = lsst.afw.table.Schema()
        schema2.addField("a", type=int, doc="doc for a", units="units for a")
        schema2.addField("b", type=float, doc="doc for b", units="units for b")
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

    def testVersions(self):
        s0 = lsst.afw.table.Schema(0)
        s1 = lsst.afw.table.Schema(1)
        self.assertEqual(s0.getVersion(), 0)
        self.assertEqual(s1.getVersion(), 1)
        s0.setVersion(1)
        s1.setVersion(0)
        self.assertEqual(s0.getVersion(), 1)
        self.assertEqual(s1.getVersion(), 0)

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

    def testVersions(self):
        s0 = lsst.afw.table.Schema(0)
        s1 = lsst.afw.table.Schema(1)
        sm0 = lsst.afw.table.SchemaMapper(s0)
        sm1 = lsst.afw.table.SchemaMapper(s1)
        self.assertEqual(sm0.getOutputSchema().getVersion(), 0)
        self.assertEqual(sm1.getOutputSchema().getVersion(), 1)
        sm0.editOutputSchema().setVersion(1)
        sm1.editOutputSchema().setVersion(0)
        self.assertEqual(sm0.getOutputSchema().getVersion(), 1)
        self.assertEqual(sm1.getOutputSchema().getVersion(), 0)


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

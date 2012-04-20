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
        abp_k = schema.addField("a.b.p", type="Point<F4>", doc="point")
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
        schema3.addField("i", type="I4", doc="int")
        schema3.addField("f", type="F4", doc="float")
        schema3.addField("d", type="Angle", doc="angle")
        schema3.addField("p", type="Point<F4>", doc="point")
        self.assertEqual(schema3, schema)
        
    def testInspection(self):
        schema = lsst.afw.table.Schema()
        keys = []
        keys.append(schema.addField("d", type=int))
        keys.append(schema.addField("c", type=float))
        keys.append(schema.addField("b", type="Array<F4>", size=3))
        keys.append(schema.addField("a", type="Cov<Point<F4>>"))
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
        arrayKey = schema.addField("a", type="Array<F4>", doc="doc for array field", size=5)
        arrayElementKey = arrayKey[1]
        self.assertEqual(lsst.afw.table.Key["F4"], type(arrayElementKey))
        covKey = schema.addField("c", type="Cov<F4>", doc="doc for cov field", size=5)
        covElementKey = covKey[1,2]
        self.assertEqual(lsst.afw.table.Key["F4"], type(covElementKey))
        pointKey = schema.addField("p", type="Point<F4>", doc="doc for point field")
        pointElementKey = pointKey.getX()
        self.assertEqual(lsst.afw.table.Key["F4"], type(pointElementKey))
        shapeKey = schema.addField("s", type="Moments<F4>", doc="doc for shape field")
        shapeElementKey = shapeKey.getIxx()
        self.assertEqual(lsst.afw.table.Key["F4"], type(shapeElementKey))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(SchemaTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

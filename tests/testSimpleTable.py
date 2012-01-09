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

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def makeRandom(shape, dtype):
    return numpy.array(numpy.random.randn(*shape), dtype=dtype)

class SimpleTableTestCase(unittest.TestCase):

    def checkScalarAccessors(self, record, key, value1, value2):
        record[key] = value1
        self.assertEqual(record[key], value1)
        self.assertEqual(record.get(key), value1)
        record.set(key, value2)
        self.assertEqual(record[key], value2)
        self.assertEqual(record.get(key), value2)

    def checkGeomAccessors(self, record, key, value):
        record.set(key, value)
        self.assertEqual(record.get(key), value)

    def checkArrayAccessors(self, record, key, value):
        record.set(key, value)
        self.assert_(numpy.all(record.get(key) == value))

    def checkCovAccessors(self, record, key, value):
        matrix = numpy.dot(value.transpose(), value)
        record.set(key, matrix)
        self.assert_(numpy.all(record.get(key) == matrix))

    def testRecordAccess(self):
        schema = lsst.afw.table.Schema(False)
        k1 = schema.addField("f1", type="I4")
        k2 = schema.addField("f2", type="I8")
        k3 = schema.addField("f3", type="F4")
        k4 = schema.addField("f4", type="F8")
        k5 = schema.addField("f5", type="Point<I4>")
        k6 = schema.addField("f6", type="Point<F4>")
        k7 = schema.addField("f7", type="Point<F8>")
        k8 = schema.addField("f8", type="Shape<F4>")
        k9 = schema.addField("f9", type="Shape<F8>")
        k10 = schema.addField("f10", type="Array<F4>", size=4)
        k11 = schema.addField("f11", type="Array<F8>", size=5)
        k12 = schema.addField("f12", type="Cov<F4>", size=3)
        k13 = schema.addField("f13", type="Cov<F8>", size=4)
        k14 = schema.addField("f14", type="Cov<Point<F4>>")
        k15 = schema.addField("f15", type="Cov<Point<F8>>")
        k16 = schema.addField("f16", type="Cov<Shape<F4>>")
        k17 = schema.addField("f17", type="Cov<Shape<F8>>")
        table = lsst.afw.table.SimpleTable(schema)
        record = table.addRecord()
        self.checkScalarAccessors(record, k1, 2, 3)
        self.checkScalarAccessors(record, k2, 2, 3)
        self.checkScalarAccessors(record, k3, 2.5, 3.5)
        self.checkScalarAccessors(record, k4, 2.5, 3.5)
        self.checkGeomAccessors(record, k5, lsst.afw.geom.Point2I(5, 3))
        self.checkGeomAccessors(record, k6, lsst.afw.geom.Point2D(5.5, 3.5))
        self.checkGeomAccessors(record, k7, lsst.afw.geom.Point2D(5.5, 3.5))
        self.checkGeomAccessors(record, k8, lsst.afw.geom.ellipses.Quadrupole(5.5, 3.5, -1.0))
        self.checkGeomAccessors(record, k9, lsst.afw.geom.ellipses.Quadrupole(5.5, 3.5, -1.0))
        self.checkArrayAccessors(record, k10, makeRandom((k10.getSize(),), dtype=numpy.float32))
        self.checkArrayAccessors(record, k11, makeRandom((k11.getSize(),), dtype=numpy.float64))
        self.checkCovAccessors(record, k12, makeRandom((k12.getSize(), k12.getSize()), dtype=numpy.float32))
        self.checkCovAccessors(record, k13, makeRandom((k13.getSize(), k13.getSize()), dtype=numpy.float64))
        self.checkCovAccessors(record, k14, makeRandom((k14.getSize(), k14.getSize()), dtype=numpy.float32))
        self.checkCovAccessors(record, k15, makeRandom((k15.getSize(), k15.getSize()), dtype=numpy.float64))
        self.checkCovAccessors(record, k16, makeRandom((k16.getSize(), k16.getSize()), dtype=numpy.float32))
        self.checkCovAccessors(record, k17, makeRandom((k17.getSize(), k17.getSize()), dtype=numpy.float64))

    def testColumnView(self):
        schema = lsst.afw.table.Schema(True)
        k1 = schema.addField("f1", type="I4")
        kb1 = schema.addField("fb1", type="Flag")
        k2 = schema.addField("f2", type="F4")
        kb2 = schema.addField("fb2", type="Flag")
        k3 = schema.addField("f3", type="F8")
        kb3 = schema.addField("fb3", type="Flag")
        k4 = schema.addField("f4", type="Array<F4>", size=2)
        k5 = schema.addField("f5", type="Array<F8>", size=3)
        table = lsst.afw.table.SimpleTable(schema, 2)
        records = []
        records.append(table.addRecord())
        records.append(table.addRecord())
        records[0].set(k1, 2)
        records[0].set(k2, 0.5)
        records[0].set(k3, 0.25)
        records[0].set(kb1, False)
        records[0].set(kb2, True)
        records[0].set(kb3, False)
        records[0].set(k4, numpy.array([-0.5, -0.25], dtype=numpy.float32))
        records[0].set(k5, numpy.array([-1.5, -1.25, 3.375], dtype=numpy.float64))
        records[1].set(k1, 3)
        records[1].set(k2, 2.5)
        records[1].set(k3, 0.75)
        records[1].set(kb1, True)
        records[1].set(kb2, False)
        records[1].set(kb3, True)
        records[1].set(k4, numpy.array([-3.25, -0.75], dtype=numpy.float32))
        records[1].set(k5, numpy.array([-1.25, -2.75, 0.625], dtype=numpy.float64))
        records[1].setParentId(1)
        self.assert_(table.isConsolidated())
        columns = table.getColumnView()
        for key in [k1, k2, k3, kb1, kb2, kb3]:
            array = columns[key]
            for i in [0, 1]:
                self.assertEqual(array[i], records[i].get(key))
        ids = columns.getId()
        parentIds = columns.getParentId();
        for i in [0, 1]:
            self.assertEqual(ids[i], records[i].getId())
            self.assertEqual(parentIds[i], records[i].getParentId())

    def testIteration(self):
        schema = lsst.afw.table.Schema(False)
        table = lsst.afw.table.SimpleTable(schema)
        table.addRecord()
        table.addRecord()
        table.addRecord()
        table.addRecord()
        for n, record in enumerate(table):
            self.assertEqual(n+1, record.getId())

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

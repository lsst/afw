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
        

    def checkGeomAccessors(self, record, key, name, value):
        record.set(key, value)
        self.assertEqual(record.get(key), value)
        record.set(name, value)
        self.assertEqual(record.get(name), value)

    def checkArrayAccessors(self, record, key, name, value):
        record.set(key, value)
        self.assert_(numpy.all(record.get(key) == value))
        record.set(name, value)
        self.assert_(numpy.all(record.get(name) == value))

    def testRecordAccess(self):
        schema = lsst.afw.table.Schema()
        k1 = schema.addField("f1", type="I4")
        k2 = schema.addField("f2", type="I8")
        k3 = schema.addField("f3", type="F4")
        k4 = schema.addField("f4", type="F8")
        k5 = schema.addField("f5", type="Point<I4>")
        k6 = schema.addField("f6", type="Point<F4>")
        k7 = schema.addField("f7", type="Point<F8>")
        k8 = schema.addField("f8", type="Moments<F4>")
        k9 = schema.addField("f9", type="Moments<F8>")
        k10 = schema.addField("f10", type="Array<F4>", size=4)
        k11 = schema.addField("f11", type="Array<F8>", size=5)
        k12 = schema.addField("f12", type="Cov<F4>", size=3)
        k13 = schema.addField("f13", type="Cov<F8>", size=4)
        k14 = schema.addField("f14", type="Cov<Point<F4>>")
        k15 = schema.addField("f15", type="Cov<Point<F8>>")
        k16 = schema.addField("f16", type="Cov<Moments<F4>>")
        k17 = schema.addField("f17", type="Cov<Moments<F8>>")
        k18 = schema.addField("f18", type="Angle")
        k19 = schema.addField("f19", type="Coord")
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
        self.checkGeomAccessors(record, k8, "f8", lsst.afw.geom.ellipses.Quadrupole(5.5, 3.5, -1.0))
        self.checkGeomAccessors(record, k9, "f9", lsst.afw.geom.ellipses.Quadrupole(5.5, 3.5, -1.0))
        self.checkArrayAccessors(record, k10, "f10", makeArray(k10.getSize(), dtype=numpy.float32))
        self.checkArrayAccessors(record, k11, "f11", makeArray(k11.getSize(), dtype=numpy.float64))
        self.checkArrayAccessors(record, k12, "f12", makeCov(k12.getSize(), dtype=numpy.float32))
        self.checkArrayAccessors(record, k13, "f13", makeCov(k13.getSize(), dtype=numpy.float64))
        self.checkArrayAccessors(record, k14, "f14", makeCov(k14.getSize(), dtype=numpy.float32))
        self.checkArrayAccessors(record, k15, "f15", makeCov(k15.getSize(), dtype=numpy.float64))
        self.checkArrayAccessors(record, k16, "f16", makeCov(k16.getSize(), dtype=numpy.float32))
        self.checkArrayAccessors(record, k17, "f17", makeCov(k17.getSize(), dtype=numpy.float64))
        self.checkGeomAccessors(record, k18, "f18", lsst.afw.geom.Angle(1.2))
        self.checkGeomAccessors(
            record, k19, "f19", 
            lsst.afw.coord.IcrsCoord(lsst.afw.geom.Angle(1.3), lsst.afw.geom.Angle(0.5))
            )

    def testBaseFits(self):
        schema = lsst.afw.table.Schema()
        k = schema.addField("f", type="F8")
        cat1 = lsst.afw.table.BaseCatalog(schema)
        for i in range(50):
            record = cat1.addNew()
            record.set(k, numpy.random.randn())
        cat1.writeFits("testBaseTable.fits")
        cat2 = lsst.afw.table.BaseCatalog.readFits("testBaseTable.fits")
        self.assertEqual(len(cat1), len(cat2))
        for r1, r2 in zip(cat1, cat2):
            self.assertEqual(r1.get(k), r2.get(k))
        os.remove("testBaseTable.fits")
        self.assertRaises(Exception, lsst.afw.table.BaseCatalog.readFits, "nonexistentfile.fits")

    def testColumnView(self):
        schema = lsst.afw.table.Schema()
        k1 = schema.addField("f1", type="I4")
        kb1 = schema.addField("fb1", type="Flag")
        k2 = schema.addField("f2", type="F4")
        kb2 = schema.addField("fb2", type="Flag")
        k3 = schema.addField("f3", type="F8")
        kb3 = schema.addField("fb3", type="Flag")
        k4 = schema.addField("f4", type="Array<F4>", size=2)
        k5 = schema.addField("f5", type="Array<F8>", size=3)
        k6 = schema.addField("f6", type="Angle")
        catalog = lsst.afw.table.BaseCatalog(schema)
        catalog.addNew()
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
        catalog[1].set(k1, 3)
        catalog[1].set(k2, 2.5)
        catalog[1].set(k3, 0.75)
        catalog[1].set(kb1, True)
        catalog[1].set(kb2, False)
        catalog[1].set(kb3, True)
        catalog[1].set(k4, numpy.array([-3.25, -0.75], dtype=numpy.float32))
        catalog[1].set(k5, numpy.array([-1.25, -2.75, 0.625], dtype=numpy.float64))
        catalog[1].set(k6, lsst.afw.geom.Angle(0.15))
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

    def testIteration(self):
        schema = lsst.afw.table.Schema()
        k = schema.addField("a", type=int)
        catalog = lsst.afw.table.BaseCatalog(schema)
        for n in range(5):
            record = catalog.addNew()
            record[k] = n
        for n, r in enumerate(catalog):
            self.assertEqual(n, r[k])
        

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

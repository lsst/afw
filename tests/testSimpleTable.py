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

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.table
import lsst.afw.geom

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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

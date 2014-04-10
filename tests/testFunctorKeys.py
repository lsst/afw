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
Tests for table FunctorKeys

Run with:
   ./testFunctorKeys.py
or
   python
   >>> import testFunctorKeys; testFunctorKeys.run()
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

class FunctorKeysTestCase(lsst.utils.tests.TestCase):

    def doTestPointKey(self, fieldType, functorKeyType, valueType):
        schema = lsst.afw.table.Schema();
        xKey = schema.addField("a.x", type=fieldType, doc="x")
        yKey = schema.addField("a.y", type=fieldType, doc="y")
        fKey1 = functorKeyType(xKey, yKey)
        fKey2 = functorKeyType(schema["a"])
        self.assertEqual(fKey1.getX(), xKey)
        self.assertEqual(fKey2.getX(), xKey)
        self.assertEqual(fKey1.getY(), yKey)
        self.assertEqual(fKey2.getY(), yKey)
        self.assertEqual(fKey1, fKey2)
        self.assertTrue(fKey1.isValid())
        self.assertTrue(fKey2.isValid())
        fKey3 = functorKeyType()
        self.assertNotEqual(fKey3, fKey1)
        self.assertFalse(fKey3.isValid())
        table = lsst.afw.table.BaseTable.make(schema)
        record = table.makeRecord()
        record.set(xKey, 4)
        record.set(yKey, 2)
        self.assertIsInstance(record.get(fKey1), valueType)
        self.assertEqual(record.get(fKey1).getX(), record.get(xKey))
        self.assertEqual(record.get(fKey1).getY(), record.get(yKey))
        p = valueType(8, 16)
        record.set(fKey1, p)
        self.assertEqual(record.get(xKey), p.getX())
        self.assertEqual(record.get(yKey), p.getY())

    def testPointKey(self):
        self.doTestPointKey("I", lsst.afw.table.Point2IKey, lsst.afw.geom.Point2I)
        self.doTestPointKey("D", lsst.afw.table.Point2DKey, lsst.afw.geom.Point2D)

    def testQuadrupoleKey(self):
        schema = lsst.afw.table.Schema();
        xxKey = schema.addField("a.xx", type=float, doc="xx")
        yyKey = schema.addField("a.yy", type=float, doc="yy")
        xyKey = schema.addField("a.xy", type=float, doc="xy")
        fKey1 = lsst.afw.table.QuadrupoleKey(xxKey, yyKey, xyKey)
        fKey2 = lsst.afw.table.QuadrupoleKey(schema["a"])
        self.assertEqual(fKey1.getIxx(), xxKey)
        self.assertEqual(fKey2.getIxx(), xxKey)
        self.assertEqual(fKey1.getIyy(), yyKey)
        self.assertEqual(fKey2.getIyy(), yyKey)
        self.assertEqual(fKey1.getIxy(), xyKey)
        self.assertEqual(fKey2.getIxy(), xyKey)
        self.assertEqual(fKey1, fKey2)
        self.assertTrue(fKey1.isValid())
        self.assertTrue(fKey2.isValid())
        fKey3 = lsst.afw.table.QuadrupoleKey()
        self.assertNotEqual(fKey3, fKey1)
        self.assertFalse(fKey3.isValid())
        table = lsst.afw.table.BaseTable.make(schema)
        record = table.makeRecord()
        record.set(xxKey, 4)
        record.set(yyKey, 2)
        record.set(xyKey, 2)
        self.assertIsInstance(record.get(fKey1), lsst.afw.geom.ellipses.Quadrupole)
        self.assertEqual(record.get(fKey1).getIxx(), record.get(xxKey))
        self.assertEqual(record.get(fKey1).getIyy(), record.get(yyKey))
        self.assertEqual(record.get(fKey1).getIxy(), record.get(xyKey))
        p = lsst.afw.geom.ellipses.Quadrupole(8, 16, 4)
        record.set(fKey1, p)
        self.assertEqual(record.get(xxKey), p.getIxx())
        self.assertEqual(record.get(yyKey), p.getIyy())
        self.assertEqual(record.get(xyKey), p.getIxy())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(FunctorKeysTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

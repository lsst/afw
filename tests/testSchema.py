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

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.table

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SchemaTestCase(unittest.TestCase):

    def testInspection(self):
        schema = lsst.afw.table.Schema(False)
        keys = []
        keys.append(schema.addField("d", type=int))
        keys.append(schema.addField("c", type=float))
        keys.append(schema.addField("b", type="Array<F4>", size=3))
        keys.append(schema.addField("a", type="Cov<Point<F4>>"))
        for key, item in zip(keys, schema):
            self.assertEqual(item.key, key)
        self.assertNotEqual(keys[0], keys[1])

    def testKeyAccessors(self):
        schema = lsst.afw.table.Schema(False)
        arrayKey = schema.addField("a", type="Array<F4>", doc="doc for array field", size=5)
        arrayElementKey = arrayKey[1]
        self.assertEqual(lsst.afw.table.Key["F4"], type(arrayElementKey))
        covKey = schema.addField("c", type="Cov<F4>", doc="doc for cov field", size=5)
        covElementKey = covKey[1,2]
        self.assertEqual(lsst.afw.table.Key["F4"], type(covElementKey))
        pointKey = schema.addField("p", type="Point<F4>", doc="doc for point field")
        pointElementKey = pointKey.getX()
        self.assertEqual(lsst.afw.table.Key["F4"], type(pointElementKey))
        shapeKey = schema.addField("s", type="Shape<F4>", doc="doc for shape field")
        shapeElementKey = shapeKey.getIXX()
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

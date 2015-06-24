#!/usr/bin/env python2
from __future__ import absolute_import, division
#
# LSST Data Management System
# Copyright 2008-2014 LSST Corporation.
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
Tests for aliases in table.schema

Run with:
   ./testTableAliases.py
or
   python
   >>> import testTableAliases; testTableAliases.run()
"""

import unittest

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

class TableAliasTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.schema = lsst.afw.table.Schema()
        self.a11 = self.schema.addField("a11", type=int, doc="")
        self.a12 = self.schema.addField("a12", type=int, doc="")
        self.ab11 = self.schema.addField("ab11", type=int, doc="")
        self.ab12 = self.schema.addField("ab12", type=int, doc="")
        self.dict = dict(q="a", r="a1", s="ab", t="ab11")
        for k, v in self.dict.iteritems():
            self.schema.getAliasMap().set(k, v)

    def tearDown(self):
        del self.schema
        del self.a11
        del self.a12
        del self.ab11
        del self.ab12
        del self.dict

    def testApply(self):
        self.assertEqual(self.schema.getAliasMap().apply("q11"), "a11")
        self.assertEqual(self.schema.getAliasMap().apply("q12"), "a12")
        self.assertEqual(self.schema.getAliasMap().apply("qb11"), "ab11")
        self.assertEqual(self.schema.getAliasMap().apply("qb12"), "ab12")
        self.assertEqual(self.schema.getAliasMap().apply("r1"), "a11")
        self.assertEqual(self.schema.getAliasMap().apply("r2"), "a12")
        self.assertEqual(self.schema.getAliasMap().apply("s11"), "ab11")
        self.assertEqual(self.schema.getAliasMap().apply("s12"), "ab12")
        self.assertEqual(self.schema.getAliasMap().apply("t"), "ab11")

    def testAccessors(self):
        aliases = self.schema.getAliasMap()

        # Check that iteration works
        self.assertEqual(dict(aliases), self.dict)
        for n, (k, v) in enumerate(aliases.iteritems()):
            self.assertEqual(aliases.get(k), v)
            self.assertEqual(aliases[k], v)
        self.assertEqual(n + 1, len(aliases))
        self.assertEqual(aliases.keys(), [k for k, v in aliases.iteritems()])
        self.assertEqual(aliases.values(), [v for k, v in aliases.iteritems()])

        # Try removing something using the C++-named methods
        self.assertFalse(aliases.erase("q"))
        del self.dict["q"]
        self.assertEqual(dict(aliases), self.dict)

        # Try removing something using the Python operators
        del aliases["r"]
        del self.dict["r"]
        self.assertEqual(dict(aliases), self.dict)

        # Try getting/setting something using the Python operators
        aliases["p"] = "ab12"
        self.assertEqual(aliases["p"], "ab12")

        # Test empty and bool conversion
        self.assertFalse(aliases.empty())
        self.assertFalse(aliases)
        aliases = lsst.afw.table.AliasMap()
        self.assertFalse(aliases.empty())
        self.assertFalse(aliases)

    def testFind(self):
        self.assertEqual(self.schema.find("q11").key, self.a11)
        self.assertEqual(self.schema.find("q12").key, self.a12)
        self.assertEqual(self.schema.find("qb11").key, self.ab11)
        self.assertEqual(self.schema.find("qb12").key, self.ab12)
        self.assertEqual(self.schema.find("r1").key, self.a11)
        self.assertEqual(self.schema.find("r2").key, self.a12)
        self.assertEqual(self.schema.find("s11").key, self.ab11)
        self.assertEqual(self.schema.find("s12").key, self.ab12)
        self.assertEqual(self.schema.find("t").key, self.ab11)

    def testRecursiveAliases(self):
        """Test that multi-level alias replacement works.
        """
        self.schema.setAliasMap(None) # remove all current aliases
        # applying the following aliases recursively should let us use 'u1' to get to the 'a11' field
        self.schema.getAliasMap().set("t2", "a1")
        self.schema.getAliasMap().set("u", "t2")
        self.assertEqual(self.schema.find("u1").key, self.a11)

    def testCycle(self):
        """Test that multi-level aliases that form a cycle produce an exception, not an infinite loop.
        """
        self.schema.setAliasMap(None) # remove all current aliases
        self.schema.getAliasMap().set("t", "a")
        self.schema.getAliasMap().set("a", "t")
        self.assertRaises(lsst.pex.exceptions.RuntimeError, self.schema.find, "t")


    def testReplace(self):
        aliases = self.schema.getAliasMap()
        self.assertEqual(aliases.get("q"), "a")
        aliases.set("q", "a1")
        self.assertEqual(aliases.get("q"), "a1")

    def testSchemaComparison(self):
        self.assertEqual(self.schema.compare(self.schema, self.schema.EQUAL_ALIASES),
                         self.schema.EQUAL_ALIASES)
        self.assertEqual(self.schema.compare(self.schema, self.schema.IDENTICAL), self.schema.IDENTICAL)
        copy = lsst.afw.table.Schema(self.schema)
        copy.disconnectAliases()
        self.assertEqual(self.schema.getAliasMap(), copy.getAliasMap())
        copy.getAliasMap().erase("q")
        self.assertNotEqual(self.schema.getAliasMap(), copy.getAliasMap())
        self.assertEqual(self.schema.compare(copy, self.schema.EQUAL_ALIASES), 0)
        self.assertEqual(self.schema.compare(copy, self.schema.IDENTICAL), self.schema.EQUAL_FIELDS)
        self.assertEqual(self.schema.contains(copy, self.schema.EQUAL_ALIASES), self.schema.EQUAL_ALIASES)
        self.assertEqual(self.schema.contains(copy, self.schema.IDENTICAL), self.schema.IDENTICAL)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(TableAliasTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

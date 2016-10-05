#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008-2014 LSST Corporation.
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
#pybind11#Tests for aliases in table.schema
#pybind11#
#pybind11#Run with:
#pybind11#   ./testTableAliases.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import testTableAliases; testTableAliases.run()
#pybind11#"""
#pybind11#
#pybind11#import unittest
#pybind11#
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
#pybind11#class TableAliasTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.schema = lsst.afw.table.Schema()
#pybind11#        self.a11 = self.schema.addField("a11", type=int, doc="")
#pybind11#        self.a12 = self.schema.addField("a12", type=int, doc="")
#pybind11#        self.ab11 = self.schema.addField("ab11", type=int, doc="")
#pybind11#        self.ab12 = self.schema.addField("ab12", type=int, doc="")
#pybind11#        self.dict = dict(q="a", r="a1", s="ab", t="ab11")
#pybind11#        for k, v in self.dict.items():
#pybind11#            self.schema.getAliasMap().set(k, v)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.schema
#pybind11#        del self.a11
#pybind11#        del self.a12
#pybind11#        del self.ab11
#pybind11#        del self.ab12
#pybind11#        del self.dict
#pybind11#
#pybind11#    def testApply(self):
#pybind11#        self.assertEqual(self.schema.getAliasMap().apply("q11"), "a11")
#pybind11#        self.assertEqual(self.schema.getAliasMap().apply("q12"), "a12")
#pybind11#        self.assertEqual(self.schema.getAliasMap().apply("qb11"), "ab11")
#pybind11#        self.assertEqual(self.schema.getAliasMap().apply("qb12"), "ab12")
#pybind11#        self.assertEqual(self.schema.getAliasMap().apply("r1"), "a11")
#pybind11#        self.assertEqual(self.schema.getAliasMap().apply("r2"), "a12")
#pybind11#        self.assertEqual(self.schema.getAliasMap().apply("s11"), "ab11")
#pybind11#        self.assertEqual(self.schema.getAliasMap().apply("s12"), "ab12")
#pybind11#        self.assertEqual(self.schema.getAliasMap().apply("t"), "ab11")
#pybind11#
#pybind11#    def testAccessors(self):
#pybind11#        aliases = self.schema.getAliasMap()
#pybind11#
#pybind11#        # Check that iteration works
#pybind11#        self.assertEqual(dict(aliases), self.dict)
#pybind11#        for n, (k, v) in enumerate(aliases.items()):
#pybind11#            self.assertEqual(aliases.get(k), v)
#pybind11#            self.assertEqual(aliases[k], v)
#pybind11#        self.assertEqual(n + 1, len(aliases))
#pybind11#        self.assertEqual(list(aliases.keys()), [k for k, v in aliases.items()])
#pybind11#        self.assertEqual(list(aliases.values()), [v for k, v in aliases.items()])
#pybind11#
#pybind11#        # Try removing something using the C++-named methods
#pybind11#        self.assertTrue(aliases.erase("q"))
#pybind11#        del self.dict["q"]
#pybind11#        self.assertEqual(dict(aliases), self.dict)
#pybind11#
#pybind11#        # Try removing something using the Python operators
#pybind11#        del aliases["r"]
#pybind11#        del self.dict["r"]
#pybind11#        self.assertEqual(dict(aliases), self.dict)
#pybind11#
#pybind11#        # Try getting/setting something using the Python operators
#pybind11#        aliases["p"] = "ab12"
#pybind11#        self.assertEqual(aliases["p"], "ab12")
#pybind11#
#pybind11#        # Test empty and bool conversion
#pybind11#        self.assertFalse(aliases.empty())
#pybind11#        self.assertTrue(aliases)
#pybind11#        aliases = lsst.afw.table.AliasMap()
#pybind11#        self.assertTrue(aliases.empty())
#pybind11#        self.assertFalse(aliases)
#pybind11#
#pybind11#    def testFind(self):
#pybind11#        self.assertEqual(self.schema.find("q11").key, self.a11)
#pybind11#        self.assertEqual(self.schema.find("q12").key, self.a12)
#pybind11#        self.assertEqual(self.schema.find("qb11").key, self.ab11)
#pybind11#        self.assertEqual(self.schema.find("qb12").key, self.ab12)
#pybind11#        self.assertEqual(self.schema.find("r1").key, self.a11)
#pybind11#        self.assertEqual(self.schema.find("r2").key, self.a12)
#pybind11#        self.assertEqual(self.schema.find("s11").key, self.ab11)
#pybind11#        self.assertEqual(self.schema.find("s12").key, self.ab12)
#pybind11#        self.assertEqual(self.schema.find("t").key, self.ab11)
#pybind11#
#pybind11#    def testRecursiveAliases(self):
#pybind11#        """Test that multi-level alias replacement works.
#pybind11#        """
#pybind11#        self.schema.setAliasMap(None)  # remove all current aliases
#pybind11#        # applying the following aliases recursively should let us use 'u1' to get to the 'a11' field
#pybind11#        self.schema.getAliasMap().set("t2", "a1")
#pybind11#        self.schema.getAliasMap().set("u", "t2")
#pybind11#        self.assertEqual(self.schema.find("u1").key, self.a11)
#pybind11#
#pybind11#    def testCycle(self):
#pybind11#        """Test that multi-level aliases that form a cycle produce an exception, not an infinite loop.
#pybind11#        """
#pybind11#        self.schema.setAliasMap(None)  # remove all current aliases
#pybind11#        self.schema.getAliasMap().set("t", "a")
#pybind11#        self.schema.getAliasMap().set("a", "t")
#pybind11#        with self.assertRaises(lsst.pex.exceptions.RuntimeError):
#pybind11#            self.schema.find("t")
#pybind11#
#pybind11#    def testReplace(self):
#pybind11#        aliases = self.schema.getAliasMap()
#pybind11#        self.assertEqual(aliases.get("q"), "a")
#pybind11#        aliases.set("q", "a1")
#pybind11#        self.assertEqual(aliases.get("q"), "a1")
#pybind11#
#pybind11#    def testSchemaComparison(self):
#pybind11#        self.assertEqual(self.schema.compare(self.schema, self.schema.EQUAL_ALIASES),
#pybind11#                         self.schema.EQUAL_ALIASES)
#pybind11#        self.assertEqual(self.schema.compare(self.schema, self.schema.IDENTICAL), self.schema.IDENTICAL)
#pybind11#        copy = lsst.afw.table.Schema(self.schema)
#pybind11#        copy.disconnectAliases()
#pybind11#        self.assertEqual(self.schema.getAliasMap(), copy.getAliasMap())
#pybind11#        copy.getAliasMap().erase("q")
#pybind11#        self.assertNotEqual(self.schema.getAliasMap(), copy.getAliasMap())
#pybind11#        self.assertEqual(self.schema.compare(copy, self.schema.EQUAL_ALIASES), 0)
#pybind11#        self.assertEqual(self.schema.compare(copy, self.schema.IDENTICAL), self.schema.EQUAL_FIELDS)
#pybind11#        self.assertEqual(self.schema.contains(copy, self.schema.EQUAL_ALIASES), self.schema.EQUAL_ALIASES)
#pybind11#        self.assertEqual(self.schema.contains(copy, self.schema.IDENTICAL), self.schema.IDENTICAL)
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

#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import str
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2012 LSST Corporation.
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
#pybind11#Test for match persistence via FITS
#pybind11#"""
#pybind11#
#pybind11#try:
#pybind11#    debug
#pybind11#except NameError:
#pybind11#    debug = False
#pybind11#
#pybind11#import unittest
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.table as afwTable
#pybind11#
#pybind11#
#pybind11#class MatchFitsTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.size = 10
#pybind11#        self.numMatches = self.size//2
#pybind11#        self.filename = "matches.fits"
#pybind11#        self.schema = afwTable.SimpleTable.makeMinimalSchema()
#pybind11#        self.cat1 = afwTable.SimpleCatalog(self.schema)
#pybind11#        self.cat2 = afwTable.SimpleCatalog(self.schema)
#pybind11#        for i in range(self.size):
#pybind11#            record1 = self.cat1.table.makeRecord()
#pybind11#            record2 = self.cat2.table.makeRecord()
#pybind11#            record1.setId(i + 1)
#pybind11#            record2.setId(self.size - i)
#pybind11#            self.cat1.append(record1)
#pybind11#            self.cat2.append(record2)
#pybind11#
#pybind11#        self.matches = afwTable.SimpleMatchVector()
#pybind11#        for i in range(self.numMatches):
#pybind11#            index = 2*i
#pybind11#            match = afwTable.SimpleMatch(self.cat1[index], self.cat2[self.size - index - 1], index)
#pybind11#            if debug:
#pybind11#                print("Inject:", match.first.getId(), match.second.getId())
#pybind11#            self.matches.push_back(match)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.schema
#pybind11#        del self.cat1
#pybind11#        del self.cat2
#pybind11#        del self.matches
#pybind11#
#pybind11#    def testMatches(self, matches=None):
#pybind11#        if matches is None:
#pybind11#            matches = self.matches
#pybind11#        self.assertEqual(len(matches), self.numMatches)
#pybind11#        for m in matches:
#pybind11#            str(m)  # Check __str__ works
#pybind11#            if debug:
#pybind11#                print("Test:", m.first.getId(), m.second.getId())
#pybind11#            self.assertEqual(m.first.getId(), m.second.getId())
#pybind11#
#pybind11#    def testIO(self):
#pybind11#        packed = afwTable.packMatches(self.matches)
#pybind11#        packed.writeFits(self.filename)
#pybind11#        matches = afwTable.BaseCatalog.readFits(self.filename)
#pybind11#        cat1 = self.cat1.copy()
#pybind11#        cat2 = self.cat2.copy()
#pybind11#        cat1.sort()
#pybind11#        cat2.sort()
#pybind11#        unpacked = afwTable.unpackMatches(matches, cat1, cat2)
#pybind11#        self.testMatches(unpacked)
#pybind11#
#pybind11#    def testTicket2080(self):
#pybind11#        packed = afwTable.packMatches(self.matches)
#pybind11#        cat1 = self.cat1.copy()
#pybind11#        cat2 = afwTable.SimpleCatalog(self.schema)
#pybind11#        cat1.sort()
#pybind11#        cat2.sort()
#pybind11#        # just test that the next line doesn't segv
#pybind11#        afwTable.unpackMatches(packed, cat1, cat2)
#pybind11#
#pybind11#
#pybind11##################################################################
#pybind11## Test suite boiler plate
#pybind11##################################################################
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

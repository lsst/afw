#
# LSST Data Management System
# Copyright 2012 LSST Corporation.
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
Test for match persistence via FITS
"""
from __future__ import absolute_import, division, print_function
import unittest

from builtins import str
from builtins import range

import lsst.utils.tests
import lsst.afw.table as afwTable

Debug = False  # set True to print some debugging information


class MatchFitsTestCase(unittest.TestCase):

    def setUp(self):
        self.size = 10
        self.numMatches = self.size//2
        self.schema = afwTable.SimpleTable.makeMinimalSchema()
        self.cat1 = afwTable.SimpleCatalog(self.schema)
        self.cat2 = afwTable.SimpleCatalog(self.schema)
        for i in range(self.size):
            record1 = self.cat1.table.makeRecord()
            record2 = self.cat2.table.makeRecord()
            record1.setId(i + 1)
            record2.setId(self.size - i)
            self.cat1.append(record1)
            self.cat2.append(record2)

        self.matches = []
        for i in range(self.numMatches):
            index = 2*i
            match = afwTable.SimpleMatch(
                self.cat1[index], self.cat2[self.size - index - 1], index)
            if Debug:
                print("Inject:", match.first.getId(), match.second.getId())
            self.matches.append(match)

    def tearDown(self):
        del self.schema
        del self.cat1
        del self.cat2
        del self.matches

    def testMatches(self, matches=None):
        if matches is None:
            matches = self.matches
        self.assertEqual(len(matches), self.numMatches)
        for m in matches:
            str(m)  # Check __str__ works
            if Debug:
                print("Test:", m.first.getId(), m.second.getId())
            self.assertEqual(m.first.getId(), m.second.getId())

    def testIO(self):
        packed = afwTable.packMatches(self.matches)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            packed.writeFits(filename)
            matches = afwTable.BaseCatalog.readFits(filename)
        cat1 = self.cat1.copy()
        cat2 = self.cat2.copy()
        cat1.sort()
        cat2.sort()
        unpacked = afwTable.unpackMatches(matches, cat1, cat2)
        self.testMatches(unpacked)

    def testTicket2080(self):
        packed = afwTable.packMatches(self.matches)
        cat1 = self.cat1.copy()
        cat2 = afwTable.SimpleCatalog(self.schema)
        cat1.sort()
        cat2.sort()
        # just test that the next line doesn't segv
        afwTable.unpackMatches(packed, cat1, cat2)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

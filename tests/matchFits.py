#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

"""
Test for match persistence via FITS
"""

try:
    debug
except NameError:
    debug = False

import unittest
import lsst.utils.tests as utilsTests
import lsst.afw.table as afwTable

class MatchFitsTestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10
        self.numMatches = self.size//2
        self.filename = "matches.fits"
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

        self.matches = afwTable.SimpleMatchVector()
        for i in range(self.numMatches):
            index = 2*i
            match = afwTable.SimpleMatch(self.cat1[index], self.cat2[self.size - index - 1], index)
            if debug: print "Inject:", match.first.getId(), match.second.getId()
            self.matches.push_back(match)
    
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
            str(m) # Check __str__ works
            if debug: print "Test:", m.first.getId(), m.second.getId()
            self.assertEqual(m.first.getId(), m.second.getId())

    def testIO(self):
        packed = afwTable.packMatches(self.matches)
        packed.writeFits(self.filename)
        matches = afwTable.BaseCatalog.readFits(self.filename)
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
        
        
        
#################################################################
# Test suite boiler plate
#################################################################
def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(MatchFitsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

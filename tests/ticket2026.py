#!/usr/bin/env python

import sys
import os
import unittest

import lsst.utils.tests

import lsst.afw.table as afwTable

# Subtract 1 so that ids == indices
def printids(c):
    print '  ', [s.getId()-1 for s in c]

class IndexingCatalogTestCase(unittest.TestCase):

    def testMinusOne(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        catalog = afwTable.SourceCatalog(table)
        catalog.addNew()
        self.assertEqual(len(catalog), 1)
        catalog[-1]
        catalog.addNew()
        catalog.addNew()
        catalog[1] = catalog[2]
        del catalog[2]
        print catalog
        for src in catalog:
            print src.getId()
        self.assertEqual(len(catalog), 2)
        self.assertEqual(catalog[0].getId(), 1)
        self.assertEqual(catalog[1].getId(), 3)
        self.assertEqual(catalog[-1].getId(), 3)
        self.assertEqual(catalog[-2].getId(), 1)

    def testSlice(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        catalog = afwTable.SourceCatalog(table)
        for i in range(10):
            catalog.addNew()
        print 'Catalog:', printids(catalog)
        print 'Empty range (4,4)'
        c = catalog[4:4]
        printids(c)
        print 'Count by 2 (1,7,2)'
        c = catalog[1:7:2]
        printids(c)
        print 'Normal range (4,7)'
        c = catalog[4:7]
        printids(c)
        print 'Normal range 2 (4,10)'
        c = catalog[4:10]
        printids(c)
        print 'Normal range 3 (4,15)'
        c = catalog[4:15]
        printids(c)
        print 'Negative indexed range (-20,-1)'
        c = catalog[-20:-1]
        printids(c)
        print 'Negative end (1,-3)'
        catalog[1:-3]
        printids(c)
        print 'Negative step (6:1:-2)'
        catalog[6:1:-2]
        printids(c)
        print 'Negative step (6:0:-2)'
        catalog[6:0:-2]
        printids(c)
        print 'Negative step (6:0:-1)'
        catalog[6:0:-1]
        printids(c)
        print 'Negative step (6:-20:-1)'
        catalog[6:-20:-1]
        printids(c)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(IndexingCatalogTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

        
        

#!/usr/bin/env python

import sys
import os
import unittest

import lsst.utils.tests

import lsst.afw.table as afwTable

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
        print 'Empty range'
        catalog[4:4]
        print 'Count by 2'
        catalog[1:7:2]
        print 'Normal range'
        catalog[4:7]
        print 'Normal range 2'
        catalog[4:10]
        print 'Negative indexed range'
        catalog[-20:-1]
        catalog[1:-1]
        catalog[6:1:-2]

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

        
        

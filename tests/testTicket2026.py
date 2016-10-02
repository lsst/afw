#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import range
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#
#pybind11#import lsst.afw.table as afwTable
#pybind11#
#pybind11## Subtract 1 so that ids == indices
#pybind11#
#pybind11#
#pybind11#def getids(c):
#pybind11#    return [s.getId()-1 for s in c]
#pybind11#
#pybind11#
#pybind11#def printids(c):
#pybind11#    print(getids(c))
#pybind11#
#pybind11#
#pybind11#class IndexingCatalogTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def testSimpleCatalogType(self):
#pybind11#        schema = afwTable.SourceTable.makeMinimalSchema()
#pybind11#        table = afwTable.SourceTable.make(schema)
#pybind11#        catalog = afwTable.SourceCatalog(table)
#pybind11#        catalog.addNew()
#pybind11#        catcopy = catalog.copy()
#pybind11#        catsub = catalog[:]
#pybind11#        catsub2 = catalog.subset(0, 1, 1)
#pybind11#        print('catalog', catalog)
#pybind11#        print('catcopy', catcopy)
#pybind11#        print('catsub', catsub)
#pybind11#        print('catsub2', catsub2)
#pybind11#        self.assertEqual(type(catalog), type(catcopy))
#pybind11#        self.assertEqual(type(catalog), type(catsub))
#pybind11#        self.assertEqual(type(catalog), type(catsub2))
#pybind11#
#pybind11#    def testMinusOne(self):
#pybind11#        schema = afwTable.SourceTable.makeMinimalSchema()
#pybind11#        table = afwTable.SourceTable.make(schema)
#pybind11#        catalog = afwTable.SourceCatalog(table)
#pybind11#        catalog.addNew()
#pybind11#        self.assertEqual(len(catalog), 1)
#pybind11#        catalog[-1]
#pybind11#        catalog.addNew()
#pybind11#        catalog.addNew()
#pybind11#        catalog[1] = catalog[2]
#pybind11#        del catalog[2]
#pybind11#        print(catalog)
#pybind11#        for src in catalog:
#pybind11#            print(src.getId())
#pybind11#        self.assertEqual(len(catalog), 2)
#pybind11#        self.assertEqual(catalog[0].getId(), 1)
#pybind11#        self.assertEqual(catalog[1].getId(), 3)
#pybind11#        self.assertEqual(catalog[-1].getId(), 3)
#pybind11#        self.assertEqual(catalog[-2].getId(), 1)
#pybind11#
#pybind11#    def assertSlice(self, cat, start, stop, step=None):
#pybind11#        if step is None:
#pybind11#            c = cat[start:stop]
#pybind11#            tru = list(range(10))[start:stop]
#pybind11#        else:
#pybind11#            c = cat[start:stop:step]
#pybind11#            tru = list(range(10))[start:stop:step]
#pybind11#        printids(c)
#pybind11#        self.assertEqual(getids(c), tru)
#pybind11#
#pybind11#    def testSlice(self):
#pybind11#        schema = afwTable.SourceTable.makeMinimalSchema()
#pybind11#        table = afwTable.SourceTable.make(schema)
#pybind11#        catalog = afwTable.SourceCatalog(table)
#pybind11#        for i in range(10):
#pybind11#            catalog.addNew()
#pybind11#        print('Catalog:', printids(catalog))
#pybind11#        print('Empty range (4,4)')
#pybind11#        self.assertSlice(catalog, 4, 4)
#pybind11#        print('Count by 2 (1,7,2)')
#pybind11#        self.assertSlice(catalog, 1, 7, 2)
#pybind11#        print('Normal range (4,7)')
#pybind11#        self.assertSlice(catalog, 4, 7)
#pybind11#        print('Normal range 2 (4,10)')
#pybind11#        self.assertSlice(catalog, 4, 10)
#pybind11#        print('Normal range 3 (4,15)')
#pybind11#        self.assertSlice(catalog, 4, 15)
#pybind11#        print('Negative indexed range (-20,-1)')
#pybind11#        self.assertSlice(catalog, -20, -1)
#pybind11#        print('Negative end (1,-3)')
#pybind11#        self.assertSlice(catalog, 1, -3)
#pybind11#        print('Negative step (6:1:-2)')
#pybind11#        self.assertSlice(catalog, 6, 1, -2)
#pybind11#        print('Negative step (6:0:-2)')
#pybind11#        self.assertSlice(catalog, 6, 0, -2)
#pybind11#        print('Negative step (-1:-12:-2)')
#pybind11#        self.assertSlice(catalog, -1, -12, -2)
#pybind11#        print('Negative step (6:0:-1)')
#pybind11#        self.assertSlice(catalog, 6, 0, -1)
#pybind11#        print('Negative step (6:-20:-1)')
#pybind11#        self.assertSlice(catalog, 6, -20, -1)
#pybind11#        print('Negative step (6:-20:-2)')
#pybind11#        self.assertSlice(catalog, 6, -20, -2)
#pybind11#        print('Negative step (5:-20:-2)')
#pybind11#        self.assertSlice(catalog, 5, -20, -2)
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

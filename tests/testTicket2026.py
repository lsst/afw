from __future__ import absolute_import, division, print_function
import unittest

from builtins import range

import lsst.utils.tests
import lsst.afw.table as afwTable

# Subtract 1 so that ids == indices


def getids(c):
    return [s.getId()-1 for s in c]


def printids(c):
    print(getids(c))


class IndexingCatalogTestCase(unittest.TestCase):

    def testSimpleCatalogType(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        catalog = afwTable.SourceCatalog(table)
        catalog.addNew()
        catcopy = catalog.copy()
        catsub = catalog[:]
        catsub2 = catalog.subset(0, 1, 1)
        print('catalog', catalog)
        print('catcopy', catcopy)
        print('catsub', catsub)
        print('catsub2', catsub2)
        self.assertEqual(type(catalog), type(catcopy))
        self.assertEqual(type(catalog), type(catsub))
        self.assertEqual(type(catalog), type(catsub2))

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
        print(catalog)
        for src in catalog:
            print(src.getId())
        self.assertEqual(len(catalog), 2)
        self.assertEqual(catalog[0].getId(), 1)
        self.assertEqual(catalog[1].getId(), 3)
        self.assertEqual(catalog[-1].getId(), 3)
        self.assertEqual(catalog[-2].getId(), 1)

    def assertSlice(self, cat, start, stop, step=None):
        if step is None:
            c = cat[start:stop]
            tru = list(range(10))[start:stop]
        else:
            c = cat[start:stop:step]
            tru = list(range(10))[start:stop:step]
        printids(c)
        self.assertEqual(getids(c), tru)

    def testSlice(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        catalog = afwTable.SourceCatalog(table)
        for i in range(10):
            catalog.addNew()
        print('Catalog:', printids(catalog))
        print('Empty range (4,4)')
        self.assertSlice(catalog, 4, 4)
        print('Count by 2 (1,7,2)')
        self.assertSlice(catalog, 1, 7, 2)
        print('Normal range (4,7)')
        self.assertSlice(catalog, 4, 7)
        print('Normal range 2 (4,10)')
        self.assertSlice(catalog, 4, 10)
        print('Normal range 3 (4,15)')
        self.assertSlice(catalog, 4, 15)
        print('Negative indexed range (-20,-1)')
        self.assertSlice(catalog, -20, -1)
        print('Negative end (1,-3)')
        self.assertSlice(catalog, 1, -3)
        print('Negative step (6:1:-2)')
        self.assertSlice(catalog, 6, 1, -2)
        print('Negative step (6:0:-2)')
        self.assertSlice(catalog, 6, 0, -2)
        print('Negative step (-1:-12:-2)')
        self.assertSlice(catalog, -1, -12, -2)
        print('Negative step (6:0:-1)')
        self.assertSlice(catalog, 6, 0, -1)
        print('Negative step (6:-20:-1)')
        self.assertSlice(catalog, 6, -20, -1)
        print('Negative step (6:-20:-2)')
        self.assertSlice(catalog, 6, -20, -2)
        print('Negative step (5:-20:-2)')
        self.assertSlice(catalog, 5, -20, -2)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

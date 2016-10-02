#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#import unittest
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.detection as afwDet
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.table as afwTable
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#
#pybind11#
#pybind11#class SourceHeavyFootprintTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def test1(self):
#pybind11#        im = afwImage.ImageF(100, 100)
#pybind11#        im += 42.
#pybind11#        fp = afwDet.Footprint(afwGeom.Point2I(50, 50), 10.)
#pybind11#        #seed = 42
#pybind11#        #rand = afwMath.Random(afwMath.Random.MT19937, seed)
#pybind11#        #afwMath.randomGaussianImage(im, rand)
#pybind11#        mi = afwImage.MaskedImageF(im)
#pybind11#        # set a mask bit before grabbing the heavyfootprint
#pybind11#        mi.getMask().set(50, 50, 1)
#pybind11#        heavy = afwDet.makeHeavyFootprint(fp, mi)
#pybind11#        # reset it
#pybind11#        mi.getMask().set(50, 50, 0)
#pybind11#
#pybind11#        schema = afwTable.SourceTable.makeMinimalSchema()
#pybind11#        table = afwTable.SourceTable.make(schema)
#pybind11#        table.preallocate(10)
#pybind11#        catalog = afwTable.SourceCatalog(table)
#pybind11#        catalog.addNew()
#pybind11#        # This used to segfault
#pybind11#        catalog[0].setFootprint(heavy)
#pybind11#
#pybind11#        # However, we still have to up-cast
#pybind11#        fp = catalog[0].getFootprint()
#pybind11#        hfp = afwDet.cast_HeavyFootprintF(fp)
#pybind11#        # change one pixel...
#pybind11#        self.assertEqual(mi.getImage().get(50, 50), 42)
#pybind11#        self.assertEqual(mi.getMask().get(50, 50), 0)
#pybind11#        mi.getImage().set(50, 50, 100)
#pybind11#        mi.getMask().set(50, 50, 2)
#pybind11#        mi.getMask().set(51, 50, 2)
#pybind11#        self.assertEqual(mi.getImage().get(50, 50), 100)
#pybind11#        self.assertEqual(mi.getMask().get(50, 50), 2)
#pybind11#        self.assertEqual(mi.getMask().get(51, 50), 2)
#pybind11#        # reinsert the heavy footprint; it should reset the pixel value.
#pybind11#        # insert(MaskedImage)
#pybind11#        hfp.insert(mi)
#pybind11#        self.assertEqual(mi.getImage().get(50, 50), 42)
#pybind11#        self.assertEqual(mi.getMask().get(50, 50), 1)
#pybind11#        self.assertEqual(mi.getMask().get(51, 50), 0)
#pybind11#
#pybind11#        # Also test insert(Image)
#pybind11#        im = mi.getImage()
#pybind11#        self.assertEqual(im.get(50, 50), 42)
#pybind11#        im.set(50, 50, 100)
#pybind11#        self.assertEqual(im.get(50, 50), 100)
#pybind11#        self.assertEqual(mi.getImage().get(50, 50), 100)
#pybind11#        # reinsert the heavy footprint; it should reset the pixel value.
#pybind11#        hfp.insert(im)
#pybind11#        self.assertEqual(im.get(50, 50), 42)
#pybind11#        self.assertEqual(mi.getImage().get(50, 50), 42)
#pybind11#        self.assertEqual(mi.getMask().get(50, 50), 1)
#pybind11#        self.assertEqual(mi.getMask().get(51, 50), 0)
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

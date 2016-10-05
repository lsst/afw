#pybind11##!/usr/bin/env python
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
#pybind11#Tests for image.ApCorrMap
#pybind11#
#pybind11#Run with:
#pybind11#   ./testApCorrMap.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import testSchema; testSchema.run()
#pybind11#"""
#pybind11#from __future__ import division
#pybind11#
#pybind11#import os
#pybind11#import unittest
#pybind11#import numpy
#pybind11#import collections
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.geom
#pybind11#import lsst.afw.math
#pybind11#import lsst.afw.image
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11#class ApCorrMapTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        numpy.random.seed(100)
#pybind11#        self.bbox = lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(-5, -5), lsst.afw.geom.Point2I(5, 5))
#pybind11#        self.map = lsst.afw.image.ApCorrMap()
#pybind11#        for name in ("a", "b", "c"):
#pybind11#            self.map.set(name, lsst.afw.math.ChebyshevBoundedField(self.bbox, numpy.random.randn(3, 3)))
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.map
#pybind11#        del self.bbox
#pybind11#
#pybind11#    def compare(self, a, b):
#pybind11#        """Compare two ApCorrMaps for equality, without assuming that their BoundedFields have the
#pybind11#        same addresses (i.e. so we can compare after serialization).
#pybind11#        """
#pybind11#        self.assertEqual(len(a), len(b))
#pybind11#        for name, value in list(a.items()):
#pybind11#            value2 = b.get(name)
#pybind11#            self.assertIsNotNone(value2)
#pybind11#            self.assertEqual(value.getBBox(), value2.getBBox())
#pybind11#            self.assertClose(lsst.afw.math.ChebyshevBoundedField.cast(value).getCoefficients(),
#pybind11#                             lsst.afw.math.ChebyshevBoundedField.cast(value2).getCoefficients(),
#pybind11#                             rtol=0.0)
#pybind11#
#pybind11#    def testAccessors(self):
#pybind11#        """Test the accessors and other custom Swig code we've added to make ApCorrMap behave like a Python
#pybind11#        mapping."""
#pybind11#        self.assertEqual(len(self.map), 3)
#pybind11#        self.assertEqual(collections.OrderedDict(self.map),
#pybind11#                         {name: value for (name, value) in list(self.map.items())})
#pybind11#        self.assertEqual(list(collections.OrderedDict(self.map).keys()), list(self.map.keys()))
#pybind11#        self.assertEqual(list(collections.OrderedDict(self.map).values()), list(self.map.values()))
#pybind11#        self.assertEqual(list(collections.OrderedDict(self.map).items()), list(self.map.items()))
#pybind11#        self.assertEqual(list(collections.OrderedDict(self.map).keys()), list(self.map))
#pybind11#        self.assertIn("b", self.map)
#pybind11#        self.assertNotIn("d", self.map)
#pybind11#        self.map["d"] = lsst.afw.math.ChebyshevBoundedField(self.bbox, numpy.random.randn(2, 2))
#pybind11#        self.assertIn("d", self.map)
#pybind11#        self.assertIsNone(self.map.get("e"))
#pybind11#        with self.assertRaises(lsst.pex.exceptions.NotFoundError):
#pybind11#            self.map["e"]
#pybind11#        self.assertEqual(self.map.get("d"), self.map["d"])
#pybind11#
#pybind11#    def testPersistence(self):
#pybind11#        """Test that we can round-trip an ApCorrMap through FITS persistence."""
#pybind11#        filename = "testApCorrMap.fits"
#pybind11#        self.map.writeFits(filename)
#pybind11#        map2 = lsst.afw.image.ApCorrMap.readFits(filename)
#pybind11#        self.compare(self.map, map2)
#pybind11#        os.remove(filename)
#pybind11#
#pybind11#    def testExposurePersistence(self):
#pybind11#        """Test that the ApCorrMap is saved with an Exposure"""
#pybind11#        filename = "testApCorrMap.fits"
#pybind11#        exposure1 = lsst.afw.image.ExposureF(self.bbox)
#pybind11#        exposure1.getInfo().setApCorrMap(self.map)
#pybind11#        exposure1.writeFits(filename)
#pybind11#        exposure2 = lsst.afw.image.ExposureF(filename)
#pybind11#        map2 = exposure2.getInfo().getApCorrMap()
#pybind11#        self.compare(self.map, map2)
#pybind11#        os.remove(filename)
#pybind11#
#pybind11#    def testExposureRecordPersistence(self):
#pybind11#        """Test that the ApCorrMap is saved with an ExposureRecord"""
#pybind11#        filename = "testApCorrMap.fits"
#pybind11#        cat1 = lsst.afw.table.ExposureCatalog(lsst.afw.table.ExposureTable.makeMinimalSchema())
#pybind11#        record1 = cat1.addNew()
#pybind11#        record1.setApCorrMap(self.map)
#pybind11#        cat1.writeFits(filename)
#pybind11#        cat2 = lsst.afw.table.ExposureCatalog.readFits(filename)
#pybind11#        record2 = cat2[0]
#pybind11#        map2 = record2.getApCorrMap()
#pybind11#        self.compare(self.map, map2)
#pybind11#        os.remove(filename)
#pybind11#
#pybind11#    def testExposureCatalogBackwardsCompatibility(self):
#pybind11#        """Test that we can read an ExposureCatalog written with an old version of the code."""
#pybind11#        filename = os.path.join(os.environ["AFW_DIR"], "tests", "data", "version-0-ExposureCatalog.fits")
#pybind11#        cat = lsst.afw.table.ExposureCatalog.readFits(filename)
#pybind11#        record = cat[0]
#pybind11#        self.assertIsNone(record.getApCorrMap())
#pybind11#
#pybind11#    def testScale(self):
#pybind11#        """Test that we can scale an ApCorrMap"""
#pybind11#        scale = 12.345
#pybind11#        new = lsst.afw.image.ApCorrMap()
#pybind11#        for name, value in self.map.items():
#pybind11#            new.set(name, value*scale)
#pybind11#        new /= scale
#pybind11#        self.compare(self.map, new)
#pybind11#        # And back the other way
#pybind11#        new = lsst.afw.image.ApCorrMap()
#pybind11#        for name, value in self.map.items():
#pybind11#            new.set(name, value/scale)
#pybind11#        new *= scale
#pybind11#        self.compare(self.map, new)
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

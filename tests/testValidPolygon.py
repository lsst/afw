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
#pybind11#Tests for ValidPolygon
#pybind11#
#pybind11#Run with:
#pybind11#   ./testValidPolygon.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import testValidPolygonTestCase; testPolygonTestCase.run()
#pybind11#"""
#pybind11#from builtins import zip
#pybind11#
#pybind11#import os
#pybind11#import unittest
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.table as afwTable
#pybind11#
#pybind11#
#pybind11#from lsst.afw.geom.polygon import Polygon
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class ValidPolygonTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Point2I(20, 20))
#pybind11#        x = [0, 0, 10, 10]
#pybind11#        y = [0, 10, 10, 0]
#pybind11#        self.polygon = Polygon([afwGeom.Point2D(xc, yc) for xc, yc in zip(x, y)])
#pybind11#
#pybind11#    def testPersistence(self):
#pybind11#        """Test that we can round-trip an ValidPolygon through FITS persistence."""
#pybind11#        filename = "testValidPolygon.fits"
#pybind11#        self.polygon.writeFits(filename)
#pybind11#        polygon2 = Polygon.readFits(filename)
#pybind11#        self.assertEqual(self.polygon, polygon2)
#pybind11#        os.remove(filename)
#pybind11#
#pybind11#    def testExposurePersistence(self):
#pybind11#        """Test that the ValidPolygon is saved with an Exposure"""
#pybind11#        filename = "testValidPolygon.fits"
#pybind11#        exposure1 = afwImage.ExposureF(self.bbox)
#pybind11#        exposure1.getInfo().setValidPolygon(self.polygon)
#pybind11#        exposure1.writeFits(filename)
#pybind11#        exposure2 = afwImage.ExposureF(filename)
#pybind11#        polygon2 = exposure2.getInfo().getValidPolygon()
#pybind11#        self.assertEqual(self.polygon, polygon2)
#pybind11#        os.remove(filename)
#pybind11#
#pybind11#    def testExposureRecordPersistence(self):
#pybind11#        """Test that the ValidPolygon is saved with an ExposureRecord"""
#pybind11#        filename = "testValidPolygon.fits"
#pybind11#        cat1 = afwTable.ExposureCatalog(afwTable.ExposureTable.makeMinimalSchema())
#pybind11#        record1 = cat1.addNew()
#pybind11#        record1.setValidPolygon(self.polygon)
#pybind11#        cat1.writeFits(filename)
#pybind11#        cat2 = afwTable.ExposureCatalog.readFits(filename)
#pybind11#        record2 = cat2[0]
#pybind11#        polygon2 = record2.getValidPolygon()
#pybind11#        self.assertEqual(self.polygon, polygon2)
#pybind11#        os.remove(filename)
#pybind11#
#pybind11#    def testExposureCatalogBackwardsCompatibility(self):
#pybind11#        """Test that we can read an ExposureCatalog written with an old version of the code."""
#pybind11#        filename = os.path.join(os.environ["AFW_DIR"], "tests", "data", "version-0-ExposureCatalog.fits")
#pybind11#        cat = afwTable.ExposureCatalog.readFits(filename)
#pybind11#        record = cat[0]
#pybind11#        self.assertIsNone(record.getValidPolygon())
#pybind11#
#pybind11#        filename2 = os.path.join(os.environ["AFW_DIR"], "tests", "data", "version-1-ExposureCatalog.fits")
#pybind11#        cat2 = afwTable.ExposureCatalog.readFits(filename2)
#pybind11#        record2 = cat2[0]
#pybind11#        self.assertIsNone(record2.getValidPolygon())
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

#!/usr/bin/env python
#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

"""
Tests for ValidPolygon

Run with:
   ./testValidPolygon.py
or
   python
   >>> import testValidPolygonTestCase; testPolygonTestCase.run()
"""

import sys
import os
import unittest
import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable


from lsst.afw.geom.polygon import Polygon

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ValidPolygonTestCase(utilsTests.TestCase):

    def setUp(self):
        self.bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Point2I(20, 20))
        x=[0,0,10,10]
        y=[0,10,10,0]
        self.polygon=Polygon([afwGeom.Point2D(xc,yc) for xc,yc in zip(x,y)])

    def testPersistence(self):
        """Test that we can round-trip an ValidPolygon through FITS persistence."""
        filename = "testValidPolygon.fits"
        self.polygon.writeFits(filename)
        polygon2 = Polygon.readFits(filename)
        self.assertEqual(self.polygon, polygon2)
        os.remove(filename)

    def testExposurePersistence(self):
        """Test that the ValidPolygon is saved with an Exposure"""
        filename = "testValidPolygon.fits"
        exposure1 = afwImage.ExposureF(self.bbox)
        exposure1.getInfo().setValidPolygon(self.polygon)
        exposure1.writeFits(filename)
        exposure2 = afwImage.ExposureF(filename)
        polygon2 = exposure2.getInfo().getValidPolygon()
        self.assertEqual(self.polygon, polygon2)
        os.remove(filename)

    def testExposureRecordPersistence(self):
        """Test that the ValidPolygon is saved with an ExposureRecord"""
        filename = "testValidPolygon.fits"
        cat1 = afwTable.ExposureCatalog(afwTable.ExposureTable.makeMinimalSchema())
        record1 = cat1.addNew()
        record1.setValidPolygon(self.polygon)
        cat1.writeFits(filename)
        cat2 = afwTable.ExposureCatalog.readFits(filename)
        record2 = cat2[0]
        polygon2 = record2.getValidPolygon()
        self.assertEqual(self.polygon, polygon2)
        os.remove(filename)

    def testExposureCatalogBackwardsCompatibility(self):
        """Test that we can read an ExposureCatalog written with an old version of the code."""
        filename = os.path.join(os.environ["AFW_DIR"], "tests", "data", "version-0-ExposureCatalog.fits")
        cat = afwTable.ExposureCatalog.readFits(filename)
        record = cat[0]
        self.assertIsNone(record.getValidPolygon())

        filename2 = os.path.join(os.environ["AFW_DIR"], "tests", "data", "version-1-ExposureCatalog.fits")
        cat2 = afwTable.ExposureCatalog.readFits(filename2)
        record2 = cat2[0]
        self.assertIsNone(record2.getValidPolygon())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ValidPolygonTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

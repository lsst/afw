#pybind11##!/usr/bin/env python
#pybind11#
#pybind11## LSST Data Management System
#pybind11## Copyright 2016 LSST Corporation.
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
#pybind11## The classes in this test are a little non-standard to reduce code
#pybind11## duplication and support automated unittest discovery.
#pybind11## A base class includes all the code that implements the testing and
#pybind11## itself inherits from unittest.TestCase. unittest automated discovery
#pybind11## will scan all classes that inherit from unittest.TestCase and invoke
#pybind11## any test methods found. To prevent this base class from being executed
#pybind11## the test methods are placed in a different class that does not inherit
#pybind11## from unittest.TestCase. The actual test classes then inherit from
#pybind11## both the testing class and the implementation class allowing test
#pybind11## discovery to only run tests found in the subclasses.
#pybind11#
#pybind11#from __future__ import absolute_import, division, print_function
#pybind11#import math
#pybind11#import unittest
#pybind11#from builtins import zip
#pybind11#
#pybind11#import numpy as np
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.coord as afwCoord
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.table as afwTable
#pybind11#
#pybind11#class UpdateTestCase(lsst.utils.tests.TestCase):
#pybind11#    """A test case for the lsst.afw.table.updateRefCentroids and updateSourceCoords
#pybind11#    """
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.crval = afwCoord.IcrsCoord(afwGeom.PointD(44., 45.))
#pybind11#        self.crpix = afwGeom.Point2D(15000, 4000)
#pybind11#
#pybind11#        arcsecPerPixel = 1/3600.0
#pybind11#        CD11 = arcsecPerPixel
#pybind11#        CD12 = 0
#pybind11#        CD21 = 0
#pybind11#        CD22 = arcsecPerPixel
#pybind11#
#pybind11#        self.wcs = afwImage.makeWcs(self.crval, self.crpix, CD11, CD12, CD21, CD22)
#pybind11#
#pybind11#        refSchema = afwTable.SimpleTable.makeMinimalSchema()
#pybind11#        self.refCentroidKey = afwTable.Point2DKey.addFields(refSchema, "centroid", "centroid", "pixels")
#pybind11#        self.refCoordKey = afwTable.CoordKey(refSchema["coord"])
#pybind11#        self.refCat = afwTable.SimpleCatalog(refSchema)
#pybind11#
#pybind11#        # an alias is required to make src.getCentroid() work;
#pybind11#        # simply defining a field named "slot_Centroid" doesn't suffice
#pybind11#        srcSchema = afwTable.SourceTable.makeMinimalSchema()
#pybind11#        self.srcCentroidKey = afwTable.Point2DKey.addFields(srcSchema, "base_SdssCentroid",
#pybind11#                                                            "centroid", "pixels")
#pybind11#        srcAliases = srcSchema.getAliasMap()
#pybind11#        srcAliases.set("slot_Centroid", "base_SdssCentroid")
#pybind11#        self.srcCoordKey = afwTable.CoordKey(srcSchema["coord"])
#pybind11#        self.sourceCat = afwTable.SourceCatalog(srcSchema)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.wcs
#pybind11#        del self.refCat
#pybind11#        del self.sourceCat
#pybind11#
#pybind11#    def testNull(self):
#pybind11#        """Check that an empty list causes no problems for either function"""
#pybind11#        afwTable.updateRefCentroids(self.wcs, [])
#pybind11#        afwTable.updateSourceCoords(self.wcs, [])
#pybind11#
#pybind11#    def testRefCenter(self):
#pybind11#        """Check that a ref obj at the center is handled as expected"""
#pybind11#        refObj = self.refCat.addNew()
#pybind11#        refObj.set(self.refCoordKey, self.crval)
#pybind11#
#pybind11#        # initial centroid should be nan
#pybind11#        nanRefCentroid = self.refCat[0].get(self.refCentroidKey)
#pybind11#        for val in nanRefCentroid:
#pybind11#            self.assertTrue(math.isnan(val))
#pybind11#
#pybind11#        # computed centroid should be crpix
#pybind11#        afwTable.updateRefCentroids(self.wcs, self.refCat)
#pybind11#        refCentroid = self.refCat[0].get(self.refCentroidKey)
#pybind11#        self.assertPairsNearlyEqual(refCentroid, self.crpix)
#pybind11#
#pybind11#        # coord should not be changed
#pybind11#        self.assertEqual(self.refCat[0].get(self.refCoordKey), self.crval)
#pybind11#
#pybind11#    def testSourceCenter(self):
#pybind11#        """Check that a source at the center is handled as expected"""
#pybind11#        src = self.sourceCat.addNew()
#pybind11#        src.set(self.srcCentroidKey, self.crpix)
#pybind11#
#pybind11#        # initial coord should be nan; as a sanity-check
#pybind11#        nanSourceCoord = self.sourceCat[0].get(self.srcCoordKey)
#pybind11#        for val in nanSourceCoord:
#pybind11#            self.assertTrue(math.isnan(val))
#pybind11#
#pybind11#        # compute coord should be crval
#pybind11#        afwTable.updateSourceCoords(self.wcs, self.sourceCat)
#pybind11#        srcCoord = self.sourceCat[0].get(self.srcCoordKey)
#pybind11#        self.assertPairsNearlyEqual(srcCoord, self.crval)
#pybind11#
#pybind11#        # centroid should not be changed; also make sure that getCentroid words
#pybind11#        self.assertEqual(self.sourceCat[0].getCentroid(), self.crpix)
#pybind11#
#pybind11#    def testLists(self):
#pybind11#        """Check updating lists of reference objects and sources"""
#pybind11#        # arbitrary but reasonable values that are intentionally different than testCatalogs
#pybind11#        maxPix = 1000
#pybind11#        numPoints = 10
#pybind11#        self.setCatalogs(maxPix=maxPix, numPoints=numPoints)
#pybind11#
#pybind11#        # update the catalogs as lists
#pybind11#        afwTable.updateSourceCoords(self.wcs, [s for s in self.sourceCat])
#pybind11#        afwTable.updateRefCentroids(self.wcs, [r for r in self.refCat])
#pybind11#
#pybind11#        self.checkCatalogs()
#pybind11#
#pybind11#    def testCatalogs(self):
#pybind11#        """Check updating catalogs of reference objects and sources"""
#pybind11#        # arbitrary but reasonable values that are intentionally different than testLists
#pybind11#        maxPix = 2000
#pybind11#        numPoints = 9
#pybind11#        self.setCatalogs(maxPix=maxPix, numPoints=numPoints)
#pybind11#
#pybind11#        # update the catalogs
#pybind11#        afwTable.updateSourceCoords(self.wcs, self.sourceCat)
#pybind11#        afwTable.updateRefCentroids(self.wcs, self.refCat)
#pybind11#
#pybind11#        # check that centroids and coords match
#pybind11#        self.checkCatalogs()
#pybind11#
#pybind11#    def checkCatalogs(self, maxPixDiff=1e-7, maxSkyDiff=0.001*afwGeom.arcseconds):
#pybind11#        """Check that the source and reference object catalogs have equal centroids and coords"""
#pybind11#        self.assertEqual(len(self.sourceCat), len(self.refCat))
#pybind11#
#pybind11#        for src, refObj in zip(self.sourceCat, self.refCat):
#pybind11#            srcCentroid = src.get(self.srcCentroidKey)
#pybind11#            refCentroid = refObj.get(self.refCentroidKey)
#pybind11#            self.assertPairsNearlyEqual(srcCentroid, refCentroid, maxDiff=maxPixDiff)
#pybind11#
#pybind11#            srcCoord = src.get(self.srcCoordKey)
#pybind11#            refCoord = refObj.get(self.refCoordKey)
#pybind11#            self.assertCoordsNearlyEqual(srcCoord, refCoord, maxDiff=maxSkyDiff)
#pybind11#
#pybind11#    def setCatalogs(self, maxPix, numPoints):
#pybind11#        """Set the source centroids and reference object coords
#pybind11#
#pybind11#        Set self.sourceCat centroids to a square grid of points
#pybind11#        and set self.refCat coords to the corresponding sky positions
#pybind11#
#pybind11#        The catalogs must be empty to start
#pybind11#
#pybind11#        @param[in] maxPix  maximum pixel position; used for both x and y;
#pybind11#                            the min is the negative of maxPix
#pybind11#        @param[in] numPoints  number of points in x or y; total points = numPoints*numPoints
#pybind11#        """
#pybind11#        if len(self.sourceCat) != 0:
#pybind11#            raise RuntimeError("self.sourceCat must be empty")
#pybind11#        if len(self.refCat) != 0:
#pybind11#            raise RuntimeError("self.refCat must be empty")
#pybind11#
#pybind11#        for i in np.linspace(-maxPix, maxPix, numPoints):
#pybind11#            for j in np.linspace(-maxPix, maxPix, numPoints):
#pybind11#                centroid = afwGeom.Point2D(i, j)
#pybind11#                src = self.sourceCat.addNew()
#pybind11#                src.set(self.srcCentroidKey, centroid)
#pybind11#
#pybind11#                refObj = self.refCat.addNew()
#pybind11#                coord = self.wcs.pixelToSky(centroid)
#pybind11#                refObj.set(self.refCoordKey, coord)
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

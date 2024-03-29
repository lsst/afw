# LSST Data Management System
# Copyright 2016 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# The classes in this test are a little non-standard to reduce code
# duplication and support automated unittest discovery.
# A base class includes all the code that implements the testing and
# itself inherits from unittest.TestCase. unittest automated discovery
# will scan all classes that inherit from unittest.TestCase and invoke
# any test methods found. To prevent this base class from being executed
# the test methods are placed in a different class that does not inherit
# from unittest.TestCase. The actual test classes then inherit from
# both the testing class and the implementation class allowing test
# discovery to only run tests found in the subclasses.

import math
import unittest

import numpy as np

import lsst.utils.tests
import lsst.geom
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable


class UpdateTestCase(lsst.utils.tests.TestCase):
    """A test case for the lsst.afw.table.updateRefCentroids and updateSourceCoords
    """

    def setUp(self):
        self.crval = lsst.geom.SpherePoint(44.0, 45.0, lsst.geom.degrees)
        self.crpix = lsst.geom.Point2D(15000, 4000)

        arcsecPerPixel = 1/3600.0
        cdMatrix = afwGeom.makeCdMatrix(arcsecPerPixel * lsst.geom.arcseconds)
        self.wcs = afwGeom.makeSkyWcs(crval=self.crval, crpix=self.crpix, cdMatrix=cdMatrix)

        refSchema = afwTable.SimpleTable.makeMinimalSchema()
        self.refCentroidKey = afwTable.Point2DKey.addFields(
            refSchema, "centroid", "centroid", "pixels")
        self.refCoordKey = afwTable.CoordKey(refSchema["coord"])
        self.refHasCentroidKey = refSchema.addField("hasCentroid", type="Flag")
        self.refCat = afwTable.SimpleCatalog(refSchema)

        # an alias is required to make src.getCentroid() work;
        # simply defining a field named "slot_Centroid" doesn't suffice
        srcSchema = afwTable.SourceTable.makeMinimalSchema()
        self.srcCentroidKey = afwTable.Point2DKey.addFields(srcSchema, "base_SdssCentroid",
                                                            "centroid", "pixels")
        self.srcCentroidErrKey = afwTable.CovarianceMatrix2fKey.addFields(srcSchema, "base_SdssCentroid",
                                                                          ["x", "y"], "pixels")
        self.srcCoordErrKey = afwTable.CoordKey.addErrorFields(srcSchema)

        srcAliases = srcSchema.getAliasMap()
        srcAliases.set("slot_Centroid", "base_SdssCentroid")
        self.srcCoordKey = afwTable.CoordKey(srcSchema["coord"])
        self.sourceCat = afwTable.SourceCatalog(srcSchema)

    def tearDown(self):
        del self.wcs
        del self.refCat
        del self.sourceCat

    def testNull(self):
        """Check that an empty list causes no problems for either function"""
        afwTable.updateRefCentroids(self.wcs, [])
        afwTable.updateSourceCoords(self.wcs, [])

    def testRefCenter(self):
        """Check that a ref obj at the center is handled as expected"""
        refObj = self.refCat.addNew()
        refObj.set(self.refCoordKey, self.crval)

        # initial centroid should be nan and hasCentroid False
        nanRefCentroid = self.refCat[0].get(self.refCentroidKey)
        for val in nanRefCentroid:
            self.assertTrue(math.isnan(val))
        self.assertFalse(self.refCat[0].get(self.refHasCentroidKey))

        # computed centroid should be crpix and hasCentroid True
        afwTable.updateRefCentroids(self.wcs, self.refCat)
        refCentroid = self.refCat[0].get(self.refCentroidKey)
        self.assertPairsAlmostEqual(refCentroid, self.crpix)
        self.assertTrue(self.refCat[0].get(self.refHasCentroidKey))

        # coord should not be changed
        self.assertEqual(self.refCat[0].get(self.refCoordKey), self.crval)

    def testSourceCenter(self):
        """Check that a source at the center is handled as expected"""
        src = self.sourceCat.addNew()
        src.set(self.srcCentroidKey, self.crpix)

        # initial coord should be nan; as a sanity-check
        nanSourceCoord = self.sourceCat[0].get(self.srcCoordKey)
        for val in nanSourceCoord:
            self.assertTrue(math.isnan(val))

        # compute coord should be crval
        afwTable.updateSourceCoords(self.wcs, self.sourceCat)
        srcCoord = self.sourceCat[0].get(self.srcCoordKey)
        self.assertPairsAlmostEqual(srcCoord, self.crval)

        # centroid should not be changed; also make sure that getCentroid words
        self.assertEqual(self.sourceCat[0].getCentroid(), self.crpix)

    def testLists(self):
        """Check updating lists of reference objects and sources"""
        # arbitrary but reasonable values that are intentionally different than
        # testCatalogs
        maxPix = 1000
        numPoints = 10
        self.setCatalogs(maxPix=maxPix, numPoints=numPoints)

        # update the catalogs as lists
        afwTable.updateSourceCoords(self.wcs, [s for s in self.sourceCat])
        afwTable.updateRefCentroids(self.wcs, [r for r in self.refCat])

        self.checkCatalogs()

    def testCatalogs(self):
        """Check updating catalogs of reference objects and sources"""
        # arbitrary but reasonable values that are intentionally different than
        # testLists
        maxPix = 2000
        numPoints = 9
        self.setCatalogs(maxPix=maxPix, numPoints=numPoints)

        # update the catalogs
        afwTable.updateSourceCoords(self.wcs, self.sourceCat)
        afwTable.updateRefCentroids(self.wcs, self.refCat)

        # check that centroids and coords match
        self.checkCatalogs()

    def testCoordErrors(self):
        """Check that updateSourceCoords has correctly propagated the centroid
        errors.
        """
        maxPix = 2000
        numPoints = 10

        self.setCatalogs(maxPix=maxPix, numPoints=numPoints)
        scale = (1.0 * lsst.geom.arcseconds).asDegrees()
        # update the catalogs
        afwTable.updateSourceCoords(self.wcs, self.sourceCat)
        for src in self.sourceCat:
            center = src.get(self.srcCentroidKey)
            skyCenter = self.wcs.pixelToSky(center)
            localGnomonicWcs = lsst.afw.geom.makeSkyWcs(
                center, skyCenter, np.diag((scale, scale)))
            measurementToLocalGnomonic = self.wcs.getTransform().then(
                localGnomonicWcs.getTransform().inverted()
            )
            localMatrix = measurementToLocalGnomonic.getJacobian(center)
            radMatrix = np.radians(localMatrix / 3600)

            centroidErr = src.get(self.srcCentroidErrKey)
            coordErr = radMatrix.dot(centroidErr.dot(radMatrix.T))
            catCoordErr = src.get(self.srcCoordErrKey)
            np.testing.assert_almost_equal(coordErr, catCoordErr)

    def checkCatalogs(self, maxPixDiff=1e-5, maxSkyDiff=0.001*lsst.geom.arcseconds):
        """Check that the source and reference object catalogs have equal centroids and coords"""
        self.assertEqual(len(self.sourceCat), len(self.refCat))

        for src, refObj in zip(self.sourceCat, self.refCat):
            self.assertTrue(refObj.get(self.refHasCentroidKey))
            srcCentroid = src.get(self.srcCentroidKey)
            refCentroid = refObj.get(self.refCentroidKey)
            self.assertPairsAlmostEqual(
                srcCentroid, refCentroid, maxDiff=maxPixDiff)

            srcCoord = src.get(self.srcCoordKey)
            refCoord = refObj.get(self.refCoordKey)
            self.assertSpherePointsAlmostEqual(
                srcCoord, refCoord, maxSep=maxSkyDiff)

    def setCatalogs(self, maxPix, numPoints):
        """Set the source centroids and reference object coords

        Set self.sourceCat centroids to a square grid of points
        and set self.refCat coords to the corresponding sky positions

        The catalogs must be empty to start

        @param[in] maxPix  maximum pixel position; used for both x and y;
                            the min is the negative of maxPix
        @param[in] numPoints  number of points in x or y; total points = numPoints*numPoints
        """
        if len(self.sourceCat) != 0:
            raise RuntimeError("self.sourceCat must be empty")
        if len(self.refCat) != 0:
            raise RuntimeError("self.refCat must be empty")

        for i in np.linspace(-maxPix, maxPix, numPoints):
            for j in np.linspace(-maxPix, maxPix, numPoints):
                centroid = lsst.geom.Point2D(i, j)
                centroidErr = np.array([[0.05, 0], [0, 0.05]])
                src = self.sourceCat.addNew()
                src.set(self.srcCentroidKey, centroid)
                src.set(self.srcCentroidErrKey, centroidErr)

                refObj = self.refCat.addNew()
                coord = self.wcs.pixelToSky(centroid)
                refObj.set(self.refCoordKey, coord)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

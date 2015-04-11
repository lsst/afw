#!/usr/bin/env python2
from __future__ import absolute_import, division
#
# LSST Data Management System
# Copyright 2008-2014 LSST Corporation.
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

"""
Tests for SourceTable slots with version > 0
"""

import unittest
import numpy

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.table
import lsst.afw.geom
import lsst.afw.coord
import lsst.afw.image
import lsst.afw.detection

numpy.random.seed(1)

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def makeArray(size, dtype):
    return numpy.array(numpy.random.randn(*size), dtype=dtype)

def makeCov(size, dtype):
    m = numpy.array(numpy.random.randn(size, size), dtype=dtype)
    return numpy.dot(m, m.transpose())

def makeWcs():
    crval = lsst.afw.coord.Coord(lsst.afw.geom.Point2D(1.606631, 5.090329))
    crpix = lsst.afw.geom.Point2D(2036., 2000.)
    return lsst.afw.image.makeWcs(crval, crpix, 5.399452e-5, -1.30770e-5, 1.30770e-5, 5.399452e-5)

class SourceTableTestCase(lsst.utils.tests.TestCase):

    def fillRecord(self, record):
        record.set(self.fluxKey, numpy.random.randn())
        record.set(self.fluxErrKey, numpy.random.randn())
        record.set(self.centroidKey, lsst.afw.geom.Point2D(*numpy.random.randn(2)))
        record.set(self.centroidErrKey, makeCov(2, numpy.float32))
        record.set(self.shapeKey, lsst.afw.geom.ellipses.Quadrupole(*numpy.random.randn(3)))
        record.set(self.shapeErrKey, makeCov(3, numpy.float32))
        record.set(self.fluxFlagKey, numpy.random.randn() > 0)
        record.set(self.centroidFlagKey, numpy.random.randn() > 0)
        record.set(self.shapeFlagKey, numpy.random.randn() > 0)

    def makeFlux(self, schema, prefix, uncertainty):
        self.fluxKey = self.schema.addField(prefix+"_flux", type="D")
        if uncertainty:
            self.fluxErrKey = self.schema.addField(prefix+"_fluxSigma", type="D")
        self.fluxFlagKey = self.schema.addField(prefix+"_flag", type="Flag")

    def makeCentroid(self, schema, prefix, uncertainty):
        self.centroidXKey = self.schema.addField(prefix+"_x", type="D")
        self.centroidYKey = self.schema.addField(prefix+"_y", type="D")
        sigmaArray = []
        covArray = []
        if uncertainty > 0:
            self.centroidXErrKey = self.schema.addField(prefix+"_xSigma", type="F")
            self.centroidYErrKey = self.schema.addField(prefix+"_ySigma", type="F")
            sigmaArray.append(self.centroidXErrKey)
            sigmaArray.append(self.centroidYErrKey)
        if uncertainty > 1:
            self.centroidXYCovKey = self.schema.addField(prefix+"_x_y_Cov", type="F")
            covArray.append(self.centroidXYCovKey)
        self.centroidKey = lsst.afw.table.Point2DKey(self.centroidXKey, self.centroidYKey)
        self.centroidErrKey = lsst.afw.table.CovarianceMatrix2fKey(sigmaArray, covArray)
        self.centroidFlagKey = self.schema.addField(prefix+"_flag", type="Flag")

    def makeShape(self, schema, prefix, uncertainty):
        self.shapeXXKey = self.schema.addField(prefix+"_xx", type="D")
        self.shapeYYKey = self.schema.addField(prefix+"_yy", type="D")
        self.shapeXYKey = self.schema.addField(prefix+"_xy", type="D")
        self.shapeKey = lsst.afw.table.QuadrupoleKey(self.shapeXXKey, self.shapeYYKey, self.shapeXYKey)
        sigmaArray = []
        covArray = []
        if uncertainty > 0:
            self.shapeXXErrKey = self.schema.addField(prefix+"_xxSigma", type="F")
            self.shapeYYErrKey = self.schema.addField(prefix+"_yySigma", type="F")
            self.shapeXYErrKey = self.schema.addField(prefix+"_xySigma", type="F")
            sigmaArray.append(self.shapeXXErrKey)
            sigmaArray.append(self.shapeYYErrKey)
            sigmaArray.append(self.shapeXYErrKey)
        if uncertainty > 1:
            self.shapeXXYYCovKey = self.schema.addField(prefix+"_xx_yy_Cov", type="F")
            self.shapeXXXYCovKey = self.schema.addField(prefix+"_xx_xy_Cov", type="F")
            self.shapeYYXYCovKey = self.schema.addField(prefix+"_yy_xy_Cov", type="F")
            covArray.append(self.shapeXXYYCovKey)
            covArray.append(self.shapeXXXYCovKey)
            covArray.append(self.shapeYYXYCovKey)
        self.shapeErrKey = lsst.afw.table.CovarianceMatrix3fKey(sigmaArray, covArray)
        self.shapeFlagKey = self.schema.addField(prefix+"_flag", type="Flag")

    def setUp(self):
        self.schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        self.schema.setVersion(1)
        self.makeFlux(self.schema, "a", 1)
        self.makeCentroid(self.schema, "b", 2)
        self.makeShape(self.schema, "c", 2)
        self.table = lsst.afw.table.SourceTable.make(self.schema)
        self.catalog = lsst.afw.table.SourceCatalog(self.table)
        self.record = self.catalog.addNew()
        self.fillRecord(self.record)
        self.record.setId(50)
        self.fillRecord(self.catalog.addNew())
        self.fillRecord(self.catalog.addNew())
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")

    def tearDown(self):
        del self.schema
        del self.record
        del self.table
        del self.catalog

    def testPersisted(self):
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            self.catalog.writeFits(filename)
            catalog = lsst.afw.table.SourceCatalog.readFits(filename)
            table = catalog.getTable()
            record = catalog[0]
            # I'm using the keys from the non-persisted table.  They should work at least in the
            # current implementation
            self.assertEqual(table.getPsfFluxDefinition(), "a")
            self.assertEqual(record.get(self.fluxKey), record.getPsfFlux())
            self.assertEqual(record.get(self.fluxFlagKey), record.getPsfFluxFlag())
            self.assertEqual(table.getCentroidDefinition(), "b")
            self.assertEqual(record.get(self.centroidKey), record.getCentroid())
            self.assertClose(record.get(self.centroidErrKey), record.getCentroidErr())
            self.assertEqual(table.getShapeDefinition(), "c")
            self.assertEqual(record.get(self.shapeKey), record.getShape())
            self.assertClose(record.get(self.shapeErrKey), record.getShapeErr())

    def testDefiner1(self):
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        self.assertEqual(self.table.getPsfFluxDefinition(), "a")
        self.assertEqual(self.record.get(self.fluxKey), self.record.getPsfFlux())
        self.assertEqual(self.record.get(self.fluxFlagKey), self.record.getPsfFluxFlag())
        self.assertEqual(self.table.getCentroidDefinition(), "b")
        self.assertEqual(self.record.get(self.centroidKey), self.record.getCentroid())
        self.assertClose(self.record.get(self.centroidErrKey), self.record.getCentroidErr())
        self.assertEqual(self.table.getShapeDefinition(), "c")
        self.assertEqual(self.record.get(self.shapeKey), self.record.getShape())
        self.assertClose(self.record.get(self.shapeErrKey), self.record.getShapeErr())

    def testCoordUpdate(self):
        self.table.defineCentroid("b")
        wcs = makeWcs()
        self.record.updateCoord(wcs)
        coord1 = self.record.getCoord()
        coord2 = wcs.pixelToSky(self.record.get(self.centroidKey))
        self.assertEqual(coord1, coord2)

    def testColumnView(self):
        cols1 = self.catalog.getColumnView()
        cols2 = self.catalog.columns
        self.assert_(cols1 is cols2)
        self.assert_(isinstance(cols1, lsst.afw.table.SourceColumnView))
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        self.assert_((cols2["a_flux"] == cols2.getPsfFlux()).all())
        self.assert_((cols2["a_fluxSigma"] == cols2.getPsfFluxErr()).all())
        self.assert_((cols2["b_x"] == cols2.getX()).all())
        self.assert_((cols2["b_y"] == cols2.getY()).all())
        self.assert_((cols2["c_xx"] == cols2.getIxx()).all())
        self.assert_((cols2["c_yy"] == cols2.getIyy()).all())
        self.assert_((cols2["c_xy"] == cols2.getIxy()).all())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(SourceTableTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

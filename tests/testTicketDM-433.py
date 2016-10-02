#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
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
#pybind11#Tests for SourceTable slots with version > 0
#pybind11#"""
#pybind11#
#pybind11#import unittest
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.table
#pybind11#import lsst.afw.geom
#pybind11#import lsst.afw.coord
#pybind11#import lsst.afw.image
#pybind11#import lsst.afw.detection
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#def makeArray(size, dtype):
#pybind11#    return numpy.array(numpy.random.randn(*size), dtype=dtype)
#pybind11#
#pybind11#
#pybind11#def makeCov(size, dtype):
#pybind11#    m = numpy.array(numpy.random.randn(size, size), dtype=dtype)
#pybind11#    return numpy.dot(m, m.transpose())
#pybind11#
#pybind11#
#pybind11#def makeWcs():
#pybind11#    crval = lsst.afw.coord.Coord(lsst.afw.geom.Point2D(1.606631, 5.090329))
#pybind11#    crpix = lsst.afw.geom.Point2D(2036., 2000.)
#pybind11#    return lsst.afw.image.makeWcs(crval, crpix, 5.399452e-5, -1.30770e-5, 1.30770e-5, 5.399452e-5)
#pybind11#
#pybind11#
#pybind11#class SourceTableTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def fillRecord(self, record):
#pybind11#        record.set(self.fluxKey, numpy.random.randn())
#pybind11#        record.set(self.fluxErrKey, numpy.random.randn())
#pybind11#        record.set(self.centroidKey, lsst.afw.geom.Point2D(*numpy.random.randn(2)))
#pybind11#        record.set(self.centroidErrKey, makeCov(2, numpy.float32))
#pybind11#        record.set(self.shapeKey, lsst.afw.geom.ellipses.Quadrupole(*numpy.random.randn(3)))
#pybind11#        record.set(self.shapeErrKey, makeCov(3, numpy.float32))
#pybind11#        record.set(self.fluxFlagKey, numpy.random.randn() > 0)
#pybind11#        record.set(self.centroidFlagKey, numpy.random.randn() > 0)
#pybind11#        record.set(self.shapeFlagKey, numpy.random.randn() > 0)
#pybind11#
#pybind11#    def makeFlux(self, schema, prefix, uncertainty):
#pybind11#        self.fluxKey = self.schema.addField(prefix+"_flux", type="D")
#pybind11#        if uncertainty:
#pybind11#            self.fluxErrKey = self.schema.addField(prefix+"_fluxSigma", type="D")
#pybind11#        self.fluxFlagKey = self.schema.addField(prefix+"_flag", type="Flag")
#pybind11#
#pybind11#    def makeCentroid(self, schema, prefix, uncertainty):
#pybind11#        self.centroidXKey = self.schema.addField(prefix+"_x", type="D")
#pybind11#        self.centroidYKey = self.schema.addField(prefix+"_y", type="D")
#pybind11#        sigmaArray = []
#pybind11#        covArray = []
#pybind11#        if uncertainty > 0:
#pybind11#            self.centroidXErrKey = self.schema.addField(prefix+"_xSigma", type="F")
#pybind11#            self.centroidYErrKey = self.schema.addField(prefix+"_ySigma", type="F")
#pybind11#            sigmaArray.append(self.centroidXErrKey)
#pybind11#            sigmaArray.append(self.centroidYErrKey)
#pybind11#        if uncertainty > 1:
#pybind11#            self.centroidXYCovKey = self.schema.addField(prefix+"_x_y_Cov", type="F")
#pybind11#            covArray.append(self.centroidXYCovKey)
#pybind11#        self.centroidKey = lsst.afw.table.Point2DKey(self.centroidXKey, self.centroidYKey)
#pybind11#        self.centroidErrKey = lsst.afw.table.CovarianceMatrix2fKey(sigmaArray, covArray)
#pybind11#        self.centroidFlagKey = self.schema.addField(prefix+"_flag", type="Flag")
#pybind11#
#pybind11#    def makeShape(self, schema, prefix, uncertainty):
#pybind11#        self.shapeXXKey = self.schema.addField(prefix+"_xx", type="D")
#pybind11#        self.shapeYYKey = self.schema.addField(prefix+"_yy", type="D")
#pybind11#        self.shapeXYKey = self.schema.addField(prefix+"_xy", type="D")
#pybind11#        self.shapeKey = lsst.afw.table.QuadrupoleKey(self.shapeXXKey, self.shapeYYKey, self.shapeXYKey)
#pybind11#        sigmaArray = []
#pybind11#        covArray = []
#pybind11#        if uncertainty > 0:
#pybind11#            self.shapeXXErrKey = self.schema.addField(prefix+"_xxSigma", type="F")
#pybind11#            self.shapeYYErrKey = self.schema.addField(prefix+"_yySigma", type="F")
#pybind11#            self.shapeXYErrKey = self.schema.addField(prefix+"_xySigma", type="F")
#pybind11#            sigmaArray.append(self.shapeXXErrKey)
#pybind11#            sigmaArray.append(self.shapeYYErrKey)
#pybind11#            sigmaArray.append(self.shapeXYErrKey)
#pybind11#        if uncertainty > 1:
#pybind11#            self.shapeXXYYCovKey = self.schema.addField(prefix+"_xx_yy_Cov", type="F")
#pybind11#            self.shapeXXXYCovKey = self.schema.addField(prefix+"_xx_xy_Cov", type="F")
#pybind11#            self.shapeYYXYCovKey = self.schema.addField(prefix+"_yy_xy_Cov", type="F")
#pybind11#            covArray.append(self.shapeXXYYCovKey)
#pybind11#            covArray.append(self.shapeXXXYCovKey)
#pybind11#            covArray.append(self.shapeYYXYCovKey)
#pybind11#        self.shapeErrKey = lsst.afw.table.CovarianceMatrix3fKey(sigmaArray, covArray)
#pybind11#        self.shapeFlagKey = self.schema.addField(prefix+"_flag", type="Flag")
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        numpy.random.seed(1)
#pybind11#        self.schema = lsst.afw.table.SourceTable.makeMinimalSchema()
#pybind11#        self.makeFlux(self.schema, "a", 1)
#pybind11#        self.makeCentroid(self.schema, "b", 2)
#pybind11#        self.makeShape(self.schema, "c", 2)
#pybind11#        self.table = lsst.afw.table.SourceTable.make(self.schema)
#pybind11#        self.catalog = lsst.afw.table.SourceCatalog(self.table)
#pybind11#        self.record = self.catalog.addNew()
#pybind11#        self.fillRecord(self.record)
#pybind11#        self.record.setId(50)
#pybind11#        self.fillRecord(self.catalog.addNew())
#pybind11#        self.fillRecord(self.catalog.addNew())
#pybind11#        self.table.definePsfFlux("a")
#pybind11#        self.table.defineCentroid("b")
#pybind11#        self.table.defineShape("c")
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.schema
#pybind11#        del self.record
#pybind11#        del self.table
#pybind11#        del self.catalog
#pybind11#
#pybind11#    def testPersisted(self):
#pybind11#        self.table.definePsfFlux("a")
#pybind11#        self.table.defineCentroid("b")
#pybind11#        self.table.defineShape("c")
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as filename:
#pybind11#            self.catalog.writeFits(filename)
#pybind11#            catalog = lsst.afw.table.SourceCatalog.readFits(filename)
#pybind11#            table = catalog.getTable()
#pybind11#            record = catalog[0]
#pybind11#            # I'm using the keys from the non-persisted table.  They should work at least in the
#pybind11#            # current implementation
#pybind11#            self.assertEqual(table.getPsfFluxDefinition(), "a")
#pybind11#            self.assertEqual(record.get(self.fluxKey), record.getPsfFlux())
#pybind11#            self.assertEqual(record.get(self.fluxFlagKey), record.getPsfFluxFlag())
#pybind11#            self.assertEqual(table.getCentroidDefinition(), "b")
#pybind11#            self.assertEqual(record.get(self.centroidKey), record.getCentroid())
#pybind11#            self.assertClose(record.get(self.centroidErrKey), record.getCentroidErr())
#pybind11#            self.assertEqual(table.getShapeDefinition(), "c")
#pybind11#            self.assertEqual(record.get(self.shapeKey), record.getShape())
#pybind11#            self.assertClose(record.get(self.shapeErrKey), record.getShapeErr())
#pybind11#
#pybind11#    def testDefiner1(self):
#pybind11#        self.table.definePsfFlux("a")
#pybind11#        self.table.defineCentroid("b")
#pybind11#        self.table.defineShape("c")
#pybind11#        self.assertEqual(self.table.getPsfFluxDefinition(), "a")
#pybind11#        self.assertEqual(self.record.get(self.fluxKey), self.record.getPsfFlux())
#pybind11#        self.assertEqual(self.record.get(self.fluxFlagKey), self.record.getPsfFluxFlag())
#pybind11#        self.assertEqual(self.table.getCentroidDefinition(), "b")
#pybind11#        self.assertEqual(self.record.get(self.centroidKey), self.record.getCentroid())
#pybind11#        self.assertClose(self.record.get(self.centroidErrKey), self.record.getCentroidErr())
#pybind11#        self.assertEqual(self.table.getShapeDefinition(), "c")
#pybind11#        self.assertEqual(self.record.get(self.shapeKey), self.record.getShape())
#pybind11#        self.assertClose(self.record.get(self.shapeErrKey), self.record.getShapeErr())
#pybind11#
#pybind11#    def testCoordUpdate(self):
#pybind11#        self.table.defineCentroid("b")
#pybind11#        wcs = makeWcs()
#pybind11#        self.record.updateCoord(wcs)
#pybind11#        coord1 = self.record.getCoord()
#pybind11#        coord2 = wcs.pixelToSky(self.record.get(self.centroidKey))
#pybind11#        self.assertEqual(coord1, coord2)
#pybind11#
#pybind11#    def testColumnView(self):
#pybind11#        cols1 = self.catalog.getColumnView()
#pybind11#        cols2 = self.catalog.columns
#pybind11#        self.assertIs(cols1, cols2)
#pybind11#        self.assertIsInstance(cols1, lsst.afw.table.SourceColumnView)
#pybind11#        self.table.definePsfFlux("a")
#pybind11#        self.table.defineCentroid("b")
#pybind11#        self.table.defineShape("c")
#pybind11#        self.assertTrue((cols2["a_flux"] == cols2.getPsfFlux()).all())
#pybind11#        self.assertTrue((cols2["a_fluxSigma"] == cols2.getPsfFluxErr()).all())
#pybind11#        self.assertTrue((cols2["b_x"] == cols2.getX()).all())
#pybind11#        self.assertTrue((cols2["b_y"] == cols2.getY()).all())
#pybind11#        self.assertTrue((cols2["c_xx"] == cols2.getIxx()).all())
#pybind11#        self.assertTrue((cols2["c_yy"] == cols2.getIyy()).all())
#pybind11#        self.assertTrue((cols2["c_xy"] == cols2.getIxy()).all())
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

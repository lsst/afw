#!/usr/bin/env python

# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
Tests for table.SourceTable

Run with:
   ./testSourceTable.py
or
   python
   >>> import testSourceTable; testSourceTable.run()
"""

import sys
import os
import unittest
import numpy

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.table
import lsst.afw.geom
import lsst.afw.coord
import lsst.afw.image

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

class SourceTableTestCase(unittest.TestCase):

    def setUp(self):
        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        self.fluxKey = schema.addField("a", type="F8")
        self.fluxErrKey = schema.addField("a.err", type="F8")
        self.fluxFlagKey = schema.addField("a.flags", type="Flag")
        self.centroidKey = schema.addField("b", type="Point<F8>")
        self.centroidErrKey = schema.addField("b.err", type="Cov<Point<F8>>")
        self.centroidFlagKey = schema.addField("b.flags", type="Flag")
        self.shapeKey = schema.addField("c", type="Moments<F8>")
        self.shapeErrKey = schema.addField("c.err", type="Cov<Moments<F8>>")
        self.shapeFlagKey = schema.addField("c.flags", type="Flag")
        self.table = lsst.afw.table.SourceTable.make(schema)
        self.record = self.table.makeRecord()
        self.record.set(self.fluxKey, numpy.random.randn())
        self.record.set(self.fluxErrKey, numpy.random.randn())
        self.record.set(self.centroidKey, lsst.afw.geom.Point2D(*numpy.random.randn(2)))
        self.record.set(self.centroidErrKey, makeCov(2, float))
        self.record.set(self.shapeKey, lsst.afw.geom.ellipses.Quadrupole(*numpy.random.randn(3)))
        self.record.set(self.shapeErrKey, makeCov(3, float))

    def tearDown(self):
        del self.record
        del self.table

    def checkCanonical(self):
        self.assertEqual(self.table.getPsfFluxDefinition(), "a")
        self.assertEqual(self.record.get(self.fluxKey), self.record.getPsfFlux())
        self.assertEqual(self.record.get(self.fluxFlagKey), self.record.getPsfFluxFlag())
        self.assertEqual(self.table.getCentroidDefinition(), "b")
        self.assertEqual(self.record.get(self.centroidKey), self.record.getCentroid())
        self.assert_(numpy.all(self.record.get(self.centroidErrKey) == self.record.getCentroidErr()))
        self.assertEqual(self.table.getShapeDefinition(), "c")
        self.assertEqual(self.record.get(self.shapeKey), self.record.getShape())
        self.assert_(numpy.all(self.record.get(self.shapeErrKey) == self.record.getShapeErr()))

    def testCanonical1(self):
        self.table.definePsfFlux(self.fluxKey, self.fluxErrKey, self.fluxFlagKey)
        self.table.defineCentroid(self.centroidKey, self.centroidErrKey, self.centroidFlagKey)
        self.table.defineShape(self.shapeKey, self.shapeErrKey, self.shapeFlagKey)
        self.checkCanonical()

    def testCanonical2(self):
        self.table.definePsfFlux("a")
        self.table.defineCentroid("b")
        self.table.defineShape("c")
        self.checkCanonical()

    def testCoordUpdate(self):
        wcs = makeWcs()
        self.record.updateCoord(wcs, self.centroidKey)
        coord1 = self.record.getCoord()
        coord2 = wcs.pixelToSky(self.record.get(self.centroidKey))
        self.assertEqual(coord1, coord2)

class SourceCatalogTestCase(unittest.TestCase):

    def setUp(self):
        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        self.catalog = lsst.afw.table.SourceCatalog(schema)
        self.catalog.addNew().setId(50)
        self.catalog.addNew()
        self.catalog.addNew()

    def tearDown(self):
        del self.catalog

    def testCustomization(self):
        self.assertFalse(self.catalog.isSorted())
        self.catalog.sort()
        self.assert_(self.catalog.isSorted())
        r = self.catalog.find(2)
        self.assertEqual(r["id"], 2)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(SourceTableTestCase)
    suites += unittest.makeSuite(SourceCatalogTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

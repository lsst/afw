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

class SourceTableTestCase(unittest.TestCase):

    def setUp(self):
        self.schema = lsst.afw.table.Schema(True)
        self.fluxKey = self.schema.addField("a", type="F8")
        self.fluxErrKey = self.schema.addField("a.err", type="F8")
        self.centroidKey = self.schema.addField("b", type="Point<F8>")
        self.centroidCovKey = self.schema.addField("b.cov", type="Cov<Point<F8>>")
        self.shapeKey = self.schema.addField("c", type="Moments<F8>")
        self.shapeCovKey = self.schema.addField("c.cov", type="Cov<Moments<F8>>")
        self.table = lsst.afw.table.SourceTable(self.schema)
        self.record = self.table.addRecord()
        self.record.set(self.fluxKey, numpy.random.randn())
        self.record.set(self.fluxErrKey, numpy.random.randn())
        self.record.set(self.centroidKey, lsst.afw.geom.Point2D(*numpy.random.randn(2)))
        self.record.set(self.centroidCovKey, makeCov(2, float))
        self.record.set(self.shapeKey, lsst.afw.geom.ellipses.Quadrupole(*numpy.random.randn(3)))
        self.record.set(self.shapeCovKey, makeCov(3, float))

    def checkCanonical(self):
        self.assertEqual(self.table.getPsfPhotometryDefinition(), "a")
        self.assertEqual(self.record.get(self.fluxKey), self.record.getPsfFlux())
        self.assertEqual(self.record.get(self.fluxErrKey), self.record.getPsfFluxErr())
        self.assertEqual(self.table.getAstrometryDefinition(), "b")
        self.assertEqual(self.record.get(self.centroidKey), self.record.getCentroid())
        self.assert_(numpy.all(self.record.get(self.centroidCovKey) == self.record.getCentroidCov()))
        self.assertEqual(self.table.getShapeDefinition(), "c")
        self.assertEqual(self.record.get(self.shapeKey), self.record.getShape())
        self.assert_(numpy.all(self.record.get(self.shapeCovKey) == self.record.getShapeCov()))

    def testCanonical1(self):
        self.table.definePsfPhotometry(self.fluxKey, self.fluxErrKey)
        self.table.defineAstrometry(self.centroidKey, self.centroidCovKey)
        self.table.defineShape(self.shapeKey, self.shapeCovKey)
        self.checkCanonical()

    def testCanonical2(self):
        self.table.definePsfPhotometry("a")
        self.table.defineAstrometry("b")
        self.table.defineShape("c")
        self.checkCanonical()

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

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

import os, os.path
import unittest

import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
import lsst.utils.tests as utilsTests


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
class MatchXyTest(unittest.TestCase):
    """Test that matching sources by centroid works as expected,
    even when some of the centroids contain NaN.
    """
    def setUp(self):
        nan = float('nan')
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        centroidKey = self.schema.addField("cen", type="PointD")
        self.table = afwTable.SourceTable.make(self.schema)
        self.table.setVersion(0)
        self.table.defineCentroid("cen")
        idKey = self.table.getIdKey()
        self.cat1 = afwTable.SourceCatalog(self.table)
        self.cat2 = afwTable.SourceCatalog(self.table)
        for i in xrange(10):
            j = 9 - i
            r1, r2 = self.cat1.addNew(), self.cat2.addNew()
            r1.set(idKey, i)
            r2.set(idKey, 10 + j)
            if i % 3 != 0:
                r1.set(centroidKey, afwGeom.Point2D(i,i))
                r2.set(centroidKey, afwGeom.Point2D(j,j))
            else:
                r1.set(centroidKey, afwGeom.Point2D(nan,nan))
                r2.set(centroidKey, afwGeom.Point2D(nan,nan))

    def tearDown(self):
        del self.cat2
        del self.cat1
        del self.table
        del self.schema

    def testMatchXy(self):
        matches = afwTable.matchXy(self.cat1, self.cat2, 0.01)
        self.assertEquals(len(matches), 6)
        for m in matches:
            self.assertEquals(m.first.getId() + 10, m.second.getId())
            self.assertEquals(m.distance, 0.0)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(MatchXyTest)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

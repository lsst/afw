#!/usr/bin/env python2
from __future__ import absolute_import, division

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
        centroidKey = afwTable.Point2DKey.addFields(self.schema, "cen", "center", "pixels")
        self.table = afwTable.SourceTable.make(self.schema)
        self.table.defineCentroid("cen")
        idKey = self.table.getIdKey()
        self.cat1 = afwTable.SourceCatalog(self.table)
        self.cat2 = afwTable.SourceCatalog(self.table)
        self.nobj = 10
        self.nUniqueMatch = 0
        for i in range(self.nobj):
            j = self.nobj - i - 1
            r1, r2 = self.cat1.addNew(), self.cat2.addNew()
            r1.set(idKey, i)
            r2.set(idKey, self.nobj + j)
            if i % 3 != 0:
                r1.set(centroidKey, afwGeom.Point2D(i,i))
                r2.set(centroidKey, afwGeom.Point2D(j,j))
                self.nUniqueMatch += 1
            elif i == 3:
                r1.set(centroidKey, afwGeom.Point2D(i,i))
                r2.set(centroidKey, afwGeom.Point2D(j + 2,j + 2))
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
        self.assertEquals(len(matches), self.nUniqueMatch)

        for m in matches:
            self.assertEquals(m.first.getId() + self.nobj, m.second.getId())
            self.assertEquals(m.distance, 0.0)

    def testMatchXyMatchControl(self):
        """Test using MatchControl to return all matches

        Also tests closest==False at the same time
        """
        for closest in (True, False):
            for includeMismatches in (True, False):
                mc = afwTable.MatchControl()
                mc.findOnlyClosest = closest
                mc.includeMismatches = includeMismatches
                matches = afwTable.matchXy(self.cat1, self.cat2, 0.01, mc)

                if False:
                    for m in matches:
                        print closest, m.first.getId(), m.second.getId(), m.distance

                if includeMismatches:
                    catMatches = afwTable.SourceCatalog(self.table)
                    catMismatches = afwTable.SourceCatalog(self.table)
                    for m in matches:
                        if m[1] is not None:
                            if not any(x == m[0] for x in catMatches):
                                catMatches.append(m[0])
                        else:
                            catMismatches.append(m[0])
                    matches = afwTable.matchXy(catMatches, self.cat2, 0.01, mc)
                    mc.includeMismatches = False
                    noMatches = afwTable.matchXy(catMismatches, self.cat2, 0.01, mc)
                    self.assertEquals(len(noMatches), 0)

                self.assertEquals(len(matches), self.nUniqueMatch if closest else self.nUniqueMatch + 1)
                for m in matches:
                    if closest:
                        self.assertEquals(m.first.getId() + self.nobj, m.second.getId())
                    self.assertEquals(m.distance, 0.0)

    def testSelfMatchXy(self):
        """Test doing a self-matches"""
        for symmetric in (True, False):
            mc = afwTable.MatchControl()
            mc.symmetricMatch = symmetric
            matches = afwTable.matchXy(self.cat2, 0.01, mc)

            if False:
                for m in matches:
                    print m.first.getId(), m.second.getId(), m.distance

            self.assertEquals(len(matches), 2 if symmetric else 1)

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

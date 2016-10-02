#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
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
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.table as afwTable
#pybind11#import lsst.utils.tests
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#class MatchXyTest(unittest.TestCase):
#pybind11#    """Test that matching sources by centroid works as expected,
#pybind11#    even when some of the centroids contain NaN.
#pybind11#    """
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        nan = float('nan')
#pybind11#        self.schema = afwTable.SourceTable.makeMinimalSchema()
#pybind11#        centroidKey = afwTable.Point2DKey.addFields(self.schema, "cen", "center", "pixels")
#pybind11#        self.table = afwTable.SourceTable.make(self.schema)
#pybind11#        self.table.defineCentroid("cen")
#pybind11#        idKey = self.table.getIdKey()
#pybind11#        self.cat1 = afwTable.SourceCatalog(self.table)
#pybind11#        self.cat2 = afwTable.SourceCatalog(self.table)
#pybind11#        self.nobj = 10
#pybind11#        self.nUniqueMatch = 0
#pybind11#        self.matchRadius = 0.1  # Matching radius to use in tests (pixels)
#pybind11#        for i in range(self.nobj):
#pybind11#            j = self.nobj - i - 1
#pybind11#            r1, r2 = self.cat1.addNew(), self.cat2.addNew()
#pybind11#            r1.set(idKey, i)
#pybind11#            r2.set(idKey, self.nobj + j)
#pybind11#            if i % 3 != 0:
#pybind11#                # These will provide the exact matches, though the two records we're setting right now won't
#pybind11#                # match each other (because cat2 counts down in reverse).
#pybind11#                r1.set(centroidKey, afwGeom.Point2D(i, i))
#pybind11#                r2.set(centroidKey, afwGeom.Point2D(j, j))
#pybind11#                self.nUniqueMatch += 1
#pybind11#            elif i == 3:
#pybind11#                # Deliberately offset position in cat2 by 2 pixels and a bit so that it will match a
#pybind11#                # different source. This gives us an extra match when we're not just getting the closest.
#pybind11#                # The "2 pixels" makes it line up with a different source.
#pybind11#                # The "a bit" is half a match radius, so that it's still within the matching radius, but it
#pybind11#                # doesn't match another source exactly. If it matches another source exactly, then it's not
#pybind11#                # clear which one will be taken as the match (in fact, it appears to depend on the compiler).
#pybind11#                offset = 2 + 0.5*self.matchRadius
#pybind11#                r1.set(centroidKey, afwGeom.Point2D(i, i))
#pybind11#                r2.set(centroidKey, afwGeom.Point2D(j + offset, j + offset))
#pybind11#            else:
#pybind11#                # Test that we can handle NANs
#pybind11#                r1.set(centroidKey, afwGeom.Point2D(nan, nan))
#pybind11#                r2.set(centroidKey, afwGeom.Point2D(nan, nan))
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.cat2
#pybind11#        del self.cat1
#pybind11#        del self.table
#pybind11#        del self.schema
#pybind11#
#pybind11#    def testMatchXy(self):
#pybind11#        matches = afwTable.matchXy(self.cat1, self.cat2, self.matchRadius)
#pybind11#        self.assertEqual(len(matches), self.nUniqueMatch)
#pybind11#
#pybind11#        for m in matches:
#pybind11#            self.assertEqual(m.first.getId() + self.nobj, m.second.getId())
#pybind11#            self.assertEqual(m.distance, 0.0)
#pybind11#
#pybind11#    def testMatchXyMatchControl(self):
#pybind11#        """Test using MatchControl to return all matches
#pybind11#
#pybind11#        Also tests closest==False at the same time
#pybind11#        """
#pybind11#        for closest in (True, False):
#pybind11#            for includeMismatches in (True, False):
#pybind11#                mc = afwTable.MatchControl()
#pybind11#                mc.findOnlyClosest = closest
#pybind11#                mc.includeMismatches = includeMismatches
#pybind11#                matches = afwTable.matchXy(self.cat1, self.cat2, self.matchRadius, mc)
#pybind11#
#pybind11#                if False:
#pybind11#                    for m in matches:
#pybind11#                        print(closest, m.first.getId(), m.second.getId(), m.distance)
#pybind11#
#pybind11#                if includeMismatches:
#pybind11#                    catMatches = afwTable.SourceCatalog(self.table)
#pybind11#                    catMismatches = afwTable.SourceCatalog(self.table)
#pybind11#                    for m in matches:
#pybind11#                        if m[1] is not None:
#pybind11#                            if not any(x == m[0] for x in catMatches):
#pybind11#                                catMatches.append(m[0])
#pybind11#                        else:
#pybind11#                            catMismatches.append(m[0])
#pybind11#                    matches = afwTable.matchXy(catMatches, self.cat2, self.matchRadius, mc)
#pybind11#                    mc.includeMismatches = False
#pybind11#                    noMatches = afwTable.matchXy(catMismatches, self.cat2, self.matchRadius, mc)
#pybind11#                    self.assertEqual(len(noMatches), 0)
#pybind11#
#pybind11#                # If we're not getting only the closest match, then we get an extra match due to the
#pybind11#                # source we offset by 2 pixels and a bit.  Everything else should match exactly.
#pybind11#                self.assertEqual(len(matches), self.nUniqueMatch if closest else self.nUniqueMatch + 1)
#pybind11#                self.assertEqual(sum(1 for m in matches if m.distance == 0.0), self.nUniqueMatch)
#pybind11#                for m in matches:
#pybind11#                    if closest:
#pybind11#                        self.assertEqual(m.first.getId() + self.nobj, m.second.getId())
#pybind11#                    else:
#pybind11#                        self.assertLessEqual(m.distance, self.matchRadius)
#pybind11#
#pybind11#    def testSelfMatchXy(self):
#pybind11#        """Test doing a self-matches"""
#pybind11#        for symmetric in (True, False):
#pybind11#            mc = afwTable.MatchControl()
#pybind11#            mc.symmetricMatch = symmetric
#pybind11#            matches = afwTable.matchXy(self.cat2, self.matchRadius, mc)
#pybind11#
#pybind11#            if False:
#pybind11#                for m in matches:
#pybind11#                    print(m.first.getId(), m.second.getId(), m.distance)
#pybind11#
#pybind11#            # There is only one source that matches another source when we do a self-match: the one
#pybind11#            # offset by 2 pixels and a bit.
#pybind11#            # If we're getting symmetric matches, that multiplies the expected number by 2 because it
#pybind11#            # produces s1,s2 and s2,s1.
#pybind11#            self.assertEqual(len(matches), 2 if symmetric else 1)
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

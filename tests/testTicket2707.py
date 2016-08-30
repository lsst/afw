#!/usr/bin/env python
from __future__ import absolute_import, division
from __future__ import print_function
from builtins import range

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
import lsst.utils.tests


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
        self.matchRadius = 0.1  # Matching radius to use in tests (pixels)
        for i in range(self.nobj):
            j = self.nobj - i - 1
            r1, r2 = self.cat1.addNew(), self.cat2.addNew()
            r1.set(idKey, i)
            r2.set(idKey, self.nobj + j)
            if i % 3 != 0:
                # These will provide the exact matches, though the two records we're setting right now won't
                # match each other (because cat2 counts down in reverse).
                r1.set(centroidKey, afwGeom.Point2D(i, i))
                r2.set(centroidKey, afwGeom.Point2D(j, j))
                self.nUniqueMatch += 1
            elif i == 3:
                # Deliberately offset position in cat2 by 2 pixels and a bit so that it will match a
                # different source. This gives us an extra match when we're not just getting the closest.
                # The "2 pixels" makes it line up with a different source.
                # The "a bit" is half a match radius, so that it's still within the matching radius, but it
                # doesn't match another source exactly. If it matches another source exactly, then it's not
                # clear which one will be taken as the match (in fact, it appears to depend on the compiler).
                offset = 2 + 0.5*self.matchRadius
                r1.set(centroidKey, afwGeom.Point2D(i, i))
                r2.set(centroidKey, afwGeom.Point2D(j + offset, j + offset))
            else:
                # Test that we can handle NANs
                r1.set(centroidKey, afwGeom.Point2D(nan, nan))
                r2.set(centroidKey, afwGeom.Point2D(nan, nan))

    def tearDown(self):
        del self.cat2
        del self.cat1
        del self.table
        del self.schema

    def testMatchXy(self):
        matches = afwTable.matchXy(self.cat1, self.cat2, self.matchRadius)
        self.assertEqual(len(matches), self.nUniqueMatch)

        for m in matches:
            self.assertEqual(m.first.getId() + self.nobj, m.second.getId())
            self.assertEqual(m.distance, 0.0)

    def testMatchXyMatchControl(self):
        """Test using MatchControl to return all matches

        Also tests closest==False at the same time
        """
        for closest in (True, False):
            for includeMismatches in (True, False):
                mc = afwTable.MatchControl()
                mc.findOnlyClosest = closest
                mc.includeMismatches = includeMismatches
                matches = afwTable.matchXy(self.cat1, self.cat2, self.matchRadius, mc)

                if False:
                    for m in matches:
                        print(closest, m.first.getId(), m.second.getId(), m.distance)

                if includeMismatches:
                    catMatches = afwTable.SourceCatalog(self.table)
                    catMismatches = afwTable.SourceCatalog(self.table)
                    for m in matches:
                        if m[1] is not None:
                            if not any(x == m[0] for x in catMatches):
                                catMatches.append(m[0])
                        else:
                            catMismatches.append(m[0])
                    matches = afwTable.matchXy(catMatches, self.cat2, self.matchRadius, mc)
                    mc.includeMismatches = False
                    noMatches = afwTable.matchXy(catMismatches, self.cat2, self.matchRadius, mc)
                    self.assertEqual(len(noMatches), 0)

                # If we're not getting only the closest match, then we get an extra match due to the
                # source we offset by 2 pixels and a bit.  Everything else should match exactly.
                self.assertEqual(len(matches), self.nUniqueMatch if closest else self.nUniqueMatch + 1)
                self.assertEqual(sum(1 for m in matches if m.distance == 0.0), self.nUniqueMatch)
                for m in matches:
                    if closest:
                        self.assertEqual(m.first.getId() + self.nobj, m.second.getId())
                    else:
                        self.assertLessEqual(m.distance, self.matchRadius)

    def testSelfMatchXy(self):
        """Test doing a self-matches"""
        for symmetric in (True, False):
            mc = afwTable.MatchControl()
            mc.symmetricMatch = symmetric
            matches = afwTable.matchXy(self.cat2, self.matchRadius, mc)

            if False:
                for m in matches:
                    print(m.first.getId(), m.second.getId(), m.distance)

            # There is only one source that matches another source when we do a self-match: the one
            # offset by 2 pixels and a bit.
            # If we're getting symmetric matches, that multiplies the expected number by 2 because it
            # produces s1,s2 and s2,s1.
            self.assertEqual(len(matches), 2 if symmetric else 1)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

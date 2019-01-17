# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Tests for matching SourceSets

Run with:
   python test_sourceMatch.py
or
   pytest test_sourceMatch.py
"""
import os
import re
import unittest

import numpy as np

import lsst.geom
import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import lsst.utils.tests
import lsst.pex.exceptions as pexExcept

try:
    afwdataDir = lsst.utils.getPackageDir("afwdata")
except pexExcept.NotFoundError:
    afwdataDir = None


class SourceMatchTestCase(lsst.utils.tests.TestCase):
    """A test case for matching SourceSets
    """

    def setUp(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField("flux_instFlux", type=np.float64)
        schema.addField("flux_instFluxErr", type=np.float64)
        schema.addField("flux_flag", type="Flag")
        self.table = afwTable.SourceTable.make(schema)
        self.table.definePsfFlux("flux")
        self.ss1 = afwTable.SourceCatalog(self.table)
        self.ss2 = afwTable.SourceCatalog(self.table)
        self.metadata = dafBase.PropertyList()

    def tearDown(self):
        del self.table
        del self.metadata
        del self.ss1
        del self.ss2

    def testIdentity(self):
        nobj = 1000
        for i in range(nobj):
            s = self.ss1.addNew()
            s.setId(i)
            s.set(afwTable.SourceTable.getCoordKey().getRa(),
                  (10 + 0.001*i) * lsst.geom.degrees)
            s.set(afwTable.SourceTable.getCoordKey().getDec(),
                  (10 + 0.001*i) * lsst.geom.degrees)

            s = self.ss2.addNew()
            s.setId(2*nobj + i)
            # Give slight offsets for Coord testing of matches to/from catalog in checkMatchToFromCatalog()
            # Chosen such that the maximum offset (nobj*1E-7 deg = 0.36 arcsec) is within the maximum
            # distance (1 arcsec) in afwTable.matchRaDec.
            s.set(afwTable.SourceTable.getCoordKey().getRa(),
                  (10 + 0.0010001*i) * lsst.geom.degrees)
            s.set(afwTable.SourceTable.getCoordKey().getDec(),
                  (10 + 0.0010001*i) * lsst.geom.degrees)

        mc = afwTable.MatchControl()
        mc.findOnlyClosest = False
        mat = afwTable.matchRaDec(
            self.ss1, self.ss2, 1.0*lsst.geom.arcseconds, mc)
        self.assertEqual(len(mat), nobj)

        cat = afwTable.packMatches(mat)

        mat2 = afwTable.unpackMatches(cat, self.ss1, self.ss2)

        for m1, m2, c in zip(mat, mat2, cat):
            self.assertEqual(m1.first, m2.first)
            self.assertEqual(m1.second, m2.second)
            self.assertEqual(m1.distance, m2.distance)
            self.assertEqual(m1.first.getId(), c["first"])
            self.assertEqual(m1.second.getId(), c["second"])
            self.assertEqual(m1.distance, c["distance"])

        self.checkMatchToFromCatalog(mat, cat)

        if False:
            s0 = mat[0][0]
            s1 = mat[0][1]
            print(s0.getRa(), s1.getRa(), s0.getId(), s1.getId())

    def testNaNPositions(self):
        ss1 = afwTable.SourceCatalog(self.table)
        ss2 = afwTable.SourceCatalog(self.table)
        for ss in (ss1, ss2):
            ss.addNew().set(afwTable.SourceTable.getCoordKey().getRa(),
                            float('nan') * lsst.geom.radians)

            ss.addNew().set(afwTable.SourceTable.getCoordKey().getDec(),
                            float('nan') * lsst.geom.radians)

            s = ss.addNew()
            s.set(afwTable.SourceTable.getCoordKey().getRa(),
                  0.0 * lsst.geom.radians)
            s.set(afwTable.SourceTable.getCoordKey().getDec(),
                  0.0 * lsst.geom.radians)

            s = ss.addNew()
            s.set(afwTable.SourceTable.getCoordKey().getRa(),
                  float('nan') * lsst.geom.radians)
            s.set(afwTable.SourceTable.getCoordKey().getDec(),
                  float('nan') * lsst.geom.radians)

        mc = afwTable.MatchControl()
        mc.findOnlyClosest = False
        mat = afwTable.matchRaDec(ss1, ss2, 1.0*lsst.geom.arcseconds, mc)
        self.assertEqual(len(mat), 1)

    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
    def testPhotometricCalib(self):
        """Test matching the CFHT catalogue (as generated using LSST code) to the SDSS catalogue
        """

        band = 2                        # SDSS r

        # Read SDSS catalogue
        with open(os.path.join(afwdataDir, "CFHT", "D2", "sdss.dat"), "r") as ifd:

            sdss = afwTable.SourceCatalog(self.table)
            sdssSecondary = afwTable.SourceCatalog(self.table)

            PRIMARY, SECONDARY = 1, 2       # values of mode

            id = 0
            for line in ifd.readlines():
                if re.search(r"^\s*#", line):
                    continue

                fields = line.split()
                objId = int(fields[0])
                fields[1]
                mode = int(fields[2])
                ra, dec = [float(f) for f in fields[3:5]]
                psfMags = [float(f) for f in fields[5:]]

                if mode == PRIMARY:
                    s = sdss.addNew()
                elif SECONDARY:
                    s = sdssSecondary.addNew()

                s.setId(objId)
                s.setRa(ra * lsst.geom.degrees)
                s.setDec(dec * lsst.geom.degrees)
                s.set(self.table.getPsfFluxSlot().getMeasKey(), psfMags[band])

        del ifd

        # Read catalalogue built from the template image
        # Read SDSS catalogue
        with open(os.path.join(afwdataDir, "CFHT", "D2", "template.dat"), "r") as ifd:

            template = afwTable.SourceCatalog(self.table)

            id = 0
            for line in ifd.readlines():
                if re.search(r"^\s*#", line):
                    continue

                fields = line.split()
                id, flags = [int(f) for f in fields[0:2]]
                ra, dec = [float(f) for f in fields[2:4]]
                flux = [float(f) for f in fields[4:]]

                if flags & 0x1:             # EDGE
                    continue

                s = template.addNew()
                s.setId(id)
                id += 1
                s.set(afwTable.SourceTable.getCoordKey().getRa(),
                      ra * lsst.geom.degrees)
                s.set(afwTable.SourceTable.getCoordKey().getDec(),
                      dec * lsst.geom.degrees)
                s.set(self.table.getPsfFluxSlot().getMeasKey(), flux[0])

        del ifd

        # Actually do the match
        mc = afwTable.MatchControl()
        mc.findOnlyClosest = False

        matches = afwTable.matchRaDec(
            sdss, template, 1.0*lsst.geom.arcseconds, mc)

        self.assertEqual(len(matches), 901)

        if False:
            for mat in matches:
                s0 = mat[0]
                s1 = mat[1]
                d = mat[2]
                print(s0.getRa(), s0.getDec(), s1.getRa(),
                      s1.getDec(), s0.getPsfInstFlux(), s1.getPsfInstFlux())

        # Actually do the match
        for s in sdssSecondary:
            sdss.append(s)

        mc = afwTable.MatchControl()
        mc.symmetricMatch = False
        matches = afwTable.matchRaDec(sdss, 1.0*lsst.geom.arcseconds, mc)
        nmiss = 1                                              # one object doesn't match
        self.assertEqual(len(matches), len(sdssSecondary) - nmiss)

        # Find the one that didn't match
        if False:
            matchIds = set()
            for s0, s1, d in matches:
                matchIds.add(s0.getId())
                matchIds.add(s1.getId())

            for s in sdssSecondary:
                if s.getId() not in matchIds:
                    print("RHL", s.getId())

        matches = afwTable.matchRaDec(sdss, 1.0*lsst.geom.arcseconds)
        self.assertEqual(len(matches), 2*(len(sdssSecondary) - nmiss))

        if False:
            for mat in matches:
                s0 = mat[0]
                s1 = mat[1]
                mat[2]
                print(s0.getId(), s1.getId(), s0.getRa(), s0.getDec(), end=' ')
                print(s1.getRa(), s1.getDec(), s0.getPsfInstFlux(), s1.getPsfInstFlux())

    def testMismatches(self):
        """ Chech that matchRaDec works as expected when using
            the includeMismatches option
        """
        cat1 = afwTable.SourceCatalog(self.table)
        cat2 = afwTable.SourceCatalog(self.table)
        nobj = 100
        for i in range(nobj):
            s1 = cat1.addNew()
            s2 = cat2.addNew()
            s1.setId(i)
            s2.setId(i)
            s1.set(afwTable.SourceTable.getCoordKey().getRa(),
                   (10 + 0.0001*i) * lsst.geom.degrees)
            s2.set(afwTable.SourceTable.getCoordKey().getRa(),
                   (10.005 + 0.0001*i) * lsst.geom.degrees)
            s1.set(afwTable.SourceTable.getCoordKey().getDec(),
                   (10 + 0.0001*i) * lsst.geom.degrees)
            s2.set(afwTable.SourceTable.getCoordKey().getDec(),
                   (10.005 + 0.0001*i) * lsst.geom.degrees)

        for closest in (True, False):
            mc = afwTable.MatchControl()
            mc.findOnlyClosest = closest
            mc.includeMismatches = False
            matches = afwTable.matchRaDec(
                cat1, cat2, 1.0*lsst.geom.arcseconds, mc)
            mc.includeMismatches = True
            matchesMismatches = afwTable.matchRaDec(
                cat1, cat2, 1.0*lsst.geom.arcseconds, mc)

            catMatches = afwTable.SourceCatalog(self.table)
            catMismatches = afwTable.SourceCatalog(self.table)
            for m in matchesMismatches:
                if m[1] is not None:
                    if not any(x == m[0] for x in catMatches):
                        catMatches.append(m[0])
                else:
                    catMismatches.append(m[0])
            if closest:
                self.assertEqual(len(catMatches), len(matches))
            matches2 = afwTable.matchRaDec(
                catMatches, cat2, 1.0*lsst.geom.arcseconds, mc)
            self.assertEqual(len(matches), len(matches2))
            mc.includeMismatches = False
            noMatches = afwTable.matchRaDec(
                catMismatches, cat2, 1.0*lsst.geom.arcseconds, mc)
            self.assertEqual(len(noMatches), 0)

    def checkMatchToFromCatalog(self, matches, catalog):
        """Check the conversion of matches to and from a catalog

        Test the functions in lsst.afw.table.catalogMatches.py
        Note that the return types and schemas of these functions do not necessarily match
        those of the catalogs passed to them, so value entries are compared as opposed to
        comparing the attributes as a whole.
        """
        catalog.setMetadata(self.metadata)
        matchMeta = catalog.getTable().getMetadata()
        matchToCat = afwTable.catalogMatches.matchesToCatalog(
            matches, matchMeta)
        matchFromCat = afwTable.catalogMatches.matchesFromCatalog(matchToCat)
        self.assertEqual(len(matches), len(matchToCat))
        self.assertEqual(len(matches), len(matchFromCat))

        for mat, cat, catM, matchC in zip(matches, catalog, matchToCat, matchFromCat):
            self.assertEqual(mat.first.getId(), catM["ref_id"])
            self.assertEqual(mat.first.getId(), matchC.first.getId())
            self.assertEqual(mat.first.getCoord(), matchC.first.getCoord())
            self.assertEqual(mat.second.getId(), cat["second"])
            self.assertEqual(mat.second.getId(), catM["src_id"])
            self.assertEqual(mat.second.getId(), matchC.second.getId())
            self.assertEqual((mat.first.getRa(), mat.first.getDec()),
                             (catM["ref_coord_ra"], catM["ref_coord_dec"]))
            self.assertEqual((mat.second.getRa(), mat.second.getDec()),
                             (catM["src_coord_ra"], catM["src_coord_dec"]))
            self.assertEqual(mat.first.getCoord(), matchC.first.getCoord())
            self.assertEqual(mat.second.getCoord(), matchC.second.getCoord())
            self.assertEqual(mat.distance, matchC.distance)
            self.assertEqual(mat.distance, cat["distance"])
            self.assertEqual(mat.distance, catM["distance"])

    def assertEqualFloat(self, value1, value2):
        """Compare floating point values, allowing for NAN
        """
        self.assertTrue(value1 == value2 or
                        (np.isnan(value1) and np.isnan(value2)))

    def testDistancePrecision(self):
        """Test for precision of the calculated distance

        Check that the distance produced by matchRaDec is the same
        as the distance produced from calculating the separation
        between the matched coordinates.

        Based on DM-13891.
        """
        num = 1000  # Number of points
        radius = 0.5*lsst.geom.arcseconds  # Matching radius
        tol = 1.0e-10  # Absolute tolerance
        rng = np.random.RandomState(12345)  # I have the same combination on my luggage
        coordKey = afwTable.SourceTable.getCoordKey()
        raKey = coordKey.getRa()
        decKey = coordKey.getDec()
        for ii in range(num):
            src1 = self.ss1.addNew()
            src1.setId(ii)
            src1.set(raKey, (10 + 0.001*ii) * lsst.geom.degrees)
            src1.set(decKey, (10 + 0.001*ii) * lsst.geom.degrees)

            src2 = self.ss2.addNew()
            src2.setId(2*num + ii)
            src2.set(coordKey,
                     src1.getCoord().offset(rng.uniform(high=360)*lsst.geom.degrees,
                                            rng.uniform(high=radius.asArcseconds())*lsst.geom.arcseconds))

        matches = afwTable.matchRaDec(self.ss1, self.ss2, radius)
        dist1 = np.array([(mm.distance*lsst.geom.radians).asArcseconds() for mm in matches])
        dist2 = np.array([mm.first.getCoord().separation(mm.second.getCoord()).asArcseconds()
                         for mm in matches])
        diff = dist1 - dist2
        self.assertLess(diff.std(), tol)  # I get 4e-12
        self.assertFloatsAlmostEqual(dist1, dist2, atol=tol)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

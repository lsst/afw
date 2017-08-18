
#
# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#

"""
Tests for matching SourceSets

Run with:
   python SourceMatch.py
or
   python
   >>> import SourceMatch; SourceMatch.run()
"""
from __future__ import absolute_import, division, print_function
import os
import re
import unittest

from builtins import zip
from builtins import range
import numpy as np

import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import lsst.utils.tests
import lsst.pex.exceptions as pexExcept

try:
    afwdataDir = lsst.utils.getPackageDir("afwdata")
except pexExcept.NotFoundError:
    afwdataDir = None


class SourceMatchTestCase(unittest.TestCase):
    """A test case for matching SourceSets
    """

    def setUp(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField("flux_flux", type=np.float64)
        schema.addField("flux_fluxSigma", type=np.float64)
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
                  (10 + 0.001*i) * afwGeom.degrees)
            s.set(afwTable.SourceTable.getCoordKey().getDec(),
                  (10 + 0.001*i) * afwGeom.degrees)

            s = self.ss2.addNew()
            s.setId(2*nobj + i)
            # Give slight offsets for Coord testing of matches to/from catalog in checkMatchToFromCatalog()
            # Chosen such that the maximum offset (nobj*1E-7 deg = 0.36 arcsec) is within the maximum
            # distance (1 arcsec) in afwTable.matchRaDec.
            s.set(afwTable.SourceTable.getCoordKey().getRa(),
                  (10 + 0.0010001*i) * afwGeom.degrees)
            s.set(afwTable.SourceTable.getCoordKey().getDec(),
                  (10 + 0.0010001*i) * afwGeom.degrees)

        mc = afwTable.MatchControl()
        mc.findOnlyClosest = False
        mat = afwTable.matchRaDec(
            self.ss1, self.ss2, 1.0*afwGeom.arcseconds, mc)
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
                            float('nan') * afwGeom.radians)

            ss.addNew().set(afwTable.SourceTable.getCoordKey().getDec(),
                            float('nan') * afwGeom.radians)

            s = ss.addNew()
            s.set(afwTable.SourceTable.getCoordKey().getRa(),
                  0.0 * afwGeom.radians)
            s.set(afwTable.SourceTable.getCoordKey().getDec(),
                  0.0 * afwGeom.radians)

            s = ss.addNew()
            s.set(afwTable.SourceTable.getCoordKey().getRa(),
                  float('nan') * afwGeom.radians)
            s.set(afwTable.SourceTable.getCoordKey().getDec(),
                  float('nan') * afwGeom.radians)

        mc = afwTable.MatchControl()
        mc.findOnlyClosest = False
        mat = afwTable.matchRaDec(ss1, ss2, 1.0*afwGeom.arcseconds, mc)
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
                s.setRa(ra * afwGeom.degrees)
                s.setDec(dec * afwGeom.degrees)
                s.set(self.table.getPsfFluxKey(), psfMags[band])

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
                      ra * afwGeom.degrees)
                s.set(afwTable.SourceTable.getCoordKey().getDec(),
                      dec * afwGeom.degrees)
                s.set(self.table.getPsfFluxKey(), flux[0])

        del ifd

        # Actually do the match
        mc = afwTable.MatchControl()
        mc.findOnlyClosest = False

        matches = afwTable.matchRaDec(
            sdss, template, 1.0*afwGeom.arcseconds, mc)

        self.assertEqual(len(matches), 901)

        if False:
            for mat in matches:
                s0 = mat[0]
                s1 = mat[1]
                d = mat[2]
                print(s0.getRa(), s0.getDec(), s1.getRa(),
                      s1.getDec(), s0.getPsfFlux(), s1.getPsfFlux())

        # Actually do the match
        for s in sdssSecondary:
            sdss.append(s)

        mc = afwTable.MatchControl()
        mc.symmetricMatch = False
        matches = afwTable.matchRaDec(sdss, 1.0*afwGeom.arcseconds, mc)
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

        matches = afwTable.matchRaDec(sdss, 1.0*afwGeom.arcseconds)
        self.assertEqual(len(matches), 2*(len(sdssSecondary) - nmiss))

        if False:
            for mat in matches:
                s0 = mat[0]
                s1 = mat[1]
                mat[2]
                print(s0.getId(), s1.getId(), s0.getRa(), s0.getDec(), end=' ')
                print(s1.getRa(), s1.getDec(), s0.getPsfFlux(), s1.getPsfFlux())

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
                   (10 + 0.0001*i) * afwGeom.degrees)
            s2.set(afwTable.SourceTable.getCoordKey().getRa(),
                   (10.005 + 0.0001*i) * afwGeom.degrees)
            s1.set(afwTable.SourceTable.getCoordKey().getDec(),
                   (10 + 0.0001*i) * afwGeom.degrees)
            s2.set(afwTable.SourceTable.getCoordKey().getDec(),
                   (10.005 + 0.0001*i) * afwGeom.degrees)

        for closest in (True, False):
            mc = afwTable.MatchControl()
            mc.findOnlyClosest = closest
            mc.includeMismatches = False
            matches = afwTable.matchRaDec(
                cat1, cat2, 1.0*afwGeom.arcseconds, mc)
            mc.includeMismatches = True
            matchesMismatches = afwTable.matchRaDec(
                cat1, cat2, 1.0*afwGeom.arcseconds, mc)

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
                catMatches, cat2, 1.0*afwGeom.arcseconds, mc)
            self.assertEqual(len(matches), len(matches2))
            mc.includeMismatches = False
            noMatches = afwTable.matchRaDec(
                catMismatches, cat2, 1.0*afwGeom.arcseconds, mc)
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


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

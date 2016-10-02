#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import zip
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008-2016 AURA/LSST.
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
#pybind11## see <https://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11#"""
#pybind11#Tests for matching SourceSets
#pybind11#
#pybind11#Run with:
#pybind11#   python SourceMatch.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import SourceMatch; SourceMatch.run()
#pybind11#"""
#pybind11#import os
#pybind11#import re
#pybind11#
#pybind11#import numpy
#pybind11#import pickle
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.table as afwTable
#pybind11#import lsst.daf.base as dafBase
#pybind11#import lsst.utils
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#
#pybind11#try:
#pybind11#    afwdataDir = lsst.utils.getPackageDir("afwdata")
#pybind11#except pexExcept.NotFoundError:
#pybind11#    afwdataDir = None
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class SourceMatchTestCase(unittest.TestCase):
#pybind11#    """A test case for matching SourceSets
#pybind11#    """
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        schema = afwTable.SourceTable.makeMinimalSchema()
#pybind11#        schema.addField("flux_flux", type=float)
#pybind11#        schema.addField("flux_fluxSigma", type=float)
#pybind11#        schema.addField("flux_flag", type="Flag")
#pybind11#        self.table = afwTable.SourceTable.make(schema)
#pybind11#        self.table.definePsfFlux("flux")
#pybind11#        self.ss1 = afwTable.SourceCatalog(self.table)
#pybind11#        self.ss2 = afwTable.SourceCatalog(self.table)
#pybind11#        self.metadata = dafBase.PropertyList()
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.table
#pybind11#        del self.metadata
#pybind11#        del self.ss1
#pybind11#        del self.ss2
#pybind11#
#pybind11#    def testIdentity(self):
#pybind11#        nobj = 1000
#pybind11#        for i in range(nobj):
#pybind11#            s = self.ss1.addNew()
#pybind11#            s.setId(i)
#pybind11#            s.set(afwTable.SourceTable.getCoordKey().getRa(), (10 + 0.001*i) * afwGeom.degrees)
#pybind11#            s.set(afwTable.SourceTable.getCoordKey().getDec(), (10 + 0.001*i) * afwGeom.degrees)
#pybind11#
#pybind11#            s = self.ss2.addNew()
#pybind11#            s.setId(2*nobj + i)
#pybind11#            # Give slight offsets for Coord testing of matches to/from catalog in checkMatchToFromCatalog()
#pybind11#            # Chosen such that the maximum offset (nobj*1E-7 deg = 0.36 arcsec) is within the maximum
#pybind11#            # distance (1 arcsec) in afwTable.matchRaDec.
#pybind11#            s.set(afwTable.SourceTable.getCoordKey().getRa(), (10 + 0.0010001*i) * afwGeom.degrees)
#pybind11#            s.set(afwTable.SourceTable.getCoordKey().getDec(), (10 + 0.0010001*i) * afwGeom.degrees)
#pybind11#
#pybind11#        # Old API (pre DM-855)
#pybind11#        mat = afwTable.matchRaDec(self.ss1, self.ss2, 1.0 * afwGeom.arcseconds, False)
#pybind11#        self.assertEqual(len(mat), nobj)
#pybind11#        # New API
#pybind11#        mc = afwTable.MatchControl()
#pybind11#        mc.findOnlyClosest = False
#pybind11#        mat = afwTable.matchRaDec(self.ss1, self.ss2, 1.0*afwGeom.arcseconds, mc)
#pybind11#        self.assertEqual(len(mat), nobj)
#pybind11#
#pybind11#        cat = afwTable.packMatches(mat)
#pybind11#
#pybind11#        mat2 = afwTable.unpackMatches(cat, self.ss1, self.ss2)
#pybind11#
#pybind11#        for m1, m2, c in zip(mat, mat2, cat):
#pybind11#            self.assertEqual(m1.first, m2.first)
#pybind11#            self.assertEqual(m1.second, m2.second)
#pybind11#            self.assertEqual(m1.distance, m2.distance)
#pybind11#            self.assertEqual(m1.first.getId(), c["first"])
#pybind11#            self.assertEqual(m1.second.getId(), c["second"])
#pybind11#            self.assertEqual(m1.distance, c["distance"])
#pybind11#
#pybind11#        self.checkPickle(mat, checkSlots=False)
#pybind11#        self.checkPickle(mat2, checkSlots=False)
#pybind11#
#pybind11#        self.checkMatchToFromCatalog(mat, cat)
#pybind11#
#pybind11#        if False:
#pybind11#            s0 = mat[0][0]
#pybind11#            s1 = mat[0][1]
#pybind11#            print(s0.getRa(), s1.getRa(), s0.getId(), s1.getId())
#pybind11#
#pybind11#    def testNaNPositions(self):
#pybind11#        ss1 = afwTable.SourceCatalog(self.table)
#pybind11#        ss2 = afwTable.SourceCatalog(self.table)
#pybind11#        for ss in (ss1, ss2):
#pybind11#            ss.addNew().set(afwTable.SourceTable.getCoordKey().getRa(), float('nan') * afwGeom.radians)
#pybind11#
#pybind11#            ss.addNew().set(afwTable.SourceTable.getCoordKey().getDec(), float('nan') * afwGeom.radians)
#pybind11#
#pybind11#            s = ss.addNew()
#pybind11#            s.set(afwTable.SourceTable.getCoordKey().getRa(), 0.0 * afwGeom.radians)
#pybind11#            s.set(afwTable.SourceTable.getCoordKey().getDec(), 0.0 * afwGeom.radians)
#pybind11#
#pybind11#            s = ss.addNew()
#pybind11#            s.set(afwTable.SourceTable.getCoordKey().getRa(), float('nan') * afwGeom.radians)
#pybind11#            s.set(afwTable.SourceTable.getCoordKey().getDec(), float('nan') * afwGeom.radians)
#pybind11#
#pybind11#        mc = afwTable.MatchControl()
#pybind11#        mc.findOnlyClosest = False
#pybind11#        mat = afwTable.matchRaDec(ss1, ss2, 1.0*afwGeom.arcseconds, mc)
#pybind11#        self.assertEqual(len(mat), 1)
#pybind11#        self.checkPickle(mat)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testPhotometricCalib(self):
#pybind11#        """Test matching the CFHT catalogue (as generated using LSST code) to the SDSS catalogue
#pybind11#        """
#pybind11#
#pybind11#        band = 2                        # SDSS r
#pybind11#
#pybind11#        # Read SDSS catalogue
#pybind11#        ifd = open(os.path.join(afwdataDir, "CFHT", "D2", "sdss.dat"), "r")
#pybind11#
#pybind11#        sdss = afwTable.SourceCatalog(self.table)
#pybind11#        sdssSecondary = afwTable.SourceCatalog(self.table)
#pybind11#
#pybind11#        PRIMARY, SECONDARY = 1, 2       # values of mode
#pybind11#
#pybind11#        id = 0
#pybind11#        for line in ifd.readlines():
#pybind11#            if re.search(r"^\s*#", line):
#pybind11#                continue
#pybind11#
#pybind11#            fields = line.split()
#pybind11#            objId = int(fields[0])
#pybind11#            fields[1]
#pybind11#            mode = int(fields[2])
#pybind11#            ra, dec = [float(f) for f in fields[3:5]]
#pybind11#            psfMags = [float(f) for f in fields[5:]]
#pybind11#
#pybind11#            if mode == PRIMARY:
#pybind11#                s = sdss.addNew()
#pybind11#            elif SECONDARY:
#pybind11#                s = sdssSecondary.addNew()
#pybind11#
#pybind11#            s.setId(objId)
#pybind11#            s.setRa(ra * afwGeom.degrees)
#pybind11#            s.setDec(dec * afwGeom.degrees)
#pybind11#            s.set(self.table.getPsfFluxKey(), psfMags[band])
#pybind11#
#pybind11#        del ifd
#pybind11#
#pybind11#        # Read catalalogue built from the template image
#pybind11#        # Read SDSS catalogue
#pybind11#        ifd = open(os.path.join(afwdataDir, "CFHT", "D2", "template.dat"), "r")
#pybind11#
#pybind11#        template = afwTable.SourceCatalog(self.table)
#pybind11#
#pybind11#        id = 0
#pybind11#        for line in ifd.readlines():
#pybind11#            if re.search(r"^\s*#", line):
#pybind11#                continue
#pybind11#
#pybind11#            fields = line.split()
#pybind11#            id, flags = [int(f) for f in fields[0:2]]
#pybind11#            ra, dec = [float(f) for f in fields[2:4]]
#pybind11#            flux = [float(f) for f in fields[4:]]
#pybind11#
#pybind11#            if flags & 0x1:             # EDGE
#pybind11#                continue
#pybind11#
#pybind11#            s = template.addNew()
#pybind11#            s.setId(id)
#pybind11#            id += 1
#pybind11#            s.set(afwTable.SourceTable.getCoordKey().getRa(), ra * afwGeom.degrees)
#pybind11#            s.set(afwTable.SourceTable.getCoordKey().getDec(), dec * afwGeom.degrees)
#pybind11#            s.set(self.table.getPsfFluxKey(), flux[0])
#pybind11#
#pybind11#        del ifd
#pybind11#
#pybind11#        # Actually do the match
#pybind11#        mc = afwTable.MatchControl()
#pybind11#        mc.findOnlyClosest = False
#pybind11#
#pybind11#        matches = afwTable.matchRaDec(sdss, template, 1.0*afwGeom.arcseconds, mc)
#pybind11#
#pybind11#        self.assertEqual(len(matches), 901)
#pybind11#        self.checkPickle(matches)
#pybind11#
#pybind11#        if False:
#pybind11#            for mat in matches:
#pybind11#                s0 = mat[0]
#pybind11#                s1 = mat[1]
#pybind11#                d = mat[2]
#pybind11#                print(s0.getRa(), s0.getDec(), s1.getRa(), s1.getDec(), s0.getPsfFlux(), s1.getPsfFlux())
#pybind11#
#pybind11#        # Actually do the match
#pybind11#        for s in sdssSecondary:
#pybind11#            sdss.append(s)
#pybind11#
#pybind11#        mc = afwTable.MatchControl()
#pybind11#        mc.symmetricMatch = False
#pybind11#        matches = afwTable.matchRaDec(sdss, 1.0*afwGeom.arcseconds, mc)
#pybind11#        nmiss = 1                                              # one object doesn't match
#pybind11#        self.assertEqual(len(matches), len(sdssSecondary) - nmiss)
#pybind11#        self.checkPickle(matches)
#pybind11#
#pybind11#        # Find the one that didn't match
#pybind11#        if False:
#pybind11#            matchIds = set()
#pybind11#            for s0, s1, d in matches:
#pybind11#                matchIds.add(s0.getId())
#pybind11#                matchIds.add(s1.getId())
#pybind11#
#pybind11#            for s in sdssSecondary:
#pybind11#                if s.getId() not in matchIds:
#pybind11#                    print("RHL", s.getId())
#pybind11#
#pybind11#        matches = afwTable.matchRaDec(sdss, 1.0*afwGeom.arcseconds)
#pybind11#        self.assertEqual(len(matches), 2*(len(sdssSecondary) - nmiss))
#pybind11#        self.checkPickle(matches)
#pybind11#
#pybind11#        if False:
#pybind11#            for mat in matches:
#pybind11#                s0 = mat[0]
#pybind11#                s1 = mat[1]
#pybind11#                mat[2]
#pybind11#                print(s0.getId(), s1.getId(), s0.getRa(), s0.getDec(), end=' ')
#pybind11#                print(s1.getRa(), s1.getDec(), s0.getPsfFlux(), s1.getPsfFlux())
#pybind11#
#pybind11#    def testMismatches(self):
#pybind11#        """ Chech that matchRaDec works as expected when using
#pybind11#            the includeMismatches option
#pybind11#        """
#pybind11#        cat1 = afwTable.SourceCatalog(self.table)
#pybind11#        cat2 = afwTable.SourceCatalog(self.table)
#pybind11#        nobj = 100
#pybind11#        for i in range(nobj):
#pybind11#            s1 = cat1.addNew()
#pybind11#            s2 = cat2.addNew()
#pybind11#            s1.setId(i)
#pybind11#            s2.setId(i)
#pybind11#            s1.set(afwTable.SourceTable.getCoordKey().getRa(), (10 + 0.0001*i) * afwGeom.degrees)
#pybind11#            s2.set(afwTable.SourceTable.getCoordKey().getRa(), (10.005 + 0.0001*i) * afwGeom.degrees)
#pybind11#            s1.set(afwTable.SourceTable.getCoordKey().getDec(), (10 + 0.0001*i) * afwGeom.degrees)
#pybind11#            s2.set(afwTable.SourceTable.getCoordKey().getDec(), (10.005 + 0.0001*i) * afwGeom.degrees)
#pybind11#
#pybind11#        for closest in (True, False):
#pybind11#            mc = afwTable.MatchControl()
#pybind11#            mc.findOnlyClosest = closest
#pybind11#            mc.includeMismatches = False
#pybind11#            matches = afwTable.matchRaDec(cat1, cat2, 1.0*afwGeom.arcseconds, mc)
#pybind11#            mc.includeMismatches = True
#pybind11#            matchesMismatches = afwTable.matchRaDec(cat1, cat2, 1.0*afwGeom.arcseconds, mc)
#pybind11#
#pybind11#            catMatches = afwTable.SourceCatalog(self.table)
#pybind11#            catMismatches = afwTable.SourceCatalog(self.table)
#pybind11#            for m in matchesMismatches:
#pybind11#                if m[1] is not None:
#pybind11#                    if not any(x == m[0] for x in catMatches):
#pybind11#                        catMatches.append(m[0])
#pybind11#                else:
#pybind11#                    catMismatches.append(m[0])
#pybind11#            if closest:
#pybind11#                self.assertEqual(len(catMatches), len(matches))
#pybind11#            matches2 = afwTable.matchRaDec(catMatches, cat2, 1.0*afwGeom.arcseconds, mc)
#pybind11#            self.assertEqual(len(matches), len(matches2))
#pybind11#            mc.includeMismatches = False
#pybind11#            noMatches = afwTable.matchRaDec(catMismatches, cat2, 1.0*afwGeom.arcseconds, mc)
#pybind11#            self.assertEqual(len(noMatches), 0)
#pybind11#
#pybind11#    def checkPickle(self, matches, checkSlots=True):
#pybind11#        """Check that a match list pickles
#pybind11#
#pybind11#        Also checks that the slots survive pickling, if checkSlots is True.
#pybind11#        """
#pybind11#        orig = afwTable.SourceMatchVector(matches)
#pybind11#        unpickled = pickle.loads(pickle.dumps(orig))
#pybind11#        self.assertEqual(len(orig), len(unpickled))
#pybind11#        for m1, m2 in zip(orig, unpickled):
#pybind11#            self.assertEqual(m1.first.getId(), m2.first.getId())
#pybind11#            self.assertEqual(m1.first.getRa(), m2.first.getRa())
#pybind11#            self.assertEqual(m1.first.getDec(), m2.first.getDec())
#pybind11#            self.assertEqual(m1.second.getId(), m2.second.getId())
#pybind11#            self.assertEqual(m1.second.getRa(), m2.second.getRa())
#pybind11#            self.assertEqual(m1.second.getDec(), m2.second.getDec())
#pybind11#            self.assertEqual(m1.distance, m2.distance)
#pybind11#            if checkSlots:
#pybind11#                self.assertEqualFloat(m1.first.getPsfFlux(), m2.first.getPsfFlux())
#pybind11#                self.assertEqualFloat(m1.second.getPsfFlux(), m2.second.getPsfFlux())
#pybind11#
#pybind11#    def checkMatchToFromCatalog(self, matches, catalog):
#pybind11#        """Check the conversion of matches to and from a catalog
#pybind11#
#pybind11#        Test the functions in lsst.afw.table.catalogMatches.py
#pybind11#        Note that the return types and schemas of these functions do not necessarily match
#pybind11#        those of the catalogs passed to them, so value entries are compared as opposed to
#pybind11#        comparing the attributes as a whole.
#pybind11#        """
#pybind11#        catalog.setMetadata(self.metadata)
#pybind11#        matchMeta = catalog.getTable().getMetadata()
#pybind11#        matchToCat = afwTable.catalogMatches.matchesToCatalog(matches, matchMeta)
#pybind11#        matchFromCat = afwTable.catalogMatches.matchesFromCatalog(matchToCat)
#pybind11#        self.assertEqual(len(matches), len(matchToCat))
#pybind11#        self.assertEqual(len(matches), len(matchFromCat))
#pybind11#
#pybind11#        for mat, cat, catM, matchC in zip(matches, catalog, matchToCat, matchFromCat):
#pybind11#            self.assertEqual(mat.first.getId(), catM["ref_id"])
#pybind11#            self.assertEqual(mat.first.getId(), matchC.first.getId())
#pybind11#            self.assertEqual(mat.first.getCoord(), matchC.first.getCoord())
#pybind11#            self.assertEqual(mat.second.getId(), cat["second"])
#pybind11#            self.assertEqual(mat.second.getId(), catM["src_id"])
#pybind11#            self.assertEqual(mat.second.getId(), matchC.second.getId())
#pybind11#            self.assertEqual((mat.first.getRa(), mat.first.getDec()),
#pybind11#                             (catM["ref_coord_ra"], catM["ref_coord_dec"]))
#pybind11#            self.assertEqual((mat.second.getRa(), mat.second.getDec()),
#pybind11#                             (catM["src_coord_ra"], catM["src_coord_dec"]))
#pybind11#            self.assertEqual(mat.first.getCoord(), matchC.first.getCoord())
#pybind11#            self.assertEqual(mat.second.getCoord(), matchC.second.getCoord())
#pybind11#            self.assertEqual(mat.distance, matchC.distance)
#pybind11#            self.assertEqual(mat.distance, cat["distance"])
#pybind11#            self.assertEqual(mat.distance, catM["distance"])
#pybind11#
#pybind11#    def assertEqualFloat(self, value1, value2):
#pybind11#        """Compare floating point values, allowing for NAN
#pybind11#        """
#pybind11#        self.assertTrue(value1 == value2 or (numpy.isnan(value1) and numpy.isnan(value2)))
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

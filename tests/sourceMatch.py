#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

"""
Tests for matching SourceSets

Run with:
   python SourceMatch.py
or
   python
   >>> import SourceMatch; SourceMatch.run()
"""
import os
import re

import numpy
import unittest
import pickle

import lsst.utils
import lsst.utils.tests as utilsTests
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SourceMatchTestCase(unittest.TestCase):
    """A test case for matching SourceSets"""

    def setUp(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        fluxKey = schema.addField("flux_flux", type=float)
        fluxErrKey = schema.addField("flux_fluxSigma", type=float)
        fluxFlagKey = schema.addField("flux_flag", type="Flag")
        self.table = afwTable.SourceTable.make(schema)
        self.table.definePsfFlux("flux")
        self.ss1 = afwTable.SourceCatalog(self.table)
        self.ss2 = afwTable.SourceCatalog(self.table)

    def tearDown(self):
        del self.table
        del self.ss1
        del self.ss2

    def testIdentity(self):
        nobj = 1000
        for i in range(nobj):
            s = self.ss1.addNew()
            s.setId(i)
            s.set(afwTable.SourceTable.getCoordKey().getRa(), (10 + 0.001*i) * afwGeom.degrees)
            s.set(afwTable.SourceTable.getCoordKey().getDec(), (10 + 0.001*i) * afwGeom.degrees)

            s = self.ss2.addNew()
            s.setId(2*nobj + i)
            s.set(afwTable.SourceTable.getCoordKey().getRa(), (10 + 0.001*i) * afwGeom.degrees)
            s.set(afwTable.SourceTable.getCoordKey().getDec(), (10 + 0.001*i) * afwGeom.degrees)

        mat = afwTable.matchRaDec(self.ss1, self.ss2, 1.0 * afwGeom.arcseconds, False)

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

        self.checkPickle(mat, checkSlots=False)
        self.checkPickle(mat2, checkSlots=False)

        if False:
            s0 = mat[0][0]
            s1 = mat[0][1]
            print s0.getRa(), s1.getRa(), s0.getId(), s1.getId()

    def testNaNPositions(self):
        ss1 = afwTable.SourceCatalog(self.table)
        ss2 = afwTable.SourceCatalog(self.table)
        for ss in (ss1, ss2):
            ss.addNew().set(afwTable.SourceTable.getCoordKey().getRa(), float('nan') * afwGeom.radians)

            ss.addNew().set(afwTable.SourceTable.getCoordKey().getDec(), float('nan') * afwGeom.radians)

            s = ss.addNew()
            s.set(afwTable.SourceTable.getCoordKey().getRa(), 0.0 * afwGeom.radians)
            s.set(afwTable.SourceTable.getCoordKey().getDec(), 0.0 * afwGeom.radians)

            s = ss.addNew()
            s.set(afwTable.SourceTable.getCoordKey().getRa(), float('nan') * afwGeom.radians)
            s.set(afwTable.SourceTable.getCoordKey().getDec(), float('nan') * afwGeom.radians)

        mat = afwTable.matchRaDec(ss1, ss2, 1.0 * afwGeom.arcseconds, False)
        self.assertEqual(len(mat), 1)
        self.checkPickle(mat)

    def testPhotometricCalib(self):
        """Test matching the CFHT catalogue (as generated using LSST code) to the SDSS catalogue"""

        band = 2                        # SDSS r
        
        #
        # Read SDSS catalogue
        #
        ifd = open(os.path.join(lsst.utils.getPackageDir("afwdata"), "CFHT", "D2", "sdss.dat"), "r")

        sdss = afwTable.SourceCatalog(self.table)
        sdssSecondary = afwTable.SourceCatalog(self.table)

        PRIMARY, SECONDARY = 1, 2       # values of mode

        id = 0
        for line in ifd.readlines():
            if re.search(r"^\s*#", line):
                continue

            fields = line.split()
            objId = int(fields[0])
            name = fields[1]
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
        #
        # Read catalalogue built from the template image
        #
        #
        # Read SDSS catalogue
        #
        ifd = open(os.path.join(lsst.utils.getPackageDir("afwdata"), "CFHT", "D2", "template.dat"), "r")

        template = afwTable.SourceCatalog(self.table)

        id = 0
        for line in ifd.readlines():
            if re.search(r"^\s*#", line):
                continue

            fields = line.split()
            id, flags = [int(f) for f in  fields[0:2]]
            ra, dec = [float(f) for f in fields[2:4]]
            flux = [float(f) for f in fields[4:]]

            if flags & 0x1:             # EDGE
                continue

            s = template.addNew()
            s.setId(id)
            id += 1
            s.set(afwTable.SourceTable.getCoordKey().getRa(), ra * afwGeom.degrees)
            s.set(afwTable.SourceTable.getCoordKey().getDec(), dec * afwGeom.degrees)
            s.set(self.table.getPsfFluxKey(), flux[0])

        del ifd
        #
        # Actually do the match
        #
        matches = afwTable.matchRaDec(sdss, template, 1.0 * afwGeom.arcseconds, False)

        self.assertEqual(len(matches), 901)
        self.checkPickle(matches)

        if False:
            for mat in matches:
                s0 = mat[0]
                s1 = mat[1]
                d = mat[2]
                print s0.getRa(), s0.getDec(), s1.getRa(), s1.getDec(), s0.getPsfFlux(), s1.getPsfFlux()
        #
        # Actually do the match
        #
        for s in sdssSecondary:
            sdss.append(s)

        matches = afwTable.matchRaDec(sdss, 1.0 * afwGeom.arcseconds, False)
        nmiss = 1                                              # one object doesn't match
        self.assertEqual(len(matches), len(sdssSecondary) - nmiss)
        self.checkPickle(matches)
        #
        # Find the one that didn't match
        #
        if False:
            matchIds = set()
            for s0, s1, d in matches:
                matchIds.add(s0.getId())
                matchIds.add(s1.getId())

            for s in sdssSecondary:
                if s.getId() not in matchIds:
                    print "RHL", s.getId()

        matches = afwTable.matchRaDec(sdss, 1.0 * afwGeom.arcseconds, True)
        self.assertEqual(len(matches), 2*(len(sdssSecondary) - nmiss))
        self.checkPickle(matches)

        if False:
            for mat in matches:
                s0 = mat[0]
                s1 = mat[1]
                d = mat[2]
                print s0.getId(), s1.getId(), s0.getRa(), s0.getDec(),
                print s1.getRa(), s1.getDec(), s0.getPsfFlux(), s1.getPsfFlux()
                
    def checkPickle(self, matches, checkSlots=True):
        """Check that a match list pickles

        Also checks that the slots survive pickling, if checkSlots is True.
        """
        orig = afwTable.SourceMatchVector(matches)
        unpickled = pickle.loads(pickle.dumps(orig))
        self.assertEqual(len(orig), len(unpickled))
        for m1, m2 in zip(orig, unpickled):
            self.assertEqual(m1.first.getId(), m2.first.getId())
            self.assertEqual(m1.first.getRa(), m2.first.getRa())
            self.assertEqual(m1.first.getDec(), m2.first.getDec())
            self.assertEqual(m1.second.getId(), m2.second.getId())
            self.assertEqual(m1.second.getRa(), m2.second.getRa())
            self.assertEqual(m1.second.getDec(), m2.second.getDec())
            self.assertEqual(m1.distance, m2.distance)
            if checkSlots:
                self.assertEqualFloat(m1.first.getPsfFlux(), m2.first.getPsfFlux())
                self.assertEqualFloat(m1.second.getPsfFlux(), m2.second.getPsfFlux())

    def assertEqualFloat(self, value1, value2):
        """Compare floating point values, allowing for NAN"""
        self.assertTrue(value1 == value2 or (numpy.isnan(value1) and numpy.isnan(value2)))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SourceMatchTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)


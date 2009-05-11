#!/usr/bin/env python
"""
Tests for matching SourceSets

Run with:
   python SourceMatch.py
or
   python
   >>> import SourceMatch; SourceMatch.run()
"""
import os, re, sys
import pdb
import unittest
import random
import time
import eups

import lsst.daf.base as dafBase
import lsst.pex.policy as dafPolicy
import lsst.daf.persistence as dafPers
import lsst.utils.tests as utilsTests
import lsst.afw.detection as afwDet

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SourceMatchTestCase(unittest.TestCase):
    """A test case for matching SourceSets"""

    def setUp(self):
        self.ss1 = afwDet.SourceSet()
        self.ss2 = afwDet.SourceSet()

    def tearDown(self):
        del self.ss1
        del self.ss2

    def testIdentity(self):
        nobj = 1000
        for i in range(nobj):
            s = afwDet.Source()
            s.setId(i)
            s.setRa(10 + 0.001*i)
            s.setDec(10 + 0.001*i)

            self.ss1.append(s)

            s = afwDet.Source()
            s.setId(2*nobj + i)
            s.setRa(10 + 0.001*i)
            s.setDec(10 + 0.001*i)

            self.ss2.append(s)

        mat = afwDet.matchRaDec(self.ss1, self.ss2, 1.0)

        self.assertEqual(len(mat), nobj)

        if False:
            s0 = mat[0][0]
            s1 = mat[0][1]
            print s0.getRa(), s1.getRa(), s0.getId(), s1.getId()

    def testPhotometricCalib(self):
        """Test matching the CFHT catalogue (as generated using LSST code) to the SDSS catalogue"""

        if not eups.productDir("afwdata"):
            print >> sys.stderr, "Failed to open sdss catalogue"
            return

        band = 2                        # SDSS r
        
        #
        # Read SDSS catalogue
        #
        ifd = open(os.path.join(eups.productDir("afwdata"), "CFHT", "D2", "sdss.dat"), "r")

        sdss = afwDet.SourceSet()
        sdssSecondary = afwDet.SourceSet()

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

            s = afwDet.Source()
            s.setId(objId);
            s.setRa(ra)
            s.setDec(dec)
            s.setPsfFlux(psfMags[band])

            if mode == PRIMARY:
                sdss.append(s)
            elif SECONDARY:
                sdssSecondary.append(s)

        del ifd
        #
        # Read catalalogue built from the template image
        #
        #
        # Read SDSS catalogue
        #
        ifd = open(os.path.join(eups.productDir("afwdata"), "CFHT", "D2", "template.dat"), "r")

        template = afwDet.SourceSet()

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

            s = afwDet.Source()
            s.setId(id); id += 1
            s.setRa(ra)
            s.setDec(dec)
            s.setPsfFlux(flux[0])

            template.append(s)

        del ifd
        #
        # Actually do the match
        #
        matches = afwDet.matchRaDec(sdss, template, 1.0)

        self.assertEqual(len(matches), 901)

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

        matches = afwDet.matchRaDec(sdss, 1.0, False)
        nmiss = 1                                              # one object doesn't match
        self.assertEqual(len(matches), len(sdssSecondary) - nmiss)
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

        matches = afwDet.matchRaDec(sdss, 1.0, True)
        self.assertEqual(len(matches), 2*(len(sdssSecondary) - nmiss))
        
        if False:
            for mat in matches:
                s0 = mat[0]
                s1 = mat[1]
                d = mat[2]
                print s0.getId(), s1.getId(), s0.getRa(), s0.getDec(), s1.getRa(), s1.getDec(), s0.getPsfFlux(), s1.getPsfFlux()


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SourceMatchTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)


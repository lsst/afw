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
Tests for matching SourceSets

Run with:
   python SourceMatch.py
or
   python
   >>> import SourceMatch; SourceMatch.run()
"""
import os, re, sys

import unittest
import eups
from math import radians

import lsst.utils.tests as utilsTests
import lsst.afw.detection as afwDetect

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SourceMatchTestCase(unittest.TestCase):
    """A test case for matching SourceSets"""

    def setUp(self):
        self.ss1 = afwDetect.SourceSet()
        self.ss2 = afwDetect.SourceSet()

    def tearDown(self):
        del self.ss1
        del self.ss2

    def testIdentity(self):
        nobj = 1000
        for i in range(nobj):
            s = afwDetect.Source()
            s.setId(i)
            s.setRa(radians(10 + 0.001*i))
            s.setDec(radians(10 + 0.001*i))

            self.ss1.append(s)

            s = afwDetect.Source()
            s.setId(2*nobj + i)
            s.setRa(radians(10 + 0.001*i))
            s.setDec(radians(10 + 0.001*i))

            self.ss2.append(s)

        mat = afwDetect.matchRaDec(self.ss1, self.ss2, 1.0, False)

        self.assertEqual(len(mat), nobj)

        if False:
            s0 = mat[0][0]
            s1 = mat[0][1]
            print s0.getRa(), s1.getRa(), s0.getId(), s1.getId()

    def testNaNPositions(self):
        ss1 = afwDetect.SourceSet()
        ss2 = afwDetect.SourceSet()
        for ss in (ss1, ss2):
            s = afwDetect.Source()
            s.setRa(float('nan'))
            ss.append(s)
            s = afwDetect.Source()
            s.setDec(float('nan'))
            ss.append(s)
            s = afwDetect.Source()
            s.setRa(0.0)
            s.setDec(0.0)
            ss.append(s)
            s = afwDetect.Source()
            s.setRa(float('nan'))
            s.setDec(float('nan'))
            ss.append(s)
        mat = afwDetect.matchRaDec(ss1, ss2, 1.0, False)
        self.assertEqual(len(mat), 1)

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

        sdss = afwDetect.SourceSet()
        sdssSecondary = afwDetect.SourceSet()

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

            s = afwDetect.Source()
            s.setId(objId)
            s.setRa(radians(ra))
            s.setDec(radians(dec))
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

        template = afwDetect.SourceSet()

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

            s = afwDetect.Source()
            s.setId(id)
            id += 1
            s.setRa(radians(ra))
            s.setDec(radians(dec))
            s.setPsfFlux(flux[0])

            template.append(s)

        del ifd
        #
        # Actually do the match
        #
        matches = afwDetect.matchRaDec(sdss, template, 1.0, False)

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

        matches = afwDetect.matchRaDec(sdss, 1.0, False)
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

        matches = afwDetect.matchRaDec(sdss, 1.0, True)
        self.assertEqual(len(matches), 2*(len(sdssSecondary) - nmiss))
        
        if False:
            for mat in matches:
                s0 = mat[0]
                s1 = mat[1]
                d = mat[2]
                print s0.getId(), s1.getId(), s0.getRa(), s0.getDec(),
                print s1.getRa(), s1.getDec(), s0.getPsfFlux(), s1.getPsfFlux()
                

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


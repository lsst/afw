#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

# -*- python -*-
"""
Check that coord and coordPtr are properly passed through swig

Run with:
   python coordptr.py
"""

import os
import unittest

import lsst.utils
import lsst.afw.image            as image
import lsst.afw.geom             as afwGeom
import lsst.afw.coord.coordLib   as coord
import lsst.utils.tests          as utilsTests


class CoordPtrTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testMakeCoord(self):
        c = coord.Coord(1 * afwGeom.degrees,2 * afwGeom.degrees)
        print type(c)
        c = coord.makeCoord(coord.FK5, 1 * afwGeom.degrees, 2 * afwGeom.degrees)
        print type(c)

    def testMakeWcs(self):
        afwdataDir = lsst.utils.getPackageDir("afw")
        path = os.path.join(afwdataDir, "tests", "data", "parent.fits")
        fitsHdr = image.readMetadata(path)

        wcs = image.makeWcs(fitsHdr)

        c = wcs.pixelToSky(0,0)
        print type(c)
        c.getPosition()

    def testCoordCast(self):
        for CoordClass in (coord.IcrsCoord, coord.Fk5Coord, coord.GalacticCoord, coord.EclipticCoord):
            derived1 = CoordClass(1 * afwGeom.degrees, 2 * afwGeom.degrees)
            self.assertEqual(type(derived1), CoordClass)
            base = derived1.clone()
            self.assertEqual(type(base), coord.Coord)
            derived2 = CoordClass.cast(base)
            self.assertEqual(type(derived2), CoordClass)

#################################################################
# Test suite boiler plate
#################################################################
def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(CoordPtrTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit = False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

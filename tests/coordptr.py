#!/usr/bin/env python
# -*- python -*-
"""
Check that coord and coordPtr are properly passed through swig

Run with:
   python coordptr.py
"""

import os
import unittest

import eups
import lsst.afw.image            as image
import lsst.afw.geom             as geom
import lsst.afw.coord.coordLib   as coord
import lsst.utils.tests          as utilsTests
import lsst.daf.base             as dafBase


class CoordPtrTestCase(unittest.TestCase):

    def setUp(self):
        pass
        
    def tearDown(self):
        pass
        
    def testMakeCoord(self):
        
        c = coord.Coord(1,2,2000, coord.DEGREES)
        print type(c)
        c = coord.makeCoord(coord.FK5, 1, 2, coord.DEGREES)
        print type(c)
        
    def testMakeWcs(self):
        path= eups.productDir("afw")
        path = os.path.join(path, "tests", "data", "parent.fits")
        fitsHdr = image.readMetadata(path)
        
        wcs = image.makeWcs(fitsHdr)
        
        c = wcs.pixelToSky(0,0)
        print type(c)
        c.getPosition()
        
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

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

import os
import math
import unittest

import lsst.afw.image
import lsst.utils.tests as utilsTests
import lsst.daf.base

class WcsFitsTableTestCase(unittest.TestCase):
    """Test that we can read and write Wcs objects saved to FITS binary tables.
    """
    
    def setUp(self):
        #metadata taken from CFHT data
        #v695856-e0/v695856-e0-c000-a00.sci.fits

        self.metadata = lsst.daf.base.PropertySet()

        self.metadata.set("SIMPLE",                    "T") 
        self.metadata.set("BITPIX",                  -32) 
        self.metadata.set("NAXIS",                    2) 
        self.metadata.set("NAXIS1",                 1024) 
        self.metadata.set("NAXIS2",                 1153) 
        self.metadata.set("RADECSYS", 'FK5')
        self.metadata.set("EQUINOX",                2000.)

        self.metadata.setDouble("CRVAL1",     215.604025685476)
        self.metadata.setDouble("CRVAL2",     53.1595451514076)
        self.metadata.setDouble("CRPIX1",     1109.99981456774)
        self.metadata.setDouble("CRPIX2",     560.018167811613)
        self.metadata.set("CTYPE1", 'RA---SIN')
        self.metadata.set("CTYPE2", 'DEC--SIN')

        self.metadata.setDouble("CD1_1", 5.10808596133527E-05)
        self.metadata.setDouble("CD1_2", 1.85579539217196E-07)
        self.metadata.setDouble("CD2_2", -5.10281493481982E-05)
        self.metadata.setDouble("CD2_1", -8.27440751733828E-07)

    def tearDown(self):
        del self.metadata

    def doFitsRoundTrip(self, wcsIn):
        fileName = "wcs-table-test.fits"
        wcsIn.writeFits(fileName)
        wcsOut = lsst.afw.image.Wcs.readFits(fileName)
        os.remove(fileName)
        return wcsOut
        
    def testSimpleWcs(self):
        wcsIn = lsst.afw.image.makeWcs(self.metadata)
        wcsOut = self.doFitsRoundTrip(wcsIn)
        self.assertEqual(wcsIn, wcsOut)
        
#####

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(WcsFitsTableTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

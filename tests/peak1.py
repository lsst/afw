#!/usr/bin/env python2
from __future__ import absolute_import, division

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
Tests for Peaks

Run with:
   python peak1.py
or
   python
   >>> import peak1; peak1.run()
"""

import unittest
import numpy
import lsst.utils.tests as tests
import lsst.pex.logging as logging
import lsst.afw.detection as afwDetect
import lsst.afw.geom as afwGeom

try:
    type(verbose)
except NameError:
    verbose = 0
    logging.Debug("afwDetect.Footprint", verbose)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class PeakTestCase(unittest.TestCase):
    """A test case for Peak"""
    def setUp(self):
        self.peak = afwDetect.Peak()

    def tearDown(self):
        del self.peak

    def testGC(self):
        """Check that Peaks are automatically garbage collected (when MemoryTestCase runs)"""
        
        f = afwDetect.Peak()

    def testToString(self):
        assert self.peak.toString() != None
        
    def testCentroidInt(self):
        x, y = 10, -10
        peak = afwDetect.Peak(x, y)
        self.assertEqual(peak.getIx(), x)
        self.assertEqual(peak.getIy(), y)

        self.assertEqual(peak.getFx(), x)
        self.assertEqual(peak.getFy(), y)

        self.assertEqual(peak.getF(),        afwGeom.PointD(x, y))
        self.assertEqual(peak.getCentroid(), afwGeom.PointD(x, y))
        self.assertEqual(peak.getI(),            afwGeom.PointI(x, y))
        self.assertEqual(peak.getCentroid(True), afwGeom.PointI(x, y))

    def testCentroidFloat(self):
        for x, y in [(5, 6), (10.5, -10.5)]:
            peak = afwDetect.Peak(x, y)
            self.assertEqual(peak.getCentroid(), afwGeom.PointD(x, y))

            self.assertEqual(peak.getIx(), int(x) if x > 0 else -int(-x) - 1)
            self.assertEqual(peak.getIy(), int(y) if y > 0 else -int(-y) - 1)

    def testPeakVal(self):
        peak = afwDetect.Peak(1, 1)
        self.assertTrue(numpy.isnan(peak.getPeakValue()))

        val = 666.0
        peak.setPeakValue(val)
        self.assertEqual(peak.getPeakValue(), val)
        del peak

        peak = afwDetect.Peak(1, 1, val)
        self.assertEqual(peak.getPeakValue(), val)
        
    def testId(self):
        """Test uniqueness of IDs"""
        
        self.assertNotEqual(self.peak.getId(), afwDetect.Peak().getId())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(PeakTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

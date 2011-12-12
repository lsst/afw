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
Tests for Angle

Run with:
   angle.py
or
   python
   >>> import angle; angle.run()
"""

import math, os, sys
import unittest
import lsst.utils.tests as tests

import lsst.afw.geom as afwGeom

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class AngleTestCase(unittest.TestCase):
    """A test case for Angle"""
    def setUp(self):
        self.pi = afwGeom.Angle(math.pi, afwGeom.radians)
        self.d = 180*afwGeom.degrees

    def tearDown(self):
        pass

    def testCtor(self):
        self.assertEqual(self.pi, math.pi)
        self.assertEqual(self.pi, afwGeom.Angle(math.pi))
        self.assertEqual(self.pi, self.d)

        dd = afwGeom.Angle(180, afwGeom.degrees)
        self.assertEqual(self.d, dd)
        dd = afwGeom.Angle(60*180, afwGeom.arcminutes)
        self.assertEqual(self.d, dd)
        dd = afwGeom.Angle(60*60*180, afwGeom.arcseconds)
        self.assertEqual(self.d, dd)

    def testArithmetic(self):
        self.assertTrue(afwGeom.isAngle(self.pi))
        self.assertFalse(afwGeom.isAngle(self.pi.asRadians()))
        self.assertFalse(afwGeom.isAngle(math.pi))
        
        def tst():
            self.pi - math.pi           # subtracting a float from an Angle
        self.assertRaises(TypeError, tst)
        self.assertEqual(self.pi - math.pi*afwGeom.radians, 0) # OK with units specified
        self.assertEqual(self.pi - self.d, 0)                  # can subtract Angles

        def tst():
            self.pi + math.pi           # adding a float to an Angle
        self.assertRaises(TypeError, tst)

        def tst():
            self.pi*afwGeom.degrees     # self.pi is already an Angle
        self.assertRaises(NotImplementedError, tst)

        self.assertEqual((self.pi + self.d).asAngularUnits(afwGeom.degrees), 360)
        self.assertEqual((self.pi).asRadians(), math.pi)
        self.assertEqual((self.pi/2).asDegrees(), 90)
        self.assertEqual((self.pi*2).asArcminutes(), 360*60)
        self.assertEqual((self.pi*2).asArcseconds(), 360*60*60)

        self.assertEqual(math.sin(self.pi/2), 1.0) # automatic conversion to double

    def testPi(self):
        self.assertEqual(afwGeom.PI, math.pi)

    def testComparison(self):
        a2 = 2.0 * afwGeom.arcseconds
        a1 = 0.5 * afwGeom.arcseconds
        a3 = 0.5 * afwGeom.arcseconds
        print 'a1', a1
        print 'a2', a2
        print 'a3', a3
        self.assertEqual(a1 == a3, True)
        self.assertEqual(a1 != a2, True)
        self.assertEqual(a1 <= a2, True)
        self.assertEqual(a1 <  a2, True)
        self.assertEqual(a2 >  a1, True)
        self.assertEqual(a2 >= a1, True)

        self.assertEqual(a1 != a3, False)
        self.assertEqual(a1 == a2, False)
        self.assertEqual(a1 >= a2, False)
        self.assertEqual(a1 >  a2, False)
        self.assertEqual(a2 <  a1, False)
        self.assertEqual(a2 <= a1, False)

    def testTrig(self):
        self.assertEqual(math.cos(self.d), -1.0)
        self.assertAlmostEqual(math.sin(self.d),  0.0, places=15)
        thirty = 30.*afwGeom.degrees
        self.assertAlmostEqual(math.sin(thirty), 0.5, places=15)
        


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(AngleTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

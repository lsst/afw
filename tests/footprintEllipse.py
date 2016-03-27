#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#


import math
import unittest
import lsst.utils.tests as tests
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDet

class FootprintTestCase(unittest.TestCase):
    def testCircle(self):
        xc, yc = 30, 50
        radius = 10
        
        test = afwDet.Footprint(afwGeom.Point2I(xc, yc), radius)

        # Here's how it used to be done using circles, before #1556
        r2 = int(radius**2 + 0.5)
        r = int(math.sqrt(r2))
        control = afwDet.Footprint()
        for i in range(-r, r+1):
            hlen = int(math.sqrt(r2 - i**2))
            control.addSpan(yc + i, xc - hlen, xc + hlen)

        self.assertEqual(len(test.getSpans()), len(control.getSpans()))
        for s0, s1 in zip(test.getSpans(), control.getSpans()):
            print s0.getY(), s0.getX0(), s0.getX1(), s1.getY(), s1.getX0(), s1.getX1()
            self.assertEqual(s0.getX0(), s1.getX0())
            self.assertEqual(s0.getX1(), s1.getX1())
            self.assertEqual(s0.getY(), s1.getY())
        self.assertEqual(test.getNpix(), control.getNpix())


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(FootprintTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

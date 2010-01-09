#!/usr/bin/env python
"""
Tests for geom.Point, geom.Extent, geom.CoordinateExpr

Run with:
   ./Coordinates.py
or
   python
   >>> import Coordinates; Coordinates.run()
"""

import pdb  # we may want to say pdb.set_trace()
import unittest
import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.geom as geom

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class EllipseTestCase(unittest.TestCase):
    
    def setUp(self):
        self.cores = (
            geom.ellipses.Axes(4, 3, 1),
            geom.ellipses.Distortion(0.5, -0.3, 3.2),
            geom.ellipses.LogShear(0.1, 0.5, 0.7),
            geom.ellipses.Quadrupole(5, 3, -1),
            )

    def assertClose(self, a, b):
        if not numpy.allclose(a, b):
            return self.assertEqual(a, b)
        else:
            return self.assert_(True)

    def testAccessors(self):
        for core in self.cores:
            self.assertRaises(IndexError, core.__getitem__, -1)
            self.assertRaises(IndexError, core.__setitem__, -1, 0)
            self.assertRaises(IndexError, core.__getitem__, 3)
            self.assertRaises(IndexError, core.__setitem__, 3, 0)
            for n in range(3):
                v = core[n]
                v += numpy.random.randn() * 1E-3
                core[n] = v
                self.assertEqual(core[n], v)
            center = geom.makePointD(*numpy.random.randn(2))
            ellipse = core.makeEllipse(center)
            for n in range(3):
                self.assertEqual(core[n], ellipse[n+2])
            self.assertEqual(ellipse[0], center[0])
            self.assertEqual(ellipse[1], center[1])
            self.assertEqual(tuple(center), tuple(ellipse.getCenter()))
            self.assertEqual(geom.Point2D, type(ellipse.getCenter()))
            self.assertEqual(tuple(core), tuple(ellipse.getCore()))
            self.assertEqual(core.__class__, type(ellipse.getCore()))
            newcore = core.__class__(geom.ellipses.LogShear(*numpy.random.randn(3)))
            ellipse.setCore(newcore)
            self.assertEqual(tuple(newcore), tuple(ellipse.getCore()))
            self.assertEqual(tuple(core.clone()), tuple(core))
            self.assert_(core is not core.clone())
            self.assertEqual(tuple(ellipse.clone()), tuple(ellipse))
            self.assert_(ellipse is not ellipse.clone())

    def testTransform(self):
        for core in self.cores:
            t1 = core.getGenerator()
            unit_circle_core = core.__class__(geom.ellipses.Axes(1, 1, 0))
            self.assertClose(tuple(unit_circle_core.transform(t1)), core)
            center = geom.makePointD(*numpy.random.randn(2))
            ellipse = core.makeEllipse(center)
            t2 = ellipse.getGenerator()
            unit_circle_ellipse = ellipse.__class__(unit_circle_core, geom.makePointD(0, 0))
            self.assertClose(tuple(unit_circle_ellipse.transform(t2)), ellipse)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(EllipseTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

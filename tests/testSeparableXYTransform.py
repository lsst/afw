#!/usr/bin/env python2
#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#
"""
Tests for lsst.afw.geom.SeparableXYTransform class.
"""
import unittest
import numpy as np
import numpy.random as random
import lsst.utils.tests
import lsst.afw.geom as afwGeom

class SeparableXYTransformTestCase(unittest.TestCase):
    def setUp(self):
        random.seed(48091)
        self.xpars = 5, 30
        self.ypars = 6, 25
        # Generate test points, drawing randomly from within 2 pixels
        # around edges of sensors
        dxy = 2.
        npts = 10
        self.xvals = np.concatenate((random.uniform(0, dxy, size=npts),
                                     random.uniform(self.xpars[-1] - dxy,
                                                    self.xpars[-1], size=npts)))
        random.shuffle(self.xvals)
        self.yvals = np.concatenate((random.uniform(0, dxy, size=npts),
                                     random.uniform(self.ypars[-1] - dxy,
                                                    self.ypars[-1], size=npts)))
        self.functorClass = afwGeom.LinearFunctor
    def tearDown(self):
        pass
    def test_roundtrip(self):
        xfunctor = self.functorClass(*self.xpars)
        yfunctor = self.functorClass(*self.ypars)
        transform = afwGeom.SeparableXYTransform(xfunctor, yfunctor)
        for x, y in zip(self.xvals, self.yvals):
            pt0 = afwGeom.Point2D(x, y)
            tmp = transform.forwardTransform(pt0)
            pt1 = transform.reverseTransform(tmp)
            self.assertAlmostEquals((pt1 - pt0).computeNorm(), 0, places=6)

def suite():
    """Return a suite containing all of the test cases in this module."""
    lsst.utils.tests.init()
    suites = []
    suites += unittest.makeSuite(SeparableXYTransformTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == '__main__':
    run(True)

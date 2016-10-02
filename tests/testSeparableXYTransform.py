#pybind11##!/usr/bin/env python
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2015 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#"""
#pybind11#Tests for lsst.afw.geom.SeparableXYTransform class.
#pybind11#"""
#pybind11#from builtins import zip
#pybind11#import unittest
#pybind11#import numpy as np
#pybind11#import numpy.random as random
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#
#pybind11#
#pybind11#class SeparableXYTransformTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        random.seed(48091)
#pybind11#        self.xpars = 5, 30
#pybind11#        self.ypars = 6, 25
#pybind11#        # Generate test points, drawing randomly from within 2 pixels
#pybind11#        # around edges of sensors
#pybind11#        dxy = 2.
#pybind11#        npts = 10
#pybind11#        self.xvals = np.concatenate((random.uniform(0, dxy, size=npts),
#pybind11#                                     random.uniform(self.xpars[-1] - dxy,
#pybind11#                                                    self.xpars[-1], size=npts)))
#pybind11#        random.shuffle(self.xvals)
#pybind11#        self.yvals = np.concatenate((random.uniform(0, dxy, size=npts),
#pybind11#                                     random.uniform(self.ypars[-1] - dxy,
#pybind11#                                                    self.ypars[-1], size=npts)))
#pybind11#        self.functorClass = afwGeom.LinearFunctor
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        pass
#pybind11#
#pybind11#    def test_roundtrip(self):
#pybind11#        xfunctor = self.functorClass(*self.xpars)
#pybind11#        yfunctor = self.functorClass(*self.ypars)
#pybind11#        transform = afwGeom.SeparableXYTransform(xfunctor, yfunctor)
#pybind11#        for x, y in zip(self.xvals, self.yvals):
#pybind11#            pt0 = afwGeom.Point2D(x, y)
#pybind11#            tmp = transform.forwardTransform(pt0)
#pybind11#            pt1 = transform.reverseTransform(tmp)
#pybind11#            self.assertAlmostEquals((pt1 - pt0).computeNorm(), 0, places=6)
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

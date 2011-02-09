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
Tests for geom.ellipses

Run with:
   ./ellipse.py
or
   python
   >>> import ellipse; ellipse.run()
"""

import unittest
import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.geom as geom
import lsst.afw.geom.ellipses as geomEllipse

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class EllipseTestCase(unittest.TestCase):
    
    def setUp(self):
        self.cores = [
            geom.ellipses.Axes(4, 3, 1),
            geom.ellipses.Quadrupole(5, 3, -1)
            ]
        for s in geomEllipse.Separable.values():
            self.cores.append(s(0.5, 0.3, 2.1))

    def assertClose(self, a, b):
        if not numpy.allclose(a, b):
            return self.assertEqual(a,b)
        else:
            return self.assert_(True)

    def testAccessors(self):
        for core in self.cores:
            vec = numpy.random.randn(3,1) * 1E-3 + core.getParameterVector()
            core.setParameterVector(vec)
            self.assert_((core.getParameterVector()==vec).all())
            center = geom.PointD(*numpy.random.randn(2))
            ellipse = geomEllipse.Ellipse(core, center)
            self.assertClose(core.getParameterVector(), ellipse.getParameterVector()[:3])
            self.assertEqual(tuple(center), tuple(ellipse.getCenter()))
            self.assertEqual(geom.Point2D, type(ellipse.getCenter()))
            core.setParameterVector(numpy.random.randn(3))
            try:
                core.normalize()
            except:
                #tried to normalize a non-normalizable core
                pass

            ellipse.setCore(core)
            self.assertClose(core.getParameterVector(), ellipse.getCore().getParameterVector())
            self.assert_((core.clone().getParameterVector()==core.getParameterVector()).all())
            self.assert_(core is not core.clone())
            self.assert_((geomEllipse.Ellipse(ellipse).getParameterVector()==ellipse.getParameterVector()).all())
            self.assert_(ellipse is not geomEllipse.Ellipse(ellipse))

    def testTransform(self):
        for core in self.cores:
            transform = geom.LinearTransform(numpy.random.randn(2,2))
            t1 = core.transform(transform)
            core.transformInPlace(transform)
            self.assert_(t1 is not core)
            self.assertClose(t1.getParameterVector(), core.getParameterVector())

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

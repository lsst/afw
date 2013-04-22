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
import lsst.afw.geom.ellipses
import lsst.afw.image

numpy.random.seed(500)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class EllipseTestCase(unittest.TestCase):

    def setUp(self):
        self.cores = [
            lsst.afw.geom.ellipses.Axes(4, 3, 1*lsst.afw.geom.radians),
            lsst.afw.geom.ellipses.Quadrupole(5, 3, -1)
            ]
        self.classes = [lsst.afw.geom.ellipses.Axes, lsst.afw.geom.ellipses.Quadrupole]
        for s in lsst.afw.geom.ellipses.Separable.values():
            self.cores.append(s(0.5, 0.3, 2.1))
            self.classes.append(s)

    def assertClose(self, a, b):
        self.assert_(numpy.allclose(a, b), "%s != %s" % (a, b))

    def testRadii(self):
        for core, det, trace in zip(self.cores, [144, 14], [25, 8]):
            detRadius = det**0.25
            traceRadius = (0.5 * trace)**0.5
            area = numpy.pi * det**0.5
            self.assertClose(core.getDeterminantRadius(), detRadius)
            self.assertClose(core.getTraceRadius(), traceRadius)
            self.assertClose(core.getArea(), area)
            for cls in self.classes:
                conv = cls(core)
                self.assertClose(conv.getDeterminantRadius(), detRadius)
                self.assertClose(conv.getTraceRadius(), traceRadius)
                self.assertClose(conv.getArea(), area)
                conv.scale(3.0)
                self.assertClose(conv.getDeterminantRadius(), detRadius * 3)
                self.assertClose(conv.getTraceRadius(), traceRadius * 3)
                self.assertClose(conv.getArea(), area * 9)


    def testAccessors(self):
        for core in self.cores:
            vec = numpy.random.randn(3) * 1E-3 + core.getParameterVector()
            core.setParameterVector(vec)
            self.assert_((core.getParameterVector()==vec).all())
            center = lsst.afw.geom.Point2D(*numpy.random.randn(2))
            ellipse = lsst.afw.geom.ellipses.Ellipse(core, center)
            self.assertClose(core.getParameterVector(), ellipse.getParameterVector()[:3])
            self.assertEqual(tuple(center), tuple(ellipse.getCenter()))
            self.assertEqual(lsst.afw.geom.Point2D, type(ellipse.getCenter()))
            newcore = lsst.afw.geom.ellipses.Axes(1,2,3*lsst.afw.geom.radians)
            newcore.normalize()
            core.assign(newcore)
            ellipse.setCore(core)
            self.assertClose(core.getParameterVector(), ellipse.getCore().getParameterVector())
            self.assert_((core.clone().getParameterVector()==core.getParameterVector()).all())
            self.assert_(core is not core.clone())
            self.assert_((lsst.afw.geom.ellipses.Ellipse(ellipse).getParameterVector()
                          == ellipse.getParameterVector()).all())
            self.assert_(ellipse is not lsst.afw.geom.ellipses.Ellipse(ellipse))

    def testTransform(self):
        for core in self.cores:
            transform = lsst.afw.geom.LinearTransform(numpy.random.randn(2,2))
            t1 = core.transform(transform)
            core.transformInPlace(transform)
            self.assert_(t1 is not core)
            self.assertClose(t1.getParameterVector(), core.getParameterVector())

    def testPixelRegion(self):
        for core in self.cores:
            e = lsst.afw.geom.ellipses.Ellipse(core, lsst.afw.geom.Point2D(*numpy.random.randn(2)))
            region = lsst.afw.geom.ellipses.PixelRegion(e)
            bbox = region.getBBox()
            bbox.grow(2)
            array = numpy.zeros((bbox.getHeight(), bbox.getWidth()), dtype=bool)
            for span in region:
                for point in span:
                    adjusted = point - bbox.getMin()
                    array[adjusted.getY(), adjusted.getX()] = True
            gt = e.getGridTransform()
            for i in range(bbox.getBeginY(), bbox.getEndY()):
                for j in range(bbox.getBeginX(), bbox.getEndX()):
                    point = lsst.afw.geom.Point2I(j, i)
                    adjusted = point - bbox.getMin()
                    transformed = gt(lsst.afw.geom.Point2D(point))
                    r = (transformed.getX()**2 + transformed.getY()**2)**0.5
                    if array[adjusted.getY(), adjusted.getX()]:
                        self.assert_(r <= 1.0, "Point %s is in region but r=%f" % (point, r))
                    else:
                        self.assert_(r > 1.0, "Point %s is outside region but r=%f" % (point, r))

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

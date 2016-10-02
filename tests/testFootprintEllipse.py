#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import zip
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
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
#pybind11#
#pybind11#import math
#pybind11#import unittest
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.detection as afwDet
#pybind11#
#pybind11#
#pybind11#class FootprintTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def testCircle(self):
#pybind11#        xc, yc = 30, 50
#pybind11#        radius = 10
#pybind11#
#pybind11#        test = afwDet.Footprint(afwGeom.Point2I(xc, yc), radius)
#pybind11#
#pybind11#        # Here's how it used to be done using circles, before #1556
#pybind11#        r2 = int(radius**2 + 0.5)
#pybind11#        r = int(math.sqrt(r2))
#pybind11#        control = afwDet.Footprint()
#pybind11#        for i in range(-r, r+1):
#pybind11#            hlen = int(math.sqrt(r2 - i**2))
#pybind11#            control.addSpan(yc + i, xc - hlen, xc + hlen)
#pybind11#
#pybind11#        self.assertEqual(len(test.getSpans()), len(control.getSpans()))
#pybind11#        for s0, s1 in zip(test.getSpans(), control.getSpans()):
#pybind11#            self.assertEqual(s0.getX0(), s1.getX0())
#pybind11#            self.assertEqual(s0.getX1(), s1.getX1())
#pybind11#            self.assertEqual(s0.getY(), s1.getY())
#pybind11#        self.assertEqual(test.getNpix(), control.getNpix())
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

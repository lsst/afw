#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2014 LSST Corporation.
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
Tests for image.ApCorrMap

Run with:
   ./testApCorrMap.py
or
   python
   >>> import testSchema; testSchema.run()
"""

import sys
import os
import unittest
import numpy
import collections
import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.geom
import lsst.afw.math
import lsst.afw.image

try:
    type(display)
except NameError:
    display = False

numpy.random.seed(5)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ApCorrMapTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.bbox = lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(-5, -5), lsst.afw.geom.Point2I(5, 5))
        self.map = lsst.afw.image.ApCorrMap()
        for name in ("a", "b", "c"):
            self.map.set(name, lsst.afw.math.ChebyshevBoundedField(self.bbox, numpy.random.randn(3, 3)))

    def tearDown(self):
        del self.map
        del self.bbox

    def testAccessors(self):
        """Test the accessors and other custom Swig code we've added to make ApCorrMap behave like a Python
        mapping."""
        self.assertEqual(len(self.map), 3)
        self.assertEqual(collections.OrderedDict(self.map),
                         {name: value for (name, value) in self.map.items()})
        self.assertEqual(collections.OrderedDict(self.map).keys(), self.map.keys())
        self.assertEqual(collections.OrderedDict(self.map).values(), self.map.values())
        self.assertEqual(collections.OrderedDict(self.map).items(), self.map.items())
        self.assertEqual(collections.OrderedDict(self.map).keys(), list(self.map))
        self.assertTrue("b" in self.map)
        self.assertTrue("d" not in self.map)
        self.map["d"] = lsst.afw.math.ChebyshevBoundedField(self.bbox, numpy.random.randn(2, 2))
        self.assertTrue("d" in self.map)
        self.assertIsNone(self.map.get("e"))
        self.assertRaisesLsstCpp(lsst.pex.exceptions.NotFoundException, self.map.__getitem__, "e")
        self.assertEqual(self.map.get("d"), self.map["d"])

    def testPersistence(self):
        """Test that we can round-trip an ApCorrMap through FITS persistence."""
        filename = "testApCorrMap.fits"
        self.map.writeFits(filename)
        map2 = lsst.afw.image.ApCorrMap.readFits(filename)
        for name, value in self.map.items():
            value2 = map2.get(name)
            self.assertIsNotNone(value2)
            self.assertEqual(value.getBBox(), value2.getBBox())
            self.assertClose(lsst.afw.math.ChebyshevBoundedField.cast(value).getCoefficients(),
                             lsst.afw.math.ChebyshevBoundedField.cast(value2).getCoefficients(),
                             rtol=0.0)
        os.remove(filename)

    def testExposurePersistence(self):
        """Test that the ApCorrMap is saved with an Exposure"""
        filename = "testApCorrMap.fits"
        exposure1 = lsst.afw.image.ExposureF(self.bbox)
        exposure1.getInfo().setApCorrMap(self.map)
        exposure1.writeFits(filename)
        exposure2 = lsst.afw.image.ExposureF(filename)
        map2 = exposure2.getInfo().getApCorrMap()
        for name, value in self.map.items():
            value2 = map2.get(name)
            self.assertIsNotNone(value2)
            self.assertEqual(value.getBBox(), value2.getBBox())
            self.assertClose(lsst.afw.math.ChebyshevBoundedField.cast(value).getCoefficients(),
                             lsst.afw.math.ChebyshevBoundedField.cast(value2).getCoefficients(),
                             rtol=0.0)
        os.remove(filename)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(ApCorrMapTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

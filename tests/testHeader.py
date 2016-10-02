#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008-2013 LSST Corporation.
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
#pybind11#import numpy
#pybind11#import unittest
#pybind11#from past.builtins import long
#pybind11#
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.utils.tests
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#class HeaderTestCase(lsst.utils.tests.TestCase):
#pybind11#    """Test that headers round-trip"""
#pybind11#
#pybind11#    def testHeaders(self):
#pybind11#        filename = "tests/header.fits"
#pybind11#        header = {"STR": "String",
#pybind11#                  "INT": 12345,
#pybind11#                  "FLOAT": 678.9,
#pybind11#                  "NAN": numpy.nan,
#pybind11#                  "PLUSINF": numpy.inf,
#pybind11#                  "MINUSINF": -numpy.inf,
#pybind11#                  "LONG": long(987654321),
#pybind11#                  }
#pybind11#
#pybind11#        exp = afwImage.ExposureI(0, 0)
#pybind11#        metadata = exp.getMetadata()
#pybind11#        for k, v in header.items():
#pybind11#            metadata.add(k, v)
#pybind11#
#pybind11#        exp.writeFits(filename)
#pybind11#
#pybind11#        exp = afwImage.ExposureI(filename)
#pybind11#        metadata = exp.getMetadata()
#pybind11#        for k, v in header.items():
#pybind11#            self.assertTrue(metadata.exists(k))
#pybind11#            if isinstance(v, float) and numpy.isnan(v):
#pybind11#                self.assertIsInstance(metadata.get(k), float)
#pybind11#                self.assertTrue(numpy.isnan(metadata.get(k)))
#pybind11#            else:
#pybind11#                self.assertEqual(metadata.get(k), v)
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()

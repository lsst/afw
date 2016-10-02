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
#pybind11#import os
#pybind11#import os.path
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.utils.tests
#pybind11#
#pybind11#testPath = os.path.abspath(os.path.dirname(__file__))
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#class Ticket2905Test(unittest.TestCase):
#pybind11#    """Test reading a FITS header that contains:
#pybind11#
#pybind11#    INR-STR =                2E-05
#pybind11#    """
#pybind11#
#pybind11#    def test(self):
#pybind11#        path = os.path.join(testPath, "data", "ticket2905.fits")
#pybind11#        md = afwImage.readMetadata(path)
#pybind11#        value = md.get("INR-STR")
#pybind11#        self.assertEqual(type(value), float)
#pybind11#        self.assertEqual(value, 2.0e-5)
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

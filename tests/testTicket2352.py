#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
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
#pybind11#import os
#pybind11#import os.path
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.utils.tests
#pybind11#
#pybind11#testPath = os.path.abspath(os.path.dirname(__file__))
#pybind11#DATA = os.path.join(testPath, "data", "ticket2352.fits")
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#class ReadMefTest(unittest.TestCase):
#pybind11#    """Test the reading of a multi-extension FITS (MEF) file"""
#pybind11#
#pybind11#    def checkExtName(self, name, value, extNum):
#pybind11#        filename = DATA + "[%s]" % name
#pybind11#
#pybind11#        header = afwImage.readMetadata(filename)
#pybind11#        self.assertEqual(header.get("EXT_NUM"), extNum)
#pybind11#        self.assertEqual(header.get("EXTNAME").strip(), name)
#pybind11#
#pybind11#        image = afwImage.ImageI(filename)
#pybind11#        self.assertEqual(image.get(0, 0), value)
#pybind11#
#pybind11#    def testExtName(self):
#pybind11#        self.checkExtName("ONE", 1, 2)
#pybind11#        self.checkExtName("TWO", 2, 3)
#pybind11#        self.checkExtName("THREE", 3, 4)
#pybind11#
#pybind11#    def checkExtNum(self, hdu, extNum):
#pybind11#        header = afwImage.readMetadata(DATA, hdu)
#pybind11#        self.assertEqual(header.get("EXT_NUM"), extNum)
#pybind11#
#pybind11#    def testExtNum(self):
#pybind11#        self.checkExtNum(0, 2)  # Should skip PHU
#pybind11#        self.checkExtNum(1, 1)
#pybind11#        self.checkExtNum(2, 2)
#pybind11#        self.checkExtNum(3, 3)
#pybind11#        self.checkExtNum(4, 4)
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

#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2016 LSST Corporation.
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
#pybind11#import unittest
#pybind11#
#pybind11#import numpy
#pybind11#import astropy.io.fits
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom
#pybind11#import lsst.afw.table
#pybind11#
#pybind11#
#pybind11#class TableIoTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def testAngleUnitWriting(self):
#pybind11#        """Test that Angle columns have TUNIT set appropriately,
#pybind11#        as per DM-7221.
#pybind11#        """
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        key = schema.addField("a", type="Angle", doc="angle field")
#pybind11#        outCat = lsst.afw.table.BaseCatalog(schema)
#pybind11#        outCat.addNew().set(key, 1.0*lsst.afw.geom.degrees)
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            outCat.writeFits(tmpFile)
#pybind11#            inFits = astropy.io.fits.open(tmpFile)
#pybind11#            self.assertEqual(inFits[1].header["TTYPE1"], "a")
#pybind11#            self.assertEqual(inFits[1].header["TUNIT1"], "rad")
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

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
#pybind11#import unittest
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.daf.base as dafBase
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#
#pybind11#
#pybind11#class TanSipTestCases(unittest.TestCase):
#pybind11#    """Tests for the existence of the bug reported in #835
#pybind11#       (Wcs class doesn't gracefully handle the case of ctypes
#pybind11#       having -SIP appended to them).
#pybind11#    """
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        # metadata taken from CFHT data
#pybind11#        # v695856-e0/v695856-e0-c000-a00.sci_img.fits
#pybind11#
#pybind11#        metadata = dafBase.PropertySet()
#pybind11#
#pybind11#        metadata.set("SIMPLE", "T")
#pybind11#        metadata.set("BITPIX", -32)
#pybind11#        metadata.set("NAXIS", 2)
#pybind11#        metadata.set("NAXIS1", 1024)
#pybind11#        metadata.set("NAXIS2", 1153)
#pybind11#        metadata.set("RADECSYS", 'FK5')
#pybind11#        metadata.set("EQUINOX", 2000.)
#pybind11#
#pybind11#        metadata.setDouble("CRVAL1", 215.604025685476)
#pybind11#        metadata.setDouble("CRVAL2", 53.1595451514076)
#pybind11#        metadata.setDouble("CRPIX1", 1109.99981456774)
#pybind11#        metadata.setDouble("CRPIX2", 560.018167811613)
#pybind11#        metadata.set("CTYPE1", 'RA---TAN-SIP')
#pybind11#        metadata.set("CTYPE2", 'DEC--TAN-SIP')
#pybind11#
#pybind11#        metadata.setDouble("CD1_1", 5.10808596133527E-05)
#pybind11#        metadata.setDouble("CD1_2", 1.85579539217196E-07)
#pybind11#        metadata.setDouble("CD2_2", -5.10281493481982E-05)
#pybind11#        metadata.setDouble("CD2_1", -8.27440751733828E-07)
#pybind11#
#pybind11#        self.metadata = metadata
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.metadata
#pybind11#
#pybind11#    def testExcept(self):
#pybind11#        with self.assertRaises(pexExcept.Exception):
#pybind11#            afwImage.makeWcs(self.metadata)
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

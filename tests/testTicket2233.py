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
#pybind11#"""
#pybind11#Sogo Mineo writes:
#pybind11#
#pybind11#'''
#pybind11#If I read Wcs from, e.g., the following file:
#pybind11#   master:/data1a/Subaru/SUPA/rerun/mineo-Abell1689/03430/W-S-I+/corr/wcs01098593.fits
#pybind11#
#pybind11#then Wcs::_nWcsInfo becomes 2.
#pybind11#
#pybind11#But WcsFormatter assumes that Wcs::_nWcsInfo is 1.
#pybind11#
#pybind11#When the stacking program tries bcasting Wcs:
#pybind11#    - In serializing Wcs, the value _nWcsInfo = 2 is recorded and so read in
#pybind11#deserialization.
#pybind11#    - But in the deserialization, the formatter allocates only a single
#pybind11#element of _wcsInfo.
#pybind11#
#pybind11#It causes inconsistency at the destructor, and SEGV arrises.
#pybind11#'''
#pybind11#
#pybind11#The example file above has been copied and is used in the below test.
#pybind11#"""
#pybind11#
#pybind11#import os
#pybind11#import os.path
#pybind11#import unittest
#pybind11#import pickle
#pybind11#
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.utils.tests
#pybind11#
#pybind11#testPath = os.path.abspath(os.path.dirname(__file__))
#pybind11#
#pybind11#DATA = os.path.join(testPath, "data", "ticket2233.fits")
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#class WcsFormatterTest(unittest.TestCase):
#pybind11#    """Test the WCS formatter, by round-trip pickling."""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        exposure = afwImage.ExposureF(DATA)
#pybind11#        self.wcs = exposure.getWcs()
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.wcs
#pybind11#
#pybind11#    def testFormat(self):
#pybind11#        dumped = pickle.dumps(self.wcs)
#pybind11#        wcs = pickle.loads(dumped)
#pybind11#        self.assertEqual(wcs.getFitsMetadata().toString(), self.wcs.getFitsMetadata().toString())
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
